#!/usr/bin/env python3
"""Build comprehensive Erdős benchmark corpus from all available sources.

Merges data from:
1. Tao's erdosproblems GitHub (problems.yaml) — metadata, tags, status, formalization state
2. gpt-erdos dataset (unsolved.jsonl) — LaTeX problem statements
3. gpt-erdos solutions/ — GPT 5.2 Pro candidate proofs + Lean formalizations
4. erdosproblems.com comments — expert discussions, partial results, Tao's comments

Output: AUTOLEAN-compatible JSON files + corpus summary

Usage:
    python benchmark/build_erdos_corpus.py \
        --tao-yaml /path/to/erdosproblems/data/problems.yaml \
        --gpt-erdos-jsonl /path/to/gpt-erdos/data/unsolved.jsonl \
        --gpt-erdos-solutions /path/to/gpt-erdos/data/solutions/ \
        --output benchmark/erdos_corpus/ \
        --scrape-comments
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ErdosBenchmark/1.0"

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# LaTeX → readable text (best effort)
# ---------------------------------------------------------------------------

def latex_to_text(latex: str) -> str:
    text = latex
    text = re.sub(r'\\\[', '', text)
    text = re.sub(r'\\\]', '', text)
    text = re.sub(r'\$\$', '', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    text = text.replace(r'\lvert', '|').replace(r'\rvert', '|')
    text = text.replace(r'\leq', '≤').replace(r'\geq', '≥')
    text = text.replace(r'\neq', '≠').replace(r'\infty', '∞')
    text = text.replace(r'\subseteq', '⊆').replace(r'\subset', '⊂')
    text = text.replace(r'\cup', '∪').replace(r'\cap', '∩')
    text = text.replace(r'\in', '∈').replace(r'\to', '→')
    text = text.replace(r'\forall', '∀').replace(r'\exists', '∃')
    text = text.replace(r'\sum', '∑').replace(r'\prod', '∏')
    text = text.replace(r'\mathbb{N}', 'ℕ').replace(r'\mathbb{Z}', 'ℤ')
    text = text.replace(r'\mathbb{R}', 'ℝ').replace(r'\mathbb{Q}', 'ℚ')
    text = re.sub(r"Erd\\H\{o\}s", "Erdős", text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Comment scraper for erdosproblems.com
# ---------------------------------------------------------------------------

class CommentExtractor(HTMLParser):
    """Extract comments/discussion from an erdosproblems.com problem page."""

    def __init__(self):
        super().__init__()
        self._in_comment = False
        self._comment_depth = 0
        self._parts: list[str] = []
        self.comments: list[dict] = []
        self._current_author = ""
        self._in_author = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = (attrs_dict.get("class") or "").split()

        if tag == "div" and "post-body" in classes:
            self._in_comment = True
            self._comment_depth = 1
            self._parts = []
        elif self._in_comment and tag == "div":
            self._comment_depth += 1
        elif tag == "a" and "post-author" in classes:
            self._in_author = True
        elif tag == "br" and self._in_comment:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if self._in_comment and tag == "div":
            self._comment_depth -= 1
            if self._comment_depth <= 0:
                self._in_comment = False
                text = "".join(self._parts).strip()
                if text:
                    self.comments.append({
                        "author": self._current_author,
                        "text": text,
                    })
                self._parts = []
        if tag == "a" and self._in_author:
            self._in_author = False

    def handle_data(self, data):
        if self._in_comment:
            self._parts.append(data)
        if self._in_author:
            self._current_author = data.strip()


def scrape_comments(problem_number: int, timeout: float = 15.0) -> list[dict]:
    """Scrape discussion comments from a problem's forum thread."""
    url = f"https://www.erdosproblems.com/forum/thread/{problem_number}"
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8")
    except Exception:
        return []

    # Parse post-meta and post-body pairs
    class ThreadParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_meta = False
            self.in_body = False
            self.meta_depth = 0
            self.body_depth = 0
            self.meta_parts: list[str] = []
            self.body_parts: list[str] = []
            self.results: list[dict] = []

        def handle_starttag(self, tag, attrs):
            cls = dict(attrs).get("class", "")
            if "post-meta" in cls.split():
                self.in_meta = True
                self.meta_depth = 1
                self.meta_parts = []
            elif self.in_meta and tag == "div":
                self.meta_depth += 1
            if "post-body" in cls.split():
                self.in_body = True
                self.body_depth = 1
                self.body_parts = []
            elif self.in_body and tag == "div":
                self.body_depth += 1
            if tag == "br" and self.in_body:
                self.body_parts.append("\n")

        def handle_endtag(self, tag):
            if self.in_meta and tag == "div":
                self.meta_depth -= 1
                if self.meta_depth <= 0:
                    self.in_meta = False
            if self.in_body and tag == "div":
                self.body_depth -= 1
                if self.body_depth <= 0:
                    self.in_body = False
                    meta = " ".join("".join(self.meta_parts).split()).strip()
                    body = "".join(self.body_parts).strip()
                    # Author is before the em dash (—) or any dash-like separator
                    author = meta
                    for sep in ("\u2014", "--", " —"):
                        if sep in meta:
                            author = meta.split(sep)[0].strip()
                            break
                    if body:
                        self.results.append({"author": author, "text": body[:1000]})

        def handle_data(self, data):
            if self.in_meta:
                self.meta_parts.append(data)
            if self.in_body:
                self.body_parts.append(data)

    parser = ThreadParser()
    parser.feed(html)
    return parser.results


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def load_tao_metadata(yaml_path: Path) -> dict[str, dict]:
    """Load problems.yaml into a dict keyed by problem number."""
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return {str(p.get("number", "")): p for p in data}


def load_gpt_erdos_latex(jsonl_path: Path) -> dict[str, dict]:
    """Load gpt-erdos unsolved.jsonl into a dict keyed by problem number."""
    result = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            number = str(record.get("number", ""))
            result[number] = record
    return result


def build_corpus(
    *,
    tao_yaml: Optional[Path] = None,
    gpt_erdos_jsonl: Optional[Path] = None,
    gpt_erdos_solutions: Optional[Path] = None,
    output_dir: Path,
    do_scrape_comments: bool = False,
    limit: Optional[int] = None,
    filter_tags: Optional[list[str]] = None,
    filter_status: Optional[list[str]] = None,
    delay: float = 1.0,
) -> dict:
    """Build the full corpus by merging all sources."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sources
    tao_data = load_tao_metadata(tao_yaml) if tao_yaml else {}
    gpt_data = load_gpt_erdos_latex(gpt_erdos_jsonl) if gpt_erdos_jsonl else {}

    # Determine problem numbers to process
    all_numbers = set(tao_data.keys()) | set(gpt_data.keys())
    # Sort numerically
    sorted_numbers = sorted(all_numbers, key=lambda x: int(x) if x.isdigit() else 99999)

    stats = {
        "total": 0, "open": 0, "proved": 0, "formalized": 0,
        "has_latex": 0, "has_lean": 0, "has_comments": 0,
        "by_tag": {}, "by_status": {},
    }
    count = 0

    for number in sorted_numbers:
        tao = tao_data.get(number, {})
        gpt = gpt_data.get(number, {})

        # Apply filters
        status_state = tao.get("status", {}).get("state", "unknown")
        tags = tao.get("tags", [])

        if filter_status and status_state not in filter_status:
            continue
        if filter_tags and not any(t in tags for t in filter_tags):
            continue

        # Build merged record
        problem_text = gpt.get("latex", "")
        additional_text = gpt.get("additional_text", "")

        record = {
            "uuid": f"erdos_{number}",
            "problem": [latex_to_text(problem_text)] if problem_text else [f"Erdős Problem #{number}"],
            "source": "erdosproblems.com",
            "erdos_number": int(number) if number.isdigit() else number,
            "status": status_state,
            "tags": tags,
            "prize": tao.get("prize", "no"),
            "formalized_on_site": tao.get("formalized", {}).get("state", "no") == "yes",
        }

        if problem_text:
            record["original_latex"] = problem_text
            stats["has_latex"] += 1
        if additional_text:
            record["additional_context"] = latex_to_text(additional_text)

        # Add gpt-erdos solutions if available
        if gpt_erdos_solutions:
            sol_dir = gpt_erdos_solutions / str(number)
            lean_path = sol_dir / "candidate_solution.lean"
            md_path = sol_dir / "candidate_solution.md"
            if lean_path.exists():
                record["reference_lean"] = lean_path.read_text(encoding="utf-8")
                stats["has_lean"] += 1
            if md_path.exists():
                md_text = md_path.read_text(encoding="utf-8")
                record["reference_proof_hint"] = md_text[:1000]

        # Scrape comments if requested
        if do_scrape_comments and number.isdigit():
            print(f"  [{number}] Scraping comments...", end="", flush=True)
            comments = scrape_comments(int(number))
            if comments:
                record["expert_comments"] = comments
                stats["has_comments"] += 1
                print(f" {len(comments)} comments")
            else:
                print(" none")
            time.sleep(delay)

        # Update stats
        stats["total"] += 1
        stats["by_status"][status_state] = stats["by_status"].get(status_state, 0) + 1
        for tag in tags:
            stats["by_tag"][tag] = stats["by_tag"].get(tag, 0) + 1
        if "open" in status_state:
            stats["open"] += 1
        if "proved" in status_state or "solved" in status_state:
            stats["proved"] += 1
        if record.get("formalized_on_site"):
            stats["formalized"] += 1

        # Write individual file
        out_path = output_dir / f"erdos_{number}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1

        if limit and count >= limit:
            break

    # Write corpus summary
    summary_path = output_dir / "_corpus_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build comprehensive Erdős benchmark corpus")
    parser.add_argument("--tao-yaml", type=Path, help="Path to erdosproblems/data/problems.yaml")
    parser.add_argument("--gpt-erdos-jsonl", type=Path, help="Path to gpt-erdos/data/unsolved.jsonl")
    parser.add_argument("--gpt-erdos-solutions", type=Path, help="Path to gpt-erdos/data/solutions/")
    parser.add_argument("--output", type=Path, default=Path("benchmark/erdos_corpus/"))
    parser.add_argument("--scrape-comments", action="store_true", help="Scrape discussion comments from erdosproblems.com")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tags", nargs="*", help="Filter by tags (e.g., 'number theory' 'combinatorics')")
    parser.add_argument("--status", nargs="*", help="Filter by status (e.g., 'open' 'proved')")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between scrape requests")
    args = parser.parse_args()

    if not args.tao_yaml and not args.gpt_erdos_jsonl:
        print("Provide at least --tao-yaml or --gpt-erdos-jsonl", file=sys.stderr)
        return 1

    print(f"Building Erdős corpus → {args.output}")
    stats = build_corpus(
        tao_yaml=args.tao_yaml,
        gpt_erdos_jsonl=args.gpt_erdos_jsonl,
        gpt_erdos_solutions=args.gpt_erdos_solutions,
        output_dir=args.output,
        do_scrape_comments=args.scrape_comments,
        limit=args.limit,
        filter_tags=args.tags,
        filter_status=args.status,
        delay=args.delay,
    )

    print(f"\nCorpus built:")
    print(f"  Total: {stats['total']}")
    print(f"  Open: {stats['open']}")
    print(f"  Proved: {stats['proved']}")
    print(f"  Formalized: {stats['formalized']}")
    print(f"  Has LaTeX: {stats['has_latex']}")
    print(f"  Has Lean: {stats['has_lean']}")
    print(f"  Has comments: {stats['has_comments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
