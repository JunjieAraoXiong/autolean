#!/usr/bin/env python3
"""Convert gpt-erdos and erdosproblems.com data into AUTOLEAN benchmark format.

Data sources:
1. gpt-erdos unsolved.jsonl (675 problems with LaTeX from erdosproblems.com)
2. gpt-erdos solutions/ (677 dirs with candidate_solution.md + .lean)
3. erdosproblems.com/latex/{n} endpoint (1184 problems total, 502 solved)

Usage:
    # Convert gpt-erdos unsolved.jsonl → AUTOLEAN JSON files
    python benchmark/convert_erdos_dataset.py \
        --source /path/to/gpt-erdos/data/unsolved.jsonl \
        --output benchmark/problems_full/ \
        --limit 50

    # Also include solved problems from gpt-erdos solutions/
    python benchmark/convert_erdos_dataset.py \
        --source /path/to/gpt-erdos/data/unsolved.jsonl \
        --solutions /path/to/gpt-erdos/data/solutions/ \
        --output benchmark/problems_full/

    # Scrape all problems directly from erdosproblems.com
    python benchmark/convert_erdos_dataset.py \
        --scrape --output benchmark/problems_full/ --limit 100
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
from urllib.request import urlopen


def latex_to_natural_language(latex: str) -> str:
    """Best-effort conversion of LaTeX math to readable natural language.

    Not perfect, but good enough for LLM consumption. The LLM will interpret
    both LaTeX and natural language, so partial conversion is fine.
    """
    text = latex
    # Remove display math delimiters
    text = re.sub(r'\\\[', '', text)
    text = re.sub(r'\\\]', '', text)
    text = re.sub(r'\$\$', '', text)
    # Keep inline math markers for LLM readability
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    # Common LaTeX commands
    text = text.replace(r'\lvert', '|').replace(r'\rvert', '|')
    text = text.replace(r'\lfloor', '⌊').replace(r'\rfloor', '⌋')
    text = text.replace(r'\lceil', '⌈').replace(r'\rceil', '⌉')
    text = text.replace(r'\leq', '≤').replace(r'\geq', '≥')
    text = text.replace(r'\neq', '≠')
    text = text.replace(r'\infty', '∞')
    text = text.replace(r'\cdots', '⋯').replace(r'\ldots', '…')
    text = text.replace(r'\cdot', '·')
    text = text.replace(r'\times', '×')
    text = text.replace(r'\subseteq', '⊆').replace(r'\subset', '⊂')
    text = text.replace(r'\cup', '∪').replace(r'\cap', '∩')
    text = text.replace(r'\in', '∈').replace(r'\notin', '∉')
    text = text.replace(r'\to', '→').replace(r'\rightarrow', '→')
    text = text.replace(r'\implies', '⟹')
    text = text.replace(r'\forall', '∀').replace(r'\exists', '∃')
    text = text.replace(r'\sum', '∑').replace(r'\prod', '∏')
    text = text.replace(r'\mathbb{N}', 'ℕ').replace(r'\mathbb{Z}', 'ℤ')
    text = text.replace(r'\mathbb{R}', 'ℝ').replace(r'\mathbb{Q}', 'ℚ')
    # Erdős-specific
    text = re.sub(r"Erd\\H\{o\}s", "Erdős", text)
    # Clean up remaining backslashes for common commands
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\operatorname\{([^}]+)\}', r'\1', text)
    # Fractions
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\tfrac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    return text.strip()


def convert_unsolved_jsonl(
    jsonl_path: Path,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
    solutions_dir: Optional[Path] = None,
) -> int:
    """Convert gpt-erdos unsolved.jsonl into AUTOLEAN JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            number = record.get("number", "")
            latex = record.get("latex", "")
            additional = record.get("additional_text", "")

            if not latex:
                continue

            # Convert to AUTOLEAN format
            problem_text = latex_to_natural_language(latex)
            problem_lines = [problem_text]
            if additional:
                problem_lines.append(latex_to_natural_language(additional))

            autolean_json = {
                "uuid": f"erdos_{number}",
                "problem": problem_lines,
                "source": "erdosproblems.com",
                "erdos_number": int(number) if number.isdigit() else number,
                "original_latex": latex,
            }

            # Check if gpt-erdos has a solution
            if solutions_dir:
                sol_dir = solutions_dir / str(number)
                lean_path = sol_dir / "candidate_solution.lean"
                md_path = sol_dir / "candidate_solution.md"
                if lean_path.exists():
                    autolean_json["reference_lean"] = lean_path.read_text(encoding="utf-8")
                if md_path.exists():
                    # Store first 500 chars as hint
                    md_text = md_path.read_text(encoding="utf-8")
                    autolean_json["reference_proof_hint"] = md_text[:500]

            out_path = output_dir / f"erdos_{number}.json"
            out_path.write_text(
                json.dumps(autolean_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            count += 1

            if limit and count >= limit:
                break

    return count


class LatexExtractor(HTMLParser):
    """Extract LaTeX content from erdosproblems.com/latex/{n} pages."""

    def __init__(self):
        super().__init__()
        self._in_content = False
        self._depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div" and attrs_dict.get("id") == "content":
            self._in_content = True
            self._depth = 1
        elif self._in_content and tag == "div":
            self._depth += 1
        elif tag == "br" and self._in_content:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if self._in_content and tag == "div":
            self._depth -= 1
            if self._depth <= 0:
                self._in_content = False

    def handle_data(self, data):
        if self._in_content:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts).strip()


def scrape_problem(number: int, timeout: float = 15.0) -> Optional[dict]:
    """Scrape a single problem from erdosproblems.com."""
    url = f"https://www.erdosproblems.com/latex/{number}"
    try:
        with urlopen(url, timeout=timeout) as resp:
            html = resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None

    parser = LatexExtractor()
    parser.feed(html)
    text = parser.get_text()
    if not text:
        return None

    return {
        "uuid": f"erdos_{number}",
        "problem": [latex_to_natural_language(text)],
        "source": "erdosproblems.com",
        "erdos_number": number,
        "original_latex": text,
    }


def scrape_all(
    output_dir: Path,
    *,
    max_number: int = 1200,
    limit: Optional[int] = None,
    delay: float = 1.0,
) -> int:
    """Scrape problems directly from erdosproblems.com."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for n in range(1, max_number + 1):
        # Skip if already exists
        out_path = output_dir / f"erdos_{n}.json"
        if out_path.exists():
            count += 1
            if limit and count >= limit:
                break
            continue

        print(f"  [{n}/{max_number}] Scraping...", end="", flush=True)
        problem = scrape_problem(n)
        if problem is None:
            print(" skipped (not found)")
            continue

        out_path.write_text(
            json.dumps(problem, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1
        print(f" OK ({len(problem['problem'][0])} chars)")

        if limit and count >= limit:
            break
        if delay > 0:
            time.sleep(delay)

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Erdős problems to AUTOLEAN format")
    parser.add_argument("--source", type=Path, help="Path to gpt-erdos unsolved.jsonl")
    parser.add_argument("--solutions", type=Path, help="Path to gpt-erdos solutions/ dir")
    parser.add_argument("--output", type=Path, default=Path("benchmark/problems_full/"))
    parser.add_argument("--limit", type=int, default=None, help="Max problems to convert")
    parser.add_argument("--scrape", action="store_true", help="Scrape from erdosproblems.com directly")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between scrape requests")
    args = parser.parse_args()

    if args.scrape:
        print(f"Scraping from erdosproblems.com → {args.output}")
        count = scrape_all(args.output, limit=args.limit, delay=args.delay)
        print(f"\nScraped {count} problems")
        return 0

    if args.source:
        print(f"Converting {args.source} → {args.output}")
        count = convert_unsolved_jsonl(
            args.source, args.output,
            limit=args.limit,
            solutions_dir=args.solutions,
        )
        print(f"\nConverted {count} problems")
        return 0

    print("Provide --source or --scrape", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
