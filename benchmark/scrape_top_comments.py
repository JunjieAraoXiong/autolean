#!/usr/bin/env python3
"""Scrape forum comments for top Erdos problems and enrich corpus files.

Scrapes erdosproblems.com forum threads for a curated list of 50 problems,
adds expert_comments to the corresponding corpus JSON files, and prints
a summary of results including Tao comment counts.
"""

import json
import glob
import sys
import time
from pathlib import Path

# Add parent so we can import from build_erdos_corpus
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_erdos_corpus import scrape_comments

CORPUS_DIR = Path(__file__).resolve().parent / "erdos_corpus"

# ── Priority lists ──────────────────────────────────────────────────────────

AI_SOLVED = [728, 481, 124, 333, 205, 401, 729]

GPT_ERDOS_FINDINGS = [281, 397, 652, 591, 847, 1129, 1130, 78, 91, 274]

def _pick_open_nt_comb(already: set[int], need: int) -> list[int]:
    """Pick *need* more problem numbers from the corpus that are open and
    tagged 'number theory' or 'combinatorics', skipping those in *already*."""
    extras: list[int] = []
    for path in sorted(CORPUS_DIR.glob("erdos_*.json")):
        if len(extras) >= need:
            break
        with open(path) as fh:
            rec = json.load(fh)
        num = rec.get("erdos_number")
        if not isinstance(num, int) or num in already:
            continue
        status = (rec.get("status") or "").lower()
        tags = [t.lower() for t in rec.get("tags", [])]
        if "open" in status and ("number theory" in tags or "combinatorics" in tags):
            extras.append(num)
    return extras


def build_problem_list() -> list[int]:
    """Return a deduplicated, ordered list of 50 problem numbers to scrape."""
    seen: set[int] = set()
    ordered: list[int] = []
    for n in AI_SOLVED + GPT_ERDOS_FINDINGS:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    remaining = 50 - len(ordered)
    extras = _pick_open_nt_comb(seen, remaining)
    ordered.extend(extras)
    return ordered[:50]


def main() -> int:
    problems = build_problem_list()
    print(f"Scraping comments for {len(problems)} problems\n")
    print(f"  AI-solved   : {AI_SOLVED}")
    print(f"  gpt-erdos   : {GPT_ERDOS_FINDINGS}")
    print(f"  open NT/comb: {[p for p in problems if p not in AI_SOLVED and p not in GPT_ERDOS_FINDINGS]}")
    print()

    total_comments = 0
    problems_with_tao = 0
    results: list[dict] = []

    for idx, num in enumerate(problems, 1):
        print(f"[{idx:2d}/50] Problem {num} ... ", end="", flush=True)
        comments = scrape_comments(num)
        n_comments = len(comments)
        total_comments += n_comments

        authors = sorted({c.get("author", "?") for c in comments}) if comments else []
        tao_count = sum(1 for c in comments if "tao" in c.get("author", "").lower())
        if tao_count > 0:
            problems_with_tao += 1

        # Print per-problem line
        if n_comments == 0:
            print("no comments")
        else:
            author_str = ", ".join(authors)
            tao_note = f" (Tao: {tao_count})" if tao_count else ""
            print(f"{n_comments} comments{tao_note}  authors=[{author_str}]")

        # Enrich corpus file if comments found
        corpus_path = CORPUS_DIR / f"erdos_{num}.json"
        if n_comments > 0 and corpus_path.exists():
            with open(corpus_path) as fh:
                rec = json.load(fh)
            rec["expert_comments"] = comments
            with open(corpus_path, "w") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)

        results.append({
            "problem": num,
            "comments": n_comments,
            "tao_comments": tao_count,
            "authors": authors,
        })

        # Polite delay between requests
        if idx < len(problems):
            time.sleep(1.5)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Problems scraped       : {len(problems)}")
    print(f"  Total comments found   : {total_comments}")
    problems_with_any = sum(1 for r in results if r["comments"] > 0)
    print(f"  Problems with comments : {problems_with_any}")
    print(f"  Problems with Tao      : {problems_with_tao}")
    total_tao = sum(r["tao_comments"] for r in results)
    print(f"  Total Tao comments     : {total_tao}")

    # Top commented problems
    by_count = sorted(results, key=lambda r: r["comments"], reverse=True)
    print("\n  Top 10 most-commented:")
    for r in by_count[:10]:
        tao_note = f" (Tao: {r['tao_comments']})" if r["tao_comments"] else ""
        print(f"    #{r['problem']:>5d} : {r['comments']:3d} comments{tao_note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
