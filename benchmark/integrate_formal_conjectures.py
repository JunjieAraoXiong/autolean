#!/usr/bin/env python3
"""
Integrate ground-truth Lean formalizations from google-deepmind/formal-conjectures
into our Erdos benchmark corpus.

For each .lean file in FormalConjectures/ErdosProblems/, this script:
  1. Extracts the problem number from the filename (e.g. 728.lean -> 728)
  2. Reads the full Lean source code
  3. Finds the matching erdos_{number}.json in benchmark/erdos_corpus/
  4. Adds a "ground_truth_lean" field containing the Lean code
  5. Writes the updated JSON back

Usage:
    python integrate_formal_conjectures.py
"""

import json
import os
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FORMAL_CONJECTURES_DIR = SCRIPT_DIR.parent.parent / "formal-conjectures"
ERDOS_LEAN_DIR = FORMAL_CONJECTURES_DIR / "FormalConjectures" / "ErdosProblems"
ERDOS_CORPUS_DIR = SCRIPT_DIR / "erdos_corpus"


def main():
    # Sanity checks
    if not ERDOS_LEAN_DIR.is_dir():
        print(f"ERROR: Lean source directory not found: {ERDOS_LEAN_DIR}")
        sys.exit(1)
    if not ERDOS_CORPUS_DIR.is_dir():
        print(f"ERROR: Corpus directory not found: {ERDOS_CORPUS_DIR}")
        sys.exit(1)

    # Collect all .lean files and extract problem numbers
    lean_files = sorted(ERDOS_LEAN_DIR.glob("*.lean"))
    print(f"Found {len(lean_files)} .lean files in {ERDOS_LEAN_DIR}\n")

    matched = 0
    unmatched_lean = []  # Lean files with no corresponding JSON
    updated_problems = []

    for lean_path in lean_files:
        stem = lean_path.stem  # e.g. "728"
        if not stem.isdigit():
            continue  # skip non-numeric files like README.md

        problem_number = stem
        json_path = ERDOS_CORPUS_DIR / f"erdos_{problem_number}.json"

        if not json_path.exists():
            unmatched_lean.append(problem_number)
            continue

        # Read the Lean source
        lean_code = lean_path.read_text(encoding="utf-8")

        # Read the existing JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Add ground-truth Lean field
        data["ground_truth_lean"] = lean_code

        # Write back
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        matched += 1
        updated_problems.append(problem_number)

    # ── Summary ──────────────────────────────────────────────────────────
    total_lean = len([f for f in lean_files if f.stem.isdigit()])
    total_corpus = len(list(ERDOS_CORPUS_DIR.glob("erdos_*.json")))

    print("=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)
    print(f"Lean files scanned:          {total_lean}")
    print(f"Corpus JSON files:           {total_corpus}")
    print(f"Matched & updated:           {matched}")
    print(f"Lean with no corpus match:   {len(unmatched_lean)}")
    print(f"Corpus without Lean:         {total_corpus - matched}")
    print("=" * 60)

    if unmatched_lean:
        print(f"\nUnmatched Lean problem numbers ({len(unmatched_lean)}):")
        # Print in rows of 15 for readability
        for i in range(0, len(unmatched_lean), 15):
            row = unmatched_lean[i : i + 15]
            print("  " + ", ".join(row))

    print(f"\nDone. {matched} corpus entries now have ground_truth_lean.")


if __name__ == "__main__":
    main()
