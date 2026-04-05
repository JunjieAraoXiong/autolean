#!/usr/bin/env python3
"""Erdős Problem Autoprover Demo.

Demonstrates the full pipeline: take an Erdős problem in natural language,
formalize it in Lean 4, prove it, and verify.

Usage:
    python demo_erdos.py
    python demo_erdos.py --problem "Prove Bertrand's postulate"
    python demo_erdos.py --problem-number 728
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Demo problems (curated for showing off)
# ──────────────────────────────────────────────────────────────

DEMO_PROBLEMS = {
    "bertrand": {
        "name": "Bertrand's Postulate (Erdős's proof)",
        "erdos_number": "N/A (classic, proved by Erdős at age 19)",
        "statement": "For every natural number n > 0, there exists a prime p such that n < p ≤ 2n.",
        "prompt": "Prove Bertrand's postulate in Lean 4 using Mathlib: for every natural number n > 0, there exists a prime p such that n < p and p ≤ 2n.",
        "mathlib_hint": "Nat.bertrand in Mathlib/NumberTheory/Bertrand",
        "difficulty": "Easy (in Mathlib)",
    },
    "even_square": {
        "name": "Square of an even number is even",
        "erdos_number": "N/A (warm-up)",
        "statement": "For all integers n, if n is even then n² is even.",
        "prompt": "Prove in Lean 4 using Mathlib: for all integers n, if n is even then n squared is even.",
        "mathlib_hint": "Even.mul, Even.pow",
        "difficulty": "Very easy",
    },
    "sqrt2": {
        "name": "Irrationality of √2",
        "erdos_number": "N/A (classic)",
        "statement": "√2 is irrational.",
        "prompt": "Prove in Lean 4 using Mathlib that the square root of 2 is irrational.",
        "mathlib_hint": "Irrational.sqrt_two",
        "difficulty": "Easy (in Mathlib)",
    },
    "inf_primes": {
        "name": "Infinitely many primes",
        "erdos_number": "N/A (Euclid)",
        "statement": "There are infinitely many prime numbers.",
        "prompt": "Prove in Lean 4 using Mathlib that there are infinitely many prime numbers.",
        "mathlib_hint": "Nat.infinite_primes",
        "difficulty": "Easy (in Mathlib)",
    },
    "primes_4k3": {
        "name": "Infinitely many primes of form 4k+3",
        "erdos_number": "Related to Dirichlet's theorem",
        "statement": "There are infinitely many primes p ≡ 3 (mod 4).",
        "prompt": "Prove in Lean 4 using Mathlib that there are infinitely many primes congruent to 3 modulo 4.",
        "mathlib_hint": "Nat.infinite_setOf_prime_and_eq_mod",
        "difficulty": "Medium",
    },
    "sum_primes": {
        "name": "Sum of prime reciprocals diverges (Erdős)",
        "erdos_number": "Classic Erdős result",
        "statement": "The series ∑(1/p) over all primes p diverges.",
        "prompt": "Prove in Lean 4 using Mathlib that the sum of reciprocals of primes is not summable (diverges).",
        "mathlib_hint": "Nat.Primes.not_summable_one_div",
        "difficulty": "Medium",
    },
}

# ──────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────

def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def green(text): return color(text, "32")
def red(text): return color(text, "31")
def yellow(text): return color(text, "33")
def cyan(text): return color(text, "36")
def bold(text): return color(text, "1")
def dim(text): return color(text, "2")

def banner():
    print()
    print(bold("╔══════════════════════════════════════════════════════════╗"))
    print(bold("║         Erdős Problem Autoprover — Demo                 ║"))
    print(bold("║         AUTOLEAN + MathCode + Lean 4                    ║"))
    print(bold("╚══════════════════════════════════════════════════════════╝"))
    print()

def show_problem(problem):
    print(cyan("┌─ Problem ─────────────────────────────────────────────┐"))
    print(cyan(f"│ {bold(problem['name'])}"))
    print(cyan(f"│ Erdős: {problem['erdos_number']}"))
    print(cyan(f"│ Difficulty: {problem['difficulty']}"))
    print(cyan(f"│"))
    print(cyan(f"│ {problem['statement']}"))
    print(cyan(f"│"))
    print(cyan(f"│ Mathlib: {problem['mathlib_hint']}"))
    print(cyan("└───────────────────────────────────────────────────────┘"))
    print()

def show_result(proved, lean_output, elapsed):
    if proved:
        print(green("┌─ Result ──────────────────────────────────────────────┐"))
        print(green(f"│ ✓ PROVED in {elapsed:.1f} seconds"))
        print(green("└───────────────────────────────────────────────────────┘"))
    else:
        print(red("┌─ Result ──────────────────────────────────────────────┐"))
        print(red(f"│ ✗ FAILED after {elapsed:.1f} seconds"))
        print(red("└───────────────────────────────────────────────────────┘"))
    print()
    if lean_output:
        # Show first 40 lines of output
        lines = lean_output.strip().split("\n")
        print(dim("─── Lean Output ───"))
        for line in lines[:40]:
            print(dim(f"  {line}"))
        if len(lines) > 40:
            print(dim(f"  ... ({len(lines) - 40} more lines)"))
        print()

# ──────────────────────────────────────────────────────────────
# Run via MathCode
# ──────────────────────────────────────────────────────────────

def find_mathcode():
    """Find the MathCode binary."""
    candidates = [
        Path.home() / "Desktop" / "math160" / "mathcode" / "run",
        Path.home() / "Desktop" / "math160" / "mathcode" / "mathcode",
        Path("./mathcode/run"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def run_problem(prompt, mathcode_cmd=None, timeout=300):
    """Run a problem through MathCode."""
    if mathcode_cmd is None:
        mathcode_cmd = find_mathcode()
    if mathcode_cmd is None:
        return False, "MathCode binary not found", 0

    print(yellow(f"  ▶ Running MathCode... (timeout: {timeout}s)"))
    print(dim(f"  cmd: {mathcode_cmd} -p \"{prompt[:60]}...\""))
    print()

    start = time.time()
    try:
        proc = subprocess.run(
            [mathcode_cmd, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start

        output = proc.stdout
        if proc.stderr:
            output += "\n" + proc.stderr

        # Check if proved (no sorry in output, successful exit)
        proved = proc.returncode == 0
        if "sorry" in output.lower():
            proved = False

        return proved, output, elapsed

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", timeout
    except Exception as exc:
        return False, str(exc), time.time() - start

# ──────────────────────────────────────────────────────────────
# Interactive menu
# ──────────────────────────────────────────────────────────────

def interactive_menu():
    """Show interactive problem picker."""
    print(bold("Available demo problems:\n"))
    keys = list(DEMO_PROBLEMS.keys())
    for i, key in enumerate(keys, 1):
        p = DEMO_PROBLEMS[key]
        print(f"  {cyan(str(i))}. {p['name']} ({dim(p['difficulty'])})")
    print(f"  {cyan('A')}. Run ALL problems")
    print(f"  {cyan('Q')}. Quit")
    print()

    choice = input(bold("Pick a problem (1-6, A, or Q): ")).strip()

    if choice.upper() == 'Q':
        return None
    if choice.upper() == 'A':
        return keys
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            return [keys[idx]]
    except ValueError:
        pass

    print(red("Invalid choice"))
    return interactive_menu()

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Erdős Problem Autoprover Demo")
    parser.add_argument("--problem", type=str, help="Custom problem to prove")
    parser.add_argument("--problem-key", type=str, choices=DEMO_PROBLEMS.keys(),
                        help="Pick a preset demo problem")
    parser.add_argument("--mathcode", type=str, help="Path to mathcode binary")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per problem (seconds)")
    parser.add_argument("--all", action="store_true", help="Run all demo problems")
    args = parser.parse_args()

    banner()

    # Check MathCode
    mathcode_cmd = args.mathcode or find_mathcode()
    if mathcode_cmd:
        print(f"  MathCode: {green(mathcode_cmd)}")
    else:
        print(red("  MathCode not found! Install it or pass --mathcode path"))
        return 1
    print()

    # Determine what to run
    if args.problem:
        # Custom problem
        custom = {
            "name": "Custom Problem",
            "erdos_number": "N/A",
            "statement": args.problem,
            "prompt": f"Prove in Lean 4 using Mathlib: {args.problem}",
            "mathlib_hint": "Unknown",
            "difficulty": "Unknown",
        }
        show_problem(custom)
        proved, output, elapsed = run_problem(custom["prompt"], mathcode_cmd, args.timeout)
        show_result(proved, output, elapsed)
        return 0 if proved else 1

    if args.all:
        problem_keys = list(DEMO_PROBLEMS.keys())
    elif args.problem_key:
        problem_keys = [args.problem_key]
    else:
        problem_keys = interactive_menu()
        if problem_keys is None:
            return 0

    # Run selected problems
    results = []
    for key in problem_keys:
        problem = DEMO_PROBLEMS[key]
        show_problem(problem)
        proved, output, elapsed = run_problem(problem["prompt"], mathcode_cmd, args.timeout)
        show_result(proved, output, elapsed)
        results.append((problem["name"], proved, elapsed))

    # Summary
    if len(results) > 1:
        print(bold("═══ Summary ═══════════════════════════════════════════"))
        proved_count = sum(1 for _, p, _ in results if p)
        for name, proved, elapsed in results:
            status = green("✓ PROVED") if proved else red("✗ FAILED")
            print(f"  {status}  {name} ({elapsed:.1f}s)")
        print()
        print(bold(f"  Total: {proved_count}/{len(results)} proved"))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
