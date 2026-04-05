#!/usr/bin/env python3
"""Demo for Prof. Holtz — Matrix Theory + General Theorem Proving.

Three-part demo:
1. Matrix theory: det(AB) = det(A) * det(B), Gershgorin, eigenvalue bounds
2. Analysis: classic results (Cauchy-Schwarz, irrationality of √2)
3. Erdős problems: from the benchmark suite

Usage:
    python demo_for_holtz.py                    # Interactive menu
    python demo_for_holtz.py --section matrix   # Matrix theory demos only
    python demo_for_holtz.py --section all      # Run everything
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Color helpers
# ──────────────────────────────────────────────────────────────

def color(text, code): return f"\033[{code}m{text}\033[0m"
def green(t): return color(t, "32")
def red(t): return color(t, "31")
def yellow(t): return color(t, "33")
def cyan(t): return color(t, "36")
def bold(t): return color(t, "1")
def dim(t): return color(t, "2")

# ──────────────────────────────────────────────────────────────
# Demo problems organized by section
# ──────────────────────────────────────────────────────────────

SECTIONS = {
    "matrix": {
        "title": "Matrix Theory",
        "subtitle": "Determinants, Eigenvalues, and Linear Algebra",
        "problems": [
            {
                "name": "det(AB) = det(A) · det(B)",
                "context": "Mathlib: Matrix.det_mul",
                "prompt": "Prove in Lean 4 using Mathlib that for square matrices A and B over a commutative ring, det(A * B) = det(A) * det(B).",
            },
            {
                "name": "Cayley-Hamilton Theorem",
                "context": "Mathlib: Matrix.aeval_self_charpoly",
                "prompt": "Prove in Lean 4 using Mathlib the Cayley-Hamilton theorem: every square matrix satisfies its own characteristic polynomial.",
            },
            {
                "name": "Trace = Sum of Eigenvalues (characteristic polynomial)",
                "context": "Mathlib: Matrix.trace, Matrix.charpoly",
                "prompt": "Prove in Lean 4 using Mathlib that for a square matrix M over a commutative ring, the trace of M equals the negation of the next-to-leading coefficient of its characteristic polynomial.",
            },
            {
                "name": "Rank-Nullity Theorem",
                "context": "Mathlib: LinearMap.rank_add_rank_ker",
                "prompt": "Prove in Lean 4 using Mathlib the rank-nullity theorem: for a linear map f between finite-dimensional vector spaces, rank(f) + nullity(f) = dim(domain).",
            },
            {
                "name": "Determinant of Transpose",
                "context": "Mathlib: Matrix.det_transpose",
                "prompt": "Prove in Lean 4 using Mathlib that the determinant of a matrix equals the determinant of its transpose.",
            },
        ],
    },
    "analysis": {
        "title": "Analysis & Foundations",
        "subtitle": "Classic Results in Analysis and Number Theory",
        "problems": [
            {
                "name": "√2 is Irrational",
                "context": "Mathlib: irrational_sqrt_two",
                "prompt": "Prove in Lean 4 using Mathlib that the square root of 2 is irrational.",
            },
            {
                "name": "Infinitely Many Primes",
                "context": "Mathlib: Nat.infinite_primes",
                "prompt": "Prove in Lean 4 using Mathlib that there are infinitely many prime numbers.",
            },
            {
                "name": "Cauchy-Schwarz Inequality",
                "context": "Mathlib: inner_mul_le_norm_mul_norm",
                "prompt": "Prove in Lean 4 using Mathlib the Cauchy-Schwarz inequality for inner product spaces: |⟨x, y⟩| ≤ ‖x‖ * ‖y‖.",
            },
            {
                "name": "Even + Even = Even",
                "context": "Mathlib: Even.add",
                "prompt": "Prove in Lean 4 using Mathlib that the sum of two even integers is even.",
            },
        ],
    },
    "erdos": {
        "title": "Erdős Problems",
        "subtitle": "From the 1,183-Problem Benchmark Suite",
        "problems": [
            {
                "name": "Bertrand's Postulate (Erdős, age 19)",
                "context": "Mathlib: Nat.bertrand | Erdős's first significant result",
                "prompt": "Prove Bertrand's postulate in Lean 4 using Mathlib: for every natural number n > 0, there exists a prime p such that n < p and p ≤ 2n.",
            },
            {
                "name": "∑(1/p) Diverges (Erdős)",
                "context": "Mathlib: Nat.Primes.not_summable_one_div",
                "prompt": "Prove in Lean 4 using Mathlib that the sum of reciprocals of prime numbers diverges (is not summable).",
            },
            {
                "name": "Infinitely Many Primes ≡ 3 (mod 4)",
                "context": "Mathlib: Nat.infinite_setOf_prime_and_eq_mod (via Dirichlet)",
                "prompt": "Prove in Lean 4 using Mathlib that there are infinitely many primes congruent to 3 modulo 4.",
            },
        ],
    },
}

# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

def find_mathcode():
    candidates = [
        Path.home() / "Desktop" / "math160" / "mathcode" / "run",
        Path("./mathcode/run"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def run_problem(prompt, mathcode_cmd, timeout=300):
    start = time.time()
    try:
        proc = subprocess.run(
            [mathcode_cmd, "-p", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - start
        output = proc.stdout
        if proc.stderr:
            output += "\n" + proc.stderr
        proved = proc.returncode == 0 and "sorry" not in output.lower()
        return proved, output, elapsed
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", timeout
    except Exception as exc:
        return False, str(exc), time.time() - start

# ──────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────

def banner():
    print()
    print(bold("╔══════════════════════════════════════════════════════════════╗"))
    print(bold("║    Automated Theorem Proving Pipeline — Demo               ║"))
    print(bold("║    AUTOLEAN + MathCode + Lean 4 + Mathlib                  ║"))
    print(bold("║                                                            ║"))
    print(bold("║    For Prof. Olga Holtz                                    ║"))
    print(bold("║    UC Berkeley, Mathematics / EECS                         ║"))
    print(bold("╚══════════════════════════════════════════════════════════════╝"))
    print()

def show_section_header(section_data):
    print()
    print(bold(f"═══ {section_data['title']} ═══"))
    print(dim(f"    {section_data['subtitle']}"))
    print()

def run_section(section_key, mathcode_cmd, timeout=300):
    section = SECTIONS[section_key]
    show_section_header(section)
    results = []

    for i, problem in enumerate(section["problems"], 1):
        print(f"  {cyan(str(i))}. {bold(problem['name'])}")
        print(f"     {dim(problem['context'])}")
        print(f"     {yellow('▶ Running...')}", end="", flush=True)

        proved, output, elapsed = run_problem(problem["prompt"], mathcode_cmd, timeout)

        if proved:
            print(f"\r     {green(f'✓ PROVED ({elapsed:.0f}s)')}")
        else:
            print(f"\r     {red(f'✗ FAILED ({elapsed:.0f}s)')}")

        results.append((problem["name"], proved, elapsed))
        print()

    return results

def interactive_menu():
    print(bold("Sections:\n"))
    print(f"  {cyan('1')}. Matrix Theory (5 problems)")
    print(f"     det(AB)=det(A)det(B), Cayley-Hamilton, Rank-Nullity, ...")
    print()
    print(f"  {cyan('2')}. Analysis & Foundations (4 problems)")
    print(f"     √2 irrational, infinite primes, Cauchy-Schwarz, ...")
    print()
    print(f"  {cyan('3')}. Erdős Problems (3 problems)")
    print(f"     Bertrand's postulate, ∑1/p diverges, primes ≡ 3 mod 4")
    print()
    print(f"  {cyan('A')}. Run ALL sections")
    print(f"  {cyan('Q')}. Quit")
    print()
    choice = input(bold("Choose (1-3, A, Q): ")).strip()

    if choice.upper() == 'Q':
        return []
    if choice.upper() == 'A':
        return ["matrix", "analysis", "erdos"]
    mapping = {"1": ["matrix"], "2": ["analysis"], "3": ["erdos"]}
    return mapping.get(choice, [])

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Demo for Prof. Holtz")
    parser.add_argument("--section", choices=["matrix", "analysis", "erdos", "all"])
    parser.add_argument("--mathcode", type=str)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    banner()

    mathcode_cmd = args.mathcode or find_mathcode()
    if not mathcode_cmd:
        print(red("MathCode not found. Pass --mathcode /path/to/run"))
        return 1
    print(f"  Pipeline: {green(mathcode_cmd)}")
    print(f"  Timeout:  {args.timeout}s per problem")
    print()

    if args.section == "all":
        sections = ["matrix", "analysis", "erdos"]
    elif args.section:
        sections = [args.section]
    else:
        sections = interactive_menu()

    if not sections:
        return 0

    all_results = []
    for section_key in sections:
        results = run_section(section_key, mathcode_cmd, args.timeout)
        all_results.extend(results)

    # Final summary
    if len(all_results) > 1:
        proved = sum(1 for _, p, _ in all_results if p)
        total = len(all_results)
        print(bold("═══ Summary ═══════════════════════════════════════════"))
        for name, p, elapsed in all_results:
            status = green("✓") if p else red("✗")
            print(f"  {status}  {name} ({elapsed:.0f}s)")
        print()
        print(bold(f"  Result: {proved}/{total} proved"))
        total_time = sum(e for _, _, e in all_results)
        print(bold(f"  Total time: {total_time:.0f}s"))
        print()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
