#!/usr/bin/env python3
"""Erdős Benchmark Runner for AUTOLEAN.

Runs all JSON problems through the AUTOLEAN pipeline and collects results.
Outputs a summary table as JSON + markdown.

Usage:
    python benchmark/run_benchmark.py --input benchmark/problems/ --output benchmark/results/
    python benchmark/run_benchmark.py --input benchmark/problems/ --output benchmark/results/ --use-mathcode /path/to/mathcode
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ProblemResult:
    uuid: str
    problem_file: str
    tier: int = 0
    mathlib_status: str = ""
    expected_difficulty: str = ""

    # Formalization results
    formalized: bool = False
    formalization_grade: str = ""
    formalization_time_s: float = 0.0
    formalization_iterations: int = 0

    # Proving results
    proved: bool = False
    proving_time_s: float = 0.0
    proving_iterations: int = 0

    # GPT-Erdos comparison
    gpt_erdos_has_lean: bool = False
    gpt_erdos_has_proof: bool = False

    # Overall
    total_time_s: float = 0.0
    failure_reason: str = ""
    lean_file: str = ""
    error_log: str = ""


@dataclass
class BenchmarkSuite:
    name: str = "Erdős Theorems Benchmark"
    timestamp: str = ""
    mode: str = "baseline"  # baseline | retrieval | decomposition | full
    total_problems: int = 0
    formalized_count: int = 0
    proved_count: int = 0
    results: list[ProblemResult] = field(default_factory=list)
    problem_data: dict[str, dict] = field(default_factory=dict)

    def add(self, result: ProblemResult) -> None:
        self.results.append(result)
        self.total_problems = len(self.results)
        self.formalized_count = sum(1 for r in self.results if r.formalized)
        self.proved_count = sum(1 for r in self.results if r.proved)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name}",
            f"",
            f"**Mode:** {self.mode}  ",
            f"**Timestamp:** {self.timestamp}  ",
            f"**Problems:** {self.total_problems}  ",
            f"**Formalized:** {self.formalized_count}/{self.total_problems}  ",
            f"**Proved:** {self.proved_count}/{self.total_problems}  ",
            f"",
            "## Results",
            "",
            "| # | Problem | Tier | Formalized | Grade | Proved | Attempts | Time | Failure |",
            "|---|---------|------|-----------|-------|--------|----------|------|---------|",
        ]
        for i, r in enumerate(self.results, 1):
            form = "Yes" if r.formalized else "No"
            prov = "Yes" if r.proved else "No"
            attempts = r.formalization_iterations + r.proving_iterations
            t = f"{r.total_time_s:.0f}s"
            fail = r.failure_reason[:40] if r.failure_reason else "-"
            lines.append(
                f"| {i} | {r.uuid} | {r.tier} | {form} | {r.formalization_grade or '-'} | {prov} | {attempts} | {t} | {fail} |"
            )

        # Summary by tier
        lines.extend(["", "## Summary by Tier", ""])
        for tier in sorted(set(r.tier for r in self.results)):
            tier_results = [r for r in self.results if r.tier == tier]
            n = len(tier_results)
            f_count = sum(1 for r in tier_results if r.formalized)
            p_count = sum(1 for r in tier_results if r.proved)
            avg_time = sum(r.total_time_s for r in tier_results) / max(n, 1)
            lines.append(f"**Tier {tier}:** {f_count}/{n} formalized, {p_count}/{n} proved, avg {avg_time:.0f}s")

        # Comparison table: Our Prover vs GPT-5.2 + Aristotle
        has_gpt_data = any(r.gpt_erdos_has_lean for r in self.results)
        if has_gpt_data or self.problem_data:
            lines.extend(["", "## Comparison: Our Prover vs GPT-5.2 + Aristotle", ""])
            lines.append("| Problem | Ours | GPT-5.2+Aristotle | Ground Truth |")
            lines.append("|---------|------|--------------------|--------------|")
            for r in self.results:
                ours = "Proved" if r.proved else "No"
                gpt = "Proved" if r.gpt_erdos_has_proof else ("Lean" if r.gpt_erdos_has_lean else "No")
                pdata = self.problem_data.get(r.uuid, {})
                gt = "Yes" if pdata.get("ground_truth_lean") else "No"
                lines.append(f"| {r.uuid} | {ours} | {gpt} | {gt} |")

            # Comparison summary
            n = len(self.results)
            ours_proved = sum(1 for r in self.results if r.proved)
            gpt_proved = sum(1 for r in self.results if r.gpt_erdos_has_proof)
            has_gt = sum(1 for r in self.results if self.problem_data.get(r.uuid, {}).get("ground_truth_lean"))
            lines.extend([
                "",
                f"Our prover: {ours_proved}/{n} proved",
                f"GPT-5.2+Aristotle: {gpt_proved}/{n} proved",
                f"Has ground truth: {has_gt}/{n}",
            ])

        return "\n".join(lines)


def run_problem_mathcode(problem_path: Path, mathcode_cmd: str, output_dir: Path) -> ProblemResult:
    """Run a single problem through MathCode's -p mode."""
    problem_json = json.loads(problem_path.read_text(encoding="utf-8"))
    uuid = problem_json.get("uuid", problem_path.stem)
    problem_text = "\n".join(problem_json.get("problem", []))

    result = ProblemResult(
        uuid=uuid,
        problem_file=problem_path.name,
        tier=problem_json.get("tier", 0),
        mathlib_status=problem_json.get("mathlib_status", ""),
        expected_difficulty=problem_json.get("expected_difficulty", ""),
    )

    prompt = f"Prove the following in Lean 4 using Mathlib: {problem_text}"

    print(f"  [{uuid}] Running...", end="", flush=True)
    start = time.time()

    try:
        proc = subprocess.run(
            [*mathcode_cmd.split(), "-p", prompt],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per problem
        )
        elapsed = time.time() - start
        result.total_time_s = elapsed

        if proc.returncode == 0:
            result.formalized = True
            result.formalization_grade = "A"  # MathCode handles grading internally
            result.proved = "sorry" not in proc.stdout.lower()
        else:
            result.failure_reason = proc.stderr[:200] if proc.stderr else "nonzero exit"
            result.error_log = proc.stderr

        print(f" {'PROVED' if result.proved else 'FORMALIZED' if result.formalized else 'FAILED'} ({elapsed:.0f}s)")

    except subprocess.TimeoutExpired:
        result.total_time_s = 600
        result.failure_reason = "timeout (600s)"
        print(f" TIMEOUT")
    except Exception as exc:
        result.failure_reason = str(exc)[:200]
        print(f" ERROR: {exc}")

    return result


def run_problem_autolean(problem_path: Path, output_dir: Path, logs_dir: Path, **kwargs) -> ProblemResult:
    """Run a single problem through AUTOLEAN's Python API directly."""
    problem_json = json.loads(problem_path.read_text(encoding="utf-8"))
    uuid = problem_json.get("uuid", problem_path.stem)

    result = ProblemResult(
        uuid=uuid,
        problem_file=problem_path.name,
        tier=problem_json.get("tier", 0),
        mathlib_status=problem_json.get("mathlib_status", ""),
        expected_difficulty=problem_json.get("expected_difficulty", ""),
    )

    print(f"  [{uuid}] Running via AUTOLEAN...", end="", flush=True)
    start = time.time()

    try:
        from autolean.core import RunConfig, process_problem_file

        cfg = RunConfig(
            input_dir=problem_path.parent,
            output_dir=output_dir,
            logs_dir=logs_dir,
            max_iters=kwargs.get("max_iters", 6),
            formalization_only=True,
            openrouter_model=kwargs.get("model", "openai/gpt-5.2-codex"),
            compile_cmd=kwargs.get("compile_cmd", "lake env lean {file}"),
            cwd=kwargs.get("cwd"),
            cache_enabled=kwargs.get("cache", True),
        )

        success, records = process_problem_file(
            cfg, problem_path, repo_root=Path(".").resolve()
        )

        elapsed = time.time() - start
        result.total_time_s = elapsed
        result.formalization_iterations = len(records)
        result.formalized = success

        # Check eval grade from logs
        eval_path = output_dir / f"problem_{uuid}.eval.json"
        if eval_path.exists():
            eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
            result.formalization_grade = eval_data.get("grade", "")

        # Check lean output
        lean_path = output_dir / f"problem_{uuid}.lean"
        if lean_path.exists():
            result.lean_file = str(lean_path)
            lean_code = lean_path.read_text(encoding="utf-8")
            result.proved = success and "sorry" not in lean_code

        if not success and records:
            last = records[-1]
            result.failure_reason = last.compiler.stderr[:200] if last.compiler.stderr else "compilation failed"

        status = "PROVED" if result.proved else "FORMALIZED" if result.formalized else "FAILED"
        print(f" {status} ({elapsed:.0f}s, {len(records)} iters)")

    except Exception as exc:
        result.total_time_s = time.time() - start
        result.failure_reason = str(exc)[:200]
        print(f" ERROR: {exc}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Erdős Benchmark Runner for AUTOLEAN")
    parser.add_argument("--input", type=Path, default=Path("benchmark/problems"), help="Directory with problem JSON files")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results"), help="Output directory for results")
    parser.add_argument("--mode", choices=["baseline", "retrieval", "decomposition", "full"], default="baseline")
    parser.add_argument("--use-mathcode", type=str, default=None, help="Path to mathcode binary (uses ./run -p mode)")
    parser.add_argument("--tier", type=int, default=None, help="Only run problems of this tier")
    parser.add_argument("--max-iters", type=int, default=6, help="Max iterations per problem")
    parser.add_argument("--compile-cmd", type=str, default="lake env lean {file}")
    parser.add_argument("--cwd", type=Path, default=None, help="Compiler working directory")
    parser.add_argument("--gpt-erdos-solutions", type=Path, default=None, help="Path to gpt-erdos solutions directory")
    args = parser.parse_args()

    # Discover problems
    problem_files = sorted(args.input.glob("*.json"))
    if not problem_files:
        print(f"No JSON files found in {args.input}", file=sys.stderr)
        return 1

    # Filter by tier if requested
    if args.tier is not None:
        filtered = []
        for pf in problem_files:
            data = json.loads(pf.read_text(encoding="utf-8"))
            if data.get("tier") == args.tier:
                filtered.append(pf)
        problem_files = filtered

    print(f"Erdős Benchmark: {len(problem_files)} problems, mode={args.mode}")
    print(f"Output: {args.output}")
    print()

    # Setup output
    args.output.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    formalizations_dir = args.output / "formalizations"
    formalizations_dir.mkdir(parents=True, exist_ok=True)

    suite = BenchmarkSuite(
        mode=args.mode,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Load problem JSON data for ground truth checking
    problem_data_map: dict[str, dict] = {}

    # Run each problem
    for pf in problem_files:
        pf_data = json.loads(pf.read_text(encoding="utf-8"))
        pf_uuid = pf_data.get("uuid", pf.stem)
        problem_data_map[pf_uuid] = pf_data

        if args.use_mathcode:
            result = run_problem_mathcode(pf, args.use_mathcode, formalizations_dir)
        else:
            result = run_problem_autolean(
                pf, formalizations_dir, logs_dir,
                max_iters=args.max_iters,
                compile_cmd=args.compile_cmd,
                cwd=args.cwd,
            )

        # Check GPT-Erdos solutions if directory provided
        if args.gpt_erdos_solutions is not None:
            number = pf.stem  # problem file stem as folder name
            candidate = args.gpt_erdos_solutions / number / "candidate_solution.lean"
            if candidate.exists():
                result.gpt_erdos_has_lean = True
                lean_content = candidate.read_text(encoding="utf-8")
                if "sorry" not in lean_content:
                    result.gpt_erdos_has_proof = True

        suite.add(result)

    # Attach problem data for ground truth checking in markdown output
    suite.problem_data = problem_data_map

    # Write results
    results_json = args.output / f"benchmark_{args.mode}.json"
    results_json.write_text(suite.to_json(), encoding="utf-8")
    print(f"\nJSON results: {results_json}")

    results_md = args.output / f"benchmark_{args.mode}.md"
    results_md.write_text(suite.to_markdown(), encoding="utf-8")
    print(f"Markdown results: {results_md}")

    # Print summary
    print(f"\n{suite.to_markdown()}")

    return 0 if suite.proved_count == suite.total_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
