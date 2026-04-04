#!/usr/bin/env python3
"""Autoresearch loop for Erdos problems.

Autonomously iterates through Erdos problems, trying multiple proving
strategies per problem. Inspired by karpathy/autoresearch.

Usage:
    python benchmark/autoresearch_erdos.py \
        --corpus benchmark/erdos_corpus/ \
        --output benchmark/autoresearch_results/ \
        --max-problems 50 \
        --max-time-per-problem 600 \
        --strategies direct,retrieval,decomposition,expert
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProblemResult:
    """Result from a single strategy attempt on a single problem."""

    uuid: str
    problem_file: str
    strategy: str
    proved: bool = False
    formalized: bool = False
    time_s: float = 0.0
    iterations: int = 0
    lean_code: str = ""
    failure_reason: str = ""
    error_log: str = ""


@dataclass
class ProblemSummary:
    """Aggregated result across all strategies tried on one problem."""

    uuid: str
    problem_file: str
    tags: list[str] = field(default_factory=list)
    status: str = ""          # "open", "solved", etc. from corpus JSON
    proved: bool = False
    winning_strategy: str = ""
    total_time_s: float = 0.0
    attempts: list[ProblemResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "uuid": self.uuid,
            "problem_file": self.problem_file,
            "tags": self.tags,
            "status": self.status,
            "proved": self.proved,
            "winning_strategy": self.winning_strategy,
            "total_time_s": self.total_time_s,
            "attempts": [asdict(a) for a in self.attempts],
        }
        return d


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES = ["direct", "retrieval", "decomposition", "expert"]


def _build_prompt_direct(problem_json: dict) -> str:
    """Plain formalize-and-prove prompt, no extras."""
    problem_text = "\n".join(problem_json.get("problem", []))
    theorem_name = problem_json.get("uuid", "erdos_theorem")

    return (
        f"Prove the following mathematical theorem in Lean 4 with Mathlib.\n\n"
        f"Theorem name: {theorem_name}\n\n"
        f"Statement:\n{problem_text}\n\n"
        f"Produce a complete Lean 4 file that compiles without sorry."
    )


def _build_prompt_retrieval(problem_json: dict) -> str:
    """Inject retrieved Mathlib lemmas (from corpus JSON) into the prompt."""
    problem_text = "\n".join(problem_json.get("problem", []))
    theorem_name = problem_json.get("uuid", "erdos_theorem")

    premises = problem_json.get("retrieved_premises", [])
    if isinstance(premises, list):
        premises_block = "\n".join(premises)
    elif isinstance(premises, str):
        premises_block = premises
    else:
        premises_block = ""

    prompt = (
        f"Prove the following mathematical theorem in Lean 4 with Mathlib.\n\n"
        f"Theorem name: {theorem_name}\n\n"
        f"Statement:\n{problem_text}\n\n"
    )
    if premises_block:
        prompt += (
            f"The following Mathlib lemmas may be useful:\n"
            f"```\n{premises_block}\n```\n\n"
        )
    prompt += "Produce a complete Lean 4 file that compiles without sorry."
    return prompt


def _build_prompt_decomposition(problem_json: dict) -> str:
    """Sketch-then-solve: ask for a proof skeleton with sorry holes."""
    problem_text = "\n".join(problem_json.get("problem", []))
    theorem_name = problem_json.get("uuid", "erdos_theorem")

    premises = problem_json.get("retrieved_premises", [])
    if isinstance(premises, list):
        premises_block = "\n".join(premises)
    elif isinstance(premises, str):
        premises_block = premises
    else:
        premises_block = ""

    prompt = (
        f"You are proving a mathematical theorem step-by-step in Lean 4 with Mathlib.\n\n"
        f"Theorem name: {theorem_name}\n\n"
        f"Statement:\n{problem_text}\n\n"
    )
    if premises_block:
        prompt += (
            f"Potentially useful Mathlib lemmas:\n"
            f"```\n{premises_block}\n```\n\n"
        )
    prompt += (
        f"Approach: produce a PROOF SKETCH first.\n"
        f"- Break the proof into intermediate `have` steps.\n"
        f"- Use `sorry` for each sub-step initially.\n"
        f"- Then fill in each sorry with a real proof.\n"
        f"- The final file must compile with NO sorry.\n\n"
        f"Produce a complete Lean 4 file."
    )
    return prompt


def _build_prompt_expert(problem_json: dict) -> str:
    """Inject expert comments from the corpus JSON into the prompt."""
    problem_text = "\n".join(problem_json.get("problem", []))
    theorem_name = problem_json.get("uuid", "erdos_theorem")

    expert_comments = problem_json.get("expert_comments", "")
    if isinstance(expert_comments, list):
        expert_comments = "\n".join(expert_comments)

    hints = problem_json.get("hints", "")
    if isinstance(hints, list):
        hints = "\n".join(hints)

    known_results = problem_json.get("known_results", "")
    if isinstance(known_results, list):
        known_results = "\n".join(known_results)

    prompt = (
        f"Prove the following mathematical theorem in Lean 4 with Mathlib.\n\n"
        f"Theorem name: {theorem_name}\n\n"
        f"Statement:\n{problem_text}\n\n"
    )
    if expert_comments:
        prompt += f"Expert commentary:\n{expert_comments}\n\n"
    if hints:
        prompt += f"Hints:\n{hints}\n\n"
    if known_results:
        prompt += f"Known related results:\n{known_results}\n\n"

    prompt += "Produce a complete Lean 4 file that compiles without sorry."
    return prompt


PROMPT_BUILDERS = {
    "direct": _build_prompt_direct,
    "retrieval": _build_prompt_retrieval,
    "decomposition": _build_prompt_decomposition,
    "expert": _build_prompt_expert,
}


# ---------------------------------------------------------------------------
# Core proving logic
# ---------------------------------------------------------------------------

def try_prove(
    problem_json: dict,
    problem_path: Path,
    strategy: str,
    mathcode_cmd: str,
    timeout: int,
) -> ProblemResult:
    """Try to prove a single problem using the given strategy.

    Calls the mathcode binary via subprocess with the constructed prompt.
    Returns a ProblemResult capturing success/failure and timing.
    """
    uuid = problem_json.get("uuid", problem_path.stem)

    result = ProblemResult(
        uuid=uuid,
        problem_file=problem_path.name,
        strategy=strategy,
    )

    # Build strategy-specific prompt
    builder = PROMPT_BUILDERS.get(strategy)
    if builder is None:
        result.failure_reason = f"unknown strategy: {strategy}"
        return result

    prompt = builder(problem_json)

    start = time.monotonic()
    try:
        cmd_parts = mathcode_cmd.split()
        proc = subprocess.run(
            [*cmd_parts, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        result.time_s = round(elapsed, 2)
        result.iterations = 1  # one subprocess call = one iteration

        if proc.returncode == 0:
            result.formalized = True
            stdout_lower = proc.stdout.lower()
            result.proved = "sorry" not in stdout_lower
            result.lean_code = proc.stdout
        else:
            result.failure_reason = (proc.stderr[:300] if proc.stderr
                                     else "nonzero exit code")
            result.error_log = proc.stderr or ""

    except subprocess.TimeoutExpired:
        result.time_s = round(time.monotonic() - start, 2)
        result.failure_reason = f"timeout ({timeout}s)"
    except FileNotFoundError:
        result.time_s = round(time.monotonic() - start, 2)
        result.failure_reason = f"mathcode command not found: {mathcode_cmd}"
    except Exception as exc:
        result.time_s = round(time.monotonic() - start, 2)
        result.failure_reason = str(exc)[:300]

    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_completed(jsonl_path: Path) -> set[str]:
    """Load UUIDs of already-completed problems from the results JSONL."""
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(record["uuid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_result(jsonl_path: Path, summary: ProblemSummary) -> None:
    """Append a single problem summary as one JSONL line."""
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")


def save_proof(output_dir: Path, uuid: str, lean_code: str, strategy: str) -> Path:
    """Save a successful proof to a .lean file in the output directory."""
    proofs_dir = output_dir / "proofs"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid}__{strategy}.lean"
    proof_path = proofs_dir / filename
    proof_path.write_text(lean_code, encoding="utf-8")
    return proof_path


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary_markdown(
    summaries: list[ProblemSummary],
    run_timestamp: str,
    output_path: Path,
) -> str:
    """Generate a markdown summary table and write it to disk."""

    total = len(summaries)
    proved_count = sum(1 for s in summaries if s.proved)
    proved_pct = (proved_count / total * 100) if total else 0.0

    lines: list[str] = []
    lines.append("# Autoresearch Results")
    lines.append("")
    lines.append(f"Run: {run_timestamp}")
    lines.append(f"Problems attempted: {total}")
    lines.append(f"Proved: {proved_count} ({proved_pct:.0f}%)")
    lines.append("")

    # --- By Strategy ---
    lines.append("## By Strategy")
    lines.append("")
    lines.append("| Strategy | Attempted | Proved | Avg Time |")
    lines.append("|----------|-----------|--------|----------|")

    strategy_stats: dict[str, dict] = {}
    for s in summaries:
        for a in s.attempts:
            strat = a.strategy
            if strat not in strategy_stats:
                strategy_stats[strat] = {"attempted": 0, "proved": 0, "time": 0.0}
            strategy_stats[strat]["attempted"] += 1
            if a.proved:
                strategy_stats[strat]["proved"] += 1
            strategy_stats[strat]["time"] += a.time_s

    for strat in STRATEGIES:
        if strat not in strategy_stats:
            continue
        st = strategy_stats[strat]
        avg_t = st["time"] / max(st["attempted"], 1)
        lines.append(
            f"| {strat} | {st['attempted']} | {st['proved']} | {avg_t:.0f}s |"
        )
    lines.append("")

    # --- By Tag ---
    lines.append("## By Tag")
    lines.append("")
    lines.append("| Tag | Attempted | Proved | Rate |")
    lines.append("|-----|-----------|--------|------|")

    tag_stats: dict[str, dict] = {}
    for s in summaries:
        tags = s.tags if s.tags else ["untagged"]
        for tag in tags:
            if tag not in tag_stats:
                tag_stats[tag] = {"attempted": 0, "proved": 0}
            tag_stats[tag]["attempted"] += 1
            if s.proved:
                tag_stats[tag]["proved"] += 1

    for tag in sorted(tag_stats.keys()):
        ts = tag_stats[tag]
        rate = (ts["proved"] / ts["attempted"] * 100) if ts["attempted"] else 0
        lines.append(
            f"| {tag} | {ts['attempted']} | {ts['proved']} | {rate:.0f}% |"
        )
    lines.append("")

    # --- Detailed Results ---
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| # | Problem | Status | Strategy | Time | Attempts |")
    lines.append("|---|---------|--------|----------|------|----------|")

    for i, s in enumerate(summaries, 1):
        status_str = "PROVED" if s.proved else "FAILED"
        strat_str = s.winning_strategy if s.proved else "-"
        total_time = f"{s.total_time_s:.0f}s"
        num_attempts = len(s.attempts)
        lines.append(
            f"| {i} | {s.uuid} | {status_str} | {strat_str} | {total_time} | {num_attempts} |"
        )
    lines.append("")

    md_text = "\n".join(lines)
    output_path.write_text(md_text, encoding="utf-8")
    return md_text


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(
    corpus_dir: Path,
    max_problems: Optional[int],
    filter_status: Optional[str],
    filter_tags: Optional[str],
    shuffle: bool,
) -> list[Path]:
    """Discover and filter problem JSON files from the corpus directory."""

    problem_files = sorted(corpus_dir.glob("*.json"))
    if not problem_files:
        return []

    # Apply filters
    if filter_status or filter_tags:
        tag_set = set()
        if filter_tags:
            tag_set = {t.strip().lower() for t in filter_tags.split(",")}

        filtered: list[Path] = []
        for pf in problem_files:
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if filter_status:
                file_status = data.get("status", "").lower()
                if file_status != filter_status.lower():
                    continue

            if tag_set:
                file_tags = {t.lower() for t in data.get("tags", [])}
                if not tag_set & file_tags:
                    continue

            filtered.append(pf)
        problem_files = filtered

    if shuffle:
        random.shuffle(problem_files)

    if max_problems is not None and max_problems > 0:
        problem_files = problem_files[:max_problems]

    return problem_files


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _format_elapsed(seconds: float) -> str:
    """Format seconds as Xm Ys for display."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_autoresearch(
    corpus_dir: Path,
    output_dir: Path,
    strategies: list[str],
    mathcode_cmd: str,
    max_problems: Optional[int],
    max_time_per_problem: int,
    filter_status: Optional[str],
    filter_tags: Optional[str],
    shuffle: bool,
) -> list[ProblemSummary]:
    """Main autoresearch loop.

    Iterates over corpus problems, tries each strategy in order,
    stops at the first successful proof for each problem.
    Writes incremental results to a JSONL file and prints live progress.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "autoresearch_results.jsonl"

    # Resume support: skip already-completed problems
    completed = load_completed(jsonl_path)
    if completed:
        print(f"Resuming: {len(completed)} problem(s) already completed.")

    # Load corpus
    problem_files = load_corpus(
        corpus_dir, max_problems, filter_status, filter_tags, shuffle
    )
    if not problem_files:
        print(f"No problem JSON files found in {corpus_dir}", file=sys.stderr)
        return []

    total = len(problem_files)
    print(f"Autoresearch: {total} problems, strategies={strategies}")
    print(f"Timeout per problem: {max_time_per_problem}s")
    print(f"Output: {output_dir}")
    print(f"mathcode cmd: {mathcode_cmd}")
    print()

    summaries: list[ProblemSummary] = []
    global_start = time.monotonic()

    for idx, problem_path in enumerate(problem_files, 1):
        try:
            problem_json = json.loads(problem_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[{idx}/{total}] SKIP {problem_path.name} (read error: {exc})")
            continue

        uuid = problem_json.get("uuid", problem_path.stem)

        # Resume: skip if already done
        if uuid in completed:
            print(f"[{idx}/{total}] {uuid} | SKIP (already completed)")
            continue

        tags = problem_json.get("tags", [])
        status = problem_json.get("status", "")

        summary = ProblemSummary(
            uuid=uuid,
            problem_file=problem_path.name,
            tags=tags,
            status=status,
        )

        problem_start = time.monotonic()

        for strategy in strategies:
            # Respect per-problem time budget
            elapsed_so_far = time.monotonic() - problem_start
            remaining = max_time_per_problem - elapsed_so_far
            if remaining <= 0:
                print(
                    f"  -> time budget exhausted before strategy={strategy}"
                )
                break

            strategy_timeout = min(int(remaining), max_time_per_problem)

            result = try_prove(
                problem_json, problem_path, strategy,
                mathcode_cmd, strategy_timeout,
            )
            summary.attempts.append(result)

            status_str = "PROVED" if result.proved else "failed"
            elapsed_str = _format_elapsed(result.time_s)
            print(
                f"[{idx}/{total}] {uuid} | strategy={strategy} | "
                f"{status_str} ({elapsed_str})"
            )

            if result.proved:
                summary.proved = True
                summary.winning_strategy = strategy
                # Save the proof file
                save_proof(output_dir, uuid, result.lean_code, strategy)
                break

        summary.total_time_s = round(time.monotonic() - problem_start, 2)

        # Persist incrementally
        append_result(jsonl_path, summary)
        summaries.append(summary)

    # Print final tally
    total_elapsed = time.monotonic() - global_start
    proved_total = sum(1 for s in summaries if s.proved)
    attempted_total = len(summaries)
    pct = (proved_total / attempted_total * 100) if attempted_total else 0
    print()
    print(
        f"Done. {proved_total}/{attempted_total} proved ({pct:.0f}%) "
        f"in {_format_elapsed(total_elapsed)}"
    )

    return summaries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoresearch loop for Erdos problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmark/erdos_corpus"),
        help="Path to the Erdos corpus directory containing problem JSONs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/autoresearch_results"),
        help="Output directory for results, proofs, and summary",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to attempt (default: all)",
    )
    parser.add_argument(
        "--max-time-per-problem",
        type=int,
        default=600,
        help="Maximum wall-clock seconds per problem across all strategies (default: 600)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="direct,retrieval,decomposition",
        help=(
            "Comma-separated list of strategies to try in order. "
            "Options: direct, retrieval, decomposition, expert, all. "
            "Use 'all' to try every strategy. (default: direct,retrieval,decomposition)"
        ),
    )
    parser.add_argument(
        "--mathcode-cmd",
        type=str,
        default="mathcode",
        help="Path to the mathcode binary (default: 'mathcode' on PATH)",
    )
    parser.add_argument(
        "--filter-status",
        type=str,
        default=None,
        help="Only attempt problems with this status (e.g., 'open')",
    )
    parser.add_argument(
        "--filter-tags",
        type=str,
        default=None,
        help="Only attempt problems matching these tags (comma-separated, e.g., 'number theory')",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize problem order",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve strategy list
    raw_strategies = [s.strip().lower() for s in args.strategies.split(",")]
    if "all" in raw_strategies:
        strategies = list(STRATEGIES)
    else:
        strategies = []
        for s in raw_strategies:
            if s not in STRATEGIES:
                print(
                    f"Unknown strategy '{s}'. "
                    f"Valid options: {', '.join(STRATEGIES)}, all",
                    file=sys.stderr,
                )
                return 1
            strategies.append(s)

    if not strategies:
        print("No strategies selected.", file=sys.stderr)
        return 1

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Run the loop
    summaries = run_autoresearch(
        corpus_dir=args.corpus,
        output_dir=args.output,
        strategies=strategies,
        mathcode_cmd=args.mathcode_cmd,
        max_problems=args.max_problems,
        max_time_per_problem=args.max_time_per_problem,
        filter_status=args.filter_status,
        filter_tags=args.filter_tags,
        shuffle=args.shuffle,
    )

    if not summaries:
        print("No problems were attempted.")
        return 0

    # Generate summary markdown
    summary_path = args.output / "autoresearch_summary.md"
    md = generate_summary_markdown(summaries, run_timestamp, summary_path)
    print()
    print(md)
    print()
    print(f"Summary written to: {summary_path}")
    print(f"JSONL log: {args.output / 'autoresearch_results.jsonl'}")

    proved_count = sum(1 for s in summaries if s.proved)
    return 0 if proved_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
