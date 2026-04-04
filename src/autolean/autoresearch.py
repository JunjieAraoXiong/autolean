"""Autoresearch loop for automated theorem proving.

Inspired by karpathy/autoresearch — an autonomous loop that iterates
through mathematical problems, trying multiple proving strategies,
and accumulating successful proofs overnight.

Unlike the standalone benchmark/autoresearch_erdos.py (subprocess-based),
this module uses AUTOLEAN's native Python modules directly:
- core.py for the compile-repair loop
- retrieval.py for Mathlib lemma retrieval
- decomposition.py for sketch → sorry2lemma → prove
- axle_provider.py for cloud Lean verification
- aristotle_provider.py for LaTeX → Lean autoformalization
- cache.py for avoiding redundant API calls

Usage:
    from autolean.autoresearch import AutoResearchLoop
    loop = AutoResearchLoop(corpus_dir="benchmark/erdos_corpus/", ...)
    loop.run()
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .util import CommandResult, ensure_dir

try:
    from .cache import ResponseCache
except ImportError:
    # Cache module may not be on this branch yet
    class ResponseCache:  # type: ignore[no-redef]
        def __init__(self, **kwargs): self.enabled = False
        def get(self, *a): return None
        def put(self, *a): pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    strategy: str
    proved: bool = False
    formalized: bool = False
    grade: str = ""
    lean_code: str = ""
    time_s: float = 0.0
    iterations: int = 0
    error: str = ""


@dataclass
class ProblemOutcome:
    uuid: str
    erdos_number: str = ""
    status: str = ""
    tags: list[str] = field(default_factory=list)
    proved: bool = False
    winning_strategy: str = ""
    strategies_tried: list[StrategyResult] = field(default_factory=list)
    total_time_s: float = 0.0
    timestamp: str = ""


@dataclass
class AutoResearchState:
    """Persistent state for resume support."""
    completed_uuids: set[str] = field(default_factory=set)
    proved_count: int = 0
    failed_count: int = 0
    total_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def strategy_direct(
    problem_json: dict,
    problem_path: Path,
    output_dir: Path,
    *,
    config_kwargs: dict,
    cache: ResponseCache,
) -> StrategyResult:
    """Strategy: direct AUTOLEAN formalize + prove."""
    from .core import RunConfig, process_problem_file

    result = StrategyResult(strategy="direct")
    start = time.time()

    try:
        cfg = RunConfig(
            input_dir=problem_path.parent,
            output_dir=output_dir / "direct",
            logs_dir=output_dir / "direct" / "logs",
            cache_enabled=True,
            cache_dir=config_kwargs.get("cache_dir"),
            **{k: v for k, v in config_kwargs.items() if k != "cache_dir"},
        )
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.logs_dir)

        success, records = process_problem_file(
            cfg, problem_path,
            repo_root=Path(".").resolve(),
            cache=cache,
        )
        result.formalized = success
        result.iterations = len(records)

        if success:
            from .prompting import build_prompts
            prompts = build_prompts(
                problem_json,
                out_dir=cfg.output_dir,
                name_hint=problem_path.stem,
                formalization_only=cfg.formalization_only,
            )
            if prompts.lean_path.exists():
                lean_code = prompts.lean_path.read_text(encoding="utf-8")
                result.lean_code = lean_code
                result.proved = "sorry" not in lean_code.lower()

    except Exception as exc:
        result.error = str(exc)[:300]
        logger.warning("[AUTORESEARCH] direct strategy error: %s", exc)

    result.time_s = time.time() - start
    return result


def strategy_retrieval(
    problem_json: dict,
    problem_path: Path,
    output_dir: Path,
    *,
    config_kwargs: dict,
    cache: ResponseCache,
    retrieval_index=None,
) -> StrategyResult:
    """Strategy: AUTOLEAN + Mathlib lemma retrieval injected into prompts."""
    result = StrategyResult(strategy="retrieval")
    start = time.time()

    try:
        # Get retrieved premises
        retrieved_text = ""
        if retrieval_index is not None:
            problem_text = "\n".join(problem_json.get("problem", []))
            premises = retrieval_index.retrieve(problem_text, k=25)
            from .retrieval import format_premises_for_prompt
            retrieved_text = format_premises_for_prompt(premises, max_tokens=3000)

        # Also use any pre-scraped premises from corpus
        if not retrieved_text and "retrieved_premises" in problem_json:
            retrieved_text = problem_json["retrieved_premises"]

        from .core import RunConfig, process_problem_file
        from .prompting import build_prompts

        cfg = RunConfig(
            input_dir=problem_path.parent,
            output_dir=output_dir / "retrieval",
            logs_dir=output_dir / "retrieval" / "logs",
            cache_enabled=True,
            cache_dir=config_kwargs.get("cache_dir"),
            **{k: v for k, v in config_kwargs.items() if k != "cache_dir"},
        )
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.logs_dir)

        # Rebuild prompts with retrieval
        success, records = process_problem_file(
            cfg, problem_path,
            repo_root=Path(".").resolve(),
            cache=cache,
        )
        result.formalized = success
        result.iterations = len(records)

        if success:
            prompts = build_prompts(
                problem_json,
                out_dir=cfg.output_dir,
                name_hint=problem_path.stem,
                formalization_only=cfg.formalization_only,
                retrieved_premises_block=retrieved_text,
            )
            if prompts.lean_path.exists():
                lean_code = prompts.lean_path.read_text(encoding="utf-8")
                result.lean_code = lean_code
                result.proved = "sorry" not in lean_code.lower()

    except Exception as exc:
        result.error = str(exc)[:300]
        logger.warning("[AUTORESEARCH] retrieval strategy error: %s", exc)

    result.time_s = time.time() - start
    return result


def strategy_decomposition(
    problem_json: dict,
    problem_path: Path,
    output_dir: Path,
    *,
    config_kwargs: dict,
    cache: ResponseCache,
    call_llm: Optional[Callable] = None,
    axle_verifier=None,
    retrieval_index=None,
) -> StrategyResult:
    """Strategy: sketch → sorry2lemma → prove sub-goals."""
    from .decomposition import run_decomposition

    result = StrategyResult(strategy="decomposition")
    start = time.time()

    try:
        # Set up callbacks
        def _call_verify(lean_code: str) -> CommandResult:
            if axle_verifier:
                return axle_verifier.verify(lean_code)
            # Fallback to local compilation
            import subprocess
            cwd = config_kwargs.get("cwd", Path(".").resolve())
            compile_cmd = config_kwargs.get("compile_cmd", "lake env lean {file}")
            tmp_path = output_dir / "decomp_tmp.lean"
            tmp_path.write_text(lean_code, encoding="utf-8")
            cmd = compile_cmd.replace("{file}", str(tmp_path.resolve()))
            import shlex
            proc = subprocess.run(shlex.split(cmd), cwd=str(cwd),
                                  capture_output=True, text=True, check=False)
            return CommandResult(argv=shlex.split(cmd), returncode=proc.returncode,
                                stdout=proc.stdout, stderr=proc.stderr)

        def _call_sorry2lemma(lean_code: str):
            if axle_verifier:
                return axle_verifier.sorry_to_lemmas(lean_code)
            # Without AXLE, return empty (no decomposition possible)
            return lean_code, []

        def _retrieve_premises(query: str) -> str:
            if retrieval_index is None:
                return ""
            premises = retrieval_index.retrieve(query, k=15)
            from .retrieval import format_premises_for_prompt
            return format_premises_for_prompt(premises, max_tokens=2000)

        if call_llm is None:
            result.error = "No LLM callable provided for decomposition"
            result.time_s = time.time() - start
            return result

        decomp_result = run_decomposition(
            problem_json=problem_json,
            theorem_name=f"problem_{problem_path.stem}",
            lean_path=output_dir / "decomp" / f"problem_{problem_path.stem}.lean",
            call_llm=call_llm,
            call_sorry2lemma=_call_sorry2lemma,
            call_verify=_call_verify,
            retrieve_premises=_retrieve_premises if retrieval_index else None,
            max_sketch_attempts=3,
            max_prove_attempts=5,
        )

        result.formalized = bool(decomp_result.sketch_code)
        result.proved = decomp_result.success
        result.lean_code = decomp_result.reassembled_code
        result.iterations = decomp_result.total_attempts

    except Exception as exc:
        result.error = str(exc)[:300]
        logger.warning("[AUTORESEARCH] decomposition strategy error: %s", exc)

    result.time_s = time.time() - start
    return result


def strategy_expert(
    problem_json: dict,
    problem_path: Path,
    output_dir: Path,
    *,
    config_kwargs: dict,
    cache: ResponseCache,
) -> StrategyResult:
    """Strategy: inject expert comments + hints into prompts."""
    result = StrategyResult(strategy="expert")
    start = time.time()

    try:
        # Build expert context from corpus data
        expert_parts = []
        comments = problem_json.get("expert_comments", [])
        if comments:
            # Prioritize Tao's comments
            tao = [c for c in comments if "Tao" in c.get("author", "")]
            others = [c for c in comments if "Tao" not in c.get("author", "")][:5]
            for c in tao[:3]:
                expert_parts.append(f"[{c['author']}]: {c['text'][:300]}")
            for c in others[:3]:
                expert_parts.append(f"[{c['author']}]: {c['text'][:200]}")

        hint = problem_json.get("reference_proof_hint", "")
        if hint:
            expert_parts.append(f"[Proof hint]: {hint[:500]}")

        additional = problem_json.get("additional_context", "")
        if additional:
            expert_parts.append(f"[Additional context]: {additional[:500]}")

        if not expert_parts:
            result.error = "No expert context available"
            result.time_s = time.time() - start
            return result

        expert_block = "\n\n".join(expert_parts)

        from .core import RunConfig, process_problem_file
        cfg = RunConfig(
            input_dir=problem_path.parent,
            output_dir=output_dir / "expert",
            logs_dir=output_dir / "expert" / "logs",
            cache_enabled=True,
            cache_dir=config_kwargs.get("cache_dir"),
            **{k: v for k, v in config_kwargs.items() if k != "cache_dir"},
        )
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.logs_dir)

        success, records = process_problem_file(
            cfg, problem_path,
            repo_root=Path(".").resolve(),
            cache=cache,
        )
        result.formalized = success
        result.iterations = len(records)

    except Exception as exc:
        result.error = str(exc)[:300]
        logger.warning("[AUTORESEARCH] expert strategy error: %s", exc)

    result.time_s = time.time() - start
    return result


def strategy_aristotle(
    problem_json: dict,
    problem_path: Path,
    output_dir: Path,
    *,
    config_kwargs: dict,
    cache: ResponseCache,
    call_llm: Optional[Callable] = None,
) -> StrategyResult:
    """Strategy: LLM → LaTeX proof → Aristotle → Lean."""
    result = StrategyResult(strategy="aristotle")
    start = time.time()

    try:
        from .aristotle_provider import AristotleAutoformalize

        if call_llm is None:
            result.error = "No LLM callable for LaTeX generation"
            result.time_s = time.time() - start
            return result

        # Step 1: Get LLM to write a LaTeX proof
        problem_text = "\n".join(problem_json.get("problem", []))
        latex_prompt = (
            "This is a complex competition-style math problem. "
            "Write a complete, rigorous proof. Do not search the internet.\n\n"
            f"Problem: {problem_text}\n\n"
            "Write your proof as a formal mathematical argument suitable for "
            "a research paper. Use LaTeX notation where appropriate."
        )
        latex_proof = call_llm(latex_prompt)

        if not latex_proof:
            result.error = "LLM returned empty LaTeX proof"
            result.time_s = time.time() - start
            return result

        # Step 2: Send to Aristotle for autoformalization
        aristotle = AristotleAutoformalize()
        formalize_result = aristotle.formalize_latex(
            latex_proof,
            problem_statement=problem_text,
            theorem_name=f"problem_{problem_path.stem}",
            output_dir=output_dir / "aristotle",
        )

        if formalize_result.returncode == 0 and formalize_result.stdout:
            result.formalized = True
            result.lean_code = formalize_result.stdout
            result.proved = "sorry" not in formalize_result.stdout.lower()
        else:
            result.error = formalize_result.stderr[:300]

    except Exception as exc:
        result.error = str(exc)[:300]
        logger.warning("[AUTORESEARCH] aristotle strategy error: %s", exc)

    result.time_s = time.time() - start
    return result


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "direct": strategy_direct,
    "retrieval": strategy_retrieval,
    "decomposition": strategy_decomposition,
    "expert": strategy_expert,
    "aristotle": strategy_aristotle,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class AutoResearchLoop:
    """Autonomous proving loop for mathematical problems.

    Iterates through problems, tries multiple strategies per problem,
    keeps the first successful proof, and logs everything.
    """

    def __init__(
        self,
        corpus_dir: Path,
        output_dir: Path,
        *,
        strategies: list[str] = None,
        max_problems: Optional[int] = None,
        max_time_per_problem: float = 600,
        filter_status: Optional[list[str]] = None,
        filter_tags: Optional[list[str]] = None,
        shuffle: bool = False,
        config_kwargs: Optional[dict] = None,
        cache: Optional[ResponseCache] = None,
        retrieval_index=None,
        axle_verifier=None,
        call_llm: Optional[Callable] = None,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.strategies = strategies or ["direct", "retrieval", "decomposition", "expert"]
        self.max_problems = max_problems
        self.max_time_per_problem = max_time_per_problem
        self.filter_status = filter_status
        self.filter_tags = filter_tags
        self.shuffle = shuffle
        self.config_kwargs = config_kwargs or {}
        self.cache = cache or ResponseCache(enabled=True)
        self.retrieval_index = retrieval_index
        self.axle_verifier = axle_verifier
        self.call_llm = call_llm

        ensure_dir(self.output_dir)
        self.log_path = self.output_dir / "autoresearch_log.jsonl"
        self.state = self._load_state()

    def _load_state(self) -> AutoResearchState:
        """Load state from previous run for resume support."""
        state = AutoResearchState()
        if self.log_path.exists():
            with self.log_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        state.completed_uuids.add(record["uuid"])
                        if record.get("proved"):
                            state.proved_count += 1
                        else:
                            state.failed_count += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
            logger.info("[AUTORESEARCH] Resumed: %d completed, %d proved",
                        len(state.completed_uuids), state.proved_count)
        return state

    def _load_problems(self) -> list[tuple[Path, dict]]:
        """Load and filter problems from corpus."""
        problems = []
        for json_path in sorted(self.corpus_dir.glob("erdos_*.json")):
            if json_path.name.startswith("_"):
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            uuid = data.get("uuid", "")
            if uuid in self.state.completed_uuids:
                continue

            # Apply filters
            if self.filter_status:
                if data.get("status", "") not in self.filter_status:
                    continue
            if self.filter_tags:
                tags = data.get("tags", [])
                if not any(t in tags for t in self.filter_tags):
                    continue

            # Skip problems without actual text
            problem_text = data.get("problem", [""])[0]
            if not problem_text or problem_text.startswith("Erdős Problem #"):
                continue

            problems.append((json_path, data))

        if self.shuffle:
            random.shuffle(problems)
        if self.max_problems:
            problems = problems[:self.max_problems]

        return problems

    def _try_strategy(
        self,
        strategy_name: str,
        problem_json: dict,
        problem_path: Path,
        problem_output_dir: Path,
    ) -> StrategyResult:
        """Run a single strategy on a problem."""
        strategy_fn = STRATEGIES.get(strategy_name)
        if strategy_fn is None:
            return StrategyResult(strategy=strategy_name, error=f"Unknown strategy: {strategy_name}")

        kwargs = {
            "config_kwargs": self.config_kwargs,
            "cache": self.cache,
        }

        # Add optional dependencies based on strategy
        if strategy_name == "retrieval":
            kwargs["retrieval_index"] = self.retrieval_index
        elif strategy_name == "decomposition":
            kwargs["call_llm"] = self.call_llm
            kwargs["axle_verifier"] = self.axle_verifier
            kwargs["retrieval_index"] = self.retrieval_index
        elif strategy_name == "aristotle":
            kwargs["call_llm"] = self.call_llm

        return strategy_fn(problem_json, problem_path, problem_output_dir, **kwargs)

    def _log_outcome(self, outcome: ProblemOutcome) -> None:
        """Append outcome to JSONL log."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")
            f.flush()

    def run(self) -> list[ProblemOutcome]:
        """Run the autoresearch loop. Returns all outcomes."""
        problems = self._load_problems()
        total = len(problems)

        if total == 0:
            logger.info("[AUTORESEARCH] No problems to process")
            return []

        logger.info("[AUTORESEARCH] Starting: %d problems, strategies=%s",
                    total, self.strategies)
        print(f"\nAutoresearch: {total} problems, strategies: {', '.join(self.strategies)}")
        print(f"Output: {self.output_dir}\n")

        outcomes: list[ProblemOutcome] = []
        loop_start = time.time()

        for idx, (problem_path, problem_json) in enumerate(problems, 1):
            uuid = problem_json.get("uuid", problem_path.stem)
            erdos_num = str(problem_json.get("erdos_number", ""))
            tags = problem_json.get("tags", [])

            outcome = ProblemOutcome(
                uuid=uuid,
                erdos_number=erdos_num,
                status=problem_json.get("status", ""),
                tags=tags,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

            problem_output_dir = self.output_dir / uuid
            ensure_dir(problem_output_dir)

            problem_start = time.time()
            print(f"[{idx}/{total}] {uuid}", end="", flush=True)

            for strategy_name in self.strategies:
                # Check time budget
                elapsed = time.time() - problem_start
                if elapsed > self.max_time_per_problem:
                    print(f" | TIMEOUT ({elapsed:.0f}s)")
                    break

                print(f" | {strategy_name}", end="", flush=True)

                strat_result = self._try_strategy(
                    strategy_name, problem_json, problem_path, problem_output_dir
                )
                outcome.strategies_tried.append(strat_result)

                if strat_result.proved:
                    outcome.proved = True
                    outcome.winning_strategy = strategy_name
                    self.state.proved_count += 1
                    print(f" PROVED ({strat_result.time_s:.0f}s)")
                    break
                elif strat_result.formalized:
                    print(f" formalized", end="", flush=True)

            if not outcome.proved:
                self.state.failed_count += 1
                total_elapsed = time.time() - problem_start
                print(f" | FAILED ({total_elapsed:.0f}s)")

            outcome.total_time_s = time.time() - problem_start
            self.state.completed_uuids.add(uuid)
            outcomes.append(outcome)
            self._log_outcome(outcome)

        self.state.total_time_s = time.time() - loop_start
        self._write_summary(outcomes)
        return outcomes

    def _write_summary(self, outcomes: list[ProblemOutcome]) -> None:
        """Write markdown summary."""
        total = len(outcomes)
        proved = sum(1 for o in outcomes if o.proved)

        lines = [
            "# Autoresearch Results",
            "",
            f"**Time:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  ",
            f"**Problems:** {total}  ",
            f"**Proved:** {proved}/{total} ({100*proved/max(total,1):.0f}%)  ",
            f"**Strategies:** {', '.join(self.strategies)}  ",
            f"**Total time:** {self.state.total_time_s:.0f}s  ",
            "",
            "## By Strategy",
            "",
            "| Strategy | Wins | Avg Time |",
            "|----------|------|----------|",
        ]

        for s in self.strategies:
            wins = sum(1 for o in outcomes if o.winning_strategy == s)
            times = [sr.time_s for o in outcomes for sr in o.strategies_tried if sr.strategy == s]
            avg = sum(times) / max(len(times), 1)
            lines.append(f"| {s} | {wins} | {avg:.0f}s |")

        # By tag
        tag_stats: dict[str, tuple[int, int]] = {}
        for o in outcomes:
            for t in o.tags[:2]:
                total_t, proved_t = tag_stats.get(t, (0, 0))
                tag_stats[t] = (total_t + 1, proved_t + (1 if o.proved else 0))

        lines.extend([
            "", "## By Tag", "",
            "| Tag | Attempted | Proved | Rate |",
            "|-----|-----------|--------|------|",
        ])
        for tag, (total_t, proved_t) in sorted(tag_stats.items(), key=lambda x: -x[1][1]):
            rate = 100 * proved_t / max(total_t, 1)
            lines.append(f"| {tag} | {total_t} | {proved_t} | {rate:.0f}% |")

        # Detailed results
        lines.extend([
            "", "## Detailed Results", "",
            "| # | Problem | Status | Proved | Strategy | Time | Attempts |",
            "|---|---------|--------|--------|----------|------|----------|",
        ])
        for i, o in enumerate(outcomes, 1):
            strat = o.winning_strategy or "-"
            attempts = sum(sr.iterations for sr in o.strategies_tried)
            p = "Yes" if o.proved else "No"
            lines.append(f"| {i} | {o.uuid} | {o.status} | {p} | {strat} | {o.total_time_s:.0f}s | {attempts} |")

        summary_path = self.output_dir / "autoresearch_summary.md"
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary written to {summary_path}")
