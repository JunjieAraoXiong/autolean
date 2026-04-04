"""Sub-goal decomposition via sketch-then-solve pipeline.

Implements the DeepSeek-Prover-V2 / Seed-Prover style workflow:
1. LLM generates a proof skeleton with `have ... := by sorry` holes
2. AXLE's sorry2lemma extracts each sorry into a standalone lemma
3. Each sub-lemma is proved independently (with retrieval if available)
4. Results are reassembled into a complete proof

Requires: axle_provider (for sorry2lemma) and optionally retrieval (for
per-subgoal premise retrieval).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .util import CommandResult, ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class SubGoal:
    """A single extracted sub-goal from a proof sketch."""
    name: str
    statement: str  # Full Lean code for this standalone lemma
    proved: bool = False
    proof_code: str = ""
    attempts: int = 0
    error: str = ""


@dataclass
class DecompositionResult:
    """Result of a sketch-then-solve attempt."""
    original_theorem: str
    sketch_code: str
    subgoals: list[SubGoal] = field(default_factory=list)
    reassembled_code: str = ""
    success: bool = False
    total_attempts: int = 0


def generate_sketch_prompt(
    *,
    problem_json: dict,
    theorem_name: str,
    lean_path: str,
    max_holes: int = 5,
    retrieved_premises: str = "",
) -> str:
    """Build a prompt asking the LLM to produce a proof skeleton with sorry holes."""
    json_blob = json.dumps(problem_json, ensure_ascii=False, indent=2)

    premises_block = ""
    if retrieved_premises:
        premises_block = f"""
<retrieved_premises>
{retrieved_premises}
</retrieved_premises>
"""

    return f"""<context>
<file_header>
import Mathlib
</file_header>

<theorem>
The following JSON describes the math problem to formalize and prove.
{json_blob}

Main theorem name: `{theorem_name}`
Lean file path: `{lean_path}`
</theorem>
{premises_block}
<do_not_change>
- Do NOT weaken the statement to True/False.
- Do NOT use by trivial/by decide.
- The theorem MUST formalize the original mathematical content.
- Keep the theorem name exactly: `{theorem_name}`.
</do_not_change>

<task>
Produce a PROOF SKETCH — a tactic-style proof outline where intermediate
steps use `have` lemmas ending in `sorry`.

Requirements:
- Use at most {max_holes} `have ... := by sorry` holes.
- Each hole must be referenced later in the proof.
- Holes should break the proof into manageable pieces.
- The sketch must COMPILE (with sorry warnings, not errors).
- Put everything in namespace `Formalizations`.
- Use `import Mathlib`.

Return ONLY JSON: {{"lean": "<Lean 4 source code with sorry holes>"}}
</task>
</context>"""


def parse_subgoals_from_sorry2lemma(
    modified_code: str,
    lemma_names: list[str],
) -> list[SubGoal]:
    """Parse sorry2lemma output into SubGoal objects."""
    subgoals = []
    for name in lemma_names:
        # Extract each lemma's code from the modified source
        # sorry2lemma lifts each sorry into a top-level lemma
        subgoals.append(SubGoal(
            name=name,
            statement=modified_code,  # Full file with all lemmas
            proved=False,
        ))
    return subgoals


def build_subgoal_prove_prompt(
    subgoal: SubGoal,
    *,
    retrieved_premises: str = "",
    previous_error: str = "",
) -> str:
    """Build a prompt to prove a single extracted sub-goal."""
    premises_block = ""
    if retrieved_premises:
        premises_block = f"""
<retrieved_premises>
{retrieved_premises}
</retrieved_premises>
"""

    error_block = ""
    if previous_error:
        error_block = f"""
<compiler_error_excerpt>
{previous_error}
</compiler_error_excerpt>
"""

    return f"""<context>
<file_header>
import Mathlib
</file_header>

<theorem>
The following lemma was extracted from a larger proof sketch.
Prove it completely — do NOT use sorry.

Lemma to prove: `{subgoal.name}`
</theorem>

<failing_proof>
{subgoal.statement}
</failing_proof>
{premises_block}{error_block}
<task>
Replace the `sorry` in lemma `{subgoal.name}` with a complete proof.
Do NOT change the lemma statement.
Do NOT add new axioms.
Output ONLY JSON: {{"lean": "<complete Lean 4 source code>"}}
</task>
</context>"""


def run_decomposition(
    *,
    problem_json: dict,
    theorem_name: str,
    lean_path: Path,
    call_llm,  # callable(prompt: str) -> str (returns model response text)
    call_sorry2lemma,  # callable(lean_code: str) -> tuple[str, list[str]]
    call_verify,  # callable(lean_code: str) -> CommandResult
    retrieve_premises=None,  # optional callable(query: str) -> str
    max_sketch_attempts: int = 3,
    max_prove_attempts: int = 5,
    max_holes: int = 5,
) -> DecompositionResult:
    """Run the full sketch-then-solve pipeline.

    Args:
        call_llm: Function that takes a prompt and returns model response text.
        call_sorry2lemma: Function that takes Lean code and returns (modified_code, lemma_names).
        call_verify: Function that takes Lean code and returns a CommandResult.
        retrieve_premises: Optional function that takes a query string and returns
            formatted premises text for prompt injection.
        max_sketch_attempts: How many times to try generating a valid sketch.
        max_prove_attempts: How many times to try proving each sub-goal.
        max_holes: Maximum sorry holes allowed in the sketch.
    """
    result = DecompositionResult(
        original_theorem=theorem_name,
        sketch_code="",
    )

    # --- Phase 1: Generate sketch ---
    logger.info("[DECOMPOSE] Phase 1: generating proof sketch for %s", theorem_name)

    retrieved = ""
    if retrieve_premises:
        problem_text = "\n".join(problem_json.get("problem", []))
        retrieved = retrieve_premises(problem_text)

    sketch_code = ""
    for sketch_attempt in range(1, max_sketch_attempts + 1):
        result.total_attempts += 1

        prompt = generate_sketch_prompt(
            problem_json=problem_json,
            theorem_name=theorem_name,
            lean_path=str(lean_path),
            max_holes=max_holes,
            retrieved_premises=retrieved,
        )

        try:
            response_text = call_llm(prompt)
            obj = json.loads(response_text) if response_text.strip().startswith("{") else {}
            lean_code = obj.get("lean", response_text)
        except Exception as exc:
            logger.warning("[DECOMPOSE] Sketch attempt %d parse error: %s", sketch_attempt, exc)
            continue

        # Verify sketch compiles (with sorry warnings OK)
        verify_result = call_verify(lean_code)
        if verify_result.returncode == 0 or "sorry" in verify_result.stderr.lower():
            sketch_code = lean_code
            result.sketch_code = sketch_code
            logger.info("[DECOMPOSE] Sketch compiled on attempt %d", sketch_attempt)
            break
        else:
            logger.warning("[DECOMPOSE] Sketch attempt %d failed: %s",
                           sketch_attempt, verify_result.stderr[:200])

    if not sketch_code:
        result.success = False
        return result

    # --- Phase 2: Extract sub-goals via sorry2lemma ---
    logger.info("[DECOMPOSE] Phase 2: extracting sub-goals via sorry2lemma")

    try:
        modified_code, lemma_names = call_sorry2lemma(sketch_code)
    except Exception as exc:
        logger.error("[DECOMPOSE] sorry2lemma failed: %s", exc)
        result.success = False
        return result

    if not lemma_names:
        logger.info("[DECOMPOSE] No sub-goals extracted (proof may already be complete)")
        # Check if the original sketch is actually complete
        verify_result = call_verify(sketch_code)
        if verify_result.returncode == 0 and "sorry" not in sketch_code.lower():
            result.reassembled_code = sketch_code
            result.success = True
        return result

    subgoals = parse_subgoals_from_sorry2lemma(modified_code, lemma_names)
    result.subgoals = subgoals
    logger.info("[DECOMPOSE] Extracted %d sub-goals: %s", len(subgoals),
                [sg.name for sg in subgoals])

    # --- Phase 3: Prove each sub-goal ---
    logger.info("[DECOMPOSE] Phase 3: proving %d sub-goals", len(subgoals))

    current_code = modified_code

    for sg in subgoals:
        logger.info("[DECOMPOSE] Proving sub-goal: %s", sg.name)

        # Retrieve premises specific to this sub-goal
        sg_retrieved = ""
        if retrieve_premises:
            sg_retrieved = retrieve_premises(sg.statement[:500])

        last_error = ""
        for attempt in range(1, max_prove_attempts + 1):
            sg.attempts += 1
            result.total_attempts += 1

            prompt = build_subgoal_prove_prompt(
                sg,
                retrieved_premises=sg_retrieved,
                previous_error=last_error,
            )

            try:
                response_text = call_llm(prompt)
                obj = json.loads(response_text) if response_text.strip().startswith("{") else {}
                proof_code = obj.get("lean", response_text)
            except Exception as exc:
                last_error = f"Parse error: {exc}"
                continue

            verify_result = call_verify(proof_code)
            if verify_result.returncode == 0 and "sorry" not in proof_code.lower():
                sg.proved = True
                sg.proof_code = proof_code
                current_code = proof_code  # Update running code
                logger.info("[DECOMPOSE] Sub-goal %s proved on attempt %d", sg.name, attempt)
                break
            else:
                last_error = verify_result.stderr[:500]
                logger.warning("[DECOMPOSE] Sub-goal %s attempt %d failed", sg.name, attempt)

        if not sg.proved:
            sg.error = last_error
            logger.warning("[DECOMPOSE] Sub-goal %s FAILED after %d attempts", sg.name, sg.attempts)

    # --- Phase 4: Reassemble ---
    all_proved = all(sg.proved for sg in subgoals)

    if all_proved:
        # Use the last successful code (which should have all sub-goals proved)
        result.reassembled_code = current_code
        verify_result = call_verify(current_code)
        result.success = verify_result.returncode == 0
        if result.success:
            logger.info("[DECOMPOSE] All sub-goals proved, reassembly verified!")
        else:
            logger.warning("[DECOMPOSE] All sub-goals proved but reassembly failed: %s",
                           verify_result.stderr[:200])
    else:
        failed = [sg.name for sg in subgoals if not sg.proved]
        logger.warning("[DECOMPOSE] %d/%d sub-goals failed: %s",
                       len(failed), len(subgoals), failed)
        result.success = False

    return result
