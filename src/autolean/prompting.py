from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .util import sanitize_identifier


@dataclass(frozen=True)
class PromptBundle:
    theorem_name: str
    lean_path: Path
    initial_thinking_prompt: str
    repair_thinking_prompt_template: str
    initial_prompt: str
    repair_prompt_template: str


def _escape_format_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _build_prior_context_block(
    *,
    prior_subproblems: list[dict],
    prior_formalizations: list[tuple[str, str]],
) -> str:
    if not prior_subproblems and not prior_formalizations:
        return ""

    sections: list[str] = [
        "Prerequisite context from earlier sub-questions of the same parent problem.",
        "Treat these as available context/lemmas for the current sub-question.",
    ]

    if prior_subproblems:
        json_items: list[str] = []
        for idx, obj in enumerate(prior_subproblems, start=1):
            blob = json.dumps(obj, ensure_ascii=False, indent=2)
            json_items.append(f"[Sub-question {idx} JSON]\n{blob}")
        sections.append("Earlier sub-question JSON objects (authoritative):\n\n" + "\n\n".join(json_items))

    if prior_formalizations:
        lean_items: list[str] = []
        for idx, (theorem_name, lean_code) in enumerate(prior_formalizations, start=1):
            lean_items.append(
                f"[Sub-question {idx} theorem `{theorem_name}`]\n```lean\n{lean_code}\n```"
            )
        sections.append(
            "Available prior Lean formalizations (reuse as prerequisites when possible):\n\n"
            + "\n\n".join(lean_items)
        )

    sections.append(
        "Do not re-formalize earlier sub-questions from scratch unless a minimal adjustment is required "
        "for coherence; preserve prior theorem meaning."
    )
    return "\n\n".join(sections)


def build_prompts(
    problem_json: dict,
    *,
    out_dir: Path,
    name_hint: str,
    formalization_only: bool = True,
    prior_subproblems: Optional[list[dict]] = None,
    prior_formalizations: Optional[list[tuple[str, str]]] = None,
    retrieved_premises_block: Optional[str] = None,
) -> PromptBundle:
    """Build deterministic prompts from an authoritative JSON problem object."""
    uuid = problem_json.get("uuid")
    if not isinstance(uuid, str) or not uuid.strip():
        raise ValueError("Missing or invalid required field: uuid (string)")

    problem_lines = problem_json.get("problem")
    if not isinstance(problem_lines, list) or not all(isinstance(x, str) for x in problem_lines):
        raise ValueError("Missing or invalid required field: problem (array of strings)")

    sanitized = sanitize_identifier(name_hint)
    theorem_name = f"problem_{sanitized}"
    lean_path = out_dir / f"{theorem_name}.lean"
    prior_subproblems = list(prior_subproblems or [])
    prior_formalizations = list(prior_formalizations or [])

    # JSON is authoritative; we embed it verbatim.
    json_blob = json.dumps(problem_json, ensure_ascii=False, indent=2)
    json_blob_for_format = json_blob.replace("{", "{{").replace("}", "}}")
    prior_context_block = _build_prior_context_block(
        prior_subproblems=prior_subproblems,
        prior_formalizations=prior_formalizations,
    )
    prior_context_block_for_format = _escape_format_braces(prior_context_block)

    initial_thinking = f"""You are in phase 5.2 (Thinking) of a Lean formalization pipeline.

Goal:
- Derive a proof idea for the math problem.
- Identify exact Mathlib definitions/lemmas likely to call.
- Flag coercion/typeclass pitfalls likely to break Lean elaboration.

Rules:
- The JSON object is authoritative. Do not invent or alter problem content.
- Do not write Lean code in this phase.
- Keep output concise and technical.

Return plain text with these sections:
1) Formal target sketch
2) Proof strategy
3) Candidate lemmas (exact names when possible)
4) Coercion/type pitfalls
5) Fallback search hints

JSON input:
{json_blob}
"""
    if prior_context_block:
        initial_thinking += f"\n\n{prior_context_block}\n"

    repair_thinking_template = f"""You are in phase 5.2 (Thinking) for a repair iteration.

Original JSON problem (authoritative):
{json_blob_for_format}

Previous Lean file:
{{prev_lean}}

Lean compiler output (verbatim):
{{compile_output}}

Task:
- Diagnose why the previous Lean file failed.
- Update the proof strategy and list concrete lemma replacements.
- Keep theorem meaning unchanged.
- Focus on holes, coercions, missing assumptions, and syntax/elaboration issues.
- Do not write Lean code in this phase.

Return plain text with these sections:
1) Root-cause analysis
2) Revised strategy
3) Candidate lemmas (exact names when possible)
4) Coercion/type pitfalls
5) Minimal patch plan for phase 5.3
"""
    if prior_context_block_for_format:
        repair_thinking_template += f"\n\n{prior_context_block_for_format}\n"

    if formalization_only:
        proof_policy = """- Do NOT provide a full proof or solution.
- This is statement-only formalization mode.
- The theorem body must be exactly:
  by
    sorry
- Do not add tactics, helper lemmas, or proof scripts."""
    else:
        proof_policy = "- Full proof is allowed if you can produce one."

    initial = f"""You are given a single math problem encoded as JSON.

<context>
The JSON object is authoritative. Do not invent content.
Do not merge or split problems.

Interpretation rules:
- "problem" is an array of natural-language statements; concatenate them in order.
- Ignore "solution", "remark", "reference", "figures", and other non-required fields.
</context>

<file_header>
- Output a complete Lean 4 file.
- Use `import Mathlib`.
- Put everything in namespace `Formalizations`.
- The Lean file path (for reference) is: `{lean_path.as_posix()}`.
</file_header>

<do_not_change>
- Main theorem name MUST be exactly: `{theorem_name}`.
- No Markdown, no explanations.
- Do NOT weaken the statement to True/False. Do NOT use by trivial/by decide. The theorem MUST formalize the original mathematical content.
- The theorem must mention the core objects (e.g., `Set`, `Real`, `Metric`, `∀`/`∃`) instead of an empty shell.
- If the problem has multiple sub-questions, combine them into a single theorem using `∧` for all parts.
{proof_policy}
</do_not_change>

<task>
Return ONLY a JSON object:
{{"lean": "<Lean 4 source code>"}}
</task>

<theorem>
{json_blob}
</theorem>
"""
    if retrieved_premises_block:
        initial += f"\n<retrieved_premises>\n{_escape_format_braces(retrieved_premises_block)}\n</retrieved_premises>\n"
    if prior_context_block:
        initial += f"\n\n{prior_context_block}\n"

    retrieved_premises_for_format = (
        _escape_format_braces(retrieved_premises_block)
        if retrieved_premises_block
        else ""
    )
    repair_template = f"""The following Lean file does not compile.

<theorem>
{json_blob_for_format}
</theorem>

<failing_proof>
{{prev_lean}}
</failing_proof>

<compiler_error_excerpt>
{{compile_output}}
</compiler_error_excerpt>

<goal_state_near_failure>
Inspect the compiler output above for unsolved goals or type mismatches near the failure site.
</goal_state_near_failure>

<do_not_change>
- Do not change the meaning of the theorem.
- Keep the theorem name exactly: `{theorem_name}`.
- Do NOT weaken the statement to True/False. Do NOT use by trivial/by decide. The theorem MUST formalize the original mathematical content.
- The theorem must mention the core objects (e.g., `Set`, `Real`, `Metric`, `∀`/`∃`) instead of an empty shell.
- If the problem has multiple sub-questions, combine them into a single theorem using `∧` for all parts.
{proof_policy}
</do_not_change>

<task>
- Fix the Lean file so it compiles.
- Return ONLY JSON: {{{{"lean": "<Lean 4 source code>"}}}}.
</task>
"""
    if retrieved_premises_for_format:
        repair_template += f"\n<retrieved_premises>\n{retrieved_premises_for_format}\n</retrieved_premises>\n"
    if prior_context_block_for_format:
        repair_template += f"\n\n{prior_context_block_for_format}\n"

    return PromptBundle(
        theorem_name=theorem_name,
        lean_path=lean_path,
        initial_thinking_prompt=initial_thinking,
        repair_thinking_prompt_template=repair_thinking_template,
        initial_prompt=initial,
        repair_prompt_template=repair_template,
    )


def build_sketch_prompt(
    problem_json: dict,
    theorem_name: str,
    lean_path: Path,
    formalization_only: bool = True,
    retrieved_premises_block: Optional[str] = None,
) -> str:
    """Build a sketch prompt that asks for a proof outline using ``have ... := by sorry`` holes.

    The sketch contains at most 5 sorry-holes, each representing a key
    intermediate lemma.  The caller can later fill these holes individually.
    """
    json_blob = json.dumps(problem_json, ensure_ascii=False, indent=2)

    if formalization_only:
        body_instruction = (
            "Leave the top-level theorem body as `by sorry` but add up to 5 "
            "`have <name> : <type> := by sorry` lines inside the proof block to "
            "outline the key intermediate steps."
        )
    else:
        body_instruction = (
            "Structure the proof with up to 5 `have <name> : <type> := by sorry` "
            "holes for the key intermediate lemmas, then close the main goal using those."
        )

    prompt = f"""You are given a math problem encoded as JSON. Produce a Lean 4 proof *sketch*.

<context>
The JSON object is authoritative. Do not invent content.
</context>

<file_header>
- Output a complete Lean 4 file.
- Use `import Mathlib`.
- Put everything in namespace `Formalizations`.
- The Lean file path (for reference) is: `{lean_path.as_posix()}`.
</file_header>

<do_not_change>
- Main theorem name MUST be exactly: `{theorem_name}`.
- No Markdown, no explanations.
- Do NOT weaken the statement to True/False. Do NOT use by trivial/by decide. The theorem MUST formalize the original mathematical content.
- The theorem must mention the core objects (e.g., `Set`, `Real`, `Metric`, `∀`/`∃`) instead of an empty shell.
- Maximum 5 sorry-holes.
</do_not_change>

<task>
{body_instruction}

Return ONLY a JSON object:
{{"lean": "<Lean 4 source code>"}}
</task>

<theorem>
{json_blob}
</theorem>
"""
    if retrieved_premises_block:
        prompt += f"\n<retrieved_premises>\n{_escape_format_braces(retrieved_premises_block)}\n</retrieved_premises>\n"

    return prompt
