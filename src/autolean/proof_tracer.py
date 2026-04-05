"""Proof tracer — diagnose exactly where and why a Lean proof fails.

Parses Lean compiler output into structured diagnostics showing:
- Which tactic step failed
- What the goal state was at failure
- What hypotheses were available
- A human-readable explanation of the error

Usage:
    from autolean.proof_tracer import trace_proof, format_trace

    trace = trace_proof(lean_code, compile_output)
    print(format_trace(trace))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProofStep:
    """A single tactic step in a proof."""
    line: int
    column: int
    tactic: str
    status: str = "ok"  # "ok", "error", "warning"
    goal_before: str = ""
    goal_after: str = ""
    error_message: str = ""
    suggestion: str = ""


@dataclass
class ProofTrace:
    """Full trace of a proof attempt."""
    theorem_name: str = ""
    lean_code: str = ""
    success: bool = False
    steps: list[ProofStep] = field(default_factory=list)
    errors: list[ProofStep] = field(default_factory=list)
    warnings: list[ProofStep] = field(default_factory=list)
    raw_output: str = ""


# ──────────────────────────────────────────────────────────────
# Lean error parsing
# ──────────────────────────────────────────────────────────────

# Pattern: filename:line:col: error: message
_ERROR_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<level>error|warning|info):\s*(?P<msg>.+)$",
    re.MULTILINE,
)

# Pattern for goal state in error context
_GOAL_RE = re.compile(
    r"(?P<hyps>(?:[a-zA-Z_]\w*(?:\s*:\s*[^\n]+)?\n)*)"
    r"⊢\s*(?P<goal>.+)",
    re.MULTILINE,
)

# Common tactic names
_TACTICS = [
    "intro", "intros", "apply", "exact", "rw", "rewrite", "simp",
    "ring", "linarith", "nlinarith", "omega", "norm_num", "norm_cast",
    "field_simp", "push_neg", "contrapose", "by_contra", "by_cases",
    "induction", "cases", "rcases", "obtain", "have", "let", "show",
    "calc", "conv", "ext", "funext", "congr", "refine", "use",
    "constructor", "left", "right", "exfalso", "trivial", "tauto",
    "aesop", "decide", "positivity", "gcongr", "rel",
    "sorry", "admit",
]


def _extract_tactic_at_line(lean_code: str, line_no: int) -> str:
    """Extract the tactic used at a given line number."""
    lines = lean_code.splitlines()
    if 0 < line_no <= len(lines):
        line = lines[line_no - 1].strip()
        # Find which tactic this line starts with
        for tactic in _TACTICS:
            if line.startswith(tactic):
                return line
        return line
    return ""


def _extract_goal_from_error(error_text: str) -> tuple[str, str]:
    """Extract hypotheses and goal from an error message containing goal state."""
    match = _GOAL_RE.search(error_text)
    if match:
        hyps = match.group("hyps").strip()
        goal = match.group("goal").strip()
        return hyps, goal
    return "", ""


def _suggest_fix(error_msg: str, tactic: str) -> str:
    """Suggest a fix based on the error message and tactic used."""
    msg_lower = error_msg.lower()

    if "unknown identifier" in msg_lower:
        name = re.search(r"unknown identifier '([^']+)'", error_msg)
        if name:
            return f"The lemma '{name.group(1)}' doesn't exist in Mathlib. Try `exact?` or `apply?` to find the right name."

    if "type mismatch" in msg_lower:
        return "Types don't match. Check if you need a coercion (↑, ↓) or a cast (Nat.cast, Int.cast)."

    if "tactic 'simp' failed" in msg_lower:
        return "simp couldn't close the goal. Try `simp?` to see what lemmas it tried, or add specific lemmas: `simp [lemma_name]`."

    if "tactic 'ring' failed" in msg_lower:
        return "The goal isn't a pure ring equation. Try `ring_nf` to normalize, or check if you need `field_simp` first."

    if "tactic 'linarith' failed" in msg_lower:
        return "linarith couldn't derive this from linear arithmetic. Try adding hypotheses with `have`, or use `nlinarith` for nonlinear goals."

    if "tactic 'omega' failed" in msg_lower:
        return "omega works on Nat/Int linear arithmetic only. Check if your goal involves non-integer types."

    if "'sorry' is a proof placeholder" in msg_lower or "declaration uses 'sorry'" in msg_lower:
        return "Proof is incomplete — there's a `sorry` that needs to be filled in."

    if "function expected" in msg_lower:
        return "You're trying to apply something that isn't a function/lemma. Check the type of what you're applying."

    if "unsolved goals" in msg_lower:
        return "There are remaining goals after the last tactic. The proof is incomplete."

    if tactic.startswith("exact") and "has type" in msg_lower:
        return "The term you provided with `exact` has the wrong type. Use `exact?` to search for the right lemma."

    return ""


# ──────────────────────────────────────────────────────────────
# Main tracing functions
# ──────────────────────────────────────────────────────────────

def trace_proof(lean_code: str, compile_output: str) -> ProofTrace:
    """Parse Lean compiler output into a structured proof trace.

    Args:
        lean_code: The Lean 4 source code that was compiled.
        compile_output: stdout + stderr from the Lean compiler.

    Returns:
        A ProofTrace with parsed steps, errors, and suggestions.
    """
    trace = ProofTrace(
        lean_code=lean_code,
        raw_output=compile_output,
    )

    # Extract theorem name
    theorem_match = re.search(r"(?:theorem|lemma)\s+(\w+)", lean_code)
    if theorem_match:
        trace.theorem_name = theorem_match.group(1)

    # Parse all diagnostics from compiler output
    combined = compile_output
    has_errors = False

    for match in _ERROR_RE.finditer(combined):
        line_no = int(match.group("line"))
        col_no = int(match.group("col"))
        level = match.group("level")
        msg = match.group("msg").strip()

        # Get the full error message (may span multiple lines)
        start = match.end()
        next_match = _ERROR_RE.search(combined, start)
        if next_match:
            full_msg = msg + "\n" + combined[start:next_match.start()].strip()
        else:
            full_msg = msg + "\n" + combined[start:].strip()

        tactic = _extract_tactic_at_line(lean_code, line_no)
        hyps, goal = _extract_goal_from_error(full_msg)
        suggestion = _suggest_fix(full_msg, tactic)

        step = ProofStep(
            line=line_no,
            column=col_no,
            tactic=tactic,
            status=level,
            goal_before=f"{hyps}\n⊢ {goal}" if goal else "",
            error_message=msg,
            suggestion=suggestion,
        )

        if level == "error":
            trace.errors.append(step)
            has_errors = True
        elif level == "warning":
            trace.warnings.append(step)

        trace.steps.append(step)

    trace.success = not has_errors
    return trace


def format_trace(trace: ProofTrace, *, show_code: bool = True, color: bool = True) -> str:
    """Format a ProofTrace into a human-readable diagnostic report.

    Args:
        trace: The proof trace to format.
        show_code: Whether to show the source code with annotations.
        color: Whether to use ANSI colors.
    """
    def c(text, code):
        return f"\033[{code}m{text}\033[0m" if color else text

    lines = []

    # Header
    if trace.success:
        lines.append(c("✓ PROOF VERIFIED", "32;1"))
        if trace.theorem_name:
            lines.append(f"  Theorem: {trace.theorem_name}")
    else:
        lines.append(c("✗ PROOF FAILED", "31;1"))
        if trace.theorem_name:
            lines.append(f"  Theorem: {trace.theorem_name}")
        lines.append(f"  Errors: {len(trace.errors)}")

    lines.append("")

    # Show code with error annotations
    if show_code and trace.lean_code and trace.errors:
        code_lines = trace.lean_code.splitlines()
        error_lines = {e.line for e in trace.errors}
        warning_lines = {w.line for w in trace.warnings}

        lines.append(c("─── Source Code ───", "2"))
        for i, code_line in enumerate(code_lines, 1):
            prefix = f"  {i:3d} │ "
            if i in error_lines:
                lines.append(c(f"{prefix}{code_line}", "31"))  # red
                # Show error details inline
                for err in trace.errors:
                    if err.line == i:
                        pointer = " " * (len(prefix) + err.column - 1) + c("^~~~", "31;1")
                        lines.append(pointer)
                        lines.append(c(f"       ERROR: {err.error_message}", "31"))
                        if err.goal_before:
                            lines.append(c(f"       Goal state:", "33"))
                            for gl in err.goal_before.splitlines():
                                lines.append(c(f"         {gl}", "33"))
                        if err.suggestion:
                            lines.append(c(f"       Suggestion: {err.suggestion}", "36"))
                        lines.append("")
            elif i in warning_lines:
                lines.append(c(f"{prefix}{code_line}", "33"))  # yellow
            else:
                lines.append(f"{prefix}{code_line}")
        lines.append("")

    # Error summary
    if trace.errors:
        lines.append(c("─── Error Summary ───", "31;1"))
        for i, err in enumerate(trace.errors, 1):
            lines.append(f"  {i}. Line {err.line}: {err.error_message}")
            if err.tactic:
                lines.append(f"     Tactic: {err.tactic}")
            if err.suggestion:
                lines.append(c(f"     Fix: {err.suggestion}", "36"))
            lines.append("")

    # Warnings
    if trace.warnings:
        lines.append(c("─── Warnings ───", "33"))
        for w in trace.warnings:
            lines.append(f"  Line {w.line}: {w.error_message}")
        lines.append("")

    return "\n".join(lines)


def trace_and_explain(
    lean_code: str,
    compile_output: str,
    *,
    call_llm=None,
) -> str:
    """Trace a proof and optionally use an LLM to explain the error in plain English.

    Args:
        lean_code: The Lean 4 source code.
        compile_output: Compiler output (stdout + stderr).
        call_llm: Optional callable(prompt) -> str for LLM explanation.
    """
    trace = trace_proof(lean_code, compile_output)
    report = format_trace(trace)

    if call_llm and trace.errors:
        # Ask LLM to explain in plain English
        error_context = "\n".join(
            f"Line {e.line}: {e.error_message}\nTactic: {e.tactic}\nGoal: {e.goal_before}"
            for e in trace.errors[:3]
        )
        prompt = f"""A student wrote a Lean 4 proof that failed. Explain the error in simple English (2-3 sentences). What went wrong and how should they fix it?

Theorem: {trace.theorem_name}

Errors:
{error_context}

Explain briefly:"""

        explanation = call_llm(prompt)
        if explanation:
            report += f"\n{'─' * 40}\n"
            report += f"Plain English explanation:\n{explanation}\n"

    return report
