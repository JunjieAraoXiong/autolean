"""Semantic fidelity evaluation: prompts, parsing, grading."""

from __future__ import annotations

import json
from typing import Optional

_EVAL_GRADES = {"A", "B", "C", "D"}
_EVAL_GRADE_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1}
_EVAL_RETRY_RESPONSE_CHARS = 4000
OPENROUTER_GEMINI_FLASH_PREVIEW_MODEL = "google/gemini-3-flash-preview"
GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_MODEL = "openai/gpt-5.2"
GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_REASONING_EFFORT = "xhigh"


def is_gemini_flash_preview_model(model: str) -> bool:
    return model.strip().lower() == OPENROUTER_GEMINI_FLASH_PREVIEW_MODEL


def grade_below_threshold(grade: str, min_grade: str) -> bool:
    g = _EVAL_GRADE_ORDER.get(grade.upper(), 0)
    t = _EVAL_GRADE_ORDER.get(min_grade.upper(), 0)
    return g < t


def build_formalization_eval_prompt(
    *,
    problem_json: dict,
    theorem_name: str,
    lean_code: str,
) -> str:
    json_blob = json.dumps(problem_json, ensure_ascii=False, indent=2)
    return f"""You are evaluating semantic fidelity of a Lean formalization against its original math problem.

Important scope:
- Evaluate ONLY the theorem statement semantics (not proof quality, style, or elegance).
- Compare the original problem requirements to the Lean theorem proposition.

Required comparison checklist (must be applied explicitly before grading):
1) Core mathematical objects and domains/types match (e.g., Set/Real/Metric/Measure).
2) Quantifier structure matches (forall/exists order and scope).
3) Hypotheses/assumptions match (none dropped or materially weakened).
4) Conclusion/claim matches (same relation/equality/inequality/content).
5) Multi-part coverage matches (if the original has multiple sub-questions, all parts are represented).

Hard grading rules (must follow exactly):
- Assign exactly one grade using this top-down decision order (stop at first match):
  1) D if theorem is trivialized/vacuous/unrelated (e.g., `True`/`False` shell) or has major semantic drift.
  2) C if any major obligation is missing/wrong/weakened (including missing any sub-question).
  3) B if all major obligations are present and correct, with at most minor wording/precision issues.
  4) A only if all checklist items pass with no material weakening.
- Never assign A or B when any major obligation is missing, wrong, or weakened.
- If uncertain between two grades, choose the lower grade.

Grading rubric:
- A: Fully faithful. Core objects/quantifiers/claims are preserved with no material weakening.
- B: Mostly faithful. Minor omissions or slight imprecision, but core meaning preserved.
- C: Partially faithful. Significant mismatch, missing subparts, or notable weakening.
- D: Not faithful. Major semantic drift, trivialization, or largely unrelated statement.

Return ONLY a JSON object:
{{
  "grade": "A|B|C|D",
  "summary": "<1-3 sentence verdict>",
  "distance_from_original": "<brief description of gaps>",
  "key_mismatches": ["<concrete mismatch>", "<concrete mismatch>"]
}}

Output-format hard constraints:
- Must be strict RFC8259 JSON (no markdown/code fences, no extra text).
- Do not use LaTeX delimiters like \\( ... \\) or \\[ ... \\].
- Avoid backslashes in field values unless required by JSON escaping.
- Prefer plain natural language like "(E)" instead of LaTeX-style escaped forms.

Original problem JSON (authoritative):
{json_blob}

Lean theorem target name: {theorem_name}

Lean file content:
```lean
{lean_code}
```"""


def build_eval_retry_prompt(
    *,
    base_prompt: str,
    failure_reason: str,
    previous_response_text: str,
    retry_no: int,
) -> str:
    snippet = previous_response_text.strip()
    if len(snippet) > _EVAL_RETRY_RESPONSE_CHARS:
        snippet = snippet[:_EVAL_RETRY_RESPONSE_CHARS] + "\n...[truncated]"
    return (
        base_prompt
        + "\n\nThe previous evaluation response could not be accepted.\n"
        + f"Failure reason: {failure_reason}\n"
        + f"Retry number: {retry_no}\n"
        + "Please regenerate and return ONLY valid JSON that matches the required schema.\n"
        + "Do not include markdown/code fences/explanations.\n"
        + "Avoid LaTeX escapes and avoid backslashes in values.\n\n"
        + "Previous invalid response (for debugging):\n"
        + snippet
    )


def _to_str_list(value: object, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def parse_formalization_eval_payload(payload: dict) -> dict[str, object]:
    raw_grade = payload.get("grade")
    if not isinstance(raw_grade, str):
        raise ValueError("Evaluation output missing 'grade' field.")
    grade = raw_grade.strip().upper()
    if grade not in _EVAL_GRADES:
        raise ValueError("Evaluation grade must be one of A/B/C/D.")

    summary = ""
    for key in ("summary", "verdict", "reasoning"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            summary = candidate.strip()
            break

    distance = ""
    for key in ("distance_from_original", "distance", "distance_summary"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            distance = candidate.strip()
            break

    mismatches: list[str] = []
    for key in ("key_mismatches", "mismatches", "gap_items"):
        items = _to_str_list(payload.get(key))
        if items:
            mismatches = items
            break

    normalized: dict[str, object] = {"grade": grade}
    if summary:
        normalized["summary"] = summary
    if distance:
        normalized["distance_from_original"] = distance
    if mismatches:
        normalized["key_mismatches"] = mismatches
    return normalized


def format_eval_failure_reason(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"{exc.msg} (line {exc.lineno}, column {exc.colno}, char {exc.pos})"
    text = str(exc).strip()
    return text or exc.__class__.__name__


def format_eval_feedback_for_repair(eval_payload: dict[str, object]) -> str:
    def _extract_parts(payload: dict[str, object]) -> list[str]:
        parts: list[str] = []
        summary_obj = payload.get("summary")
        if isinstance(summary_obj, str) and summary_obj.strip():
            parts.append(f"summary={summary_obj.strip()}")
        distance_obj = payload.get("distance_from_original")
        if isinstance(distance_obj, str) and distance_obj.strip():
            parts.append(f"distance_from_original={distance_obj.strip()}")
        mismatches_obj = payload.get("key_mismatches")
        if isinstance(mismatches_obj, list):
            clean_items = [
                str(x).strip() for x in mismatches_obj if isinstance(x, str) and str(x).strip()
            ]
            if clean_items:
                parts.append("key_mismatches=" + "; ".join(clean_items[:8]))
        return parts

    def _grade_rank(payload: dict[str, object]) -> int:
        grade_obj = payload.get("grade")
        if not isinstance(grade_obj, str):
            return 999
        grade = grade_obj.strip().upper()
        if grade not in _EVAL_GRADES:
            return 999
        return _EVAL_GRADE_ORDER.get(grade, 999)

    candidates: list[tuple[str, dict[str, object]]] = [("primary", eval_payload)]
    double_check_obj = eval_payload.get("double_check")
    if isinstance(double_check_obj, dict):
        primary_obj = double_check_obj.get("primary")
        if isinstance(primary_obj, dict):
            candidates.append(("double_check:primary", primary_obj))
        secondary_obj = double_check_obj.get("secondary")
        if isinstance(secondary_obj, dict):
            candidates.append(("double_check:secondary", secondary_obj))

    best_label = "primary"
    best_parts: list[str] = []
    best_rank = 999
    best_detail_score = -1
    for label, payload in candidates:
        parts = _extract_parts(payload)
        rank = _grade_rank(payload)
        detail_score = 0
        if parts:
            detail_score = sum(
                1 for part in parts if part.startswith("key_mismatches=")
            ) * 10 + len(parts)
        if (rank < best_rank) or (rank == best_rank and detail_score > best_detail_score):
            best_label = label
            best_parts = parts
            best_rank = rank
            best_detail_score = detail_score

    if not best_parts:
        return ""
    if best_label == "primary":
        return "Evaluator feedback: " + " | ".join(best_parts)
    return f"Evaluator feedback ({best_label}): " + " | ".join(best_parts)
