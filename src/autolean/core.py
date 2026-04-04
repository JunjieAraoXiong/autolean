"""Core orchestration: RunConfig, iteration loop, and pipeline coordination.

This module ties together providers, compiler, evaluation, and caching
into the main process_problem_file() loop.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .cache import ResponseCache
from .compiler import (
    REPAIR_ERROR_MEMORY_LIMIT,
    compile_lean,
    detect_trivialized_statement,
    format_error_memory,
    inject_imports,
    module_name_from_lean_path,
    update_error_memory,
)
from .evaluation import (
    GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_MODEL,
    GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_REASONING_EFFORT,
    build_eval_retry_prompt,
    build_formalization_eval_prompt,
    format_eval_failure_reason,
    format_eval_feedback_for_repair,
    grade_below_threshold,
    is_gemini_flash_preview_model,
    parse_formalization_eval_payload,
)
from .prompting import build_prompts
from .providers import (
    _CODEX_EXEC_CODING_MODEL,
    _CODEX_EXEC_CODING_REASONING_EFFORT,
    call_codex_exec,
    call_openrouter_chat,
    extract_model_response_text,
    parse_json_object_from_model_text,
)
from .util import CommandResult, ensure_dir


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    max_iters: int = 6
    formalization_only: bool = True
    require_no_sorry: bool = False
    openrouter_model: str = "openai/gpt-5.2-codex"
    openrouter_thinking_model: Optional[str] = None
    openrouter_eval_model: str = "openai/gpt-5.2"
    openrouter_coding_reasoning_effort: str = "xhigh"
    openrouter_thinking_reasoning_effort: str = "xhigh"
    openrouter_eval_reasoning_effort: str = "xhigh"
    openrouter_eval_repair_retries: int = 2
    min_eval_grade: Optional[str] = "B"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key_env: str = "AUTOLEAN_API"
    openrouter_timeout_s: int = 180
    openrouter_max_retries: int = 2
    openrouter_web_search: bool = False
    openrouter_web_search_engine: Optional[str] = None
    openrouter_web_search_max_results: Optional[int] = None
    use_codex_exec: bool = False
    codex_exec_sandbox: str = "read-only"
    live_logs: bool = False
    compile_cmd: str = "lake env lean {file}"
    cwd: Optional[Path] = None
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None

    def compile_argv(self, lean_file: Path) -> list[str]:
        import shlex
        cmd = self.compile_cmd.replace("{file}", str(lean_file.resolve()))
        return shlex.split(cmd)


@dataclass
class IterationRecord:
    iter_no: int
    thinking: CommandResult
    coding: CommandResult
    compiler: CommandResult
    lean_path: Path


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=True, indent=2))


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _write_iteration_meta(
    *,
    logs_dir: Path,
    theorem_name: str,
    iter_no: int,
    thinking_prompt: str,
    coding_prompt: Optional[str],
    thinking_res: CommandResult,
    coding_res: Optional[CommandResult],
    compiler_res: Optional[CommandResult] = None,
    evaluation_prompt: Optional[str] = None,
    evaluation_res: Optional[CommandResult] = None,
    evaluation_payload: Optional[dict[str, object]] = None,
) -> None:
    meta: dict[str, object] = {
        "iteration": iter_no,
        "thinking_prompt_sha256": _prompt_hash(thinking_prompt),
        "thinking_prompt_chars": len(thinking_prompt),
        "thinking": {"argv": thinking_res.argv, "returncode": thinking_res.returncode},
    }
    if coding_prompt is not None:
        meta["coding_prompt_sha256"] = _prompt_hash(coding_prompt)
        meta["coding_prompt_chars"] = len(coding_prompt)
    if coding_res is not None:
        meta["coding"] = {"argv": coding_res.argv, "returncode": coding_res.returncode}
    if compiler_res is not None:
        meta["compiler"] = {"argv": compiler_res.argv, "returncode": compiler_res.returncode}
    if evaluation_prompt is not None:
        meta["evaluation_prompt_sha256"] = _prompt_hash(evaluation_prompt)
        meta["evaluation_prompt_chars"] = len(evaluation_prompt)
    if evaluation_res is not None:
        meta["evaluation"] = {"argv": evaluation_res.argv, "returncode": evaluation_res.returncode}
    if evaluation_payload is not None:
        meta["evaluation_payload"] = evaluation_payload
    _write_json(logs_dir / f"{theorem_name}.iter{iter_no}.meta.json", meta)


def _write_compiler_logs(
    logs_dir: Path, theorem_name: str, iter_no: int, compiler_res: CommandResult
) -> None:
    _write_text(logs_dir / f"{theorem_name}.iter{iter_no}.compile_stdout.log", compiler_res.stdout)
    _write_text(logs_dir / f"{theorem_name}.iter{iter_no}.compile_stderr.log", compiler_res.stderr)


# ---------------------------------------------------------------------------
# Model dispatch (with caching)
# ---------------------------------------------------------------------------

def _call_model(
    cfg: RunConfig,
    cache: ResponseCache,
    *,
    prompt: str,
    model: str,
    reasoning_effort: Optional[str],
    stage: str,
    iter_no: int,
    theorem_name: str,
    attempt_no: Optional[int] = None,
    enable_web_search: bool = False,
    logs_dir: Path,
) -> CommandResult:
    # Check cache first
    cached = cache.get(prompt, model)
    if cached is not None:
        return cached

    codex_exec_model = model
    codex_exec_reasoning_effort = reasoning_effort
    if stage == "coding":
        codex_exec_model = _CODEX_EXEC_CODING_MODEL
        codex_exec_reasoning_effort = _CODEX_EXEC_CODING_REASONING_EFFORT

    if cfg.use_codex_exec:
        suffix = f"{theorem_name}.iter{iter_no}.{stage}"
        if attempt_no is not None:
            suffix += f"_attempt{attempt_no}"
        out_message_path = logs_dir / f"{suffix}.codex_last_message.log"
        result = call_codex_exec(
            prompt=prompt,
            out_message_path=out_message_path,
            model=codex_exec_model,
            reasoning_effort=codex_exec_reasoning_effort,
            sandbox=cfg.codex_exec_sandbox,
            workdir=cfg.cwd or cfg.input_dir.parent,
            live_logs=False,
        )
    else:
        web_kwargs = {}
        if enable_web_search and cfg.openrouter_web_search:
            web_kwargs = {
                "openrouter_web_search": True,
                "openrouter_web_search_engine": cfg.openrouter_web_search_engine,
                "openrouter_web_search_max_results": cfg.openrouter_web_search_max_results,
            }
        result = call_openrouter_chat(
            prompt=prompt,
            model=model,
            base_url=cfg.openrouter_base_url,
            api_key_env=cfg.openrouter_api_key_env,
            timeout_s=cfg.openrouter_timeout_s,
            max_retries=cfg.openrouter_max_retries,
            reasoning_effort=reasoning_effort,
            **web_kwargs,
        )

    # Cache successful responses
    cache.put(prompt, model, result)
    return result


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _run_eval_with_retries(
    cfg: RunConfig,
    cache: ResponseCache,
    *,
    base_eval_prompt: str,
    model: str,
    reasoning_effort: str,
    stage_name: str,
    log_stem: str,
    iter_no: int,
    theorem_name: str,
) -> tuple[str, Optional[CommandResult], dict[str, object]]:
    max_eval_attempts = max(1, cfg.openrouter_eval_repair_retries + 1)
    stage_prompt = base_eval_prompt
    stage_attempts: list[dict[str, object]] = []
    stage_payload: Optional[dict[str, object]] = None
    stage_last_res: Optional[CommandResult] = None

    for eval_try in range(1, max_eval_attempts + 1):
        eval_res_try = _call_model(
            cfg, cache,
            prompt=stage_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            stage=stage_name,
            iter_no=iter_no,
            theorem_name=theorem_name,
            attempt_no=eval_try,
            enable_web_search=False,
            logs_dir=cfg.logs_dir,
        )
        stage_last_res = eval_res_try
        _write_text(
            cfg.logs_dir / f"{theorem_name}.iter{iter_no}.{log_stem}_attempt{eval_try}_stdout.log",
            eval_res_try.stdout,
        )
        _write_text(
            cfg.logs_dir / f"{theorem_name}.iter{iter_no}.{log_stem}_attempt{eval_try}_stderr.log",
            eval_res_try.stderr,
        )

        if eval_res_try.returncode != 0:
            reason = eval_res_try.stderr.strip() or "evaluation request failed"
            stage_attempts.append({"attempt": eval_try, "status": "request_failed", "error": reason})
            if eval_try < max_eval_attempts:
                stage_prompt = build_eval_retry_prompt(
                    base_prompt=base_eval_prompt,
                    failure_reason=reason,
                    previous_response_text=eval_res_try.stdout,
                    retry_no=eval_try + 1,
                )
                continue
            stage_payload = {"status": "request_failed", "error": reason}
            break

        try:
            eval_text = extract_model_response_text(eval_res_try.stdout)
            eval_obj = parse_json_object_from_model_text(eval_text)
            normalized_eval = parse_formalization_eval_payload(eval_obj)
            stage_payload = {"status": "ok", **normalized_eval}
            stage_attempts.append({"attempt": eval_try, "status": "ok"})
            break
        except ValueError as exc:
            reason = format_eval_failure_reason(exc)
            stage_attempts.append({"attempt": eval_try, "status": "parse_failed", "error": reason})
            if eval_try < max_eval_attempts:
                stage_prompt = build_eval_retry_prompt(
                    base_prompt=base_eval_prompt,
                    failure_reason=reason,
                    previous_response_text=eval_res_try.stdout,
                    retry_no=eval_try + 1,
                )
                continue
            stage_payload = {"status": "parse_failed", "error": reason}
            break

    if stage_payload is None:
        stage_payload = {"status": "request_failed", "error": "evaluation finished without a result payload"}
    if stage_attempts:
        stage_payload["attempt_count"] = len(stage_attempts)
        stage_payload["attempts"] = stage_attempts
    if stage_last_res is not None:
        _write_text(cfg.logs_dir / f"{theorem_name}.iter{iter_no}.{log_stem}_stdout.log", stage_last_res.stdout)
        _write_text(cfg.logs_dir / f"{theorem_name}.iter{iter_no}.{log_stem}_stderr.log", stage_last_res.stderr)
    _write_json(cfg.logs_dir / f"{theorem_name}.iter{iter_no}.{log_stem}.json", stage_payload)
    return stage_prompt, stage_last_res, stage_payload


# ---------------------------------------------------------------------------
# Post-compile evaluation (including double-check)
# ---------------------------------------------------------------------------

def _run_post_compile_evaluation(
    cfg: RunConfig,
    cache: ResponseCache,
    *,
    problem_json: dict,
    theorem_name: str,
    lean_code: str,
    iter_no: int,
) -> tuple[Optional[str], Optional[CommandResult], Optional[dict[str, object]]]:
    base_eval_prompt = build_formalization_eval_prompt(
        problem_json=problem_json,
        theorem_name=theorem_name,
        lean_code=lean_code,
    )
    gemini_double_check_enabled = (
        not cfg.use_codex_exec
    ) and is_gemini_flash_preview_model(cfg.openrouter_model)

    primary_eval_model = cfg.openrouter_eval_model
    primary_eval_reasoning_effort = cfg.openrouter_eval_reasoning_effort
    if gemini_double_check_enabled:
        primary_eval_model = cfg.openrouter_model

    eval_prompt, eval_res, primary_eval_payload = _run_eval_with_retries(
        cfg, cache,
        base_eval_prompt=base_eval_prompt,
        model=primary_eval_model,
        reasoning_effort=primary_eval_reasoning_effort,
        stage_name="eval",
        log_stem="eval",
        iter_no=iter_no,
        theorem_name=theorem_name,
    )
    eval_payload = dict(primary_eval_payload)

    if gemini_double_check_enabled:
        _, _, secondary_eval_payload = _run_eval_with_retries(
            cfg, cache,
            base_eval_prompt=base_eval_prompt,
            model=GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_MODEL,
            reasoning_effort=GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_REASONING_EFFORT,
            stage_name="eval_gpt52",
            log_stem="eval_gpt52",
            iter_no=iter_no,
            theorem_name=theorem_name,
        )
        _apply_double_check(eval_payload, primary_eval_payload, secondary_eval_payload,
                            primary_eval_model, primary_eval_reasoning_effort)

    return eval_prompt, eval_res, eval_payload


def _apply_double_check(
    eval_payload: dict,
    primary_eval_payload: dict,
    secondary_eval_payload: dict,
    primary_eval_model: str,
    primary_eval_reasoning_effort: str,
) -> None:
    """Merge double-check results into eval_payload in-place."""
    _EVAL_GRADES = {"A", "B", "C", "D"}

    primary_grade_obj = primary_eval_payload.get("grade")
    secondary_grade_obj = secondary_eval_payload.get("grade")
    primary_grade = primary_grade_obj.upper() if isinstance(primary_grade_obj, str) else ""
    secondary_grade = secondary_grade_obj.upper() if isinstance(secondary_grade_obj, str) else ""
    primary_ok = str(primary_eval_payload.get("status", "")).strip() == "ok" and primary_grade in _EVAL_GRADES
    secondary_ok = str(secondary_eval_payload.get("status", "")).strip() == "ok" and secondary_grade in _EVAL_GRADES
    both_a_pass = primary_ok and secondary_ok and primary_grade == "A" and secondary_grade == "A"

    if primary_ok and secondary_ok:
        aggregate_grade = primary_grade
        if grade_below_threshold(secondary_grade, aggregate_grade):
            aggregate_grade = secondary_grade
        eval_payload["grade"] = aggregate_grade

    if both_a_pass:
        eval_payload["status"] = "ok"
        eval_payload["grade"] = "A"
        eval_payload.pop("error", None)
    else:
        eval_payload["status"] = "double_check_failed"
        details: list[str] = []
        if primary_ok:
            details.append(f"Gemini Flash={primary_grade}")
        else:
            details.append("Gemini Flash status=" + str(primary_eval_payload.get("status", "unknown")).strip())
        if secondary_ok:
            details.append(f"GPT-5.2={secondary_grade}")
        else:
            details.append("GPT-5.2 status=" + str(secondary_eval_payload.get("status", "unknown")).strip())
        eval_payload["error"] = (
            "Double-check policy failure: requires grade A from both Gemini Flash and GPT-5.2 evaluator. "
            + "; ".join(details)
        )

    eval_payload["double_check"] = {
        "enabled": True,
        "required_grade": "A",
        "primary_model": primary_eval_model,
        "primary_reasoning_effort": primary_eval_reasoning_effort,
        "secondary_model": GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_MODEL,
        "secondary_reasoning_effort": GEMINI_DOUBLE_CHECK_SECONDARY_EVAL_REASONING_EFFORT,
        "both_a_pass": both_a_pass,
        "primary": primary_eval_payload,
        "secondary": secondary_eval_payload,
    }


# ---------------------------------------------------------------------------
# Policy checks
# ---------------------------------------------------------------------------

def _check_policies(
    lean_code: str,
    *,
    theorem_name: str,
    formalization_only: bool,
    require_no_sorry: bool,
) -> Optional[CommandResult]:
    """Run policy checks on generated Lean code. Returns a failure CommandResult or None if all pass."""
    trivial_token = detect_trivialized_statement(lean_code, theorem_name=theorem_name)
    if trivial_token is not None:
        return CommandResult(
            argv=["(policy)"], returncode=1, stdout="",
            stderr=f"Policy failure: theorem statement was trivialized as `{trivial_token}` "
                   f"(expected faithful formalization of the original problem).",
        )

    if formalization_only and "sorry" not in lean_code:
        return CommandResult(
            argv=["(policy)"], returncode=1, stdout="",
            stderr="Policy failure: formalization-only mode requires a placeholder proof (`sorry`).",
        )

    if require_no_sorry and "sorry" in lean_code:
        return CommandResult(
            argv=["(policy)"], returncode=1, stdout="",
            stderr="Policy failure: generated Lean contains 'sorry'.",
        )

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_problem_file(
    cfg: RunConfig,
    json_path: Path,
    *,
    repo_root: Path,
    prior_json_paths: Optional[list[Path]] = None,
    override_min_eval_grade: Optional[str] = None,
    cache: Optional[ResponseCache] = None,
) -> tuple[bool, list[IterationRecord]]:
    if cache is None:
        cache = ResponseCache(cache_dir=cfg.cache_dir, enabled=cfg.cache_enabled)

    problem_json = json.loads(json_path.read_text(encoding="utf-8"))
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.logs_dir)
    run_cwd = cfg.cwd or repo_root
    prior_json_paths = list(prior_json_paths or [])
    prior_problem_jsons: list[dict] = []
    prior_formalizations: list[tuple[str, str]] = []
    prior_module_imports: list[str] = []

    for prior_json_path in prior_json_paths:
        prior_problem_json = json.loads(prior_json_path.read_text(encoding="utf-8"))
        prior_problem_jsons.append(prior_problem_json)
        prior_prompts = build_prompts(
            prior_problem_json,
            out_dir=cfg.output_dir,
            name_hint=prior_json_path.stem,
            formalization_only=cfg.formalization_only,
        )
        try:
            prior_lean_code = prior_prompts.lean_path.read_text(encoding="utf-8")
        except OSError:
            continue
        prior_formalizations.append((prior_prompts.theorem_name, prior_lean_code))
        module_name = module_name_from_lean_path(prior_prompts.lean_path, run_cwd=run_cwd)
        if module_name:
            prior_module_imports.append(module_name)

    prompts = build_prompts(
        problem_json,
        out_dir=cfg.output_dir,
        name_hint=json_path.stem,
        formalization_only=cfg.formalization_only,
        prior_subproblems=prior_problem_jsons,
        prior_formalizations=prior_formalizations,
    )

    effective_min_eval_grade = (
        override_min_eval_grade if override_min_eval_grade is not None else cfg.min_eval_grade
    )
    prev_lean = ""
    initial_thinking_notes = ""
    records: list[IterationRecord] = []
    error_memory: OrderedDict[str, tuple[str, int, int]] = OrderedDict()

    def _finalize_iteration(
        *,
        iter_no: int,
        thinking_prompt: str,
        coding_prompt: str,
        thinking_res: CommandResult,
        coding_res: CommandResult,
        compiler_res: CommandResult,
        evaluation_prompt: Optional[str] = None,
        evaluation_res: Optional[CommandResult] = None,
        evaluation_payload: Optional[dict[str, object]] = None,
    ) -> None:
        if compiler_res.returncode != 0:
            update_error_memory(error_memory, compiler_res, iter_no=iter_no)
        records.append(
            IterationRecord(iter_no, thinking_res, coding_res, compiler_res, prompts.lean_path)
        )
        _write_iteration_meta(
            logs_dir=cfg.logs_dir,
            theorem_name=prompts.theorem_name,
            iter_no=iter_no,
            thinking_prompt=thinking_prompt,
            coding_prompt=coding_prompt,
            thinking_res=thinking_res,
            coding_res=coding_res,
            compiler_res=compiler_res,
            evaluation_prompt=evaluation_prompt,
            evaluation_res=evaluation_res,
            evaluation_payload=evaluation_payload,
        )

    for it in range(1, cfg.max_iters + 1):
        # --- Build prompts for this iteration ---
        if it == 1:
            thinking_prompt = prompts.initial_thinking_prompt
            coding_base_prompt = prompts.initial_prompt
        else:
            compile_output = (records[-1].compiler.stdout + "\n" + records[-1].compiler.stderr).strip()
            thinking_prompt = "Skipped: phase 5.2 thinking runs only on iteration 1."
            if prev_lean.strip():
                coding_base_prompt = prompts.repair_prompt_template.format(
                    prev_lean=prev_lean, compile_output=compile_output
                )
            else:
                coding_base_prompt = (
                    prompts.initial_prompt
                    + "\n\nPrevious attempt failed before producing a usable Lean file.\n"
                    + "Failure summary:\n" + compile_output
                )
            compact_error_memory = format_error_memory(error_memory, limit=REPAIR_ERROR_MEMORY_LIMIT)
            if compact_error_memory:
                coding_base_prompt += (
                    "\n\nCompact repair memory (recent recurring failures):\n"
                    + compact_error_memory
                    + "\nAvoid reusing these known-failing API names, argument names, and syntax forms."
                )

        # --- Phase 5.2: Thinking ---
        thinking_stdout_path = cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.thinking_stdout.log"
        thinking_stderr_path = cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.thinking_stderr.log"

        if it == 1:
            thinking_model = cfg.openrouter_thinking_model or cfg.openrouter_model
            thinking_res = _call_model(
                cfg, cache,
                prompt=thinking_prompt,
                model=thinking_model,
                reasoning_effort=cfg.openrouter_thinking_reasoning_effort,
                stage="thinking", iter_no=it, theorem_name=prompts.theorem_name,
                enable_web_search=True, logs_dir=cfg.logs_dir,
            )
            _write_text(thinking_stdout_path, thinking_res.stdout)
            _write_text(thinking_stderr_path, thinking_res.stderr)

            if thinking_res.returncode != 0:
                initial_thinking_notes = (
                    "Phase 5.2 thinking request failed on iteration 1. "
                    "Proceed with direct implementation and self-correction.\n"
                    f"Failure: {thinking_res.stderr.strip()}"
                )
            else:
                try:
                    initial_thinking_notes = extract_model_response_text(thinking_res.stdout).strip()
                except ValueError as exc:
                    initial_thinking_notes = (
                        "Phase 5.2 thinking output was unparseable on iteration 1. "
                        "Proceed with direct implementation and self-correction.\n"
                        f"Parse error: {exc}"
                    )
        else:
            thinking_res = CommandResult(
                argv=["(skipped)"], returncode=0,
                stdout="Skipped phase 5.2 thinking on repair iteration; reused iteration-1 planning notes.",
                stderr="",
            )
            _write_text(thinking_stdout_path, thinking_res.stdout)
            _write_text(thinking_stderr_path, thinking_res.stderr)

        # --- Phase 5.3: Coding ---
        coding_prompt = (
            "You are in phase 5.3 (Codex implementation) of the pipeline.\n"
            "Use the phase 5.2 planning notes (from iteration 1) to implement Lean with strict syntax "
            "and to fix holes/coercions.\n\n"
            "Phase 5.2 planning notes (iteration 1):\n"
            f"{initial_thinking_notes}\n\n"
            "Phase 5.3 task:\n"
            f"{coding_base_prompt}"
        )

        coding_stdout_path = cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.coding_stdout.log"
        coding_stderr_path = cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.coding_stderr.log"

        coding_res = _call_model(
            cfg, cache,
            prompt=coding_prompt,
            model=cfg.openrouter_model,
            reasoning_effort=cfg.openrouter_coding_reasoning_effort,
            stage="coding", iter_no=it, theorem_name=prompts.theorem_name,
            enable_web_search=True, logs_dir=cfg.logs_dir,
        )
        _write_text(coding_stdout_path, coding_res.stdout)
        _write_text(coding_stderr_path, coding_res.stderr)

        # --- Extract Lean code ---
        if coding_res.returncode != 0:
            compiler_res = CommandResult(
                argv=[], returncode=1, stdout="",
                stderr=f"Coding stage failed before producing Lean output: {coding_res.stderr.strip() or 'see logs.'}",
            )
            _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)
            _finalize_iteration(
                iter_no=it, thinking_prompt=thinking_prompt, coding_prompt=coding_prompt,
                thinking_res=thinking_res, coding_res=coding_res, compiler_res=compiler_res,
            )
            continue

        try:
            model_text = extract_model_response_text(coding_res.stdout)
            obj = parse_json_object_from_model_text(model_text)
        except ValueError as exc:
            compiler_res = CommandResult(
                argv=[], returncode=1, stdout="",
                stderr=f"Coding output parse failure: {exc}",
            )
            _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)
            _finalize_iteration(
                iter_no=it, thinking_prompt=thinking_prompt, coding_prompt=coding_prompt,
                thinking_res=thinking_res, coding_res=coding_res, compiler_res=compiler_res,
            )
            continue

        lean_code = obj.get("lean")
        if not isinstance(lean_code, str):
            compiler_res = CommandResult(
                argv=[], returncode=1, stdout="",
                stderr="Coding output missing 'lean' field.",
            )
            _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)
            _finalize_iteration(
                iter_no=it, thinking_prompt=thinking_prompt, coding_prompt=coding_prompt,
                thinking_res=thinking_res, coding_res=coding_res, compiler_res=compiler_res,
            )
            continue

        lean_code = inject_imports(lean_code, prior_module_imports)
        _write_text(prompts.lean_path, lean_code)
        prev_lean = lean_code

        # --- Policy checks ---
        policy_failure = _check_policies(
            lean_code,
            theorem_name=prompts.theorem_name,
            formalization_only=cfg.formalization_only,
            require_no_sorry=cfg.require_no_sorry,
        )
        if policy_failure is not None:
            compiler_res = policy_failure
            _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)
            _finalize_iteration(
                iter_no=it, thinking_prompt=thinking_prompt, coding_prompt=coding_prompt,
                thinking_res=thinking_res, coding_res=coding_res, compiler_res=compiler_res,
            )
            continue

        # --- Compile ---
        comp_argv = cfg.compile_argv(prompts.lean_path)
        compiler_stdout_path = cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.compile_stdout.log"
        compiler_stderr_path = cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.compile_stderr.log"

        if cfg.live_logs:
            with (
                compiler_stdout_path.open("w", encoding="utf-8") as comp_out,
                compiler_stderr_path.open("w", encoding="utf-8") as comp_err,
            ):
                compiler_res = compile_lean(
                    comp_argv, cwd=run_cwd, live=True,
                    stdout_sink=comp_out, stderr_sink=comp_err,
                )
        else:
            compiler_res = compile_lean(comp_argv, cwd=run_cwd)

        if not cfg.live_logs:
            _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)

        # --- Post-compile evaluation ---
        eval_prompt: Optional[str] = None
        eval_res: Optional[CommandResult] = None
        eval_payload: Optional[dict[str, object]] = None

        if compiler_res.returncode == 0:
            eval_prompt, eval_res, eval_payload = _run_post_compile_evaluation(
                cfg, cache,
                problem_json=problem_json,
                theorem_name=prompts.theorem_name,
                lean_code=lean_code,
                iter_no=it,
            )

            if eval_res is not None:
                _write_text(cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.eval_stdout.log", eval_res.stdout)
                _write_text(cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.eval_stderr.log", eval_res.stderr)

            _write_json(cfg.logs_dir / f"{prompts.theorem_name}.iter{it}.eval.json", eval_payload)
            _write_json(cfg.output_dir / f"{prompts.theorem_name}.iter{it}.eval.json", eval_payload)
            _write_json(cfg.output_dir / f"{prompts.theorem_name}.eval.json", eval_payload)

            # --- Grade enforcement ---
            gemini_double_check_enabled = (
                not cfg.use_codex_exec
            ) and is_gemini_flash_preview_model(cfg.openrouter_model)

            if gemini_double_check_enabled:
                double_check_obj = eval_payload.get("double_check")
                both_a_pass = isinstance(double_check_obj, dict) and bool(double_check_obj.get("both_a_pass"))
                if not both_a_pass:
                    fail_reason = str(eval_payload.get("error", "")).strip()
                    if not fail_reason:
                        fail_reason = "Policy failure: double-check evaluation requires grade A from both evaluators."
                    eval_feedback = format_eval_feedback_for_repair(eval_payload)
                    if eval_feedback:
                        fail_reason += f"\n{eval_feedback}"
                    compiler_res = CommandResult(
                        argv=["(policy)"], returncode=1,
                        stdout=compiler_res.stdout,
                        stderr=(compiler_res.stderr + "\n" + fail_reason).strip(),
                    )
                    _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)

            if compiler_res.returncode == 0 and effective_min_eval_grade is not None:
                min_grade = effective_min_eval_grade.upper()
                eval_status = str(eval_payload.get("status", "")).strip()
                eval_feedback = format_eval_feedback_for_repair(eval_payload)
                if eval_status != "ok":
                    fail_reason = (
                        f"Policy failure: evaluation result unavailable while enforcing minimum grade {min_grade}. "
                        f"Last status={eval_status or 'unknown'}"
                    )
                    if "error" in eval_payload:
                        fail_reason += f"; error={eval_payload['error']}"
                    if eval_feedback:
                        fail_reason += f"\n{eval_feedback}"
                    compiler_res = CommandResult(
                        argv=["(policy)"], returncode=1,
                        stdout=compiler_res.stdout,
                        stderr=(compiler_res.stderr + "\n" + fail_reason).strip(),
                    )
                    _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)
                else:
                    eval_grade_obj = eval_payload.get("grade")
                    eval_grade = eval_grade_obj.upper() if isinstance(eval_grade_obj, str) else ""
                    if eval_grade not in {"A", "B", "C", "D"}:
                        fail_reason = f"Policy failure: evaluator grade missing/invalid while enforcing minimum grade {min_grade}."
                        if eval_feedback:
                            fail_reason += f"\n{eval_feedback}"
                        compiler_res = CommandResult(
                            argv=["(policy)"], returncode=1,
                            stdout=compiler_res.stdout,
                            stderr=(compiler_res.stderr + "\n" + fail_reason).strip(),
                        )
                        _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)
                    elif grade_below_threshold(eval_grade, min_grade):
                        fail_reason = f"Policy failure: evaluation grade {eval_grade} is below required minimum {min_grade}."
                        if eval_feedback:
                            fail_reason += f"\n{eval_feedback}"
                        compiler_res = CommandResult(
                            argv=["(policy)"], returncode=1,
                            stdout=compiler_res.stdout,
                            stderr=(compiler_res.stderr + "\n" + fail_reason).strip(),
                        )
                        _write_compiler_logs(cfg.logs_dir, prompts.theorem_name, it, compiler_res)

        _finalize_iteration(
            iter_no=it, thinking_prompt=thinking_prompt, coding_prompt=coding_prompt,
            thinking_res=thinking_res, coding_res=coding_res, compiler_res=compiler_res,
            evaluation_prompt=eval_prompt, evaluation_res=eval_res, evaluation_payload=eval_payload,
        )

        if compiler_res.returncode == 0:
            return True, records

    return False, records


def iter_problem_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.json"))
