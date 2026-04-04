"""LLM provider backends: OpenRouter, Codex Exec, Claude CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from http.client import IncompleteRead
from pathlib import Path
from typing import Optional, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .util import CommandResult, ensure_dir

CODEX_EXEC_CODING_MODEL = "gpt-5.3-codex-spark"
CODEX_EXEC_CODING_FALLBACK_MODEL = "gpt-5.3-codex"
CODEX_EXEC_CODING_REASONING_EFFORT = "xhigh"


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    stdin_text: Optional[str] = None,
    live: bool = False,
    stdout_sink: Optional[TextIO] = None,
    stderr_sink: Optional[TextIO] = None,
) -> CommandResult:
    if not live:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            input=stdin_text,
            text=True,
            capture_output=True,
            check=False,
        )
        return CommandResult(
            argv=argv, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
        )

    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if proc.stdin is not None:
        if stdin_text is not None:
            proc.stdin.write(stdin_text)
        proc.stdin.close()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _reader(stream, chunks: list[str], sink: Optional[TextIO]) -> None:
        if stream is None:
            return
        for line in stream:
            chunks.append(line)
            if sink is not None:
                sink.write(line)
                sink.flush()
        stream.close()

    t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks, stdout_sink))
    t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks, stderr_sink))
    t_out.start()
    t_err.start()
    returncode = proc.wait()
    t_out.join()
    t_err.join()

    return CommandResult(
        argv=argv,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backoff_sleep(attempt: int) -> None:
    delay_s = min(4.0, 0.5 * (2 ** attempt))
    time.sleep(delay_s)


def _decode_incomplete_read_partial(exc: IncompleteRead) -> str:
    partial = exc.partial
    if isinstance(partial, bytes):
        return partial.decode("utf-8", errors="replace")
    if isinstance(partial, str):
        return partial
    return ""


def _parse_shell_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].strip()
    if "=" not in stripped:
        return None
    name, raw_value = stripped.split("=", 1)
    name = name.strip()
    if not name:
        return None
    value = raw_value.strip()
    if not value:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    else:
        value = value.split("#", 1)[0].strip()
    if not value:
        return None
    return name, value


def _read_var_from_zshrc(var_name: str, *, zshrc_path: Optional[Path] = None) -> Optional[str]:
    path = zshrc_path or (Path.home() / ".zshrc")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        parsed = _parse_shell_assignment(line)
        if parsed is None:
            continue
        name, value = parsed
        if name == var_name:
            return value
    return None


def _resolve_openrouter_api_key(var_name: str) -> Optional[str]:
    env_value = os.environ.get(var_name)
    if env_value:
        return env_value
    return _read_var_from_zshrc(var_name)


def _normalize_codex_model_name(model: Optional[str]) -> Optional[str]:
    if model is None:
        return None
    normalized = model.strip()
    if not normalized:
        return None
    if normalized.lower().startswith("openai/"):
        _, suffix = normalized.split("/", 1)
        normalized = suffix.strip()
    return normalized or None


def _is_codex_model_not_found(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    return "model_not_found" in lowered or "does not exist" in lowered


# ---------------------------------------------------------------------------
# OpenRouter Chat API
# ---------------------------------------------------------------------------

def call_openrouter_chat(
    *,
    prompt: str,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout_s: int,
    max_retries: int,
    reasoning_effort: Optional[str] = None,
    openrouter_web_search: bool = False,
    openrouter_web_search_engine: Optional[str] = None,
    openrouter_web_search_max_results: Optional[int] = None,
) -> CommandResult:
    api_key = _resolve_openrouter_api_key(api_key_env)
    endpoint = base_url.rstrip("/") + "/chat/completions"
    argv = ["POST", endpoint]

    if not api_key:
        return CommandResult(
            argv=argv,
            returncode=1,
            stdout="",
            stderr=(
                f"Missing OpenRouter API key. Set env var '{api_key_env}' or add "
                f"'{api_key_env}=...' to ~/.zshrc."
            ),
        )

    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if openrouter_web_search:
        web_plugin: dict[str, object] = {"id": "web"}
        if openrouter_web_search_engine:
            web_plugin["engine"] = openrouter_web_search_engine
        if openrouter_web_search_max_results is not None:
            web_plugin["max_results"] = openrouter_web_search_max_results
        payload["plugins"] = [web_plugin]
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    app_title = os.environ.get("OPENROUTER_APP_TITLE")
    if app_title:
        headers["X-Title"] = app_title

    attempts = max(0, max_retries) + 1
    last_err = "OpenRouter request failed."
    last_stdout = ""

    for attempt in range(attempts):
        req = Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                try:
                    raw_bytes = resp.read()
                except IncompleteRead as exc:
                    partial_text = _decode_incomplete_read_partial(exc)
                    if partial_text:
                        try:
                            json.loads(partial_text)
                            return CommandResult(argv=argv, returncode=0, stdout=partial_text, stderr="")
                        except json.JSONDecodeError:
                            pass
                    last_stdout = partial_text
                    last_err = "OpenRouter response ended early (IncompleteRead) and payload was incomplete JSON."
                    if attempt + 1 < attempts:
                        _backoff_sleep(attempt)
                        continue
                    return CommandResult(argv=argv, returncode=1, stdout=last_stdout, stderr=last_err)
                raw = raw_bytes.decode("utf-8", errors="replace")
                return CommandResult(argv=argv, returncode=0, stdout=raw, stderr="")
        except HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_stdout = err_body
            last_err = f"OpenRouter HTTP {exc.code}: {exc.reason}"
            if exc.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                _backoff_sleep(attempt)
                continue
            return CommandResult(argv=argv, returncode=1, stdout=err_body, stderr=last_err)
        except URLError as exc:
            last_err = f"OpenRouter request failed: {exc.reason}"
            if attempt + 1 < attempts:
                _backoff_sleep(attempt)
                continue
            return CommandResult(argv=argv, returncode=1, stdout=last_stdout, stderr=last_err)
        except IncompleteRead as exc:
            partial_text = _decode_incomplete_read_partial(exc)
            last_stdout = partial_text
            last_err = "OpenRouter response ended early (IncompleteRead)."
            if attempt + 1 < attempts:
                _backoff_sleep(attempt)
                continue
            return CommandResult(argv=argv, returncode=1, stdout=last_stdout, stderr=last_err)
        except OSError as exc:
            last_err = f"OpenRouter request failed: {exc}"
            if attempt + 1 < attempts:
                _backoff_sleep(attempt)
                continue
            return CommandResult(argv=argv, returncode=1, stdout=last_stdout, stderr=last_err)

    return CommandResult(argv=argv, returncode=1, stdout=last_stdout, stderr=last_err)


# ---------------------------------------------------------------------------
# Codex Exec CLI
# ---------------------------------------------------------------------------

def call_codex_exec(
    *,
    prompt: str,
    out_message_path: Path,
    model: Optional[str],
    reasoning_effort: Optional[str],
    sandbox: str,
    workdir: Path,
    live_logs: bool = False,
    stdout_sink: Optional[TextIO] = None,
    stderr_sink: Optional[TextIO] = None,
) -> CommandResult:
    ensure_dir(out_message_path.parent)
    normalized_model = _normalize_codex_model_name(model)

    def _build_argv(target_model: Optional[str]) -> list[str]:
        argv = ["codex", "exec"]
        if target_model:
            argv += ["--model", target_model]
        if reasoning_effort:
            argv += ["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"]
        argv += [
            "--color", "never",
            "--skip-git-repo-check",
            "--sandbox", sandbox,
            "--output-last-message", str(out_message_path),
            "-",
        ]
        return argv

    argv = _build_argv(normalized_model)
    codex_res = run_subprocess(
        argv, cwd=workdir, stdin_text=prompt,
        live=live_logs, stdout_sink=stdout_sink, stderr_sink=stderr_sink,
    )
    if (
        codex_res.returncode != 0
        and normalized_model == CODEX_EXEC_CODING_MODEL
        and _is_codex_model_not_found(codex_res.stderr)
    ):
        argv = _build_argv(CODEX_EXEC_CODING_FALLBACK_MODEL)
        codex_res = run_subprocess(
            argv, cwd=workdir, stdin_text=prompt,
            live=live_logs, stdout_sink=stdout_sink, stderr_sink=stderr_sink,
        )
    if codex_res.returncode != 0:
        return codex_res

    try:
        message = out_message_path.read_text(encoding="utf-8")
    except OSError as exc:
        return CommandResult(
            argv=argv, returncode=1, stdout=codex_res.stdout,
            stderr=f"codex exec succeeded but --output-last-message was unreadable: {out_message_path}: {exc}",
        )
    return CommandResult(argv=argv, returncode=0, stdout=message, stderr=codex_res.stderr)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_openrouter_message_content(response_obj: dict) -> str:
    choices = response_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response missing choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenRouter response contains invalid choice payload.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response missing message object.")

    content = message.get("content")
    if isinstance(content, str):
        if content.strip():
            return content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        merged = "".join(parts).strip()
        if merged:
            return merged

    for candidate in (message.get("reasoning"), first.get("reasoning")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    for details in (message.get("reasoning_details"), first.get("reasoning_details")):
        if not isinstance(details, list):
            continue
        parts = []
        for item in details:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary")
            if isinstance(summary, str) and summary.strip():
                parts.append(summary.strip())
        if parts:
            return "\n\n".join(parts)
    raise ValueError("OpenRouter response message content is empty or not text.")


def extract_model_response_text(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped:
        raise ValueError("Model response was empty.")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if not isinstance(parsed, dict):
        return stripped
    try:
        content = extract_openrouter_message_content(parsed).strip()
    except ValueError:
        return stripped
    if not content:
        raise ValueError("Model response message content was empty.")
    return content


def parse_json_object_from_model_text(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model response was empty.")
    candidates = [stripped]
    if stripped.startswith("```"):
        chunks = stripped.split("```")
        if len(chunks) >= 3:
            candidates.append(chunks[1].removeprefix("json").strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("Could not parse a JSON object from model response text.")
