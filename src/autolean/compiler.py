"""Lean compiler interaction, error extraction, and error memory."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .util import CommandResult
from .providers import run_subprocess

_LEAN_LOCATION_PREFIX_RE = re.compile(r"^(?:[A-Za-z]:)?[^:\s]*\.lean:\d+:\d+:\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_LEAN_MODULE_PART_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")

REPAIR_ERROR_MEMORY_LIMIT = 6


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def compile_lean(
    argv: list[str],
    *,
    cwd: Path,
    live: bool = False,
    stdout_sink=None,
    stderr_sink=None,
) -> CommandResult:
    return run_subprocess(
        argv, cwd=cwd, live=live,
        stdout_sink=stdout_sink, stderr_sink=stderr_sink,
    )


# ---------------------------------------------------------------------------
# Error extraction and memory
# ---------------------------------------------------------------------------

def extract_compact_error_lines(compiler_res: CommandResult) -> list[str]:
    combined = (compiler_res.stdout + "\n" + compiler_res.stderr).strip()
    if not combined:
        return []

    lines: list[str] = []
    for raw in combined.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if (
            "error" in lowered
            or "parse failure" in lowered
            or "policy failure" in lowered
            or "failed before producing lean output" in lowered
        ):
            lines.append(line)

    if lines:
        return lines

    for raw in combined.splitlines():
        line = raw.strip()
        if line:
            return [line]
    return []


def normalize_error_line(line: str) -> str:
    normalized = _LEAN_LOCATION_PREFIX_RE.sub("", line.strip())
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def update_error_memory(
    memory: OrderedDict[str, tuple[str, int, int]],
    compiler_res: CommandResult,
    *,
    iter_no: int,
) -> None:
    for line in extract_compact_error_lines(compiler_res):
        key = normalize_error_line(line)
        if not key:
            continue
        if key in memory:
            _last_line, count, _last_iter = memory[key]
            memory[key] = (key, count + 1, iter_no)
            memory.move_to_end(key)
        else:
            memory[key] = (key, 1, iter_no)


def format_error_memory(memory: OrderedDict[str, tuple[str, int, int]], *, limit: int) -> str:
    if limit <= 0 or not memory:
        return ""
    recent_items = list(memory.items())[-limit:]
    lines: list[str] = []
    for idx, (_key, (display, count, last_iter)) in enumerate(reversed(recent_items), start=1):
        if count > 1:
            lines.append(f"{idx}. [seen {count}x, last iter {last_iter}] {display}")
        else:
            lines.append(f"{idx}. [iter {last_iter}] {display}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lean code analysis
# ---------------------------------------------------------------------------

def extract_top_level_prop_from_theorem_header(header: str) -> Optional[str]:
    depth = 0
    last_colon = -1
    for i, ch in enumerate(header):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            if i + 1 < len(header) and header[i + 1] == "=":
                continue
            last_colon = i
    if last_colon < 0:
        return None
    return header[last_colon + 1:].strip()


def detect_trivialized_statement(lean_code: str, *, theorem_name: str) -> Optional[str]:
    start_re = re.compile(rf"\b(?:theorem|lemma)\s+{re.escape(theorem_name)}\b")
    m = start_re.search(lean_code)
    if m is None:
        return None
    end = lean_code.find(":=", m.end())
    if end < 0:
        return None
    header = lean_code[m.start():end]
    prop = extract_top_level_prop_from_theorem_header(header)
    if not prop:
        return None
    match = re.match(r"^\(?\s*(True|False)\b", prop)
    if match is None:
        return None
    return match.group(1)


def module_name_from_lean_path(lean_path: Path, *, run_cwd: Path) -> Optional[str]:
    try:
        rel = lean_path.resolve().relative_to(run_cwd.resolve())
    except ValueError:
        return None
    if rel.suffix != ".lean":
        return None
    parts = rel.with_suffix("").parts
    if not parts:
        return None
    for part in parts:
        if not _LEAN_MODULE_PART_RE.fullmatch(part):
            return None
    return ".".join(parts)


def inject_imports(lean_code: str, module_names: list[str]) -> str:
    if not module_names:
        return lean_code

    ordered_modules: list[str] = []
    seen: set[str] = set()
    for module in module_names:
        module = module.strip()
        if not module or module in seen:
            continue
        seen.add(module)
        ordered_modules.append(module)
    if not ordered_modules:
        return lean_code

    lines = lean_code.splitlines()
    existing_imports: set[str] = set()
    insert_at = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            insert_at = idx + 1
            continue
        if stripped.startswith("--"):
            insert_at = idx + 1
            continue
        if stripped.startswith("import "):
            module = stripped[len("import "):].strip()
            if module:
                existing_imports.add(module)
            insert_at = idx + 1
            continue
        break

    missing_import_lines = [
        f"import {module}" for module in ordered_modules if module not in existing_imports
    ]
    if not missing_import_lines:
        return lean_code

    merged_lines = lines[:insert_at] + missing_import_lines + lines[insert_at:]
    merged = "\n".join(merged_lines)
    if lean_code.endswith("\n"):
        return merged + "\n"
    return merged
