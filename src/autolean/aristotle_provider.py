"""Aristotle (Harmonic) integration for autoformalization.

Provides LaTeX → Lean 4 autoformalization via the Aristotle API.
This complements AUTOLEAN's direct LLM→Lean generation with a
specialized autoformalizer, matching the workflow used by Kevin Barreto
to solve Erdős problems (GPT generates proof → Aristotle formalizes → Lean verifies).

Requires: pip install aristotlelib
API key: export ARISTOTLE_API_KEY="your-key" or get one at aristotle.harmonic.fun
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .util import CommandResult

logger = logging.getLogger(__name__)

_aristotle_mod = None


def _import_aristotle():
    global _aristotle_mod
    if _aristotle_mod is not None:
        return _aristotle_mod
    try:
        import aristotlelib
        _aristotle_mod = aristotlelib
        return aristotlelib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "aristotlelib is required for Aristotle integration. "
            "Install with: pip install aristotlelib"
        ) from exc


class AristotleAutoformalize:
    """Synchronous wrapper around Aristotle's async API for autoformalization."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        polling_interval: int = 30,
        timeout_seconds: int = 1800,  # 30 min default
    ):
        lib = _import_aristotle()
        key = api_key or os.environ.get("ARISTOTLE_API_KEY")
        if key:
            lib.set_api_key(key)
        self.polling_interval = polling_interval
        self.timeout_seconds = timeout_seconds
        self._lib = lib

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(coro)).result()
        else:
            return asyncio.run(coro)

    def formalize_latex(
        self,
        latex_proof: str,
        *,
        problem_statement: str = "",
        theorem_name: str = "main_theorem",
        output_dir: Optional[Path] = None,
    ) -> CommandResult:
        """Submit a LaTeX proof to Aristotle for autoformalization into Lean 4.

        Args:
            latex_proof: The mathematical proof in LaTeX format.
            problem_statement: Optional problem statement for context.
            theorem_name: Desired theorem name in the output.
            output_dir: Where to save the Lean output. Uses temp dir if None.

        Returns:
            CommandResult with stdout=Lean code if successful, returncode=0.
        """
        prompt = self._build_prompt(latex_proof, problem_statement, theorem_name)
        output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aristotle_"))

        async def _run():
            Project = self._lib.Project

            project = await Project.create(prompt=prompt)
            logger.info("[ARISTOTLE] Project created: %s (status: %s)",
                        project.id, project.status)

            solution_path = await project.wait_for_completion(
                destination=str(output_dir),
                polling_interval_seconds=self.polling_interval,
            )

            await project.refresh()
            return project, solution_path

        try:
            project, solution_path = self._run_async(_run())

            ProjectStatus = self._lib.ProjectStatus

            if project.status == ProjectStatus.COMPLETE:
                lean_code = self._extract_lean_from_output(output_dir)
                return CommandResult(
                    argv=["aristotle", "formalize"],
                    returncode=0,
                    stdout=lean_code,
                    stderr="",
                )
            else:
                return CommandResult(
                    argv=["aristotle", "formalize"],
                    returncode=1,
                    stdout="",
                    stderr=f"Aristotle project finished with status: {project.status}",
                )
        except Exception as exc:
            return CommandResult(
                argv=["aristotle", "formalize"],
                returncode=1,
                stdout="",
                stderr=f"Aristotle error: {exc}",
            )

    def formalize_lean_file(
        self,
        lean_code: str,
        *,
        instruction: str = "Fix and complete this Lean 4 proof. Fill all sorry placeholders.",
        output_dir: Optional[Path] = None,
    ) -> CommandResult:
        """Submit an existing Lean file to Aristotle for proof completion.

        This is useful for fixing compilation errors or filling sorry holes
        that AUTOLEAN's compile-repair loop couldn't resolve.
        """
        output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aristotle_"))

        # Create a tar with the lean file
        tar_path = output_dir / "input.tar"
        lean_path = output_dir / "Main.lean"
        lean_path.write_text(lean_code, encoding="utf-8")

        with tarfile.open(tar_path, "w") as tar:
            tar.add(lean_path, arcname="Main.lean")

        async def _run():
            Project = self._lib.Project
            project = await Project.create(
                prompt=instruction,
                tar_file_path=tar_path,
            )
            logger.info("[ARISTOTLE] Project created with Lean file: %s", project.id)
            solution_path = await project.wait_for_completion(
                destination=str(output_dir / "solution"),
                polling_interval_seconds=self.polling_interval,
            )
            await project.refresh()
            return project, solution_path

        try:
            project, solution_path = self._run_async(_run())

            ProjectStatus = self._lib.ProjectStatus
            if project.status == ProjectStatus.COMPLETE:
                lean_code = self._extract_lean_from_output(output_dir / "solution")
                return CommandResult(
                    argv=["aristotle", "complete"],
                    returncode=0,
                    stdout=lean_code,
                    stderr="",
                )
            else:
                return CommandResult(
                    argv=["aristotle", "complete"],
                    returncode=1,
                    stdout="",
                    stderr=f"Aristotle status: {project.status}",
                )
        except Exception as exc:
            return CommandResult(
                argv=["aristotle", "complete"],
                returncode=1,
                stdout="",
                stderr=f"Aristotle error: {exc}",
            )

    @staticmethod
    def _build_prompt(
        latex_proof: str,
        problem_statement: str,
        theorem_name: str,
    ) -> str:
        parts = []
        if problem_statement:
            parts.append(f"Problem statement:\n{problem_statement}\n")
        parts.append(f"Formalize the following proof into Lean 4 using Mathlib.")
        parts.append(f"The main theorem should be named `{theorem_name}`.")
        parts.append(f"Use `import Mathlib` and put everything in namespace `Formalizations`.")
        parts.append(f"\nProof:\n{latex_proof}")
        return "\n".join(parts)

    @staticmethod
    def _extract_lean_from_output(output_dir: Path) -> str:
        """Find and read Lean files from Aristotle's output."""
        output_dir = Path(output_dir)

        # Aristotle may output a tar or directory with .lean files
        # Check for tar first
        for tar_path in output_dir.rglob("*.tar"):
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".lean"):
                        f = tar.extractfile(member)
                        if f:
                            return f.read().decode("utf-8")

        # Check for direct .lean files
        for lean_file in sorted(output_dir.rglob("*.lean")):
            return lean_file.read_text(encoding="utf-8")

        # Check for any text output
        for txt_file in sorted(output_dir.rglob("*.txt")):
            content = txt_file.read_text(encoding="utf-8")
            if "theorem" in content or "import" in content:
                return content

        return ""


def formalize_with_aristotle(
    latex_proof: str,
    *,
    problem_statement: str = "",
    theorem_name: str = "main_theorem",
    api_key: Optional[str] = None,
) -> CommandResult:
    """Convenience function: formalize a LaTeX proof via Aristotle.

    Returns CommandResult with Lean code in stdout if successful.
    """
    client = AristotleAutoformalize(api_key=api_key)
    return client.formalize_latex(
        latex_proof,
        problem_statement=problem_statement,
        theorem_name=theorem_name,
    )
