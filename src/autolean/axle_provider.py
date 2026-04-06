"""AXLE integration module for AUTOLEAN's compile-check-repair loop.

Wraps Axiom's AXLE API (axiom-axle Python package) behind a synchronous
interface that returns ``CommandResult`` objects compatible with the rest
of the AUTOLEAN pipeline.

The ``axiom-axle`` package is **not** imported at module level so that
the rest of the codebase can be imported and tested even when the package
is not installed.  A clear error is raised at construction time if the
dependency is missing.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .util import CommandResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_axle_mod: Any = None  # cached reference to the axiom.axle package


def _import_axle() -> Any:
    """Return the ``axiom.axle`` module, importing it lazily on first call."""
    global _axle_mod
    if _axle_mod is not None:
        return _axle_mod
    try:
        import axiom.axle as axle  # type: ignore[import-untyped]
        _axle_mod = axle
        return axle
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The axiom-axle package is required for AXLE integration but is "
            "not installed.  Install it with:  pip install axiom-axle"
        ) from exc


# ---------------------------------------------------------------------------
# Internal async helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 4
_RETRY_BASE_DELAY: float = 1.0  # seconds; doubles each attempt


def _is_retryable(exc: BaseException) -> bool:
    """Return True if *exc* is an AXLE transient/rate-limit error."""
    axle = _import_axle()
    retryable_types: Tuple[type, ...] = ()
    for name in ("AxleIsUnavailable", "AxleRateLimitedError"):
        cls = getattr(axle, name, None)
        if cls is not None:
            retryable_types = (*retryable_types, cls)
    if not retryable_types:
        return False
    return isinstance(exc, retryable_types)


async def _retry_async(
    coro_factory: Any,
    *,
    semaphore: asyncio.Semaphore,
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """Execute an async call with concurrency limiting and retry logic.

    *coro_factory* must be a zero-argument callable that returns a new
    awaitable each time it is called (so that retries get a fresh coroutine).
    """
    delay = _RETRY_BASE_DELAY
    last_exc: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < max_retries:
                logger.warning(
                    "AXLE retryable error (attempt %d/%d): %s – retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise
    # Unreachable, but keeps mypy happy.
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AxleVerifier
# ---------------------------------------------------------------------------

class AxleVerifier:
    """Synchronous wrapper around the AXLE async client.

    Parameters
    ----------
    api_key:
        Axiom API key.  When *None* the ``AXIOM_API_KEY`` environment
        variable is read.
    environment:
        Lean toolchain environment tag understood by AXLE
        (e.g. ``"lean-4.28.0"``).
    max_concurrency:
        Maximum number of in-flight AXLE requests.  Defaults to 20.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str = "lean-4.28.0",
        max_concurrency: int = 20,
    ) -> None:
        axle = _import_axle()

        resolved_key = api_key or os.environ.get("AXIOM_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No AXLE API key provided.  Pass api_key= or set the "
                "AXIOM_API_KEY environment variable."
            )

        self._api_key: str = resolved_key
        self._environment: str = environment
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrency)
        self._client_cls: type = axle.AxleClient

    # -- private helpers ---------------------------------------------------

    def _run_async(self, coro_factory: Any) -> Any:
        """Run an async coroutine factory synchronously.

        Handles the common case where an event loop may or may not already
        be running in the current thread.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside an already-running loop (e.g. Jupyter).
            # Spin up a helper thread so we don't deadlock.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _retry_async(coro_factory, semaphore=self._semaphore))
                return future.result()
        else:
            return asyncio.run(
                _retry_async(coro_factory, semaphore=self._semaphore)
            )

    def _make_client(self) -> Any:
        """Create a fresh ``AxleClient`` instance."""
        return self._client_cls(api_key=self._api_key)

    # -- public synchronous API -------------------------------------------

    def verify(self, lean_code: str) -> CommandResult:
        """Compile / verify *lean_code* via AXLE and return a ``CommandResult``.

        ``returncode`` is 0 when the proof is valid, 1 otherwise.  Compiler
        diagnostics (errors, warnings) are placed in ``stderr``.
        """

        async def _call() -> Any:
            client = self._make_client()
            return await client.verify_proof(
                code=lean_code,
                environment=self._environment,
            )

        try:
            result = self._run_async(_call)
        except Exception as exc:
            return CommandResult(
                argv=["axle", "verify_proof"],
                returncode=1,
                stdout="",
                stderr=f"AXLE verify_proof failed: {exc}",
            )

        # The AXLE SDK returns an object with .is_valid, .errors, .warnings
        # (exact attribute names may vary; adapt defensively).
        is_valid: bool = getattr(result, "is_valid", False)
        errors: str = getattr(result, "errors", "") or ""
        warnings: str = getattr(result, "warnings", "") or ""
        stdout_text: str = getattr(result, "stdout", "") or ""

        stderr_parts: List[str] = []
        if errors:
            stderr_parts.append(errors if isinstance(errors, str) else str(errors))
        if warnings:
            stderr_parts.append(warnings if isinstance(warnings, str) else str(warnings))

        return CommandResult(
            argv=["axle", "verify_proof"],
            returncode=0 if is_valid else 1,
            stdout=stdout_text,
            stderr="\n".join(stderr_parts),
        )

    def sorry_to_lemmas(self, lean_code: str) -> Tuple[str, List[str]]:
        """Replace ``sorry`` placeholders with extracted lemmas.

        Calls AXLE's ``sorry2lemma`` endpoint with aggressive extraction
        flags and returns a ``(modified_code, lemma_names)`` pair.
        """

        async def _call() -> Any:
            client = self._make_client()
            return await client.sorry2lemma(
                code=lean_code,
                environment=self._environment,
                extract_sorries=True,
                extract_errors=True,
                include_whole_context=True,
                reconstruct_callsite=True,
            )

        result = self._run_async(_call)

        modified_code: str = getattr(result, "code", lean_code)
        lemma_names: List[str] = list(getattr(result, "lemma_names", []))

        return modified_code, lemma_names

    def extract_theorems(self, lean_code: str) -> Dict[str, Any]:
        """Extract theorem metadata from *lean_code* via AXLE.

        Returns a dict keyed by theorem name with whatever metadata the
        AXLE SDK provides (statement, proof, dependencies, …).
        """

        async def _call() -> Any:
            client = self._make_client()
            return await client.extract_theorems(
                code=lean_code,
                environment=self._environment,
            )

        result = self._run_async(_call)

        # Normalise to a plain dict regardless of SDK return type.
        if isinstance(result, dict):
            return result
        # Some SDK versions return a list of theorem objects.
        if isinstance(result, (list, tuple)):
            out: Dict[str, Any] = {}
            for item in result:
                name = getattr(item, "name", None) or str(item)
                out[name] = item if not hasattr(item, "__dict__") else vars(item)
            return out
        # Fallback: wrap the raw result.
        return {"_raw": result}

    def repair(self, lean_code: str) -> str:
        """Attempt to automatically repair *lean_code* via AXLE.

        Returns the repaired Lean source (which may be identical to the
        input if no repair was possible).
        """

        async def _call() -> Any:
            client = self._make_client()
            return await client.repair_proofs(
                code=lean_code,
                environment=self._environment,
            )

        result = self._run_async(_call)

        repaired: str = getattr(result, "code", None) or lean_code
        return repaired


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def compile_via_axle(lean_code: str, verifier: AxleVerifier) -> CommandResult:
    """Compile *lean_code* through AXLE and return a ``CommandResult``.

    This is a drop-in replacement for the subprocess-based compilation
    used elsewhere in AUTOLEAN.  The returned ``CommandResult`` follows
    the same conventions:

    * ``returncode == 0`` — code compiled / verified successfully.
    * ``returncode == 1`` — compilation failed; diagnostics in ``stderr``.
    * ``argv`` is set to ``["axle", "compile"]`` for logging purposes.
    """
    result = verifier.verify(lean_code)
    # Re-wrap so that argv always reads "compile" in log output, even if
    # the verifier used a different label internally.
    return CommandResult(
        argv=["axle", "compile"],
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
