"""Mathlib lemma retrieval for AUTOLEAN's proving pipeline.

Embeds Mathlib theorem signatures with a sentence-transformer model and
performs nearest-neighbor lookup via a FAISS index so that the proving loop
can inject relevant premises into its prompts.

Heavy dependencies (torch, sentence_transformers, faiss) are imported lazily
so the rest of the package stays lightweight.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# Lazy imports for heavy optional deps
# ---------------------------------------------------------------------------

def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss is required for the retrieval index.  "
            "Install it with:  pip install faiss-cpu   (or faiss-gpu for CUDA)"
        ) from exc
    return faiss


def _import_sentence_transformers() -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedding theorems.  "
            "Install it with:  pip install sentence-transformers"
        ) from exc
    return sentence_transformers


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for the retrieval index.  "
            "Install it with:  pip install numpy"
        ) from exc
    return numpy


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedPremise:
    """A single Mathlib theorem returned by the retrieval index."""
    name: str
    signature: str
    module_path: str
    score: float


# ---------------------------------------------------------------------------
# Corpus building
# ---------------------------------------------------------------------------

# Matches declarations like:
#   theorem foo : ... := by
#   lemma bar (n : Nat) : ... := sorry
# We capture *name* and *signature* (the type between the name and the
# definition body).  The regex is intentionally loose to tolerate varied
# formatting.

_DECL_RE = re.compile(
    r"^(?:theorem|lemma)\s+"       # keyword
    r"(\S+)"                       # name (group 1)
    r"(.*?)"                       # signature (group 2) -- non-greedy
    r"\s*:=\s*(?:by|sorry)\b",     # end sentinel
    re.DOTALL,
)


def _extract_declarations(source: str) -> list[tuple[str, str]]:
    """Return (name, signature) pairs found in a Lean source string."""
    results: list[tuple[str, str]] = []
    # We split on top-level `theorem`/`lemma` keywords to avoid a single
    # giant regex match across the whole file.
    parts = re.split(r"(?=^(?:theorem|lemma)\s)", source, flags=re.MULTILINE)
    for part in parts:
        m = _DECL_RE.match(part)
        if m:
            name = m.group(1).strip()
            sig = " ".join(m.group(2).split())  # normalise whitespace
            if name and sig:
                results.append((name, sig))
    return results


def build_corpus_from_lean_project(project_dir: Path, output_path: Path) -> int:
    """Scan ``.lean`` files under *project_dir* for theorem/lemma declarations.

    Writes one JSON object per line to *output_path* with keys
    ``name``, ``signature``, ``module_path``.

    Returns the total number of extracted theorems.
    """
    project_dir = Path(project_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for lean_file in sorted(project_dir.rglob("*.lean")):
            # Derive a dotted module path from the file's relative location.
            try:
                rel = lean_file.relative_to(project_dir)
            except ValueError:
                continue
            module_path = ".".join(rel.with_suffix("").parts)

            try:
                source = lean_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for name, sig in _extract_declarations(source):
                record = {
                    "name": name,
                    "signature": sig,
                    "module_path": module_path,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return count


# ---------------------------------------------------------------------------
# FAISS-backed retrieval index
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_INDEX_FILENAME = "index.faiss"
_CORPUS_FILENAME = "corpus.jsonl"


class MathLibIndex:
    """Embeds Mathlib theorems and answers nearest-neighbor queries.

    Typical lifecycle::

        # Build from a corpus JSONL and persist to disk.
        idx = MathLibIndex.from_corpus("corpus.jsonl", model_name="all-MiniLM-L6-v2")
        idx.save("./index_dir")

        # Reload later without re-embedding.
        idx = MathLibIndex.load("./index_dir")
        premises = idx.retrieve("even number squared is even", k=10)
    """

    def __init__(
        self,
        entries: list[dict[str, str]],
        index: Any,  # faiss.Index
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._entries = entries
        self._index = index
        self._model_name = model_name
        # The encoder is loaded lazily on first query so that ``load()``
        # stays fast when only the FAISS index is needed right away.
        self._encoder: Any = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        *,
        model_name: str = _DEFAULT_MODEL,
        show_progress: bool = True,
    ) -> MathLibIndex:
        """Build an index by embedding every entry in a JSONL corpus file."""
        faiss = _import_faiss()
        np = _import_numpy()
        st = _import_sentence_transformers()

        corpus_path = Path(corpus_path)
        entries: list[dict[str, str]] = []
        with open(corpus_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        if not entries:
            raise ValueError(f"Corpus file is empty: {corpus_path}")

        # Build textual representations for embedding.
        texts = [
            f"{e['name']} : {e['signature']}"
            for e in entries
        ]

        encoder = st.SentenceTransformer(model_name)
        embeddings = encoder.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner-product (cosine after L2-norm)
        index.add(embeddings)

        obj = cls(entries, index, model_name=model_name)
        obj._encoder = encoder
        return obj

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Write the FAISS index and corpus metadata to *directory*."""
        faiss = _import_faiss()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / _INDEX_FILENAME))

        with open(directory / _CORPUS_FILENAME, "w", encoding="utf-8") as fout:
            for entry in self._entries:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Persist model name so we reload the correct encoder.
        meta = {"model_name": self._model_name}
        with open(directory / "meta.json", "w", encoding="utf-8") as fout:
            json.dump(meta, fout)

    @classmethod
    def load(cls, directory: str | Path) -> MathLibIndex:
        """Load a previously saved index from *directory*."""
        faiss = _import_faiss()
        directory = Path(directory)

        index = faiss.read_index(str(directory / _INDEX_FILENAME))

        entries: list[dict[str, str]] = []
        with open(directory / _CORPUS_FILENAME, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        model_name = _DEFAULT_MODEL
        meta_path = directory / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fin:
                meta = json.load(fin)
            model_name = meta.get("model_name", _DEFAULT_MODEL)

        return cls(entries, index, model_name=model_name)

    # -- querying -----------------------------------------------------------

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            st = _import_sentence_transformers()
            self._encoder = st.SentenceTransformer(self._model_name)
        return self._encoder

    def retrieve(self, query: str, k: int = 25) -> list[RetrievedPremise]:
        """Return the top-*k* premises most similar to *query*."""
        np = _import_numpy()
        encoder = self._get_encoder()

        vec = encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vec = np.asarray(vec, dtype="float32")

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results: list[RetrievedPremise] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            e = self._entries[int(idx)]
            results.append(
                RetrievedPremise(
                    name=e["name"],
                    signature=e["signature"],
                    module_path=e["module_path"],
                    score=float(score),
                )
            )
        return results

    def retrieve_batch(
        self,
        queries: list[str],
        k: int = 25,
    ) -> list[list[RetrievedPremise]]:
        """Batch version of :meth:`retrieve` -- one result list per query."""
        np = _import_numpy()
        encoder = self._get_encoder()

        vecs = encoder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vecs = np.asarray(vecs, dtype="float32")

        k = min(k, self._index.ntotal)
        all_scores, all_indices = self._index.search(vecs, k)

        batch_results: list[list[RetrievedPremise]] = []
        for scores, indices in zip(all_scores, all_indices):
            results: list[RetrievedPremise] = []
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                e = self._entries[int(idx)]
                results.append(
                    RetrievedPremise(
                        name=e["name"],
                        signature=e["signature"],
                        module_path=e["module_path"],
                        score=float(score),
                    )
                )
            batch_results.append(results)
        return batch_results

    def __len__(self) -> int:
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"MathLibIndex(entries={len(self._entries)}, "
            f"model={self._model_name!r})"
        )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4  # rough estimate used for budget checks


def format_premises_for_prompt(
    premises: list[RetrievedPremise],
    max_tokens: int = 3000,
) -> str:
    """Format retrieved premises as a text block for prompt injection.

    Each premise is rendered as::

        -- {module_path}
        {name} : {signature}

    Premises are appended in order until the estimated token budget
    (*max_tokens*, at ~4 characters per token) is exhausted.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    lines: list[str] = []
    used = 0
    for p in premises:
        block = f"-- {p.module_path}\n{p.name} : {p.signature}"
        # +1 for the blank line separator between blocks
        cost = len(block) + 1
        if used + cost > max_chars:
            break
        lines.append(block)
        used += cost
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_build(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir)
    output = Path(args.output)
    if not project_dir.is_dir():
        print(f"Error: project directory does not exist: {project_dir}", file=sys.stderr)
        sys.exit(1)
    count = build_corpus_from_lean_project(project_dir, output)
    print(f"Extracted {count} theorem/lemma declarations -> {output}")


def _cli_index(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus)
    index_dir = Path(args.index_dir)
    model_name = args.model or _DEFAULT_MODEL
    if not corpus_path.is_file():
        print(f"Error: corpus file does not exist: {corpus_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Building index from {corpus_path} with model {model_name!r} ...")
    idx = MathLibIndex.from_corpus(corpus_path, model_name=model_name)
    idx.save(index_dir)
    print(f"Saved index ({len(idx)} vectors) -> {index_dir}")


def _cli_query(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    query = args.query
    k = args.k
    if not index_dir.is_dir():
        print(f"Error: index directory does not exist: {index_dir}", file=sys.stderr)
        sys.exit(1)
    idx = MathLibIndex.load(index_dir)
    results = idx.retrieve(query, k=k)
    if not results:
        print("No results found.")
        return
    for r in results:
        print(f"[{r.score:.4f}]  {r.name} : {r.signature}")
        print(f"         -- {r.module_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="autolean.retrieval",
        description="Mathlib lemma retrieval for AUTOLEAN.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- build --------------------------------------------------------------
    p_build = sub.add_parser(
        "build",
        help="Scan a Lean project and extract theorem/lemma declarations to JSONL.",
    )
    p_build.add_argument(
        "--project-dir", required=True,
        help="Root directory of the Lean project to scan.",
    )
    p_build.add_argument(
        "--output", required=True,
        help="Output JSONL file path.",
    )
    p_build.set_defaults(func=_cli_build)

    # -- index --------------------------------------------------------------
    p_index = sub.add_parser(
        "index",
        help="Build a FAISS index from a corpus JSONL file.",
    )
    p_index.add_argument(
        "--corpus", required=True,
        help="Path to the corpus JSONL file.",
    )
    p_index.add_argument(
        "--index-dir", required=True,
        help="Directory to save the FAISS index and metadata.",
    )
    p_index.add_argument(
        "--model", default=None,
        help=f"Sentence-transformer model name (default: {_DEFAULT_MODEL}).",
    )
    p_index.set_defaults(func=_cli_index)

    # -- query --------------------------------------------------------------
    p_query = sub.add_parser(
        "query",
        help="Query a saved index with a natural-language string.",
    )
    p_query.add_argument(
        "--index-dir", required=True,
        help="Directory containing a saved FAISS index.",
    )
    p_query.add_argument(
        "--query", required=True,
        help="Natural-language query string.",
    )
    p_query.add_argument(
        "--k", type=int, default=25,
        help="Number of results to return (default: 25).",
    )
    p_query.set_defaults(func=_cli_query)

    # -- dispatch -----------------------------------------------------------
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
