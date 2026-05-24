"""Embedding comparison utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any, Protocol, cast

import click
import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

console = Console()

FloatArray = NDArray[np.float32]


class EmbeddingModel(Protocol):
    """Minimal protocol for optional sentence-transformers models."""

    def encode(self, texts: list[str]) -> Iterable[Any]:
        """Encode texts into array-like vectors."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product between two vectors."""
    return float(np.dot(a, b))


def _get_embeddings(texts: list[str], model_name: str) -> list[FloatArray]:
    """Generate embeddings for a list of texts.

    Falls back to a simple hash-based embedding if sentence-transformers
    is not installed, so the CLI remains functional without heavy deps.
    """
    try:
        module = import_module("sentence_transformers")
        model_factory = cast(
            Callable[[str], EmbeddingModel],
            getattr(module, "SentenceTransformer"),
        )
        model = model_factory(model_name)
        embeddings = model.encode(texts)
        return [np.asarray(e, dtype=np.float32) for e in embeddings]
    except ImportError:
        console.print(
            "[yellow]sentence-transformers not installed. "
            'Using hash-based fallback (from a checkout: pip install -e ".[embeddings]")[/yellow]'
        )
        return [_hash_embedding(t) for t in texts]


def _hash_embedding(text: str, dim: int = 128) -> FloatArray:
    """Create a deterministic pseudo-embedding from text hash.

    This is NOT a real embedding — it's a fallback for demo/testing
    when sentence-transformers isn't installed.
    """
    import hashlib

    h = hashlib.sha512(text.lower().encode()).digest()
    # Expand hash to fill the dimension
    while len(h) < dim * 4:
        h += hashlib.sha512(h).digest()
    raw = np.frombuffer(h[: dim * 4], dtype=np.uint8).copy()
    arr: FloatArray = (raw.astype(np.float32) / np.float32(255.0)) - np.float32(0.5)
    arr = arr[:dim]
    # Normalize to unit length
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return np.asarray(arr, dtype=np.float32)


@click.command()
@click.argument("text_a")
@click.argument("text_b")
@click.option(
    "--model",
    default="all-MiniLM-L6-v2",
    help="Sentence transformer model to use.",
)
@click.option(
    "--metric",
    type=click.Choice(["cosine", "euclidean", "dot"], case_sensitive=False),
    default="cosine",
    help="Similarity metric.",
)
@click.option("--json-output", is_flag=True, help="Output as JSON.")
def compare(text_a: str, text_b: str, model: str, metric: str, json_output: bool) -> None:
    """Compare similarity between two text strings."""
    embeddings = _get_embeddings([text_a, text_b], model)
    emb_a, emb_b = embeddings[0], embeddings[1]

    metrics = {
        "cosine": ("Cosine Similarity", cosine_similarity),
        "euclidean": ("Euclidean Distance", euclidean_distance),
        "dot": ("Dot Product", dot_product),
    }

    label, func = metrics[metric]
    score = func(emb_a, emb_b)

    if json_output:
        import json

        result = {
            "text_a": text_a,
            "text_b": text_b,
            "metric": metric,
            "score": round(score, 6),
            "model": model,
            "dimensions": len(emb_a),
        }
        click.echo(json.dumps(result, indent=2))
    else:
        table = Table(title="Embedding Comparison")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Text A", text_a[:80] + ("..." if len(text_a) > 80 else ""))
        table.add_row("Text B", text_b[:80] + ("..." if len(text_b) > 80 else ""))
        table.add_row("Model", model)
        table.add_row("Dimensions", str(len(emb_a)))
        table.add_row(label, f"{score:.6f}")
        console.print(table)
