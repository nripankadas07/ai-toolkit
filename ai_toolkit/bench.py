"""Prompt benchmarking utilities."""

from __future__ import annotations

import statistics
import time
from typing import Callable

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def benchmark_function(
    func: Callable[[], str],
    runs: int = 5,
    warmup: int = 1,
) -> dict[str, object]:
    """Benchmark a callable, measuring latency and output variance.

    Returns timing statistics and output samples.
    """
    # Warmup runs (not counted)
    for _ in range(warmup):
        func()

    latencies: list[float] = []
    outputs: list[str] = []

    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        outputs.append(result)

    # Compute output variance using pairwise similarity
    from ai_toolkit.diff import compute_similarity

    similarities: list[float] = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            similarities.append(compute_similarity(outputs[i], outputs[j]))

    avg_similarity = statistics.mean(similarities) if similarities else 1.0

    return {
        "runs": runs,
        "latencies": latencies,
        "avg_latency": statistics.mean(latencies),
        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "avg_output_similarity": avg_similarity,
        "output_variance": 1.0 - avg_similarity,
        "outputs": outputs,
    }


def _simulate_llm_call(prompt: str) -> str:
    """Simulate an LLM call for benchmarking without an API key."""
    import hashlib
    import random

    base_hash = hashlib.md5(prompt.encode()).hexdigest()
    random.seed(base_hash + str(time.time_ns() % 1000))

    templates = [
        f"Response to: {prompt[:50]}. This is a simulated output for benchmarking.",
        f"Given the prompt about {prompt[:30]}, here is a generated response for testing.",
        f"Simulated LLM output for: {prompt[:40]}. Benchmarking mode active.",
    ]

    response = random.choice(templates)
    time.sleep(random.uniform(0.05, 0.15))
    return response


@click.command()
@click.argument("prompt_text")
@click.option("--runs", "-r", default=5, help="Number of benchmark runs.")
@click.option("--warmup", "-w", default=1, help="Warmup runs (not counted).")
@click.option("--json-output", is_flag=True, help="Output as JSON.")
def prompt(prompt_text: str, runs: int, warmup: int, json_output: bool) -> None:
    """Benchmark a prompt across multiple runs."""
    console.print(f"[cyan]Benchmarking prompt:[/cyan] {prompt_text[:60]}...")
    console.print(f"[dim]Runs: {runs} | Warmup: {warmup} | Mode: simulated[/dim]\n")

    results = benchmark_function(
        lambda: _simulate_llm_call(prompt_text),
        runs=runs,
        warmup=warmup,
    )

    if json_output:
        import json
        output = {
            "prompt": prompt_text,
            "runs": results["runs"],
            "avg_latency_s": round(results["avg_latency"], 4),
            "std_latency_s": round(results["std_latency"], 4),
            "min_latency_s": round(results["min_latency"], 4),
            "max_latency_s": round(results["max_latency"], 4),
            "output_variance": round(results["output_variance"], 4),
            "avg_output_similarity": round(results["avg_output_similarity"], 4),
        }
        click.echo(json.dumps(output, indent=2))
    else:
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Runs", str(results["runs"]))
        table.add_row("Avg Latency", f"{results['avg_latency']:.4f}s")
        table.add_row("Std Dev", f"{results['std_latency']:.4f}s")
        table.add_row("Min Latency", f"{results['min_latency']:.4f}s")
        table.add_row("Max Latency", f"{results['max_latency']:.4f}s")
        table.add_row("Output Variance", f"{results['output_variance']:.4f}")
        table.add_row("Avg Similarity", f"{results['avg_output_similarity']:.4f}")
        console.print(table)
