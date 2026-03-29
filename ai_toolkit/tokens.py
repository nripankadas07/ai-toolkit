"""Token counting and splitting utilities."""

from __future__ import annotations

from pathlib import Path

import click
import tiktoken
from rich.console import Console
from rich.table import Table

console = Console()

# Pricing per 1M tokens (input) for common models — approximate, for estimation only
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4": {"encoding": "cl100k_base", "input_per_1m": 30.0, "output_per_1m": 60.0},
    "gpt-4-turbo": {"encoding": "cl100k_base", "input_per_1m": 10.0, "output_per_1m": 30.0},
    "gpt-4o": {"encoding": "cl100k_base", "input_per_1m": 2.50, "output_per_1m": 10.0},
    "gpt-4o-mini": {"encoding": "cl100k_base", "input_per_1m": 0.15, "output_per_1m": 0.60},
    "gpt-3.5-turbo": {"encoding": "cl100k_base", "input_per_1m": 0.50, "output_per_1m": 1.50},
    "claude-3-opus": {"encoding": "cl100k_base", "input_per_1m": 15.0, "output_per_1m": 75.0},
    "claude-3-sonnet": {"encoding": "cl100k_base", "input_per_1m": 3.0, "output_per_1m": 15.0},
    "claude-3-haiku": {"encoding": "cl100k_base", "input_per_1m": 0.25, "output_per_1m": 1.25},
}


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def split_tokens(text: str, encoding_name: str = "cl100k_base") -> list[str]:
    """Split text into individual token strings."""
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)
    return [enc.decode([tid]) for tid in token_ids]


def estimate_cost(
    token_count: int, model: str, direction: str = "input"
) -> float | None:
    """Estimate cost in USD for a given token count and model."""
    if model not in MODEL_PRICING:
        return None
    key = f\"{direction}_per_1m\"
    rate = MODEL_PRICING[model].get(key, 0)
    return (token_count / 1_000_000) * rate


@click.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file.")
@click.option(
    "--encoding",
    "-e",
    default="cl100k_base",
    help="Tokenizer encoding (default: cl100k_base).",
)
@click.option("--cost", is_flag=True, help="Show estimated cost.")
@click.option("--model", "-m", default="gpt-4o", help="Model for cost estimation.")
@click.option("--json-output", is_flag=True, help="Output as JSON.")
def count(
    text: str | None,
    file: str | None,
    encoding: str,
    cost: bool,
    model: str,
    json_output: bool,
) -> None:
    """Count tokens in text or a file."""
    if file:
        content = Path(file).read_text(encoding="utf-8")
        source = file
    elif text:
        content = text
        source = "inline"
    else:
        content = click.get_text_stream(\"stdin\").read()
        source = \"stdin\"

    token_count = count_tokens(content, encoding)

    if json_output:
        import json

        result: dict[str, object] = {
            \"source\": source,
            \"encoding\": encoding,
            \"tokens\": token_count,
            \"characters\": len(content),
        }
        if cost:
            est = estimate_cost(token_count, model)
            result[\"estimated_cost_usd\"] = round(est, 6) if est else None
            result[\"model\"] = model
        click.echo(json.dumps(result, indent=2))
    else:
        table = Table(title=\"Token Count\")
        table.add_column(\"Property\", style=\"cyan\")
        table.add_column(\"Value\", style=\"green\")
        table.add_row(\"Source\", source)
        table.add_row(\"Encoding\", encoding)
        table.add_row(\"Tokens\", f\"{token_count:,}\")
        table.add_row(\"Characters\", f\"{len(content):,}\")

        if cost:
            est = estimate_cost(token_count, model)
            if est is not None:
                table.add_row(\"Model\", model)
                table.add_row(\"Est. Input Cost\", f\"${est:.6f}\")
            else:
                table.add_row(\"Cost\", f\"[yellow]Unknown model: {model}[/yellow]\")

        console.print(table)


@click.command()
@click.argument(\"text\")
@click.option(
    \"--encoding\",
    \"-e\",
    default=\"cl100k_base\",
    help=\"Tokenizer encoding (default: cl100k_base).\",
)
@click.option(\"--numbered\", \"-n\", is_flag=True, help=\"Show token IDs alongside strings.\")
def split(text: str, encoding: str, numbered: bool) -> None:
    \"\"\"Visualize how text is split into tokens.\"\"\"
    enc = tiktoken.get_encoding(encoding)
    token_ids = enc.encode(text)
    token_strings = [enc.decode([tid]) for tid in token_ids]

    if numbered:
        table = Table(title=f\"Token Split ({encoding})\")
        table.add_column(\"#\", style=\"dim\")
        table.add_column(\"Token ID\", style=\"cyan\")
        table.add_column(\"Token\", style=\"green\")
        for i, (tid, ts) in enumerate(zip(token_ids, token_strings)):
            table.add_row(str(i), str(tid), repr(ts))
        console.print(table)
    else:
        console.print(f\"[cyan]Encoding:[/cyan] {encoding}\")
        console.print(f\"[cyan]Tokens ({len(token_strings)}):[/cyan] {token_strings}\")
