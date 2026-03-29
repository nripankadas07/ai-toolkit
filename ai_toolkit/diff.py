"""Model output diffing utilities."""

from __future__ import annotations

import difflib
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

console = Console()


def word_diff(text_a: str, text_b: str) -> list[tuple[str, str]]:
    """Compute a word-level diff between two texts.

    Returns a list of (tag, word) tuples where tag is one of:
    'equal', 'insert', 'delete', 'replace'.
    """
    words_a = text_a.split()
    words_b = text_b.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    result: list[tuple[str, str]] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for w in words_a[i1:i2]:
                result.append(("equal", w))
        elif op == "delete":
            for w in words_a[i1:i2]:
                result.append(("delete", w))
        elif op == "insert":
            for w in words_b[j1:j2]:
                result.append(("insert", w))
        elif op == "replace":
            for w in words_a[i1:i2]:
                result.append(("delete", w))
            for w in words_b[j1:j2]:
                result.append(("insert", w))

    return result


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute similarity ratio between two texts (0.0 to 1.0)."""
    return difflib.SequenceMatcher(None, text_a.split(), text_b.split()).ratio()


def format_diff_rich(diff_result: list[tuple[str, str]]) -> Text:
    """Format a word diff as Rich Text with colors."""
    output = Text()
    for tag, word in diff_result:
        if tag == "equal":
            output.append(word + " ")
        elif tag == "delete":
            output.append(word + " ", style="red strike")
        elif tag == "insert":
            output.append(word + " ", style="green bold")
    return output


def diff_stats(diff_result: list[tuple[str, str]]) -> dict[str, int]:
    """Compute statistics about a diff."""
    stats = {"equal": 0, "insert": 0, "delete": 0}
    for tag, _ in diff_result:
        if tag in stats:
            stats[tag] += 1
    stats["total_changes"] = stats["insert"] + stats["delete"]
    return stats


@click.command()
@click.argument("file_a", required=False)
@click.argument("file_b", required=False)
@click.option("--text", "-t", nargs=2, help="Compare two text strings directly.")
@click.option("--context", "-c", default=0, type=int, help="Lines of context around changes.")
@click.option("--stats-only", is_flag=True, help="Show only statistics, no diff.")
@click.option("--json-output", is_flag=True, help="Output as JSON.")
def outputs(
    file_a: str | None,
    file_b: str | None,
    text: tuple[str, str] | None,
    context: int,
    stats_only: bool,
    json_output: bool,
) -> None:
    """Compare two model outputs with word-level diffs."""
    if text:
        content_a, content_b = text
        label_a, label_b = "Text A", "Text B"
    elif file_a and file_b:
        content_a = Path(file_a).read_text(encoding="utf-8")
        content_b = Path(file_b).read_text(encoding="utf-8")
        label_a, label_b = file_a, file_b
    else:
        raise click.UsageError("Provide two files or use --text 'a' 'b'")

    diff_result = word_diff(content_a, content_b)
    similarity = compute_similarity(content_a, content_b)
    stats = diff_stats(diff_result)

    if json_output:
        import json

        result = {
            "source_a": label_a,
            "source_b": label_b,
            "similarity": round(similarity, 4),
            "words_unchanged": stats["equal"],
            "words_inserted": stats["insert"],
            "words_deleted": stats["delete"],
            "total_changes": stats["total_changes"],
        }
        click.echo(json.dumps(result, indent=2))
        return

    console.print(f"\n[bold]Comparing:[/bold] {label_a} vs {label_b}")
    console.print(f"[cyan]Similarity:[/cyan] {similarity:.1%}")
    console.print(
        f"[cyan]Changes:[/cyan] [red]-{stats['delete']} words[/red] "
        f"[green]+{stats['insert']} words[/green] "
        f"({stats['equal']} unchanged)\n"
    )

    if not stats_only:
        console.print("[bold]Diff:[/bold]")
        console.print(format_diff_rich(diff_result))
        console.print()
