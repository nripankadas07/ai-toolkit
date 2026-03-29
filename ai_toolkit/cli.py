"""Main CLI entry point for ai-toolkit."""

import click

from ai_toolkit import __version__


@click.group()
@click.version_option(version=__version__, prog_name="ai-toolkit")
def cli() -> None:
    """AI Toolkit — command-line utilities for everyday AI/ML tasks."""
    pass


@cli.group()
def embeddings() -> None:
    """Embedding comparison utilities."""
    pass


@cli.group()
def tokens() -> None:
    """Token counting and splitting."""
    pass


@cli.group()
def diff() -> None:
    """Compare model outputs."""
    pass


@cli.group()
def bench() -> None:
    """Benchmark prompts and models."""
    pass


@cli.group()
def text() -> None:
    """Text analysis utilities."""
    pass


# Register subcommands
from ai_toolkit.embeddings import compare  # noqa: E402
from ai_toolkit.tokens import count, split  # noqa: E402
from ai_toolkit.diff import outputs  # noqa: E402
from ai_toolkit.bench import prompt  # noqa: E402
from ai_toolkit.text_stats import stats  # noqa: E402

embeddings.add_command(compare)
tokens.add_command(count)
tokens.add_command(split)
diff.add_command(outputs)
bench.add_command(prompt)
text.add_command(stats)


if __name__ == "__main__":
    cli()
