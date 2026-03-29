"""Text statistics utilities."""

from __future__ import annotations

import math
import re
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def analyze_text(text: str) -> dict[str, object]:
    """Compute comprehensive text statistics."""
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    words = text.split()
    word_count = len(words)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    avg_word_length = (
        sum(len(w.strip(".,!?;:\"'()[]{}")) for w in words) / word_count
        if word_count > 0 else 0
    )
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

    total_syllables = sum(_count_syllables(w) for w in words)
    avg_syllables_per_word = total_syllables / word_count if word_count > 0 else 0

    if sentence_count > 0 and word_count > 0:
        fk_grade = (0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59)
        fk_grade = max(0, round(fk_grade, 1))
    else:
        fk_grade = 0

    if sentence_count > 0 and word_count > 0:
        fre = (206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (total_syllables / word_count))
        fre = max(0, min(100, round(fre, 1)))
    else:
        fre = 0

    unique_words = len(set(w.lower().strip(".,!?;:\"'()[]{}") for w in words))
    vocabulary_richness = unique_words / word_count if word_count > 0 else 0
    reading_time_minutes = word_count / 238

    return {
        "characters": characters,
        "characters_no_spaces": characters_no_spaces,
        "words": word_count,
        "sentences": sentence_count,
        "paragraphs": paragraph_count,
        "avg_word_length": round(avg_word_length, 1),
        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2),
        "unique_words": unique_words,
        "vocabulary_richness": round(vocabulary_richness, 3),
        "flesch_kincaid_grade": fk_grade,
        "flesch_reading_ease": fre,
        "reading_level": _grade_to_level(fk_grade),
        "reading_time_minutes": round(reading_time_minutes, 1),
    }


def _count_syllables(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.lower().strip(".,!?;:\"'()[]{}0123456789")
    if not word:
        return 0
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def _grade_to_level(grade: float) -> str:
    """Convert Flesch-Kincaid grade to a readable level."""
    if grade <= 1:
        return "Kindergarten"
    elif grade <= 5:
        return f"Grade {math.ceil(grade)} (Elementary)"
    elif grade <= 8:
        return f"Grade {math.ceil(grade)} (Middle School)"
    elif grade <= 12:
        return f"Grade {math.ceil(grade)} (High School)"
    elif grade <= 16:
        return "College Level"
    else:
        return "Graduate Level"


@click.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file.")
@click.option("--json-output", is_flag=True, help="Output as JSON.")
def stats(text: str | None, file: str | None, json_output: bool) -> None:
    """Compute text statistics."""
    if file:
        content = Path(file).read_text(encoding="utf-8")
        source = file
    elif text:
        content = text
        source = "inline"
    else:
        content = click.get_text_stream("stdin").read()
        source = "stdin"

    result = analyze_text(content)

    if json_output:
        import json
        result["source"] = source
        click.echo(json.dumps(result, indent=2))
    else:
        table = Table(title="Text Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Source", source)
        table.add_row("Words", f"{result['words']:,}")
        table.add_row("Characters", f"{result['characters']:,}")
        table.add_row("Sentences", str(result["sentences"]))
        table.add_row("Paragraphs", str(result["paragraphs"]))
        table.add_row("Avg Word Length", f"{result['avg_word_length']} chars")
        table.add_row("Avg Sentence Length", f"{result['avg_words_per_sentence']} words")
        table.add_row("Unique Words", f"{result['unique_words']:,}")
        table.add_row("Vocabulary Richness", f"{result['vocabulary_richness']:.1%}")
        table.add_row("Reading Level", str(result["reading_level"]))
        table.add_row("Flesch-Kincaid Grade", str(result["flesch_kincaid_grade"]))
        table.add_row("Flesch Reading Ease", f"{result['flesch_reading_ease']}/100")
        table.add_row("Est. Reading Time", f"{result['reading_time_minutes']} min")
        console.print(table)
