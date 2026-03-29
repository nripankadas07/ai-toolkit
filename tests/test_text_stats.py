"""Tests for text statistics utilities."""

import pytest

from ai_toolkit.text_stats import analyze_text, _count_syllables


class TestAnalyzeText:
    def test_word_count(self):
        result = analyze_text("Hello world foo bar")
        assert result["words"] == 4

    def test_character_count(self):
        result = analyze_text("abc")
        assert result["characters"] == 3

    def test_sentence_count(self):
        result = analyze_text("First sentence. Second sentence. Third one.")
        assert result["sentences"] == 3

    def test_paragraph_count(self):
        result = analyze_text("First paragraph.\n\nSecond paragraph.\n\nThird.")
        assert result["paragraphs"] == 3

    def test_empty_text(self):
        result = analyze_text("")
        assert result["words"] == 0
        assert result["characters"] == 0

    def test_reading_level_exists(self):
        text = "The cat sat on the mat. It was a good day."
        result = analyze_text(text)
        assert "reading_level" in result
        assert isinstance(result["reading_level"], str)

    def test_vocabulary_richness(self):
        # All unique words
        result = analyze_text("alpha beta gamma delta")
        assert result["vocabulary_richness"] == pytest.approx(1.0)

        # Repeated words
        result = analyze_text("the the the the")
        assert result["vocabulary_richness"] == pytest.approx(0.25)

    def test_reading_time(self):
        # 238 words = ~1 minute
        words = " ".join(["word"] * 238)
        result = analyze_text(words)
        assert result["reading_time_minutes"] == pytest.approx(1.0, abs=0.1)


class TestCountSyllables:
    def test_one_syllable(self):
        assert _count_syllables("cat") == 1
        assert _count_syllables("dog") == 1

    def test_two_syllables(self):
        assert _count_syllables("hello") == 2
        assert _count_syllables("python") == 2

    def test_three_syllables(self):
        assert _count_syllables("beautiful") == 3

    def test_empty_string(self):
        assert _count_syllables("") == 0

    def test_with_punctuation(self):
        assert _count_syllables("hello!") == 2
        assert _count_syllables("world.") == 1
