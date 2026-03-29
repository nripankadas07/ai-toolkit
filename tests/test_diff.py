"""Tests for output diffing utilities."""

import pytest

from ai_toolkit.diff import word_diff, compute_similarity, diff_stats


class TestWordDiff:
    def test_identical_texts(self):
        result = word_diff("hello world", "hello world")
        assert all(tag == "equal" for tag, _ in result)

    def test_insertion(self):
        result = word_diff("hello world", "hello beautiful world")
        tags = [tag for tag, _ in result]
        assert "insert" in tags
        words = [w for tag, w in result if tag == "insert"]
        assert "beautiful" in words

    def test_deletion(self):
        result = word_diff("hello beautiful world", "hello world")
        tags = [tag for tag, _ in result]
        assert "delete" in tags

    def test_empty_texts(self):
        result = word_diff("", "")
        assert result == []

    def test_completely_different(self):
        result = word_diff("alpha beta", "gamma delta")
        tags = set(tag for tag, _ in result)
        assert "equal" not in tags


class TestComputeSimilarity:
    def test_identical(self):
        assert compute_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        sim = compute_similarity("alpha beta gamma", "one two three")
        assert sim < 0.3

    def test_partial_overlap(self):
        sim = compute_similarity("the quick brown fox", "the slow brown fox")
        assert 0.5 < sim < 1.0

    def test_range(self):
        sim = compute_similarity("any text here", "other text there")
        assert 0.0 <= sim <= 1.0


class TestDiffStats:
    def test_no_changes(self):
        diff = word_diff("same text", "same text")
        stats = diff_stats(diff)
        assert stats["total_changes"] == 0
        assert stats["equal"] == 2

    def test_counts_changes(self):
        diff = word_diff("hello world", "hello beautiful world")
        stats = diff_stats(diff)
        assert stats["insert"] > 0
        assert stats["total_changes"] > 0
