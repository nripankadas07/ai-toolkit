"""Tests for token counting and splitting utilities."""

import pytest

from ai_toolkit.tokens import count_tokens, split_tokens, estimate_cost

# tiktoken requires downloading encoding files — skip if unavailable
try:
    import tiktoken
    tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

needs_tiktoken = pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken encoding not available")


class TestCountTokens:
    @needs_tiktoken
    def test_empty_string(self):
        assert count_tokens("") == 0

    @needs_tiktoken
    def test_simple_text(self):
        count = count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    @needs_tiktoken
    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hello")
        long = count_tokens("Hello, this is a much longer sentence with many more words.")
        assert long > short

    @needs_tiktoken
    def test_encoding_parameter(self):
        count_cl100k = count_tokens("test text", "cl100k_base")
        count_p50k = count_tokens("test text", "p50k_base")
        assert count_cl100k > 0
        assert count_p50k > 0


class TestSplitTokens:
    @needs_tiktoken
    def test_round_trip(self):
        text = "The quick brown fox"
        tokens = split_tokens(text)
        reconstructed = "".join(tokens)
        assert reconstructed == text

    @needs_tiktoken
    def test_returns_list(self):
        tokens = split_tokens("Hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    @needs_tiktoken
    def test_count_matches_split(self):
        text = "This is a test sentence."
        count = count_tokens(text)
        tokens = split_tokens(text)
        assert len(tokens) == count


class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost(1_000_000, "gpt-4o")
        assert cost is not None
        assert cost == pytest.approx(2.50)

    def test_unknown_model(self):
        cost = estimate_cost(1000, "unknown-model-xyz")
        assert cost is None

    def test_zero_tokens(self):
        cost = estimate_cost(0, "gpt-4o")
        assert cost == pytest.approx(0.0)

    def test_scales_linearly(self):
        cost_1k = estimate_cost(1000, "gpt-4o")
        cost_2k = estimate_cost(2000, "gpt-4o")
        assert cost_2k == pytest.approx(cost_1k * 2)
