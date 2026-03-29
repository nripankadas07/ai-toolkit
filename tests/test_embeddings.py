"""Tests for embedding comparison utilities."""

import numpy as np
import pytest

from ai_toolkit.embeddings import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    _hash_embedding,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert cosine_similarity(a, b) == 0.0

    def test_range(self):
        a = np.random.randn(128)
        b = np.random.randn(128)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


class TestEuclideanDistance:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_known_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_non_negative(self):
        a = np.random.randn(64)
        b = np.random.randn(64)
        assert euclidean_distance(a, b) >= 0


class TestDotProduct:
    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert dot_product(a, b) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        assert dot_product(a, b) == pytest.approx(23.0)


class TestHashEmbedding:
    def test_deterministic(self):
        emb1 = _hash_embedding("test text")
        emb2 = _hash_embedding("test text")
        np.testing.assert_array_equal(emb1, emb2)

    def test_correct_dimension(self):
        emb = _hash_embedding("test", dim=256)
        assert len(emb) == 256

    def test_normalized(self):
        emb = _hash_embedding("hello world")
        norm = np.linalg.norm(emb)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_different_texts_differ(self):
        emb1 = _hash_embedding("hello")
        emb2 = _hash_embedding("world")
        assert not np.allclose(emb1, emb2)
