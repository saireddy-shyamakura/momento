"""Unit tests for fuse_scores() and FusionWeights in retrieval/fusion.py.

Tests the cross-modal fusion layer including:
- Default and custom weights
- Missing signals (no object/OCR scores)
- All signals combined
- Missing paths defaulting to 0.0
- Edge cases (empty, single result, extreme weights)
"""
import pytest


def _import_fusion():
    from momento.retrieval.fusion import fuse_scores, FusionWeights
    return fuse_scores, FusionWeights


class TestFusionWeights:
    """FusionWeights dataclass defaults and construction."""

    def test_default_weights(self):
        _, FusionWeights = _import_fusion()
        w = FusionWeights()
        assert w.embedding == 0.6
        assert w.object_match == 0.2
        assert w.ocr_relevance == 0.2

    def test_custom_weights(self):
        _, FusionWeights = _import_fusion()
        w = FusionWeights(embedding=0.5, object_match=0.3, ocr_relevance=0.2)
        assert w.embedding == 0.5
        assert w.object_match == 0.3
        assert w.ocr_relevance == 0.2

    def test_weights_sum_to_one(self):
        _, FusionWeights = _import_fusion()
        w = FusionWeights()
        assert abs(w.embedding + w.object_match + w.ocr_relevance - 1.0) < 1e-6


class TestFusionDefaults:
    """With default weights (0.6, 0.2, 0.2)."""

    def test_embedding_only(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.8, "/a.jpg"), (0.6, "/b.jpg")]
        results = fuse_scores(emb_scores)
        # Without object/OCR scores, these default to 0.0
        # fused = 0.6 * emb + 0.2 * 0 + 0.2 * 0 = 0.6 * emb
        assert len(results) == 2
        assert abs(results[0][0] - 0.48) < 1e-6  # 0.6 * 0.8
        assert abs(results[1][0] - 0.36) < 1e-6  # 0.6 * 0.6

    def test_with_object_scores(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.8, "/a.jpg"), (0.6, "/b.jpg")]
        obj_scores = {"/a.jpg": 0.5, "/b.jpg": 0.0}
        results = fuse_scores(emb_scores, object_scores=obj_scores)
        # /a.jpg: 0.6*0.8 + 0.2*0.5 + 0.2*0 = 0.48 + 0.1 = 0.58
        # /b.jpg: 0.6*0.6 + 0.2*0.0 + 0.2*0 = 0.36
        assert abs(results[0][0] - 0.58) < 1e-6
        assert abs(results[1][0] - 0.36) < 1e-6

    def test_with_ocr_scores(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.8, "/a.jpg"), (0.6, "/b.jpg")]
        ocr_scores = {"/a.jpg": 0.0, "/b.jpg": 0.7}
        results = fuse_scores(emb_scores, ocr_scores=ocr_scores)
        # /a.jpg: 0.6*0.8 + 0.2*0 + 0.2*0.0 = 0.48
        # /b.jpg: 0.6*0.6 + 0.2*0 + 0.2*0.7 = 0.36 + 0.14 = 0.50
        assert abs(results[0][0] - 0.50) < 1e-6
        assert abs(results[1][0] - 0.48) < 1e-6

    def test_all_signals_present(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.8, "/a.jpg"), (0.6, "/b.jpg")]
        obj_scores = {"/a.jpg": 0.5, "/b.jpg": 1.0}
        ocr_scores = {"/a.jpg": 0.3, "/b.jpg": 0.0}
        results = fuse_scores(emb_scores, object_scores=obj_scores, ocr_scores=ocr_scores)
        # /a.jpg: 0.6*0.8 + 0.2*0.5 + 0.2*0.3 = 0.48 + 0.10 + 0.06 = 0.64
        # /b.jpg: 0.6*0.6 + 0.2*1.0 + 0.2*0.0 = 0.36 + 0.20 + 0.00 = 0.56
        assert abs(results[0][0] - 0.64) < 1e-6
        assert abs(results[1][0] - 0.56) < 1e-6


class TestFusionCustomWeights:
    """Custom weight overrides."""

    def test_custom_weights_all_embedding(self):
        fuse_scores, FusionWeights = _import_fusion()
        weights = FusionWeights(embedding=1.0, object_match=0.0, ocr_relevance=0.0)
        emb_scores = [(0.8, "/a.jpg"), (0.6, "/b.jpg")]
        obj_scores = {"/a.jpg": 1.0, "/b.jpg": 1.0}
        results = fuse_scores(emb_scores, object_scores=obj_scores, weights=weights)
        # Only embedding matters
        assert abs(results[0][0] - 0.80) < 1e-6
        assert abs(results[1][0] - 0.60) < 1e-6


class TestFusionEdgeCases:
    """Edge cases for fusion."""

    def test_empty_embeddings(self):
        fuse_scores, FusionWeights = _import_fusion()
        results = fuse_scores([])
        assert results == []

    def test_single_embedding(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.9, "/single.jpg")]
        results = fuse_scores(emb_scores)
        assert len(results) == 1
        assert results[0][1] == "/single.jpg"

    def test_missing_path_in_object_scores_defaults_zero(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.8, "/a.jpg"), (0.6, "/b.jpg")]
        obj_scores = {}  # No object scores at all
        results = fuse_scores(emb_scores, object_scores=obj_scores)
        for score, path in results:
            # Only embedding contributes
            expected = 0.6 * (0.8 if path == "/a.jpg" else 0.6)
            assert abs(score - expected) < 1e-6

    def test_partial_signal_coverage(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.8, "/a.jpg"), (0.6, "/b.jpg")]
        ocr_scores = {"/a.jpg": 0.9}  # /b.jpg not in OCR scores
        results = fuse_scores(emb_scores, ocr_scores=ocr_scores)
        # /a.jpg: 0.6*0.8 + 0.2*0 + 0.2*0.9 = 0.48 + 0.18 = 0.66
        # /b.jpg: 0.6*0.6 + 0.2*0 + 0.2*0.0 = 0.36
        assert abs(results[0][0] - 0.66) < 1e-6
        assert abs(results[1][0] - 0.36) < 1e-6

    def test_results_descending(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.3, "/c.jpg"), (0.9, "/a.jpg"), (0.6, "/b.jpg")]
        results = fuse_scores(emb_scores)
        scores = [s for s, p in results]
        assert scores == sorted(scores, reverse=True)

    def test_zero_scores(self):
        fuse_scores, FusionWeights = _import_fusion()
        emb_scores = [(0.0, "/a.jpg"), (0.0, "/b.jpg")]
        results = fuse_scores(emb_scores)
        for score, path in results:
            assert score == 0.0