"""
测试 RRF 融合
"""

from src.retrieval.milvus_store import RetrievalResult
from src.retrieval.rrf_fusion import RRFFusion


class TestRRFFusion:
    def test_duplicate_id_merges_scores(self):
        rrf = RRFFusion(k=60)

        vec_results = [
            RetrievalResult(chunk_id="a", text="doc a", score=0.9),
            RetrievalResult(chunk_id="b", text="doc b", score=0.8),
        ]
        bm25_results = [
            RetrievalResult(chunk_id="a", text="doc a", score=2.5),
            RetrievalResult(chunk_id="c", text="doc c", score=1.0),
        ]

        fused = rrf.fuse(vec_results, bm25_results)

        ids = [f.chunk_id for f in fused]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        assert len(fused) == 3

        a_score = [f for f in fused if f.chunk_id == "a"][0].score
        b_score = [f for f in fused if f.chunk_id == "b"][0].score
        assert a_score > b_score

    def test_empty_inputs(self):
        rrf = RRFFusion()
        fused = rrf.fuse([], [])
        assert len(fused) == 0
