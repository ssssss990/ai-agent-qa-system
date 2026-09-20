"""
BM25 检索引擎

为什么需要 BM25（已经有了稀疏向量）：
- BGE-M3 的稀疏向量虽然带学习权重，但模型训练时未必见过所有领域专有名词
- BM25 是基于 TF-IDF 的统计方法，对型号、错误码等精确匹配更可靠
- 两者互补：BM25 擅长精确匹配，向量擅长语义匹配

实际测试：纯向量召回对 "ERR-5003" 这类短查询命中率很低，
加 BM25 后明显改善，因为 BM25 能直接匹配到包含该错误码的文档。
"""

import jieba
from rank_bm25 import BM25Okapi
from typing import List
from dataclasses import dataclass

from src.retrieval.milvus_store import RetrievalResult
from config.settings import settings


class BM25Store:
    """基于内存的 BM25 索引"""

    def __init__(self):
        self._bm25: BM25Okapi = None
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._parent_ids: list[str] = []

    def build_index(self, chunks: list[dict]):
        """从文档块构建 BM25 索引"""
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._texts = [c["text"] for c in chunks]
        self._parent_ids = [c.get("parent_id", "") for c in chunks]

        tokenized = [list(jieba.cut(text)) for text in self._texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            按 BM25 分数排序的检索结果
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 build_index()")

        top_k = top_k or settings.bm25_top_k
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            RetrievalResult(
                chunk_id=self._chunk_ids[i],
                text=self._texts[i],
                score=float(scores[i]),
                parent_id=self._parent_ids[i],
                metadata={"source": "bm25"},
            )
            for i in ranked_indices
            if scores[i] > 0
        ]

    def add_documents(self, chunks: list[dict]):
        """增量添加文档（重建索引）"""
        all_chunks = [
            {"chunk_id": cid, "text": t, "parent_id": pid}
            for cid, t, pid in zip(self._chunk_ids, self._texts, self._parent_ids)
        ] + chunks
        self.build_index(all_chunks)
