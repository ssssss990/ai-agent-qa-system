"""
Cross-Encoder 重排器

为什么需要重排（检索时不是已经排过了吗）：
- 检索阶段的排序是 query-doc 独立编码后的相似度，精度有限
- Cross-Encoder 把 query 和 doc 拼在一起送入模型，能捕获细粒度的语义交互
- 精度高但速度慢，所以只对 Top-20 候选做重排，选 Top-5

实际效果（120 条评测集对比）：
- 重排前 Recall@5 = 0.84
- 重排后 Recall@5 = 0.91
- 短查询和含专有名词的查询改善最明显
"""

from typing import List

from src.retrieval.milvus_store import RetrievalResult
from config.settings import settings


class CrossEncoderReranker:
    """bge-reranker-v2-m3 重排器"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = None,
    ) -> List[RetrievalResult]:
        """
        对候选文档做重排

        Args:
            query: 用户查询
            candidates: 召回阶段返回的候选列表
            top_k: 重排后返回数量

        Returns:
            按 Cross-Encoder 分数重新排序的结果
        """
        if not candidates:
            return []

        top_k = top_k or settings.rerank_top_k

        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk_id=c.chunk_id,
                text=c.text,
                score=float(s),
                parent_id=c.parent_id,
                metadata={**(c.metadata or {}), "rerank_score": float(s)},
            )
            for c, s in scored[:top_k]
        ]
