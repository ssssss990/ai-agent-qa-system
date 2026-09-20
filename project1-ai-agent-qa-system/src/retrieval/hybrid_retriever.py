"""
混合检索器：整合向量召回 + BM25 + RRF 融合

完整的检索链路：
1. BGE-M3 编码查询 -> 稠密向量 + 稀疏向量
2. 稠密向量在 Milvus 做 ANN 搜索
3. 稀疏向量在 Milvus 做稀疏搜索（可选，本项目用 BM25 替代）
4. BM25 做关键词召回
5. RRF 融合两路结果
6. Cross-Encoder 重排（见 reranking 模块）
7. 回溯父块获取完整上下文
"""

from typing import List, Optional

from src.embedding.bge_embedder import BGEEmbedder
from src.retrieval.milvus_store import MilvusStore, RetrievalResult
from src.retrieval.bm25_store import BM25Store
from src.retrieval.rrf_fusion import RRFFusion
from config.settings import settings


class HybridRetriever:
    """混合检索器"""

    def __init__(self, embedder: BGEEmbedder = None, milvus: MilvusStore = None,
                 bm25: BM25Store = None, rrf: RRFFusion = None):
        self.embedder = embedder or BGEEmbedder()
        self.milvus = milvus or MilvusStore()
        self.bm25 = bm25 or BM25Store()
        self.rrf = rrf or RRFFusion()

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        混合检索

        Args:
            query: 用户查询
            top_k: 最终返回数量（重排前）

        Returns:
            融合 + 重排后的检索结果
        """
        top_k = top_k or settings.rerank_top_k

        # 1. 编码查询
        encoded = self.embedder.encode_query(query)
        dense_vec = encoded["dense"][0]

        # 2. 稠密向量召回
        dense_results = self.milvus.dense_search(dense_vec)

        # 3. BM25 召回
        bm25_results = self.bm25.search(query)

        # 4. RRF 融合
        fused = self.rrf.fuse(dense_results, bm25_results)

        # 5. 取 Top-20 送入重排
        top_candidates = fused[:20]

        # 6. 回溯父块（如果是子块命中，替换为父块文本）
        resolved = self._resolve_parents(top_candidates)

        return resolved[:top_k]

    def _resolve_parents(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        子块命中时回溯父块，提供完整上下文

        去重逻辑：如果多个子块属于同一个父块，只保留分数最高的那个。
        """
        seen_parents: dict[str, RetrievalResult] = {}
        resolved: List[RetrievalResult] = []

        for result in results:
            if result.parent_id and not result.metadata.get("is_parent"):
                if result.parent_id in seen_parents:
                    if result.score > seen_parents[result.parent_id].score:
                        seen_parents[result.parent_id] = result
                    continue

                parent = self.milvus.get_parent(result.parent_id)
                if parent:
                    resolved_result = RetrievalResult(
                        chunk_id=result.parent_id,
                        text=parent.get("text", result.text),
                        score=result.score,
                        parent_id=result.parent_id,
                        metadata={
                            **(result.metadata or {}),
                            "resolved_from": result.chunk_id,
                        },
                    )
                    seen_parents[result.parent_id] = resolved_result
                    resolved.append(resolved_result)
                else:
                    resolved.append(result)
            else:
                resolved.append(result)

        return resolved
