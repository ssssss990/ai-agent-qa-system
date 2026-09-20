"""
Milvus 向量存储

存储稠密 + 稀疏双向量，支持：
- 稠密向量搜索（ANN）
- 稀疏向量搜索（类似 BM25 但带学习权重）
- 父子块回溯

HNSW 参数选择理由：
- M=16：每个节点保留 16 条邻接边，平衡内存和召回率
- efConstruction=200：建图时搜索宽度，越大越精确但建图慢
- efSearch=64：查询时搜索宽度，设为 efConstruction 的 1/3
"""

from typing import List, Optional
from dataclasses import dataclass

from config.settings import settings


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    parent_id: Optional[str] = None
    metadata: dict = None


class MilvusStore:
    """Milvus 双向量存储"""

    def __init__(self):
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            from pymilvus import MilvusClient
            client = MilvusClient(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
            )
            self._collection = client
        return self._collection

    def create_collection(self, dim: int = 1024):
        """创建 Collection，包含稠密和稀疏向量字段"""
        from pymilvus import DataType

        schema = self.collection.create_schema(
            auto_id=False, enable_dynamic_field=True
        )
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("dense_vec", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse_vec", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", DataType.VARCHAR, max_length=4096)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("is_parent", DataType.BOOL)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vec",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": settings.hnsw_m, "efConstruction": settings.hnsw_ef_construction},
        )
        index_params.add_index(
            field_name="sparse_vec",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        self.client.create_collection(
            collection_name=settings.milvus_collection,
            schema=schema,
            index_params=index_params,
        )

    def insert(self, chunks: list[dict]):
        """插入文档块"""
        self.collection.insert(
            collection_name=settings.milvus_collection,
            data=chunks,
        )

    def dense_search(self, query_vec, top_k: int = None) -> List[RetrievalResult]:
        """稠密向量检索"""
        top_k = top_k or settings.vector_top_k
        results = self.collection.search(
            collection_name=settings.milvus_collection,
            data=[query_vec],
            anns_field="dense_vec",
            limit=top_k,
            output_fields=["chunk_id", "text", "parent_id", "is_parent"],
            search_params={"params": {"ef": settings.hnsw_ef_search}},
        )
        return self._parse_results(results[0])

    def sparse_search(self, query_sparse: dict, top_k: int = None) -> List[RetrievalResult]:
        """稀疏向量检索"""
        top_k = top_k or settings.vector_top_k
        results = self.collection.search(
            collection_name=settings.milvus_collection,
            data=[query_sparse],
            anns_field="sparse_vec",
            limit=top_k,
            output_fields=["chunk_id", "text", "parent_id", "is_parent"],
        )
        return self._parse_results(results[0])

    def get_parent(self, parent_id: str) -> Optional[dict]:
        """根据 parent_id 获取父块完整文本"""
        results = self.collection.get(
            collection_name=settings.milvus_collection,
            ids=[parent_id],
            output_fields=["text", "parent_id", "is_parent"],
        )
        return results[0] if results else None

    def _parse_results(self, raw_results) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=hit.entity.get("chunk_id"),
                text=hit.entity.get("text"),
                score=hit.score,
                parent_id=hit.entity.get("parent_id"),
                metadata={"is_parent": hit.entity.get("is_parent")},
            )
            for hit in raw_results
        ]
