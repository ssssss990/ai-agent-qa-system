"""
BGE-M3 向量化模块

BGE-M3 同时产出稠密向量和稀疏向量：
- 稠密向量：用于语义相似度搜索
- 稀疏向量：类似 BM25 的词级匹配，但带学习权重

两者写入 Milvus 的同一个 Collection，检索时可以做混合召回。
"""

from typing import List
import numpy as np

from config.settings import settings


class BGEEmbedder:
    """BGE-M3 双向量化器"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,
            )
        return self._model

    def encode(self, texts: List[str]) -> dict:
        """
        编码文本，返回稠密向量和稀疏表示

        Returns:
            {
                "dense": np.ndarray,          # (n, 1024)
                "sparse": list[dict],         # [{token_id: weight, ...}, ...]
                "colbert": ...               # 可选，本项目不用
            }
        """
        if isinstance(texts, str):
            texts = [texts]

        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = np.array(output["dense_vecs"])
        sparse = output["lexical_weights"]

        return {"dense": dense, "sparse": sparse}

    def encode_query(self, query: str) -> dict:
        """单条查询编码"""
        return self.encode([query])
