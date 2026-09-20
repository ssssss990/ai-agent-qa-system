"""
RAG 检索工具：封装为可被 Agent 调用的工具接口

两阶段提交设计（安全护栏）：
- 第一次调用返回预览结果（不执行副作用操作）
- 用户确认后才执行（如修改订单、发退款等）
- 本工具只做只读检索，没有副作用，所以直接返回
"""

from typing import Any
from dataclasses import dataclass

from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from config.settings import settings


@dataclass
class ToolResult:
    documents: list[dict]
    total: int
    metadata: dict


class RAGSearchTool:
    """RAG 检索工具"""

    def __init__(self, retriever: HybridRetriever = None,
                 reranker: CrossEncoderReranker = None):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.requires_confirmation = False

    async def execute(self, tool_input: dict) -> dict:
        """
        执行检索

        Args:
            tool_input: {"query": "查询文本", "top_k": 5}

        Returns:
            {"documents": [...], "total": int}
        """
        query = tool_input.get("query", "")
        if not query.strip():
            return {"documents": [], "total": 0}

        # 混合召回 + RRF 融合
        candidates = self.retriever.retrieve(query)

        # Cross-Encoder 重排
        reranked = self.reranker.rerank(query, candidates)

        return {
            "documents": [
                {"chunk_id": r.chunk_id, "text": r.text, "score": r.score}
                for r in reranked
            ],
            "total": len(reranked),
        }

    def describe(self) -> dict:
        """工具描述，供 LLM 理解工具用途"""
        return {
            "name": "rag_search",
            "description": "检索技术文档和售后 FAQ，获取与用户问题相关的参考信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询文本",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }
