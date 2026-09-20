"""
FastAPI 服务接口

提供：
- POST /chat：多轮对话接口
- POST /documents：上传文档接口
- GET /health：健康检查
- GET /stats：缓存和检索统计

安全设计：
- 用户输入与检索内容分角色拼接，降低间接 Prompt 注入风险
- 输出用正则脱敏手机号等 PII
"""

import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.react_agent import ReActAgent
from src.cache.semantic_cache import SemanticCache
from src.tools.rag_tool import RAGSearchTool
from src.utils.llm_client import LLMClient
from src.utils.tracing import setup_tracing
from config.settings import settings

app = FastAPI(title="AI Agent QA System", version="1.0.0")

# 初始化组件
setup_tracing(settings.otel_endpoint, settings.otel_service_name)
llm = LLMClient()
rag_tool = RAGSearchTool()
cache = SemanticCache()
agent = ReActAgent(llm=llm, tools={"rag_search": rag_tool})


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    use_cache: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    cached: bool = False
    steps: list[dict] = []


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 语义缓存
    if req.use_cache:
        cached = cache.get(req.query)
        if cached:
            return ChatResponse(
                answer=cached["answer"],
                sources=cached["sources"],
                cached=True,
                steps=[],
            )

    # Agent 执行
    result = await agent.run(req.query, req.session_id)

    # 脱敏 PII
    answer = _sanitize_pii(result["answer"])

    # 写入缓存
    if req.use_cache:
        cache.put(req.query, answer, result.get("sources", []))

    return ChatResponse(
        answer=answer,
        sources=result.get("sources", []),
        cached=False,
        steps=result.get("steps", []),
    )


class DocumentRequest(BaseModel):
    text: str
    doc_id: Optional[str] = ""


@app.post("/documents")
async def add_document(req: DocumentRequest):
    """上传文档到知识库"""
    from src.chunking.parent_child_chunker import ParentChildChunker
    chunker = ParentChildChunker()
    chunks = chunker.chunk_document(req.text, req.doc_id)

    # 写入 Milvus 和 BM25 索引
    rag_tool.retriever.milvus.insert([c.to_dict() for c in chunks])
    rag_tool.retriever.bm25.add_documents([c.to_dict() for c in chunks])

    return {"status": "ok", "chunks_created": len(chunks)}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/stats")
async def stats():
    return {
        "cache": cache.stats(),
        "milvus_collection": settings.milvus_collection,
    }


def _sanitize_pii(text: str) -> str:
    """脱敏 PII：手机号、邮箱、身份证号"""
    text = re.sub(r'1[3-9]\d{9}', '[手机号]', text)
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[邮箱]', text
    )
    text = re.sub(r'\d{17}[\dXx]', '[身份证号]', text)
    return text
