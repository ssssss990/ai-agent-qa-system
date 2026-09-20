"""
全局配置：所有可调参数集中管理，便于实验对比和回归测试。
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_docs"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    parent_chunk_size: int = 512
    child_chunk_size: int = 128
    chunk_overlap: int = 32

    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5

    agent_max_steps: int = 8
    agent_retry_limit: int = 1

    local_model: str = "Qwen/Qwen2.5-7B-Instruct"
    local_model_base_url: str = "http://localhost:8001/v1"
    remote_model: str = "deepseek-chat"
    remote_model_base_url: str = "https://api.deepseek.com/v1"

    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "ai-agent-qa"

    eval_dataset_path: str = "data/eval/eval_dataset.json"
    cache_similarity_threshold: float = 0.92

    class Config:
        env_file = ".env"
        env_prefix = "APP_"


settings = Settings()
