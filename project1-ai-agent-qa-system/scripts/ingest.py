"""
文档入库脚本

用法：
python scripts/ingest.py --input data/docs/

将 data/docs/ 目录下所有 .txt 和 .md 文件入库到 Milvus + BM25
"""

import argparse
import os
from pathlib import Path

from src.chunking.parent_child_chunker import ParentChildChunker
from src.embedding.bge_embedder import BGEEmbedder
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.bm25_store import BM25Store


def ingest(input_dir: str):
    chunker = ParentChildChunker()
    embedder = BGEEmbedder()
    milvus = MilvusStore()
    bm25 = BM25Store()

    files = list(Path(input_dir).glob("**/*.txt")) + \
            list(Path(input_dir).glob("**/*.md"))

    if not files:
        print(f"No files found in {input_dir}")
        return

    print(f"Found {len(files)} files to ingest")

    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunker.chunk_document(text, doc_id=f.stem)
        print(f"  {f.name}: {len(chunks)} chunks")

        # 向量化
        child_chunks = [c for c in chunks if not c.is_parent]
        if child_chunks:
            texts = [c.text for c in child_chunks]
            encoded = embedder.encode(texts)

            milvus_data = []
            for i, c in enumerate(child_chunks):
                milvus_data.append({
                    "chunk_id": c.chunk_id,
                    "dense_vec": encoded["dense"][i].tolist(),
                    "sparse_vec": encoded["sparse"][i],
                    "text": c.text,
                    "parent_id": c.parent_id,
                    "is_parent": False,
                })
            milvus.insert(milvus_data)

        # BM25 索引
        all_chunk_dicts = [c.to_dict() for c in chunks]
        bm25.add_documents(all_chunk_dicts)

    print("Ingest complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/docs/")
    args = parser.parse_args()
    ingest(args.input)
