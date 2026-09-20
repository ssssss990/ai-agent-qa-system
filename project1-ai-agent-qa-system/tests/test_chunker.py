"""
测试父子块切分器
"""

import pytest
from src.chunking.parent_child_chunker import ParentChildChunker, Chunk


class TestParentChildChunker:
    def setup_method(self):
        self.chunker = ParentChildChunker(
            parent_size=100, child_size=30, overlap=10
        )

    def test_short_text_returns_single_parent(self):
        text = "这是一段很短的文本。"
        chunks = self.chunker.chunk_document(text)
        assert len(chunks) >= 1
        assert chunks[0].is_parent is True

    def test_child_has_parent_id(self):
        text = "这是一段比较长的文本" * 20
        chunks = self.chunker.chunk_document(text)
        children = [c for c in chunks if not c.is_parent]
        assert len(children) > 0
        assert all(c.parent_id for c in children)

    def test_empty_text_returns_empty(self):
        chunks = self.chunker.chunk_document("")
        assert len(chunks) == 0

    def test_doc_id_propagation(self):
        text = "测试文档内容" * 10
        chunks = self.chunker.chunk_document(text, doc_id="test_doc")
        assert all(c.metadata.get("doc_id") == "test_doc" for c in chunks)
