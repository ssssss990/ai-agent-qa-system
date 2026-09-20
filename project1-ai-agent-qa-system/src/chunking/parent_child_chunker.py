"""
父子块切分策略

为什么用父子块而非固定长度切块：
- 固定长度切分会把一个完整的技术描述切成两半，检索时只命中半截，上下文断裂。
- 父子块策略：子块（128 字符）用于精确匹配，父块（512 字符）用于提供完整上下文。
- 检索时先命中子块，再回溯到父块返回给 LLM，兼顾匹配精度和上下文完整性。

走过的弯路：最初直接用固定 256 字符切块，实测型号、错误码这类短查询命中很差。
因为固定切块会把型号和它的描述切到不同块里。改成父子块后明显改善。
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str
    parent_id: Optional[str] = None
    is_parent: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "parent_id": self.parent_id,
            "is_parent": self.is_parent,
            "metadata": self.metadata,
        }


class ParentChildChunker:
    """
    父子块切分器

    流程：
    1. 按 parent_chunk_size 切父块（带 overlap）
    2. 每个父块再按 child_chunk_size 切子块（带 overlap）
    3. 子块记录 parent_id，检索时先命中子块，再回溯父块
    """

    def __init__(
        self,
        parent_size: int = None,
        child_size: int = None,
        overlap: int = None,
    ):
        self.parent_size = parent_size or settings.parent_chunk_size
        self.child_size = child_size or settings.child_chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def _make_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _sliding_window(self, text: str, size: int, overlap: int):
        """通用滑动窗口切分"""
        if len(text) <= size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += size - overlap
        return chunks

    def chunk_document(self, text: str, doc_id: str = "") -> list[Chunk]:
        """
        将文档切分为父子块结构

        Args:
            text: 原始文档文本
            doc_id: 文档标识，用于溯源

        Returns:
            按顺序排列的 Chunk 列表（父块和子块混合）
        """
        if not text or not text.strip():
            return []

        doc_id = doc_id or self._make_id(text)

        parent_texts = self._sliding_window(text, self.parent_size, self.overlap)
        results: list[Chunk] = []

        for i, p_text in enumerate(parent_texts):
            parent_id = f"{doc_id}_p{i}"
            parent_chunk = Chunk(
                chunk_id=parent_id,
                text=p_text,
                is_parent=True,
                metadata={"doc_id": doc_id, "parent_index": i},
            )
            results.append(parent_chunk)

            child_texts = self._sliding_window(p_text, self.child_size, self.overlap)
            for j, c_text in enumerate(child_texts):
                child_chunk = Chunk(
                    chunk_id=f"{parent_id}_c{j}",
                    text=c_text,
                    parent_id=parent_id,
                    is_parent=False,
                    metadata={"doc_id": doc_id, "child_index": j},
                )
                results.append(child_chunk)

        return results

    def get_parent_text(
        self, child: Chunk, all_chunks: dict[str, Chunk]
    ) -> str:
        """从子块回溯到父块文本"""
        if child.is_parent:
            return child.text
        parent = all_chunks.get(child.parent_id)
        if parent:
            return parent.text
        return child.text
