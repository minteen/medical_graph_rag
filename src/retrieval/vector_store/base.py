# base.py
# 向量化文档检索基础数据结构
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("VectorStore")


class VectorStoreType(Enum):
    """向量数据库类型"""
    QDRANT = "qdrant"
    MILVUS = "milvus"
    FAISS = "faiss"  # 本地 FAISS 索引
    NUMPY = "numpy"  # 纯 NumPy 实现（无额外依赖）


class SearchMode(Enum):
    """检索模式"""
    VECTOR_ONLY = "vector_only"  # 纯向量检索
    KEYWORD_ONLY = "keyword_only"  # 纯关键词检索
    HYBRID = "hybrid"  # 混合检索


@dataclass
class DocumentChunk:
    """文档块（用于索引）"""
    chunk_id: str  # 唯一标识
    text: str  # 原始文本
    embedding: Optional[List[float]] = None  # 向量（可选，可在线生成）

    # 元数据（用于过滤）
    kg_id: Optional[str] = None  # 关联的图谱节点 ID
    entity_type: Optional[str] = None  # 实体类型
    section_type: Optional[str] = None  # 章节类型
    involved_relations: List[str] = field(default_factory=list)  # 涉及的关系
    safety_flags: List[str] = field(default_factory=list)  # 安全标记

    # 其他元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """转换为向量数据库的 payload"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "kg_id": self.kg_id,
            "entity_type": self.entity_type,
            "section_type": self.section_type,
            "involved_relations": self.involved_relations,
            "safety_flags": self.safety_flags,
            "metadata": self.metadata
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DocumentChunk":
        """从 payload 加载"""
        return cls(
            chunk_id=payload.get("chunk_id", ""),
            text=payload.get("text", ""),
            kg_id=payload.get("kg_id"),
            entity_type=payload.get("entity_type"),
            section_type=payload.get("section_type"),
            involved_relations=payload.get("involved_relations", []),
            safety_flags=payload.get("safety_flags", []),
            metadata=payload.get("metadata", {})
        )


@dataclass
class SearchResult:
    """检索结果"""
    chunk: DocumentChunk  # 文档块
    score: float  # 相似度分数 (0.0~1.0)
    rank: int = 0  # 排名
    search_mode: SearchMode = SearchMode.VECTOR_ONLY  # 检索模式

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class SearchQuery:
    """检索查询"""
    query_text: str  # 查询文本
    query_embedding: Optional[List[float]] = None  # 查询向量（可选）

    # 过滤条件
    filter_kg_ids: Optional[List[str]] = None  # 图谱 ID 过滤
    filter_entity_types: Optional[List[str]] = None  # 实体类型过滤
    filter_section_types: Optional[List[str]] = None  # 章节类型过滤
    filter_relations: Optional[List[str]] = None  # 关系过滤

    # 检索参数
    top_k: int = 10  # 返回 top K
    score_threshold: float = 0.0  # 最低分数阈值
    search_mode: SearchMode = SearchMode.VECTOR_ONLY  # 检索模式

    # 混合检索权重
    hybrid_vector_weight: float = 0.7  # 向量检索权重
    hybrid_keyword_weight: float = 0.3  # 关键词检索权重


@dataclass
class SearchResponse:
    """检索响应"""
    query: SearchQuery  # 原始查询
    results: List[SearchResult]  # 检索结果
    total_count: int = 0  # 命中总数
    latency_ms: float = 0.0  # 检索耗时（毫秒）

    def to_text_context(self, max_chars: int = 4000) -> str:
        """转换为 LLM 友好的文本上下文"""
        if not self.results:
            return "【文档检索】未找到相关文档。"

        parts = ["【文档知识增强】"]
        total_len = len(parts[0])

        for i, result in enumerate(self.results, 1):
            chunk_text = result.text.strip()
            score_str = f" (相似度: {result.score:.3f})"

            # 截断过长的文本
            if total_len + len(chunk_text) + len(score_str) + 20 > max_chars:
                remaining = max_chars - total_len - len(score_str) - 30
                if remaining > 50:
                    chunk_text = chunk_text[:remaining] + "..."
                else:
                    break

            parts.append(f"\n[{i}]{score_str}")
            parts.append(chunk_text)
            total_len += len(chunk_text) + len(score_str) + 20

        return "".join(parts)
