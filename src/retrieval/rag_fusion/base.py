# base.py
# RAG 融合基础数据结构
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("RAGFusion")


class SourceType(Enum):
    """知识来源类型"""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR_STORE = "vector_store"
    BOTH = "both"


class FusionStrategy(Enum):
    """融合策略"""
    CONCATENATION = "concatenation"  # 简单拼接
    WEIGHTED = "weighted"  # 加权融合
    RERANK = "rerank"  # 重排序
    RECIPROCAL_RANK = "reciprocal_rank"  # 倒数排序融合 (RRF)


@dataclass
class RAGSource:
    """单个知识来源"""
    source_type: SourceType
    content: str  # 文本内容
    score: float = 1.0  # 相关性分数 (0.0~1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    # 图谱特有
    node_ids: List[str] = field(default_factory=list)
    edge_count: int = 0

    # 向量文档特有
    chunk_id: Optional[str] = None
    entity_type: Optional[str] = None


@dataclass
class RAGQuery:
    """RAG 查询"""
    query_text: str  # 用户问题
    query_embedding: Optional[List[float]] = None  # 查询向量（可选）

    # 实体信息（来自 NER）
    entities: List[Dict] = field(default_factory=list)

    # 检索配置
    use_graph: bool = True
    use_vector: bool = True
    graph_max_hops: int = 1
    vector_top_k: int = 5
    vector_score_threshold: float = 0.0

    # 融合配置
    fusion_strategy: FusionStrategy = FusionStrategy.CONCATENATION
    graph_weight: float = 0.5
    vector_weight: float = 0.5
    max_total_chars: int = 6000


@dataclass
class RAGResult:
    """RAG 结果"""
    query: RAGQuery  # 原始查询
    sources: List[RAGSource]  # 所有知识来源
    fused_context: str  # 融合后的上下文
    source_type_counts: Dict[str, int] = field(default_factory=dict)

    # 统计信息
    graph_node_count: int = 0
    graph_edge_count: int = 0
    vector_doc_count: int = 0
    latency_ms: float = 0.0

    def to_llm_prompt(self, include_sources: bool = False) -> str:
        """
        转换为 LLM 提示词

        Args:
            include_sources: 是否包含来源标记

        Returns:
            LLM 提示词字符串
        """
        parts = []

        # 系统提示
        parts.append("【背景知识】")
        parts.append("以下是相关的医疗知识，请基于这些知识回答问题：")
        parts.append("")

        # 融合的上下文
        parts.append(self.fused_context)

        # 来源标记（可选）
        if include_sources and self.sources:
            parts.append("")
            parts.append("【知识来源】")
            for i, source in enumerate(self.sources, 1):
                type_str = "图谱" if source.source_type == SourceType.KNOWLEDGE_GRAPH else "文档"
                parts.append(f"[{i}] {type_str} (score: {source.score:.3f})")

        return "\n".join(parts)
