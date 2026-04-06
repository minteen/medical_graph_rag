# base.py
# 图谱检索基础数据结构
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("GraphRetriever")


class RelationType(Enum):
    """医疗关系类型枚举（与知识库保持一致）"""
    RECOMMEND_EAT = "recommand_eat"
    NO_EAT = "no_eat"
    DO_EAT = "do_eat"
    BELONGS_TO = "belongs_to"
    COMMON_DRUG = "common_drug"
    DRUGS_OF = "drugs_of"
    RECOMMEND_DRUG = "recommand_drug"
    NEED_CHECK = "need_check"
    HAS_SYMPTOM = "has_symptom"
    ACCOMPANY_WITH = "acompany_with"
    CURE_WAY = "cure_way"


# 关系类型权重（用于置信度计算）
RELATION_WEIGHTS = {
    RelationType.RECOMMEND_DRUG.value: 1.0,
    RelationType.COMMON_DRUG.value: 0.9,
    RelationType.HAS_SYMPTOM.value: 0.95,
    RelationType.NEED_CHECK.value: 0.85,
    RelationType.CURE_WAY.value: 0.9,
    RelationType.DO_EAT.value: 0.8,
    RelationType.NO_EAT.value: 0.85,
    RelationType.RECOMMEND_EAT.value: 0.75,
    RelationType.ACCOMPANY_WITH.value: 0.7,
    RelationType.DRUGS_OF.value: 0.8,
    RelationType.BELONGS_TO.value: 0.6,
}


@dataclass
class GraphNode:
    """图谱节点"""
    id: str  # 节点唯一标识（使用 name 或 Neo4j elementId）
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def name(self) -> str:
        return self.properties.get("name", self.id)

    @property
    def primary_label(self) -> str:
        return self.labels[0] if self.labels else "Unknown"


@dataclass
class GraphEdge:
    """图谱边"""
    id: str
    type: str
    start_id: str
    end_id: str
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.properties.get("name", self.type)


@dataclass
class SubgraphResult:
    """标准化子图输出，供下游 Context Builder 直接消费"""
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pruned_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "labels": n.labels,
                    "properties": n.properties,
                    "confidence": n.confidence
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "type": e.type,
                    "start_id": e.start_id,
                    "end_id": e.end_id,
                    "properties": e.properties
                }
                for e in self.edges
            ],
            "metadata": self.metadata,
            "pruned_count": self.pruned_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubgraphResult":
        """从字典加载"""
        return cls(
            nodes=[GraphNode(**n) for n in data.get("nodes", [])],
            edges=[GraphEdge(**e) for e in data.get("edges", [])],
            metadata=data.get("metadata", {}),
            pruned_count=data.get("pruned_count", 0)
        )
