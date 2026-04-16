
"""
增强版图谱检索器 - 整合 NER、意图识别和自然语言返回
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import (
    SubgraphResult,
    GraphNode,
    GraphEdge,
    RelationType,
    RELATION_WEIGHTS
)
from .retriever import GraphRetriever

logger = logging.getLogger("EnhancedGraphRetriever")


@dataclass
class IntentQueryTemplate:
    """意图对应的查询模板"""
    intent_type: str
    description: str
    relations: List[str]
    max_hops: int = 1
    prompt_template: str = ""


# 意图查询模板配置
INTENT_QUERY_TEMPLATES = {
    "symptom_inquiry": IntentQueryTemplate(
        intent_type="symptom_inquiry",
        description="症状查询",
        relations=[
            RelationType.HAS_SYMPTOM.value,
            RelationType.ACCOMPANY_WITH.value,
            RelationType.RECOMMEND_DRUG.value,
            RelationType.CURE_WAY.value,
        ],
        max_hops=1,
        prompt_template="根据查询的症状，以下是相关的医疗知识：\n{knowledge}"
    ),
    "disease_inquiry": IntentQueryTemplate(
        intent_type="disease_inquiry",
        description="疾病查询",
        relations=[
            RelationType.HAS_SYMPTOM.value,
            RelationType.RECOMMEND_DRUG.value,
            RelationType.COMMON_DRUG.value,
            RelationType.NEED_CHECK.value,
            RelationType.CURE_WAY.value,
            RelationType.DO_EAT.value,
            RelationType.NO_EAT.value,
            RelationType.ACCOMPANY_WITH.value,
        ],
        max_hops=2,
        prompt_template="关于{entity}，以下是相关的医疗知识：\n{knowledge}"
    ),
    "medication_inquiry": IntentQueryTemplate(
        intent_type="medication_inquiry",
        description="用药咨询",
        relations=[
            RelationType.RECOMMEND_DRUG.value,
            RelationType.COMMON_DRUG.value,
            RelationType.DRUGS_OF.value,
            RelationType.CURE_WAY.value,
        ],
        max_hops=1,
        prompt_template="关于用药咨询，以下是相关的医疗知识：\n{knowledge}"
    ),
    "examination_inquiry": IntentQueryTemplate(
        intent_type="examination_inquiry",
        description="检查咨询",
        relations=[
            RelationType.NEED_CHECK.value,
        ],
        max_hops=1,
        prompt_template="关于检查咨询，以下是相关的医疗知识：\n{knowledge}"
    ),
    "diet_lifestyle": IntentQueryTemplate(
        intent_type="diet_lifestyle",
        description="饮食/生活",
        relations=[
            RelationType.DO_EAT.value,
            RelationType.NO_EAT.value,
            RelationType.RECOMMEND_EAT.value,
        ],
        max_hops=1,
        prompt_template="关于饮食和生活注意事项，以下是相关的医疗知识：\n{knowledge}"
    ),
    "other": IntentQueryTemplate(
        intent_type="other",
        description="其他",
        relations=[],  # 空列表表示使用所有关系
        max_hops=1,
        prompt_template="以下是相关的医疗知识：\n{knowledge}"
    ),
}


class EnhancedGraphRetriever:
    """
    增强版图谱检索器

    功能：
    1. 通过 NER 识别实体
    2. 通过意图识别确定查询模板
    3. 执行图谱检索
    4. 返回自然语言数据
    """

    def __init__(self,
                 graph_retriever: GraphRetriever,
                 ner_pipeline=None,
                 intent_classifier=None,
                 config: Optional[Dict] = None):
        """
        初始化增强版图谱检索器

        Args:
            graph_retriever: 基础图谱检索器
            ner_pipeline: NER 流水线（可选）
            intent_classifier: 意图分类器（可选）
            config: 配置字典
        """
        self.graph_retriever = graph_retriever
        self.ner_pipeline = ner_pipeline
        self.intent_classifier = intent_classifier
        self.config = config or {}

        logger.info("初始化增强版图谱检索器")
        if ner_pipeline:
            logger.info("  ✓ NER 流水线已配置")
        if intent_classifier:
            logger.info("  ✓ 意图分类器已配置")

    def retrieve(
        self,
        query: str,
        entities: Optional[List[Dict]] = None,
        intent_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        完整的检索流程

        Args:
            query: 用户查询
            entities: 预识别的实体列表（可选，不提供则使用 NER 识别）
            intent_type: 预识别的意图类型（可选，不提供则使用意图分类器）
            **kwargs: 其他传递给图谱检索器的参数

        Returns:
            包含检索结果的字典
        """
        logger.info(f"🔍 增强版图谱检索 | 查询: {query}")

        result = {
            "query": query,
            "entities": [],
            "intent_type": None,
            "intent_name": None,
            "subgraph": None,
            "natural_language": "",
            "success": False
        }

        # Step 1: 识别实体
        if entities is not None:
            result["entities"] = entities
            logger.info(f"  使用预提供的实体: {[e.get('text', e.get('name', '')) for e in entities]}")
        elif self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(query, return_dict=True)
                result["entities"] = ner_results
                logger.info(f"  NER 识别到实体: {[e.get('text', '') for e in ner_results]}")
            except Exception as e:
                logger.warning(f"  NER 识别失败: {e}")

        # Step 2: 识别意图
        if intent_type is not None:
            result["intent_type"] = intent_type
            template = INTENT_QUERY_TEMPLATES.get(intent_type)
            if template:
                result["intent_name"] = template.description
            logger.info(f"  使用预提供的意图: {intent_type}")
        elif self.intent_classifier:
            try:
                intent_result = self.intent_classifier(query)
                result["intent_type"] = intent_result.intent_type
                result["intent_name"] = intent_result.intent_name
                logger.info(f"  意图识别结果: {intent_result.intent_name}")
            except Exception as e:
                logger.warning(f"  意图识别失败: {e}")
                result["intent_type"] = "other"
                result["intent_name"] = "其他"
        else:
            result["intent_type"] = "other"
            result["intent_name"] = "其他"

        # Step 3: 获取查询模板
        template = INTENT_QUERY_TEMPLATES.get(result["intent_type"], INTENT_QUERY_TEMPLATES["other"])

        # Step 4: 执行图谱检索
        if result["entities"]:
            try:
                relations = kwargs.pop("relations", None)
                if relations is None:
                    relations = template.relations if template.relations else None

                max_hops = kwargs.pop("max_hops", None)
                if max_hops is None:
                    max_hops = template.max_hops

                subgraph = self.graph_retriever.retrieve_by_entities(
                    entities=result["entities"],
                    relations=relations,
                    max_hops=max_hops,
                    **kwargs
                )
                result["subgraph"] = subgraph
                logger.info(f"  图谱检索完成 | 节点: {len(subgraph.nodes)} | 边: {len(subgraph.edges)}")
            except Exception as e:
                logger.error(f"  图谱检索失败: {e}")

        # Step 5: 生成自然语言
        if result["subgraph"]:
            result["natural_language"] = self._generate_natural_language(
                result["subgraph"],
                template,
                result["entities"]
            )
            result["success"] = True

        return result

    def _generate_natural_language(
        self,
        subgraph: SubgraphResult,
        template: IntentQueryTemplate,
        entities: List[Dict]
    ) -> str:
        """
        生成自然语言描述

        Args:
            subgraph: 子图结果
            template: 查询模板
            entities: 识别到的实体

        Returns:
            自然语言字符串
        """
        if not subgraph.nodes:
            return "未找到相关的医疗知识。"

        knowledge_parts = []

        # 按实体类型分组
        grouped = {}
        for node in subgraph.nodes:
            label = node.primary_label
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(node)

        # 生成各类型的描述
        entity_names = [e.get("text", e.get("name", "")) for e in entities]
        entity_name_str = "、".join(entity_names[:3])
        if len(entity_names) > 3:
            entity_name_str += "等"

        # 疾病
        if "Disease" in grouped:
            diseases = [n.name for n in grouped["Disease"]]
            knowledge_parts.append(f"相关疾病：{', '.join(diseases)}")

        # 症状
        if "Symptom" in grouped:
            symptoms = [n.name for n in grouped["Symptom"]]
            knowledge_parts.append(f"相关症状：{', '.join(symptoms)}")

        # 药品
        if "Drug" in grouped:
            drugs = [n.name for n in grouped["Drug"]]
            knowledge_parts.append(f"推荐药品：{', '.join(drugs)}")

        # 检查
        if "Check" in grouped:
            checks = [n.name for n in grouped["Check"]]
            knowledge_parts.append(f"相关检查：{', '.join(checks)}")

        # 食物
        if "Food" in grouped:
            foods = [n.name for n in grouped["Food"]]
            knowledge_parts.append(f"相关食物：{', '.join(foods)}")

        # 关系描述
        rel_descriptions = []
        rel_counts = {}
        for edge in subgraph.edges:
            rel_type = edge.type
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

        rel_type_names = {
            RelationType.HAS_SYMPTOM.value: "有症状",
            RelationType.RECOMMEND_DRUG.value: "推荐药品",
            RelationType.COMMON_DRUG.value: "常用药品",
            RelationType.NEED_CHECK.value: "需要检查",
            RelationType.CURE_WAY.value: "治疗方法",
            RelationType.DO_EAT.value: "宜吃",
            RelationType.NO_EAT.value: "忌吃",
            RelationType.RECOMMEND_EAT.value: "推荐食用",
            RelationType.ACCOMPANY_WITH.value: "伴随",
            RelationType.BELONGS_TO.value: "属于",
            RelationType.DRUGS_OF.value: "药品所属",
        }

        for rel_type, count in rel_counts.items():
            name = rel_type_names.get(rel_type, rel_type)
            rel_descriptions.append(f"{name}({count}条)")

        if rel_descriptions:
            knowledge_parts.append(f"关系类型：{', '.join(rel_descriptions)}")

        # 组合最终文本
        knowledge_text = "\n".join([f"  - {part}" for part in knowledge_parts])

        # 使用模板
        if template.prompt_template:
            try:
                return template.prompt_template.format(
                    entity=entity_name_str,
                    knowledge=knowledge_text
                )
            except KeyError:
                pass

        # 兜底模板
        return f"根据您的查询，找到以下相关医疗知识：\n{knowledge_text}"

    def to_text_context(self, result: Dict[str, Any]) -> str:
        """
        将检索结果转换为 LLM 友好的文本上下文

        Args:
            result: retrieve() 返回的结果

        Returns:
            文本上下文
        """
        if not result.get("success"):
            return "【图谱知识】未找到相关医疗知识。"

        parts = ["【图谱知识增强】"]

        if result.get("intent_name"):
            parts.append(f"查询意图：{result['intent_name']}")

        if result.get("natural_language"):
            parts.append(result["natural_language"])

        return "\n".join(parts)


def create_enhanced_graph_retriever(
    graph_retriever: GraphRetriever,
    ner_pipeline=None,
    intent_classifier=None,
    **kwargs
) -> EnhancedGraphRetriever:
    """
    创建增强版图谱检索器的便捷函数

    Args:
        graph_retriever: 基础图谱检索器
        ner_pipeline: NER 流水线（可选）
        intent_classifier: 意图分类器（可选）
        **kwargs: 其他配置

    Returns:
        EnhancedGraphRetriever 实例
    """
    return EnhancedGraphRetriever(
        graph_retriever=graph_retriever,
        ner_pipeline=ner_pipeline,
        intent_classifier=intent_classifier,
        config=kwargs
    )

