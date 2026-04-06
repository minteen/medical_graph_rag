# retriever.py
# 图谱检索器 - 主实现
import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

try:
    from neo4j import GraphDatabase, Record, Driver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None
    Record = None
    Driver = None

from .base import (
    SubgraphResult,
    GraphNode,
    GraphEdge,
    RelationType,
    RELATION_WEIGHTS
)

logger = logging.getLogger("GraphRetriever")


class GraphRetriever:
    """
    图谱检索器：封装医疗知识图谱 Cypher 模板，支持 1~2 跳安全子图提取。

    核心机制：
    - 参数化 Cypher 模板（防注入）
    - 置信度动态加权（关系权重 + 节点完整度）
    - 三重安全剪枝（防爆炸/防循环/防超点）
    """

    # 所有支持的关系类型
    ALL_RELATIONS = [
        RelationType.RECOMMEND_EAT.value,
        RelationType.NO_EAT.value,
        RelationType.DO_EAT.value,
        RelationType.BELONGS_TO.value,
        RelationType.COMMON_DRUG.value,
        RelationType.DRUGS_OF.value,
        RelationType.RECOMMEND_DRUG.value,
        RelationType.NEED_CHECK.value,
        RelationType.HAS_SYMPTOM.value,
        RelationType.ACCOMPANY_WITH.value,
        RelationType.CURE_WAY.value,
    ]

    # 按实体类型分组的常用关系（用于智能推荐）
    RELATIONS_BY_ENTITY = {
        "Disease": [
            RelationType.HAS_SYMPTOM.value,
            RelationType.RECOMMEND_DRUG.value,
            RelationType.COMMON_DRUG.value,
            RelationType.NEED_CHECK.value,
            RelationType.CURE_WAY.value,
            RelationType.DO_EAT.value,
            RelationType.NO_EAT.value,
            RelationType.ACCOMPANY_WITH.value,
            RelationType.BELONGS_TO.value,
        ],
        "Drug": [
            RelationType.DRUGS_OF.value,
        ],
        "Symptom": [],  # 症状通常作为目标节点
        "Check": [],
        "Department": [
            RelationType.BELONGS_TO.value,
        ],
        "Food": [],
        "Producer": [
            RelationType.DRUGS_OF.value,
        ],
        "Cure": [],
    }

    def __init__(self,
                 driver: Optional[Driver] = None,
                 uri: Optional[str] = None,
                 auth: Optional[Tuple[str, str]] = None,
                 config: Optional[Dict] = None):
        """
        初始化图谱检索器

        Args:
            driver: Neo4j Driver 实例（优先使用）
            uri: Neo4j URI（如 bolt://localhost:7687）
            auth: Neo4j 认证信息 (username, password)
            config: 配置字典
                - max_hops: 最大跳数 (默认 1)
                - confidence_threshold: 置信度阈值 (默认 0.5)
                - max_results_per_seed: 每个种子节点的最大结果数 (默认 30)
                - degree_limit: 超级节点剪枝阈值 (默认 50)
                - max_total_nodes: 最大总节点数 (默认 100)
        """
        if not HAS_NEO4J:
            raise ImportError("neo4j 未安装，请运行: pip install neo4j")

        self.config = config or {}
        self.max_hops = self.config.get("max_hops", 1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.max_results_per_seed = self.config.get("max_results_per_seed", 30)
        self.degree_limit = self.config.get("degree_limit", 50)
        self.max_total_nodes = self.config.get("max_total_nodes", 100)

        # 初始化驱动
        self._owns_driver = False
        if driver:
            self.driver = driver
        elif uri and auth:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self._owns_driver = True
        else:
            # 尝试从环境变量读取
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._owns_driver = True

        self._closed = False

    def close(self):
        """关闭连接（仅当自己创建的 driver）"""
        if self._owns_driver and self.driver and not self._closed:
            self.driver.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ================= 核心检索接口 =================

    def retrieve_by_entities(self,
                             entities: List[Dict],
                             relations: Optional[List[str]] = None,
                             max_hops: Optional[int] = None) -> SubgraphResult:
        """
        通过实体列表检索子图（主入口）

        Args:
            entities: 实体列表，每个实体需要包含:
                - kg_id 或 id: 图谱节点 ID
                - type: 实体类型（可选，用于智能选择关系）
                - confidence: 置信度（可选）
            relations: 要检索的关系类型列表，None 则根据实体类型自动选择
            max_hops: 最大跳数，None 则使用默认值

        Returns:
            SubgraphResult: 子图结果
        """
        if not entities:
            logger.warning("⚠️  实体列表为空，跳过图谱检索")
            return SubgraphResult()

        # 过滤有效实体
        valid_entities = []
        seed_ids = []
        for e in entities:
            kg_id = e.get("kg_id") or e.get("id")
            if kg_id:
                valid_entities.append(e)
                seed_ids.append(kg_id)

        if not seed_ids:
            logger.warning("⚠️  无有效图谱 ID，跳过图谱检索")
            return SubgraphResult()

        # 智能选择关系
        if relations is None:
            relations = self._infer_relations(valid_entities)

        if not relations:
            logger.warning("⚠️  未指定关系类型，使用全部关系")
            relations = self.ALL_RELATIONS

        max_hops = max_hops or self.max_hops

        logger.info(f"🔍 图谱检索 | 种子: {len(seed_ids)} | 关系: {relations} | 跳数: {max_hops}")

        return self.execute(seed_ids, relations, max_hops)

    def retrieve_by_names(self,
                          names: List[str],
                          labels: Optional[List[str]] = None,
                          relations: Optional[List[str]] = None,
                          max_hops: Optional[int] = None) -> SubgraphResult:
        """
        通过实体名称检索（先查找节点 ID，再检索子图）

        Args:
            names: 实体名称列表
            labels: 节点标签限制（可选）
            relations: 关系类型列表（可选）
            max_hops: 最大跳数（可选）

        Returns:
            SubgraphResult: 子图结果
        """
        if not names:
            return SubgraphResult()

        # 先查找节点
        seed_ids = self._find_nodes_by_names(names, labels)
        if not seed_ids:
            logger.warning(f"⚠️  未找到节点: {names}")
            return SubgraphResult()

        if relations is None:
            relations = self.ALL_RELATIONS

        max_hops = max_hops or self.max_hops

        return self.execute(seed_ids, relations, max_hops)

    def execute(self,
                seed_ids: List[str],
                relations: List[str],
                max_hops: int = 1) -> SubgraphResult:
        """
        执行图谱检索（底层接口）

        Args:
            seed_ids: 种子节点 ID 列表
            relations: 关系类型列表
            max_hops: 最大跳数

        Returns:
            SubgraphResult: 子图结果
        """
        if not seed_ids or not relations:
            return SubgraphResult()

        # 构建 Cypher
        cypher = self._build_cypher(relations, max_hops)
        if not cypher:
            return SubgraphResult()

        params = {
            "seed_ids": seed_ids,
            "degree_limit": self.degree_limit,
            "limit": self.max_results_per_seed * len(seed_ids)
        }

        try:
            with self.driver.session() as session:
                result = session.run(cypher, params)
                raw_records = list(result)
        except Exception as e:
            logger.error(f"❌ Cypher 执行失败: {e}")
            return SubgraphResult()

        return self._process_and_prune(raw_records, relations, max_hops)

    # ================= Cypher 模板引擎 =================

    def _build_cypher(self, relations: List[str], max_hops: int = 1) -> str:
        """
        生成参数化 Cypher 模板

        设计原则：
        1. 使用 UNWIND + MATCH 替代字符串拼接
        2. 提前进行超级节点剪枝
        3. 1跳/2跳分开处理，避免隐式爆炸
        """
        if not relations:
            return ""

        rel_str = "|".join(relations)

        if max_hops == 1:
            return f"""
            UNWIND $seed_ids AS sid
            MATCH (seed) WHERE seed.name = sid OR seed.id = sid
            MATCH (seed)-[r:{rel_str}]->(n)
            WHERE n <> seed
              AND (NOT exists(n.__degree) OR n.__degree < $degree_limit)
            RETURN seed, r, n, sid as seed_id
            LIMIT $limit
            """

        if max_hops == 2:
            # 2跳查询：显式分步，避免路径爆炸
            return f"""
            UNWIND $seed_ids AS sid
            MATCH (seed) WHERE seed.name = sid OR seed.id = sid

            // 第1跳
            MATCH (seed)-[r1:{rel_str}]->(n1)
            WHERE n1 <> seed
              AND (NOT exists(n1.__degree) OR n1.__degree < $degree_limit)

            // 第2跳（可选，只扩展有用的路径）
            OPTIONAL MATCH (n1)-[r2:{rel_str}]->(n2)
            WHERE n2 IS NULL OR (
                n2 <> seed
                AND n2 <> n1
                AND (NOT exists(n2.__degree) OR n2.__degree < $degree_limit)
            )

            WITH seed, r1, n1, r2, n2, sid
            LIMIT $limit
            RETURN seed, r1, n1, r2, n2, sid as seed_id
            """

        return ""

    def _find_nodes_by_names(self, names: List[str], labels: Optional[List[str]] = None) -> List[str]:
        """通过名称查找节点 ID"""
        label_str = f":{labels[0]}" if labels else ""
        cypher = f"""
        UNWIND $names AS name
        MATCH (n{label_str})
        WHERE n.name = name
        RETURN n.name AS name
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, names=names)
                return [record["name"] for record in result]
        except Exception as e:
            logger.error(f"❌ 查找节点失败: {e}")
            return []

    # ================= 结果处理与剪枝 =================

    def _process_and_prune(self,
                           records: List,
                           relations: List[str],
                           max_hops: int) -> SubgraphResult:
        """
        置信度计算 + 路径剪枝 + 结果去重

        剪枝策略：
        1. 置信度阈值过滤
        2. 总节点数限制
        3. 重复边/节点去重
        """
        subgraph = SubgraphResult()
        nodes_map: Dict[str, GraphNode] = {}
        edges_map: Dict[str, GraphEdge] = {}
        pruned = 0

        for rec in records:
            # 解析记录（兼容1跳和2跳）
            seed_node = rec.get("seed")
            seed_id = rec.get("seed_id")

            # 处理第1跳
            r1 = rec.get("r1") or rec.get("r")
            n1 = rec.get("n1") or rec.get("n")

            if r1 and n1:
                # 计算置信度
                confidence = self._calculate_confidence(r1, n1)

                # 置信度过滤
                if confidence < self.confidence_threshold:
                    pruned += 1
                    continue

                # 添加种子节点（如果还没有）
                if seed_node and seed_id not in nodes_map:
                    nodes_map[seed_id] = self._to_graph_node(seed_node, 1.0)

                # 添加第1跳节点
                n1_id = n1.get("name") or getattr(n1, "element_id", str(id(n1)))
                if n1_id not in nodes_map:
                    nodes_map[n1_id] = self._to_graph_node(n1, confidence)

                # 添加第1跳边
                edge1 = self._to_graph_edge(r1)
                if edge1.id not in edges_map:
                    edges_map[edge1.id] = edge1

                # 处理第2跳
                r2 = rec.get("r2")
                n2 = rec.get("n2")
                if r2 and n2 and max_hops >= 2:
                    # 第2跳置信度略低
                    confidence2 = self._calculate_confidence(r2, n2) * 0.8

                    if confidence2 >= self.confidence_threshold:
                        n2_id = n2.get("name") or getattr(n2, "element_id", str(id(n2)))
                        if n2_id not in nodes_map:
                            nodes_map[n2_id] = self._to_graph_node(n2, confidence2)

                        edge2 = self._to_graph_edge(r2)
                        if edge2.id not in edges_map:
                            edges_map[edge2.id] = edge2
                    else:
                        pruned += 1

        # 总节点数限制
        if len(nodes_map) > self.max_total_nodes:
            # 按置信度排序，保留最高的
            sorted_nodes = sorted(nodes_map.values(), key=lambda x: -x.confidence)
            kept_nodes = {n.id: n for n in sorted_nodes[:self.max_total_nodes]}
            pruned += len(nodes_map) - len(kept_nodes)
            nodes_map = kept_nodes

            # 同时移除引用了被删除节点的边
            kept_node_ids = set(nodes_map.keys())
            edges_map = {
                eid: e for eid, e in edges_map.items()
                if e.start_id in kept_node_ids and e.end_id in kept_node_ids
            }

        subgraph.nodes = list(nodes_map.values())
        subgraph.edges = list(edges_map.values())
        subgraph.metadata = {
            "total_raw": len(records),
            "pruned_count": pruned,
            "confidence_threshold": self.confidence_threshold,
            "applied_relations": relations,
            "max_hops": max_hops
        }
        subgraph.pruned_count = pruned

        return subgraph

    def _calculate_confidence(self, rel, node) -> float:
        """
        计算置信度：关系权重 (70%) + 节点完整度 (30%)

        节点完整度：有值属性的比例
        """
        # 关系权重
        rel_type = rel.type if rel else None
        base_weight = RELATION_WEIGHTS.get(rel_type, 0.5)

        # 节点完整度
        props = dict(node.items()) if node else {}
        if props:
            # 只统计有意义的属性
            meaningful_props = [k for k in props.keys() if k not in ("name", "id")]
            if meaningful_props:
                completeness = sum(1 for k in meaningful_props if props.get(k)) / len(meaningful_props)
            else:
                completeness = 0.5
        else:
            completeness = 0.5

        return (base_weight * 0.7) + (completeness * 0.3)

    def _to_graph_node(self, node, confidence: float = 1.0) -> GraphNode:
        """转换 Neo4j Node 为 GraphNode"""
        node_id = node.get("name") or getattr(node, "element_id", str(id(node)))
        labels = list(getattr(node, "labels", ["Unknown"]))
        properties = dict(node)
        return GraphNode(
            id=node_id,
            labels=labels,
            properties=properties,
            confidence=confidence
        )

    def _to_graph_edge(self, rel) -> GraphEdge:
        """转换 Neo4j Relationship 为 GraphEdge"""
        edge_id = getattr(rel, "element_id", f"rel_{id(rel)}")

        # 获取起点和终点 ID
        start_node = rel.start_node if hasattr(rel, "start_node") else None
        end_node = rel.end_node if hasattr(rel, "end_node") else None

        start_id = start_node.get("name") if start_node else None
        end_id = end_node.get("name") if end_node else None

        return GraphEdge(
            id=edge_id,
            type=rel.type,
            start_id=start_id,
            end_id=end_id,
            properties=dict(rel)
        )

    def _infer_relations(self, entities: List[Dict]) -> List[str]:
        """根据实体类型智能推断关系"""
        relations = set()
        for e in entities:
            etype = e.get("type")
            if etype and etype in self.RELATIONS_BY_ENTITY:
                relations.update(self.RELATIONS_BY_ENTITY[etype])
        return list(relations) if relations else self.ALL_RELATIONS

    # ================= 便捷工具方法 =================

    def to_text_context(self, subgraph: SubgraphResult) -> str:
        """将子图转换为 LLM 友好的自然语言上下文"""
        if not subgraph.nodes:
            return "【图谱知识】未找到相关医疗知识。"

        ctx_parts = ["【图谱知识增强】"]

        # 按实体类型分组
        grouped: Dict[str, List[GraphNode]] = {}
        for node in subgraph.nodes:
            typ = node.primary_label
            grouped.setdefault(typ, []).append(node)

        for typ, nodes in grouped.items():
            names = [n.name for n in nodes[:10]]  # 最多显示10个
            if len(nodes) > 10:
                names.append(f"...等{len(nodes)}个")
            ctx_parts.append(f"- **{typ}**: {', '.join(names)}")

        # 关系摘要（按类型统计）
        if subgraph.edges:
            rel_counts: Dict[str, int] = {}
            for e in subgraph.edges:
                rel_counts[e.type] = rel_counts.get(e.type, 0) + 1

            rel_summary = ", ".join([f"{t}({c})" for t, c in rel_counts.items()])
            ctx_parts.append(f"- **关系**: {rel_summary}")

        return "\n".join(ctx_parts)


# ================= 便捷工厂函数 =================

def create_graph_retriever(uri: Optional[str] = None,
                           auth: Optional[Tuple[str, str]] = None,
                           **kwargs) -> GraphRetriever:
    """
    创建图谱检索器的便捷函数

    Args:
        uri: Neo4j URI
        auth: (username, password)
        **kwargs: 其他配置传递给 GraphRetriever

    Returns:
        GraphRetriever 实例
    """
    return GraphRetriever(uri=uri, auth=auth, config=kwargs)
