# fuser.py
# RAG 融合器主实现
import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from .base import (
    RAGQuery,
    RAGResult,
    RAGSource,
    SourceType,
    FusionStrategy
)

logger = logging.getLogger("RAGFuser")


class RAGFuser:
    """
    RAG 融合器：结合知识图谱和向量文档检索

    特点：
    - 支持多种融合策略（拼接、加权、重排序、RRF）
    - 支持仅图谱、仅向量、混合模式
    - 自动长度截断和去重
    - 可配置权重
    """

    def __init__(self,
                 graph_retriever=None,
                 vector_store=None,
                 ner_router=None,
                 entity_linker=None):
        """
        Args:
            graph_retriever: 图谱检索器 (GraphRetriever)
            vector_store: 向量存储 (NumpyVectorStore/QdrantVectorStore/MilvusVectorStore)
            ner_router: NER 路由器 (ConfidenceRouter)
            entity_linker: 实体链接器 (EntityLinker)
        """
        self.graph_retriever = graph_retriever
        self.vector_store = vector_store
        self.ner_router = ner_router
        self.entity_linker = entity_linker

    def query(self, rag_query: RAGQuery) -> RAGResult:
        """
        执行 RAG 查询（主入口）

        Args:
            rag_query: RAG 查询对象

        Returns:
            RAGResult: RAG 结果
        """
        start_time = time.time()

        # Step 1: NER 识别（如果有 router 且没有提供实体）
        if self.ner_router and not rag_query.entities:
            rag_query.entities = self._do_ner(rag_query.query_text)

        # Step 2: 实体链接（如果有 linker）
        if self.entity_linker and rag_query.entities:
            rag_query.entities = self._do_entity_linking(rag_query.entities)

        # Step 3: 检索知识
        sources = []
        graph_node_count = 0
        graph_edge_count = 0
        vector_doc_count = 0

        # 图谱检索
        if rag_query.use_graph and self.graph_retriever:
            graph_sources = self._retrieve_from_graph(rag_query)
            sources.extend(graph_sources)
            for s in graph_sources:
                graph_node_count += len(s.node_ids)
                graph_edge_count += s.edge_count

        # 向量检索
        if rag_query.use_vector and self.vector_store:
            vector_sources = self._retrieve_from_vector(rag_query)
            sources.extend(vector_sources)
            vector_doc_count = len(vector_sources)

        # Step 4: 融合
        fused_context = self._fuse_sources(
            sources,
            rag_query.fusion_strategy,
            rag_query.graph_weight,
            rag_query.vector_weight,
            rag_query.max_total_chars
        )

        # 统计来源类型
        source_counts = {
            SourceType.KNOWLEDGE_GRAPH.value: 0,
            SourceType.VECTOR_STORE.value: 0
        }
        for s in sources:
            source_counts[s.source_type.value] = source_counts.get(s.source_type.value, 0) + 1

        latency = (time.time() - start_time) * 1000

        return RAGResult(
            query=rag_query,
            sources=sources,
            fused_context=fused_context,
            source_type_counts=source_counts,
            graph_node_count=graph_node_count,
            graph_edge_count=graph_edge_count,
            vector_doc_count=vector_doc_count,
            latency_ms=latency
        )

    def query_simple(self,
                     query_text: str,
                     use_graph: bool = True,
                     use_vector: bool = True,
                     **kwargs) -> RAGResult:
        """
        简化版查询接口

        Args:
            query_text: 用户问题
            use_graph: 是否使用图谱检索
            use_vector: 是否使用向量检索
            **kwargs: 其他参数传递给 RAGQuery

        Returns:
            RAGResult: RAG 结果
        """
        rag_query = RAGQuery(
            query_text=query_text,
            use_graph=use_graph,
            use_vector=use_vector,
            **kwargs
        )
        return self.query(rag_query)

    def _do_ner(self, text: str) -> List[Dict]:
        """执行 NER 识别"""
        try:
            ner_results = self.ner_router.extract(text, return_dict=False)
            entities = []
            for r in ner_results:
                entity = {
                    "text": r.text,
                    "type": r.type,
                    "confidence": getattr(r, 'confidence', 1.0),
                }
                if hasattr(r, 'kg_id') and r.kg_id:
                    entity["kg_id"] = r.kg_id
                entities.append(entity)
            return entities
        except Exception as e:
            logger.warning(f"⚠️  NER 识别失败: {e}")
            return []

    def _do_entity_linking(self, entities: List[Dict]) -> List[Dict]:
        """执行实体链接"""
        try:
            from src.retrieval.entity_linking import link_ner_results
            return link_ner_results(entities, linker=self.entity_linker)
        except Exception as e:
            logger.warning(f"⚠️  实体链接失败: {e}")
            return entities

    def _retrieve_from_graph(self, rag_query: RAGQuery) -> List[RAGSource]:
        """从知识图谱检索"""
        sources = []
        try:
            linked_entities = [
                e for e in rag_query.entities
                if e.get("kg_id") or e.get("id")
            ]

            if not linked_entities:
                return sources

            subgraph = self.graph_retriever.retrieve_by_entities(
                linked_entities,
                max_hops=rag_query.graph_max_hops
            )

            if subgraph.nodes or subgraph.edges:
                # 转换为文本上下文
                context = self.graph_retriever.to_text_context(subgraph)
                sources.append(RAGSource(
                    source_type=SourceType.KNOWLEDGE_GRAPH,
                    content=context,
                    score=1.0,
                    node_ids=[n.id for n in subgraph.nodes],
                    edge_count=len(subgraph.edges)
                ))

        except Exception as e:
            logger.warning(f"⚠️  图谱检索失败: {e}")

        return sources

    def _retrieve_from_vector(self, rag_query: RAGQuery) -> List[RAGSource]:
        """从向量存储检索"""
        sources = []
        try:
            from src.retrieval.vector_store import SearchQuery, SearchMode

            # 构建向量查询
            query = SearchQuery(
                query_text=rag_query.query_text,
                query_embedding=rag_query.query_embedding,
                filter_kg_ids=[e.get("kg_id") for e in rag_query.entities if e.get("kg_id")],
                filter_entity_types=[e.get("type") for e in rag_query.entities if e.get("type")],
                top_k=rag_query.vector_top_k,
                score_threshold=rag_query.vector_score_threshold,
                search_mode=SearchMode.VECTOR_ONLY
            )

            response = self.vector_store.search(query)

            for r in response.results:
                sources.append(RAGSource(
                    source_type=SourceType.VECTOR_STORE,
                    content=r.chunk.text,
                    score=r.score,
                    chunk_id=r.chunk.chunk_id,
                    entity_type=r.chunk.entity_type
                ))

        except Exception as e:
            logger.warning(f"⚠️  向量检索失败: {e}")

        return sources

    def _fuse_sources(self,
                      sources: List[RAGSource],
                      strategy: FusionStrategy,
                      graph_weight: float,
                      vector_weight: float,
                      max_chars: int) -> str:
        """
        融合多个知识来源

        策略：
        - CONCATENATION: 简单按类型拼接
        - WEIGHTED: 按权重重新排序
        - RECIPROCAL_RANK: 倒数排序融合
        """
        if not sources:
            return "【背景知识】暂无相关知识。"

        # 按策略排序
        if strategy == FusionStrategy.WEIGHTED:
            sources = self._weighted_sort(sources, graph_weight, vector_weight)
        elif strategy == FusionStrategy.RECIPROCAL_RANK:
            sources = self._rrf_sort(sources)
        elif strategy == FusionStrategy.RERANK:
            sources = self._rerank(sources)

        # 拼接并截断
        parts = ["【背景知识】"]

        # 分组显示
        graph_sources = [s for s in sources if s.source_type == SourceType.KNOWLEDGE_GRAPH]
        vector_sources = [s for s in sources if s.source_type == SourceType.VECTOR_STORE]

        if graph_sources:
            parts.append("\n【知识图谱】")
            for s in graph_sources:
                parts.append(s.content)

        if vector_sources:
            parts.append("\n【文档资料】")
            for i, s in enumerate(vector_sources, 1):
                parts.append(f"\n[{i}] (相似度: {s.score:.3f})")
                parts.append(s.content)

        # 合并并截断
        full_text = "\n".join(parts)

        if len(full_text) <= max_chars:
            return full_text

        # 智能截断：保留开头，然后从后往前删
        # 优先保留图谱内容，然后是高相似度的文档
        return full_text[:max_chars] + "\n...（内容已截断）"

    def _weighted_sort(self,
                       sources: List[RAGSource],
                       graph_weight: float,
                       vector_weight: float) -> List[RAGSource]:
        """加权排序"""
        for s in sources:
            if s.source_type == SourceType.KNOWLEDGE_GRAPH:
                s.score *= graph_weight
            else:
                s.score *= vector_weight
        return sorted(sources, key=lambda x: -x.score)

    def _rrf_sort(self, sources: List[RAGSource], k: int = 60) -> List[RAGSource]:
        """
        倒数排序融合 (Reciprocal Rank Fusion)

        RRF 公式: score = sum(1 / (k + rank))
        """
        # 先分组排序
        graph_sources = sorted(
            [s for s in sources if s.source_type == SourceType.KNOWLEDGE_GRAPH],
            key=lambda x: -x.score
        )
        vector_sources = sorted(
            [s for s in sources if s.source_type == SourceType.VECTOR_STORE],
            key=lambda x: -x.score
        )

        # 计算 RRF 分数
        rrf_scores = {}

        for rank, s in enumerate(graph_sources, 1):
            rrf_scores[s] = rrf_scores.get(s, 0.0) + 1.0 / (k + rank)

        for rank, s in enumerate(vector_sources, 1):
            rrf_scores[s] = rrf_scores.get(s, 0.0) + 1.0 / (k + rank)

        # 排序
        sorted_sources = sorted(sources, key=lambda x: -rrf_scores.get(x, 0.0))
        for s in sorted_sources:
            s.score = rrf_scores.get(s, s.score)

        return sorted_sources

    def _rerank(self, sources: List[RAGSource]) -> List[RAGSource]:
        """
        重排序（占位实现）

        可以使用 cross-encoder 等模型进行重排序
        """
        # 目前简单按分数排序
        return sorted(sources, key=lambda x: -x.score)
