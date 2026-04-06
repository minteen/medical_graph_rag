# qdrant_store.py
# Qdrant 向量数据库实现
import os
import logging
import time
from typing import List, Dict, Any, Optional

from .base import (
    DocumentChunk,
    SearchResult,
    SearchQuery,
    SearchResponse,
    SearchMode
)

logger = logging.getLogger("QdrantVectorStore")


class QdrantVectorStore:
    """
    Qdrant 向量存储

    特点：
    - 高性能向量检索
    - 支持丰富的过滤条件
    - 支持 Payload 索引
    - 支持批量插入
    """

    def __init__(self,
                 url: str = "http://localhost:6333",
                 api_key: Optional[str] = None,
                 collection_name: str = "medical_kg_chunks",
                 embedding_model: Optional[Any] = None,
                 vector_dim: int = 1024):
        """
        Args:
            url: Qdrant 服务地址
            api_key: API Key
            collection_name: 集合名称
            embedding_model: Embedding 模型
            vector_dim: 向量维度
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim

        self._client = None
        self._qdrant = None

    def _lazy_import(self):
        """延迟导入 qdrant_client"""
        if self._qdrant is None:
            try:
                import qdrant_client
                from qdrant_client.models import (
                    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
                )
                self._qdrant = qdrant_client
                self._Distance = Distance
                self._VectorParams = VectorParams
                self._PointStruct = PointStruct
                self._Filter = Filter
                self._FieldCondition = FieldCondition
                self._MatchAny = MatchAny
            except ImportError:
                raise ImportError("qdrant_client 未安装，请运行: pip install qdrant-client")

    @property
    def client(self):
        if self._client is None:
            self._lazy_import()
            self._client = self._qdrant.QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def initialize(self, create_if_not_exists: bool = True):
        """初始化"""
        self._lazy_import()

        if create_if_not_exists:
            self.ensure_collection()

    def ensure_collection(self, recreate: bool = False):
        """确保集合存在"""
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️ 已删除旧集合: {self.collection_name}")

        if not self.client.collection_exists(self.collection_name):
            logger.info(f"📦 创建集合: {self.collection_name} | 维度: {self.vector_dim}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._VectorParams(
                    size=self.vector_dim,
                    distance=self._Distance.COSINE
                )
            )
            # 创建索引
            self._create_payload_indexes()
        else:
            logger.info(f"✅ 集合已存在: {self.collection_name}")

    def _create_payload_indexes(self):
        """创建 Payload 索引"""
        index_fields = [
            "kg_id", "entity_type", "section_type",
            "involved_relations", "safety_flags"
        ]
        for field in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword"
                )
            except Exception:
                pass

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """添加文档块"""
        if not chunks:
            return

        # 生成 embedding（如果需要）
        need_embed = any(c.embedding is None for c in chunks)
        if need_embed:
            if self.embedding_model is None:
                raise ValueError("部分文档块缺少 embedding，且未提供 embedding_model")
            self._generate_embeddings(chunks)

        # 批量插入
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []
            for chunk in batch:
                points.append(self._PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload=chunk.to_payload()
                ))
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"📤 已插入 {len(points)} 条")

    def _generate_embeddings(self, chunks: List[DocumentChunk]):
        """生成 embeddings"""
        texts = [c.text for c in chunks if c.embedding is None]
        if not texts:
            return

        logger.info(f"📦 生成 {len(texts)} 个 embeddings...")
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)

        emb_idx = 0
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = embeddings[emb_idx].tolist() if hasattr(embeddings[emb_idx], 'tolist') else list(embeddings[emb_idx])
                emb_idx += 1

    def search(self, query: SearchQuery) -> SearchResponse:
        """检索"""
        start_time = time.time()

        # 获取查询向量
        query_emb = self._get_query_embedding(query)

        # 构建过滤器
        query_filter = self._build_filter(query)

        # 执行搜索
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_emb,
            query_filter=query_filter,
            limit=query.top_k,
            score_threshold=query.score_threshold
        )

        # 转换结果
        results = []
        for i, hit in enumerate(search_result):
            chunk = DocumentChunk.from_payload(hit.payload)
            results.append(SearchResult(
                chunk=chunk,
                score=hit.score,
                rank=i + 1,
                search_mode=SearchMode.VECTOR_ONLY
            ))

        latency = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            latency_ms=latency
        )

    def _get_query_embedding(self, query: SearchQuery) -> List[float]:
        """获取查询向量"""
        if query.query_embedding is not None:
            return query.query_embedding

        if self.embedding_model is None:
            raise ValueError("query_embedding 为 None，且未提供 embedding_model")

        emb = self.embedding_model.encode([query.query_text], normalize_embeddings=True)[0]
        return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

    def _build_filter(self, query: SearchQuery) -> Optional[Any]:
        """构建 Qdrant 过滤器"""
        conditions = []

        if query.filter_kg_ids:
            conditions.append(self._FieldCondition(
                key="kg_id",
                match=self._MatchAny(any=query.filter_kg_ids)
            ))

        if query.filter_entity_types:
            conditions.append(self._FieldCondition(
                key="entity_type",
                match=self._MatchAny(any=query.filter_entity_types)
            ))

        if query.filter_section_types:
            conditions.append(self._FieldCondition(
                key="section_type",
                match=self._MatchAny(any=query.filter_section_types)
            ))

        if conditions:
            return self._Filter(must=conditions)
        return None

    def delete_collection(self):
        """删除集合"""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️ 已删除集合: {self.collection_name}")
