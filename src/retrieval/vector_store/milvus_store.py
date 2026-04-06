# milvus_store.py
# Milvus 向量数据库实现
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

logger = logging.getLogger("MilvusVectorStore")


class MilvusVectorStore:
    """
    Milvus 向量存储

    特点：
    - 高性能向量检索
    - 支持丰富的过滤条件
    - 支持标量/向量索引
    - 支持批量插入
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 token: Optional[str] = None,
                 collection_name: str = "medical_kg_chunks",
                 embedding_model: Optional[Any] = None,
                 vector_dim: int = 1024):
        """
        Args:
            host: Milvus 主机
            port: Milvus 端口
            token: Milvus token
            collection_name: 集合名称
            embedding_model: Embedding 模型
            vector_dim: 向量维度
        """
        self.host = host
        self.port = port
        self.token = token
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim

        self._collection = None
        self._connected = False
        self._pymilvus = None

    def _lazy_import(self):
        """延迟导入 pymilvus"""
        if self._pymilvus is None:
            try:
                import pymilvus
                from pymilvus import (
                    connections, Collection, utility,
                    CollectionSchema, FieldSchema, DataType
                )
                self._pymilvus = pymilvus
                self._connections = connections
                self._Collection = Collection
                self._utility = utility
                self._CollectionSchema = CollectionSchema
                self._FieldSchema = FieldSchema
                self._DataType = DataType
            except ImportError:
                raise ImportError("pymilvus 未安装，请运行: pip install pymilvus")

    def connect(self):
        """连接 Milvus"""
        self._lazy_import()
        if not self._connected:
            logger.info(f"🔌 连接 Milvus: {self.host}:{self.port}")
            self._connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                token=self.token
            )
            self._connected = True

    def disconnect(self):
        """断开 Milvus 连接"""
        if self._connected:
            self._connections.disconnect("default")
            self._connected = False
            self._collection = None
            logger.info("🔌 已断开 Milvus 连接")

    def initialize(self, create_if_not_exists: bool = True):
        """初始化"""
        self.connect()

        if create_if_not_exists:
            self.ensure_collection()

    @property
    def collection(self):
        """获取 Collection 对象"""
        if self._collection is None:
            self._collection = self._Collection(self.collection_name)
            self._collection.load()  # 加载到内存
        return self._collection

    def ensure_collection(self, recreate: bool = False):
        """确保集合存在"""
        if recreate and self._utility.has_collection(self.collection_name):
            self._utility.drop_collection(self.collection_name)
            logger.info(f"🗑️ 已删除旧集合: {self.collection_name}")

        if not self._utility.has_collection(self.collection_name):
            logger.warning(f"⚠️  集合不存在: {self.collection_name}")
            logger.info("请使用 milvus_writer.py 先创建集合并导入数据")
        else:
            logger.info(f"✅ 集合已存在: {self.collection_name}")
            self._collection = self._Collection(self.collection_name)
            self._collection.load()

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 500):
        """添加文档块（需要集合已存在）"""
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
            self._upsert_batch(batch)
            logger.info(f"📤 已插入 {len(batch)} 条")

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

    def _upsert_batch(self, chunks: List[DocumentChunk]):
        """批量插入"""
        # 转换为列格式
        column_data = {
            "chunk_id": [],
            "embedding": [],
            "kg_id": [],
            "entity_type": [],
            "section_type": [],
            "involved_relations": [],
            "safety_flags": [],
            "cured_prob": [],
            "yibao_status": [],
            "category_path": [],
            "version": [],
            "chunk_text": [],
            "metadata_json": []
        }

        for chunk in chunks:
            column_data["chunk_id"].append(chunk.chunk_id)
            column_data["embedding"].append(chunk.embedding)
            column_data["kg_id"].append(chunk.kg_id)
            column_data["entity_type"].append(chunk.entity_type)
            column_data["section_type"].append(chunk.section_type)
            column_data["involved_relations"].append(chunk.involved_relations)
            column_data["safety_flags"].append(chunk.safety_flags)
            column_data["cured_prob"].append(chunk.metadata.get("cured_prob"))
            column_data["yibao_status"].append(chunk.metadata.get("yibao_status"))
            column_data["category_path"].append(chunk.metadata.get("category_path"))
            column_data["version"].append(chunk.metadata.get("version", "v1.0"))
            column_data["chunk_text"].append(chunk.text)
            column_data["metadata_json"].append(chunk.metadata)

        # 按 schema 字段顺序排列
        ordered_fields = [field.name for field in self.collection.schema.fields]
        ordered_data = []
        for field_name in ordered_fields:
            if field_name in column_data:
                ordered_data.append(column_data[field_name])

        self.collection.insert(ordered_data)

    def search(self, query: SearchQuery) -> SearchResponse:
        """检索"""
        start_time = time.time()

        # 获取查询向量
        query_emb = self._get_query_embedding(query)

        # 构建过滤表达式
        filter_expr = self._build_filter(query)

        # 执行搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }

        results = self.collection.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=query.top_k,
            expr=filter_expr,
            output_fields=["chunk_id", "kg_id", "entity_type", "section_type", "chunk_text", "metadata_json"]
        )

        # 转换结果
        search_results = []
        if results and len(results) > 0:
            for i, hit in enumerate(results[0]):
                if hit.score < query.score_threshold:
                    continue

                # 构建 DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=hit.entity.get("chunk_id", str(hit.id)),
                    text=hit.entity.get("chunk_text", ""),
                    kg_id=hit.entity.get("kg_id"),
                    entity_type=hit.entity.get("entity_type"),
                    section_type=hit.entity.get("section_type"),
                    metadata=hit.entity.get("metadata_json", {})
                )

                search_results.append(SearchResult(
                    chunk=chunk,
                    score=hit.score,
                    rank=i + 1,
                    search_mode=SearchMode.VECTOR_ONLY
                ))

        latency = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            results=search_results,
            total_count=len(search_results),
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

    def _build_filter(self, query: SearchQuery) -> Optional[str]:
        """构建 Milvus 过滤表达式"""
        conditions = []

        if query.filter_kg_ids:
            ids_str = ', '.join(f"'{id}'" for id in query.filter_kg_ids)
            conditions.append(f"kg_id in [{ids_str}]")

        if query.filter_entity_types:
            types_str = ', '.join(f"'{t}'" for t in query.filter_entity_types)
            conditions.append(f"entity_type in [{types_str}]")

        if query.filter_section_types:
            sections_str = ', '.join(f"'{s}'" for s in query.filter_section_types)
            conditions.append(f"section_type in [{sections_str}]")

        if conditions:
            return " and ".join(conditions)
        return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
