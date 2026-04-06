# numpy_store.py
# 纯 NumPy 实现的向量存储（无额外依赖，适用于中小规模数据）
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import (
    DocumentChunk,
    SearchResult,
    SearchQuery,
    SearchResponse,
    SearchMode
)

logger = logging.getLogger("NumpyVectorStore")


class NumpyVectorStore:
    """
    纯 NumPy 向量存储

    特点：
    - 无额外依赖（仅需 numpy）
    - 适合中小规模数据（<10万条）
    - 支持持久化到磁盘
    - 支持关键词过滤 + 向量检索
    """

    def __init__(self,
                 index_path: Optional[str] = None,
                 embedding_model: Optional[Any] = None,
                 normalize: bool = True):
        """
        Args:
            index_path: 索引保存路径（JSONL 格式）
            embedding_model: Embedding 模型（需要有 encode 方法）
            normalize: 是否归一化向量
        """
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.normalize = normalize

        # 索引数据
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Any = None  # numpy array
        self._np = None

        # 关键词索引（简单倒排）
        self._keyword_index: Dict[str, List[int]] = {}

        self._loaded = False

    def _lazy_import(self):
        """延迟导入 numpy"""
        if self._np is None:
            try:
                import numpy as np
                self._np = np
            except ImportError:
                raise ImportError("numpy 未安装，请运行: pip install numpy")

    def initialize(self, load_from_disk: bool = True):
        """初始化索引"""
        self._lazy_import()

        if load_from_disk and self.index_path and os.path.exists(self.index_path):
            self.load()
        else:
            self.chunks = []
            self.embeddings = None
            self._keyword_index = {}

        self._loaded = True

    def add_chunks(self, chunks: List[DocumentChunk]):
        """添加文档块"""
        self._lazy_import()

        if not chunks:
            return

        # 检查是否有 embedding
        need_embed = any(c.embedding is None for c in chunks)
        if need_embed:
            if self.embedding_model is None:
                raise ValueError("部分文档块缺少 embedding，且未提供 embedding_model")
            self._generate_embeddings(chunks)

        # 添加到索引
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)

        # 构建向量矩阵
        new_embeddings = self._np.array([c.embedding for c in chunks], dtype=self._np.float32)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = self._np.vstack([self.embeddings, new_embeddings])

        # 构建关键词索引
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            words = self._tokenize(chunk.text)
            for word in words:
                if word not in self._keyword_index:
                    self._keyword_index[word] = []
                self._keyword_index[word].append(idx)

    def _generate_embeddings(self, chunks: List[DocumentChunk]):
        """为文档块生成 embedding"""
        texts = [c.text for c in chunks if c.embedding is None]
        if not texts:
            return

        logger.info(f"📦 生成 {len(texts)} 个 embeddings...")
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=self.normalize)

        # 填充回 chunks
        emb_idx = 0
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = embeddings[emb_idx].tolist() if hasattr(embeddings[emb_idx], 'tolist') else list(embeddings[emb_idx])
                emb_idx += 1

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（按空格和标点）"""
        import re
        # 中文按字，英文按词
        words = []
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text)
        words.extend([w.lower() for w in english_words])
        # 提取中文单字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        words.extend(chinese_chars)
        return words

    def search(self, query: SearchQuery) -> SearchResponse:
        """检索"""
        self._lazy_import()
        start_time = time.time()

        if not self._loaded:
            self.initialize()

        # 1. 获取查询向量
        query_emb = self._get_query_embedding(query)

        # 2. 过滤候选集
        candidate_indices = self._filter_candidates(query)

        if not candidate_indices:
            latency = (time.time() - start_time) * 1000
            return SearchResponse(
                query=query,
                results=[],
                total_count=0,
                latency_ms=latency
            )

        # 3. 根据检索模式计算分数
        if query.search_mode == SearchMode.VECTOR_ONLY:
            results = self._search_vector(query_emb, candidate_indices, query)
        elif query.search_mode == SearchMode.KEYWORD_ONLY:
            results = self._search_keyword(query, candidate_indices)
        else:  # HYBRID
            results_vec = self._search_vector(query_emb, candidate_indices, query)
            results_kw = self._search_keyword(query, candidate_indices)
            results = self._merge_results(
                results_vec, results_kw,
                query.hybrid_vector_weight,
                query.hybrid_keyword_weight
            )

        # 4. 截断和排序
        results = results[:query.top_k]
        for i, r in enumerate(results):
            r.rank = i + 1

        latency = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            latency_ms=latency
        )

    def _get_query_embedding(self, query: SearchQuery) -> Any:
        """获取查询向量"""
        if query.query_embedding is not None:
            return self._np.array(query.query_embedding, dtype=self._np.float32)

        if self.embedding_model is None:
            raise ValueError("query_embedding 为 None，且未提供 embedding_model")

        emb = self.embedding_model.encode([query.query_text], normalize_embeddings=self.normalize)[0]
        return self._np.array(emb, dtype=self._np.float32)

    def _filter_candidates(self, query: SearchQuery) -> List[int]:
        """根据过滤条件筛选候选"""
        candidates = set(range(len(self.chunks)))

        # 图谱 ID 过滤
        if query.filter_kg_ids:
            kg_set = set(query.filter_kg_ids)
            candidates = {
                i for i in candidates
                if self.chunks[i].kg_id in kg_set
            }

        # 实体类型过滤
        if query.filter_entity_types:
            et_set = set(query.filter_entity_types)
            candidates = {
                i for i in candidates
                if self.chunks[i].entity_type in et_set
            }

        # 章节类型过滤
        if query.filter_section_types:
            st_set = set(query.filter_section_types)
            candidates = {
                i for i in candidates
                if self.chunks[i].section_type in st_set
            }

        # 关键词预过滤（可选加速）
        if query.search_mode != SearchMode.VECTOR_ONLY:
            query_words = self._tokenize(query.query_text)
            if query_words:
                keyword_candidates = set()
                for word in query_words:
                    if word in self._keyword_index:
                        keyword_candidates.update(self._keyword_index[word])
                if keyword_candidates:
                    # 取交集，或者如果没有交集则保留原候选
                    if candidates & keyword_candidates:
                        candidates = candidates & keyword_candidates

        return list(candidates)

    def _search_vector(self, query_emb: Any, candidate_indices: List[int],
                       query: SearchQuery) -> List[SearchResult]:
        """纯向量检索"""
        if not candidate_indices:
            return []

        # 提取候选向量
        candidate_embs = self.embeddings[candidate_indices]

        # 计算余弦相似度（已归一化）
        scores = candidate_embs.dot(query_emb)

        # 组合结果
        results = []
        for idx_in_candidate, idx_in_all in enumerate(candidate_indices):
            score = float(scores[idx_in_candidate])
            if score >= query.score_threshold:
                results.append(SearchResult(
                    chunk=self.chunks[idx_in_all],
                    score=score,
                    search_mode=SearchMode.VECTOR_ONLY
                ))

        # 排序
        results.sort(key=lambda x: -x.score)
        return results

    def _search_keyword(self, query: SearchQuery, candidate_indices: List[int]) -> List[SearchResult]:
        """纯关键词检索（BM25-like 简单实现）"""
        query_words = self._tokenize(query.query_text)
        if not query_words:
            return []

        results = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            chunk_words = set(self._tokenize(chunk.text))

            # 简单的词频匹配
            match_count = sum(1 for w in query_words if w in chunk_words)
            if match_count == 0:
                continue

            # 归一化分数
            score = match_count / max(len(query_words), len(chunk_words))
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                search_mode=SearchMode.KEYWORD_ONLY
            ))

        results.sort(key=lambda x: -x.score)
        return results

    def _merge_results(self, results_vec: List[SearchResult], results_kw: List[SearchResult],
                       vec_weight: float, kw_weight: float) -> List[SearchResult]:
        """合并混合检索结果"""
        # 建立分数映射
        vec_scores = {r.chunk_id: r.score for r in results_vec}
        kw_scores = {r.chunk_id: r.score for r in results_kw}

        # 所有文档 ID
        all_ids = set(vec_scores.keys()) | set(kw_scores.keys())

        results = []
        for chunk_id in all_ids:
            vec_score = vec_scores.get(chunk_id, 0.0)
            kw_score = kw_scores.get(chunk_id, 0.0)
            combined_score = vec_score * vec_weight + kw_score * kw_weight

            # 获取 chunk
            chunk = None
            for r in results_vec:
                if r.chunk_id == chunk_id:
                    chunk = r.chunk
                    break
            if chunk is None:
                for r in results_kw:
                    if r.chunk_id == chunk_id:
                        chunk = r.chunk
                        break

            if chunk:
                results.append(SearchResult(
                    chunk=chunk,
                    score=combined_score,
                    search_mode=SearchMode.HYBRID
                ))

        results.sort(key=lambda x: -x.score)
        return results

    def save(self, path: Optional[str] = None):
        """保存索引到磁盘"""
        save_path = path or self.index_path
        if save_path is None:
            raise ValueError("未指定保存路径")

        self._lazy_import()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        # 保存 chunks 和 embeddings
        data = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "embedding": c.embedding,
                    "kg_id": c.kg_id,
                    "entity_type": c.entity_type,
                    "section_type": c.section_type,
                    "involved_relations": c.involved_relations,
                    "safety_flags": c.safety_flags,
                    "metadata": c.metadata
                }
                for c in self.chunks
            ]
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        # 同时保存 numpy 数组（加速加载）
        if self.embeddings is not None:
            np_path = save_path + ".npy"
            self._np.save(np_path, self.embeddings)

        logger.info(f"💾 索引已保存: {save_path}")

    def load(self, path: Optional[str] = None):
        """从磁盘加载索引"""
        load_path = path or self.index_path
        if load_path is None or not os.path.exists(load_path):
            logger.warning(f"⚠️  索引文件不存在: {load_path}")
            return

        self._lazy_import()

        logger.info(f"📖 加载索引: {load_path}")

        # 先尝试加载 numpy 数组
        np_path = load_path + ".npy"
        if os.path.exists(np_path):
            self.embeddings = self._np.load(np_path)

        # 加载 chunks
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.chunks = []
        for chunk_data in data["chunks"]:
            chunk = DocumentChunk(
                chunk_id=chunk_data["chunk_id"],
                text=chunk_data["text"],
                embedding=chunk_data.get("embedding"),
                kg_id=chunk_data.get("kg_id"),
                entity_type=chunk_data.get("entity_type"),
                section_type=chunk_data.get("section_type"),
                involved_relations=chunk_data.get("involved_relations", []),
                safety_flags=chunk_data.get("safety_flags", []),
                metadata=chunk_data.get("metadata", {})
            )
            self.chunks.append(chunk)

        # 如果没有 numpy 文件，重建
        if self.embeddings is None and self.chunks:
            if self.chunks[0].embedding:
                self.embeddings = self._np.array(
                    [c.embedding for c in self.chunks],
                    dtype=self._np.float32
                )

        # 重建关键词索引
        self._keyword_index = {}
        for i, chunk in enumerate(self.chunks):
            words = self._tokenize(chunk.text)
            for word in words:
                if word not in self._keyword_index:
                    self._keyword_index[word] = []
                self._keyword_index[word].append(i)

        logger.info(f"✅ 索引加载完成: {len(self.chunks)} 个文档")

    @property
    def size(self) -> int:
        return len(self.chunks)
