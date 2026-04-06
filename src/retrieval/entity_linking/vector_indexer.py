# vector_indexer.py
import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .base import BaseLinker, MatchResult, MatchStage, EntityInfo

logger = logging.getLogger("VectorIndexer")


class VectorIndexer(BaseLinker):
    """
    基于向量检索的实体链接器

    使用语义相似度进行匹配，作为最后的兜底方案

    注意: 这是一个可选组件，需要额外的依赖:
        - sentence-transformers (文本编码)
        - faiss 或 sklearn (向量索引)
    """

    def __init__(self,
                 entity_dir: Optional[str] = None,
                 model_name: Optional[str] = None,
                 use_faiss: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Args:
            entity_dir: 实体词典目录
            model_name: Sentence-BERT 模型名称
            use_faiss: 是否使用 Faiss 进行快速向量检索
            cache_dir: 向量索引缓存目录
        """
        super().__init__(entity_dir)
        self.model_name = model_name or "shibing624/text2vec-base-chinese"
        self.use_faiss = use_faiss
        self.cache_dir = cache_dir or self._get_default_cache_dir()

        # 模型和索引
        self._model = None
        self._indices: Dict[str, Any] = {}  # {type: index}
        self._entity_lists: Dict[str, List[str]] = {}  # {type: [entity_names]}

        # 检查依赖
        self._has_deps = self._check_dependencies()

    def _get_default_cache_dir(self) -> str:
        """获取默认缓存目录"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            ".cache", "entity_vectors"
        )

    def _check_dependencies(self) -> bool:
        """检查依赖是否安装"""
        has_sentence_transformers = False
        try:
            import sentence_transformers
            has_sentence_transformers = True
        except ImportError:
            logger.warning("⚠️  sentence-transformers 未安装，向量检索不可用")

        has_faiss = False
        if self.use_faiss:
            try:
                import faiss
                has_faiss = True
            except ImportError:
                logger.warning("⚠️  faiss 未安装，将使用 sklearn 替代")
                self.use_faiss = False

        return has_sentence_transformers

    def initialize(self, build_index: bool = False):
        """
        初始化

        Args:
            build_index: 是否立即构建向量索引
        """
        super().initialize()

        if not self._has_deps:
            return

        # 加载模型
        if self._model is None:
            self._load_model()

        # 构建索引
        if build_index:
            self._build_all_indices()

    def _load_model(self):
        """加载 Sentence-BERT 模型"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"🔧 加载向量模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("✅ 向量模型加载完成")
        except Exception as e:
            logger.error(f"❌ 向量模型加载失败: {e}")
            self._has_deps = False

    def _build_all_indices(self):
        """为所有实体类型构建向量索引"""
        os.makedirs(self.cache_dir, exist_ok=True)

        for entity_type, entities in self.entity_dict.items():
            self._build_index_for_type(entity_type, entities)

    def _build_index_for_type(self, entity_type: str, entities: Dict[str, EntityInfo]):
        """为单个实体类型构建向量索引"""
        if not self._has_deps or not self._model:
            return

        logger.info(f"🔧 构建 {entity_type} 向量索引 ({len(entities)} 个实体)...")

        # 检查缓存
        cache_path = os.path.join(self.cache_dir, f"{entity_type}.pkl")
        if os.path.exists(cache_path):
            try:
                self._load_index_from_cache(entity_type, cache_path)
                logger.info(f"✅ {entity_type} 向量索引已从缓存加载")
                return
            except Exception as e:
                logger.warning(f"⚠️  加载缓存失败: {e}, 将重新构建")

        # 编码实体
        entity_names = list(entities.keys())
        embeddings = self._model.encode(entity_names, batch_size=32, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)

        # 归一化（用于余弦相似度）
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        # 构建索引
        if self.use_faiss:
            index = self._build_faiss_index(embeddings)
        else:
            index = self._build_sklearn_index(embeddings)

        # 保存
        self._indices[entity_type] = index
        self._entity_lists[entity_type] = entity_names

        # 缓存
        self._save_index_to_cache(entity_type, cache_path, index, entity_names)

        logger.info(f"✅ {entity_type} 向量索引构建完成")

    def _build_faiss_index(self, embeddings: np.ndarray):
        """使用 Faiss 构建索引"""
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product = 余弦相似度（归一化后）
        index.add(embeddings)
        return index

    def _build_sklearn_index(self, embeddings: np.ndarray):
        """使用 sklearn 构建索引（回退方案）"""
        from sklearn.neighbors import NearestNeighbors
        index = NearestNeighbors(n_neighbors=10, metric='cosine')
        index.fit(embeddings)
        return index

    def _save_index_to_cache(self, entity_type: str, cache_path: str, index, entity_names: List[str]):
        """保存索引到缓存"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'index': index,
                    'entity_names': entity_names,
                    'use_faiss': self.use_faiss
                }, f)
        except Exception as e:
            logger.warning(f"⚠️  缓存索引失败: {e}")

    def _load_index_from_cache(self, entity_type: str, cache_path: str):
        """从缓存加载索引"""
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            self._indices[entity_type] = data['index']
            self._entity_lists[entity_type] = data['entity_names']

    def match(self,
              entity_text: str,
              entity_type: Optional[str] = None,
              return_best_only: bool = True,
              top_k: int = 3,
              similarity_threshold: float = 0.7) -> List[MatchResult]:
        """
        使用向量检索匹配

        Args:
            entity_text: 待匹配的实体文本
            entity_type: 实体类型限制
            return_best_only: 是否只返回最佳匹配
            top_k: 返回 top K
            similarity_threshold: 最小相似度阈值

        Returns:
            匹配结果列表
        """
        if not self._has_deps or not self._model:
            return []

        if not entity_text or not entity_text.strip():
            return []

        entity_text = entity_text.strip()

        # 确定搜索类型
        search_types = [entity_type] if entity_type else list(self.entity_dict.keys())

        all_results: List[MatchResult] = []

        # 编码查询
        query_embedding = self._model.encode([entity_text])[0]
        query_embedding = query_embedding.astype(np.float32)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.reshape(1, -1)

        for etype in search_types:
            if etype not in self._indices or etype not in self._entity_lists:
                # 尝试构建索引
                if etype in self.entity_dict:
                    self._build_index_for_type(etype, self.entity_dict[etype])
                continue

            # 搜索
            if self.use_faiss:
                results = self._search_faiss(etype, query_embedding, top_k, similarity_threshold)
            else:
                results = self._search_sklearn(etype, query_embedding, top_k, similarity_threshold)

            all_results.extend(results)

        # 排序
        all_results.sort(key=lambda x: -x.similarity)

        # 校准置信度
        for result in all_results:
            result.calibrated_confidence = self.calibrate_confidence(result)

        if return_best_only and all_results:
            return [all_results[0]]

        return all_results[:top_k]

    def _search_faiss(self, entity_type: str, query_embedding: np.ndarray,
                     top_k: int, threshold: float) -> List[MatchResult]:
        """使用 Faiss 搜索"""
        index = self._indices[entity_type]
        entity_names = self._entity_lists[entity_type]
        entities = self.entity_dict[entity_type]

        scores, indices = index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            score = scores[0][i]
            if score >= threshold:
                name = entity_names[idx]
                info = entities[name]
                results.append(MatchResult(
                    entity_name=name,
                    entity_type=entity_type,
                    kg_id=info.kg_id,
                    distance=0,
                    similarity=float(score),
                    stage=MatchStage.VECTOR
                ))

        return results

    def _search_sklearn(self, entity_type: str, query_embedding: np.ndarray,
                       top_k: int, threshold: float) -> List[MatchResult]:
        """使用 sklearn 搜索（回退方案）"""
        index = self._indices[entity_type]
        entity_names = self._entity_lists[entity_type]
        entities = self.entity_dict[entity_type]

        distances, indices = index.kneighbors(query_embedding, n_neighbors=top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            # 余弦距离转相似度
            cosine_distance = distances[0][i]
            similarity = 1.0 - cosine_distance
            if similarity >= threshold:
                name = entity_names[idx]
                info = entities[name]
                results.append(MatchResult(
                    entity_name=name,
                    entity_type=entity_type,
                    kg_id=info.kg_id,
                    distance=0,
                    similarity=float(similarity),
                    stage=MatchStage.VECTOR
                ))

        return results
