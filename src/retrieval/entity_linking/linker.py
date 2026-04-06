# linker.py
import os
import logging
from typing import List, Dict, Any, Optional

from .base import BaseLinker, MatchResult, MatchStage
from .fuzzy_matcher import FuzzyMatcher

logger = logging.getLogger("EntityLinker")


class EntityLinker:
    """
    实体链接器 - 主入口

    整合多种匹配方法：
    1. 精确匹配
    2. 别名匹配
    3. 编辑距离匹配
    4. 向量检索（可选）
    """

    def __init__(self,
                 entity_dir: Optional[str] = None,
                 use_fuzzy: bool = True,
                 use_vector: bool = False,
                 enable_cache: bool = True):
        """
        Args:
            entity_dir: 实体词典目录
            use_fuzzy: 是否使用编辑距离匹配
            use_vector: 是否使用向量检索（需要额外依赖）
            enable_cache: 是否启用缓存
        """
        self.entity_dir = entity_dir
        self.use_fuzzy = use_fuzzy
        self.use_vector = use_vector
        self.enable_cache = enable_cache

        # 匹配器
        self._fuzzy_matcher: Optional[FuzzyMatcher] = None
        self._vector_indexer = None  # 延迟初始化

        # 缓存
        self._cache: Dict[Tuple[str, Optional[str]], List[MatchResult]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._initialized = False

    def initialize(self):
        """初始化"""
        if self._initialized:
            return

        logger.info("🔧 初始化实体链接器...")

        # 初始化模糊匹配器
        if self.use_fuzzy:
            self._fuzzy_matcher = FuzzyMatcher(entity_dir=self.entity_dir)
            self._fuzzy_matcher.initialize()

        # 向量索引器延迟初始化

        self._initialized = True
        logger.info("✅ 实体链接器初始化完成")

    def link(self,
             entity_text: str,
             entity_type: Optional[str] = None,
             return_best_only: bool = True) -> Optional[MatchResult]:
        """
        链接实体到词典

        Args:
            entity_text: 实体文本
            entity_type: 实体类型（如果提供，只在该类型中搜索）
            return_best_only: 是否只返回最佳匹配

        Returns:
            MatchResult（单个或列表）
        """
        if not self._initialized:
            self.initialize()

        if not entity_text or not entity_text.strip():
            return None

        entity_text = entity_text.strip()

        # 检查缓存
        cache_key = (entity_text, entity_type)
        if self.enable_cache and cache_key in self._cache:
            self._cache_hits += 1
            results = self._cache[cache_key]
            if return_best_only and results:
                return results[0]
            return results if not return_best_only else None

        self._cache_misses += 1

        # 阶段 1: 模糊匹配（精确 + 别名 + 编辑距离）
        all_results: List[MatchResult] = []
        if self._fuzzy_matcher:
            fuzzy_results = self._fuzzy_matcher.match(
                entity_text,
                entity_type=entity_type,
                return_best_only=False
            )
            all_results.extend(fuzzy_results)

            # 如果有精确/别名匹配，直接返回
            for r in fuzzy_results:
                if r.stage in (MatchStage.EXACT, MatchStage.ALIAS):
                    if self.enable_cache:
                        self._cache[cache_key] = fuzzy_results
                    return r if return_best_only else fuzzy_results

        # 阶段 2: 向量检索（可选）
        if self.use_vector and not all_results:
            vector_results = self._try_vector_match(entity_text, entity_type)
            all_results.extend(vector_results)

        # 排序
        if all_results:
            all_results = self._sort_results(all_results)

            # 缓存
            if self.enable_cache:
                self._cache[cache_key] = all_results

            if return_best_only:
                return all_results[0]
            return all_results

        # 没有匹配结果
        if self.enable_cache:
            self._cache[cache_key] = []
        return None

    def _try_vector_match(self, entity_text: str, entity_type: Optional[str]) -> List[MatchResult]:
        """尝试向量匹配"""
        if not self.use_vector:
            return []

        # 延迟初始化向量索引器
        if self._vector_indexer is None:
            try:
                from .vector_indexer import VectorIndexer
                self._vector_indexer = VectorIndexer(entity_dir=self.entity_dir)
                self._vector_indexer.initialize(build_index=True)
            except Exception as e:
                logger.warning(f"⚠️  向量索引器初始化失败: {e}")
                self.use_vector = False
                return []

        return self._vector_indexer.match(
            entity_text,
            entity_type=entity_type,
            return_best_only=False
        )

    def _sort_results(self, results: List[MatchResult]) -> List[MatchResult]:
        """排序匹配结果"""
        stage_order = {
            MatchStage.EXACT: 0,
            MatchStage.ALIAS: 1,
            MatchStage.VECTOR: 2,
            MatchStage.EDIT_DISTANCE: 3,
            MatchStage.NONE: 4,
        }

        return sorted(
            results,
            key=lambda x: (
                stage_order.get(x.stage, 99),
                -x.similarity,
                -len(x.entity_name)
            )
        )

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# ================= 便捷函数 =================
def create_entity_linker(**kwargs) -> EntityLinker:
    """创建实体链接器"""
    linker = EntityLinker(**kwargs)
    linker.initialize()
    return linker


def link_ner_results(ner_results: List[Any], linker: Optional[EntityLinker] = None) -> List[Any]:
    """
    链接 NER 结果（自动跳过 Layer 1 的结果）

    Args:
        ner_results: NER 识别结果列表
        linker: 实体链接器（如果不提供则创建一个新的）

    Returns:
        链接后的结果列表
    """
    if linker is None:
        linker = create_entity_linker()

    linked_results = []
    for entity in ner_results:
        # 检查是否是 Layer 1 的结果（已有 kg_id）
        if hasattr(entity, 'source') and entity.source == 'keyword_match':
            linked_results.append(entity)
            continue

        # 提取属性（兼容字典和对象）
        if isinstance(entity, dict):
            text = entity.get('text')
            etype = entity.get('type')
        else:
            text = getattr(entity, 'text', None)
            etype = getattr(entity, 'type', None)

        if not text:
            linked_results.append(entity)
            continue

        # 链接
        match_result = linker.link(text, entity_type=etype)
        if match_result:
            # 更新 kg_id
            if isinstance(entity, dict):
                entity['kg_id'] = match_result.kg_id
                entity['linked_name'] = match_result.entity_name
                entity['link_stage'] = match_result.stage.value
                entity['link_confidence'] = match_result.calibrated_confidence
            else:
                setattr(entity, 'kg_id', match_result.kg_id)
                setattr(entity, 'linked_name', match_result.entity_name)
                setattr(entity, 'link_stage', match_result.stage.value)
                setattr(entity, 'link_confidence', match_result.calibrated_confidence)

        linked_results.append(entity)

    return linked_results
