# fuzzy_matcher.py
import heapq
from typing import List, Optional, Tuple
from .base import BaseLinker, MatchResult, MatchStage, EntityInfo

logger = __import__('logging').getLogger("FuzzyMatcher")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算 Levenshtein 编辑距离

    Args:
        s1: 字符串1
        s2: 字符串2

    Returns:
        编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_max_distance(text_length: int) -> int:
    """
    根据文本长度计算最大允许编辑距离

    Args:
        text_length: 文本长度

    Returns:
        最大允许编辑距离
    """
    if text_length <= 2:
        return 0  # 短词不允许错误
    elif text_length <= 4:
        return 1  # 中等长度允许1个错误
    else:
        return 2  # 长词允许2个错误


class FuzzyMatcher(BaseLinker):
    """
    基于编辑距离的模糊匹配器

    匹配策略:
    1. 精确匹配 (最高优先级)
    2. 别名匹配
    3. 编辑距离匹配 (可选)
    """

    def __init__(self,
                 entity_dir: Optional[str] = None,
                 use_edit_distance: bool = True,
                 max_edit_distance: Optional[int] = None,
                 top_k: int = 3):
        """
        Args:
            entity_dir: 实体词典目录
            use_edit_distance: 是否使用编辑距离匹配
            max_edit_distance: 最大编辑距离 (None 则自动计算)
            top_k: 返回 top K 个匹配结果
        """
        super().__init__(entity_dir)
        self.use_edit_distance = use_edit_distance
        self.max_edit_distance = max_edit_distance
        self.top_k = top_k

    def match(self,
              entity_text: str,
              entity_type: Optional[str] = None,
              return_best_only: bool = True) -> List[MatchResult]:
        """
        匹配实体

        Args:
            entity_text: 待匹配的实体文本
            entity_type: 实体类型限制 (None 则在所有类型中查找)
            return_best_only: 是否只返回最佳匹配

        Returns:
            匹配结果列表
        """
        if not entity_text or not entity_text.strip():
            return []

        entity_text = entity_text.strip()

        # 确定要搜索的类型范围
        search_types = [entity_type] if entity_type else list(self.entity_dict.keys())

        all_results: List[MatchResult] = []

        for etype in search_types:
            if etype not in self.entity_dict:
                continue

            # 阶段 1: 精确匹配
            exact_result = self._try_exact_match(entity_text, etype)
            if exact_result:
                all_results.append(exact_result)
                if return_best_only:
                    return [exact_result]

            # 阶段 2: 别名匹配
            alias_result = self._try_alias_match(entity_text, etype)
            if alias_result:
                all_results.append(alias_result)
                if return_best_only:
                    return [alias_result]

        # 阶段 3: 编辑距离匹配
        if self.use_edit_distance:
            for etype in search_types:
                if etype not in self.entity_dict:
                    continue

                edit_results = self._try_edit_distance_match(entity_text, etype)
                all_results.extend(edit_results)

        # 排序并返回
        if not all_results:
            return []

        # 按阶段和相似度排序
        stage_order = {
            MatchStage.EXACT: 0,
            MatchStage.ALIAS: 1,
            MatchStage.EDIT_DISTANCE: 2,
            MatchStage.VECTOR: 3,
            MatchStage.NONE: 4,
        }

        all_results.sort(
            key=lambda x: (
                stage_order.get(x.stage, 99),
                -x.similarity,
                -len(x.entity_name)  # 同名优先长的
            )
        )

        # 校准置信度
        for result in all_results:
            result.calibrated_confidence = self.calibrate_confidence(result)

        if return_best_only:
            return [all_results[0]]

        return all_results[:self.top_k]

    def _try_exact_match(self, entity_text: str, entity_type: str) -> Optional[MatchResult]:
        """尝试精确匹配"""
        entities = self.entity_dict.get(entity_type, {})
        if entity_text in entities:
            info = entities[entity_text]
            return MatchResult(
                entity_name=info.name,
                entity_type=entity_type,
                kg_id=info.kg_id,
                distance=0,
                similarity=1.0,
                stage=MatchStage.EXACT
            )
        return None

    def _try_alias_match(self, entity_text: str, entity_type: str) -> Optional[MatchResult]:
        """尝试别名匹配"""
        if entity_text in self.alias_map:
            for etype, standard_name in self.alias_map[entity_text]:
                if etype == entity_type:
                    info = self.entity_dict[etype][standard_name]
                    return MatchResult(
                        entity_name=standard_name,
                        entity_type=etype,
                        kg_id=info.kg_id,
                        alias=entity_text,
                        distance=0,
                        similarity=1.0,
                        stage=MatchStage.ALIAS
                    )
        return None

    def _try_edit_distance_match(self, entity_text: str, entity_type: str) -> List[MatchResult]:
        """尝试编辑距离匹配"""
        results: List[MatchResult] = []
        entities = self.entity_dict.get(entity_type, {})

        max_dist = self.max_edit_distance or get_max_distance(len(entity_text))
        text_len = len(entity_text)

        # 使用堆找出最匹配的 top_k
        candidates = []

        for name, info in entities.items():
            # 快速过滤：长度差异太大直接跳过
            name_len = len(name)
            if abs(name_len - text_len) > max_dist:
                continue

            dist = levenshtein_distance(entity_text, name)
            if dist <= max_dist:
                # 计算相似度
                max_len = max(text_len, name_len)
                similarity = 1.0 - (dist / max_len) if max_len > 0 else 0.0

                # 使用最小堆（存负相似度，因为 heapq 是最小堆）
                heapq.heappush(candidates, (-similarity, dist, name, info))

                if len(candidates) > self.top_k * 2:
                    heapq.heappop(candidates)

        # 构建结果
        for neg_sim, dist, name, info in sorted(candidates, key=lambda x: x[0]):
            similarity = -neg_sim
            results.append(MatchResult(
                entity_name=name,
                entity_type=entity_type,
                kg_id=info.kg_id,
                distance=dist,
                similarity=similarity,
                stage=MatchStage.EDIT_DISTANCE
            ))

        return results
