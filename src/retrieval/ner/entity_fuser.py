# entity_fuser.py
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("EntityFuser")


class SourcePriority(Enum):
    """来源优先级（数值越大优先级越高）"""
    KEYWORD = 3
    NER_MODEL = 2
    LLM_EXTRACT = 1


@dataclass
class MergedEntity:
    """融合后的实体"""
    text: str
    type: str
    start: int
    end: int
    kg_id: Optional[str] = None
    source: str = "merged"
    confidence: float = 1.0
    sources: List[str] = None  # 所有来源
    alias: Optional[str] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = [self.source]


class EntityFuser:
    """
    实体结果融合器
    处理多层结果的去重、冲突消解、置信度校准
    """

    def __init__(self,
                 prioritize_longest: bool = True,
                 confidence_threshold: float = 0.5):
        self.prioritize_longest = prioritize_longest
        self.confidence_threshold = confidence_threshold

    def fuse(self,
             keyword_matches: List[Dict] = None,
             ner_matches: List[Dict] = None,
             llm_matches: List[Dict] = None) -> List[MergedEntity]:
        """
        融合各层结果

        Args:
            keyword_matches: 关键词匹配结果 (dict 列表)
            ner_matches: NER模型结果 (dict 列表)
            llm_matches: LLM提取结果 (dict 列表)

        Returns:
            MergedEntity 列表
        """
        all_candidates = []

        # 收集所有候选
        if keyword_matches:
            for m in keyword_matches:
                m['_priority'] = SourcePriority.KEYWORD.value
                m['_source'] = 'keyword_match'
                all_candidates.append(m)

        if ner_matches:
            for m in ner_matches:
                m['_priority'] = SourcePriority.NER_MODEL.value
                m['_source'] = 'ner_model'
                all_candidates.append(m)

        if llm_matches:
            for m in llm_matches:
                m['_priority'] = SourcePriority.LLM_EXTRACT.value
                m['_source'] = 'llm_extract'
                all_candidates.append(m)

        if not all_candidates:
            return []

        # 按起始位置排序
        all_candidates.sort(key=lambda x: (x['start'], -x.get('end', 0), -x['_priority']))

        # 融合
        merged = self._merge_candidates(all_candidates)

        # 过滤低置信度
        merged = [e for e in merged if e.confidence >= self.confidence_threshold]

        return merged

    def _merge_candidates(self, candidates: List[Dict]) -> List[MergedEntity]:
        """合并候选实体"""
        result = []
        used_positions = set()  # (start, end)

        for cand in candidates:
            start = cand['start']
            end = cand.get('end', start + len(cand['text']))
            key = (start, end)

            # 检查是否已被覆盖
            skip = False
            for (s, e) in used_positions:
                if s <= start and end <= e:
                    # 被已有结果完全包含，跳过
                    skip = True
                    break
            if skip:
                continue

            # 检查是否与已有结果冲突（重叠但不包含）
            conflicting = []
            for i, entity in enumerate(result):
                if not (end <= entity.start or start >= entity.end):
                    conflicting.append((i, entity))

            if conflicting:
                # 处理冲突
                for (i, existing) in conflicting:
                    if self._should_replace(cand, existing):
                        # 替换
                        result[i] = self._to_merged_entity(cand)
                        used_positions.remove((existing.start, existing.end))
                        used_positions.add(key)
            else:
                # 无冲突，直接添加
                result.append(self._to_merged_entity(cand))
                used_positions.add(key)

        # 按位置排序
        result.sort(key=lambda x: (x.start, -x.end))
        return result

    def _should_replace(self, new_cand: Dict, existing: MergedEntity) -> bool:
        """判断是否应该替换已有实体"""
        # 1. 优先看来源优先级
        new_prio = new_cand['_priority']
        existing_prio = self._source_to_priority(existing.source)

        if new_prio > existing_prio:
            return True
        if new_prio < existing_prio:
            return False

        # 2. 同来源，看置信度
        new_conf = new_cand.get('confidence', 1.0)
        if new_conf > existing.confidence:
            return True
        if new_conf < existing.confidence:
            return False

        # 3. 同置信度，选更长的
        new_len = new_cand.get('end', new_cand['start'] + len(new_cand['text'])) - new_cand['start']
        existing_len = existing.end - existing.start
        return new_len > existing_len

    def _source_to_priority(self, source: str) -> int:
        if 'keyword' in source:
            return SourcePriority.KEYWORD.value
        elif 'ner' in source:
            return SourcePriority.NER_MODEL.value
        else:
            return SourcePriority.LLM_EXTRACT.value

    def _to_merged_entity(self, cand: Dict) -> MergedEntity:
        """转换为 MergedEntity"""
        return MergedEntity(
            text=cand['text'],
            type=cand['type'],
            start=cand['start'],
            end=cand.get('end', cand['start'] + len(cand['text'])),
            kg_id=cand.get('kg_id'),
            source=cand['_source'],
            confidence=cand.get('confidence', 1.0),
            sources=[cand['_source']],
            alias=cand.get('alias')
        )

    def to_dict_list(self, entities: List[MergedEntity]) -> List[Dict]:
        """转换为字典列表"""
        return [asdict(e) for e in entities]
