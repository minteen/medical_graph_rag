# base.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("EntityLinking")


class MatchStage(Enum):
    """匹配阶段"""
    EXACT = "exact"           # 精确匹配
    ALIAS = "alias"           # 别名匹配
    EDIT_DISTANCE = "edit"    # 编辑距离匹配
    VECTOR = "vector"         # 向量检索匹配
    NONE = "none"             # 未匹配


@dataclass
class MatchResult:
    """匹配结果"""
    entity_name: str          # 匹配到的标准实体名
    entity_type: str          # 实体类型
    kg_id: Optional[str] = None  # 知识图谱ID
    alias: Optional[str] = None  # 匹配到的别名（如果是别名匹配）
    distance: int = 0         # 编辑距离（如果有）
    similarity: float = 0.0    # 相似度 (0-1)
    stage: MatchStage = MatchStage.NONE  # 匹配阶段
    calibrated_confidence: float = 0.0  # 校准后的置信度

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['stage'] = self.stage.value if self.stage else None
        return result


@dataclass
class EntityInfo:
    """实体信息（从词典加载）"""
    name: str              # 主名
    aliases: List[str]     # 别名列表
    kg_id: Optional[str]   # 知识图谱ID
    entity_type: str       # 实体类型


class BaseLinker:
    """实体链接器基类"""

    # 实体类型映射（与 KeywordMatcher 一致）
    ENTITY_TYPES = {
        "disease": "Disease",
        "drug": "Drug",
        "food": "Food",
        "check": "Check",
        "department": "Department",
        "producer": "Producer",
        "symptom": "Symptom",
        "deny": "Deny"
    }

    def __init__(self, entity_dir: Optional[str] = None):
        """
        Args:
            entity_dir: 实体词典目录
        """
        self.entity_dir = entity_dir or self._get_default_entity_dir()

        # 实体词典: {type: {name: EntityInfo}}
        self.entity_dict: Dict[str, Dict[str, EntityInfo]] = {}
        # 别名映射: {alias: List[(type, standard_name)]}
        self.alias_map: Dict[str, List[Tuple[str, str]]] = {}

        self._initialized = False

    def _get_default_entity_dir(self) -> str:
        """获取默认实体目录"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data", "entity"
        )

    def initialize(self):
        """初始化（加载实体词典）"""
        if self._initialized:
            return

        logger.info(f"📖 加载实体词典: {self.entity_dir}")

        for filename in os.listdir(self.entity_dir):
            if not filename.endswith(".txt"):
                continue

            entity_type_key = os.path.splitext(filename)[0]
            entity_type = self.ENTITY_TYPES.get(entity_type_key)

            if not entity_type:
                logger.warning(f"⚠️  未知实体类型: {entity_type_key}, 跳过")
                continue

            filepath = os.path.join(self.entity_dir, filename)
            self._load_single_file(filepath, entity_type)

        self._initialized = True
        total_entities = sum(len(entities) for entities in self.entity_dict.values())
        logger.info(f"✅ 实体加载完成: {total_entities} 个主实体, "
                   f"{len(self.alias_map)} 个别名")

    def _load_single_file(self, filepath: str, entity_type: str):
        """加载单个实体文件"""
        if entity_type not in self.entity_dict:
            self.entity_dict[entity_type] = {}

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 支持格式：实体名|别名1,别名2|kg_id
                parts = line.split('|')
                main_name = parts[0].strip()
                aliases = parts[1].strip().split(',') if len(parts) > 1 and parts[1].strip() else []
                kg_id = parts[2].strip() if len(parts) > 2 else None

                # 清理别名
                aliases = [a.strip() for a in aliases if a.strip()]

                # 存储实体信息
                entity_info = EntityInfo(
                    name=main_name,
                    aliases=aliases,
                    kg_id=kg_id,
                    entity_type=entity_type
                )
                self.entity_dict[entity_type][main_name] = entity_info

                # 建立别名映射
                for alias in aliases:
                    if alias not in self.alias_map:
                        self.alias_map[alias] = []
                    self.alias_map[alias].append((entity_type, main_name))

                count += 1

        logger.info(f"   {entity_type}: {count} 条")

    def calibrate_confidence(self, result: MatchResult) -> float:
        """
        校准置信度

        Args:
            result: 匹配结果

        Returns:
            校准后的置信度 (0-1)
        """
        # 基础分
        base_scores = {
            MatchStage.EXACT: 1.0,
            MatchStage.ALIAS: 0.95,
            MatchStage.VECTOR: 0.85,
            MatchStage.EDIT_DISTANCE: 0.75,
            MatchStage.NONE: 0.0,
        }

        base_conf = base_scores.get(result.stage, 0.5)

        # 相似度加分
        conf = base_conf + result.similarity * 0.2

        # 长度加分（长词更可靠）
        len_bonus = min(len(result.entity_name) / 10, 0.1)
        conf += len_bonus

        # 确保在 0-1 范围内
        return max(0.0, min(1.0, conf))
