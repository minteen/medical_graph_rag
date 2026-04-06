# keyword_matcher.py
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ================= 配置日志 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("KeywordMatcher")


@dataclass
class EntityMatch:
    """实体匹配结果"""
    text: str          # 匹配到的实体文本
    type: str          # 实体类型 (Disease/Drug/Symptom/...)
    start: int         # 起始位置
    end: int           # 结束位置
    kg_id: Optional[str] = None   # 图谱ID（如果有）
    source: str = "keyword_match"  # 来源
    confidence: float = 1.0        # 置信度
    alias: Optional[str] = None    # 匹配到的别名（如果是别名匹配）


class KeywordMatcher:
    """
    Layer 1: 关键词匹配器
    使用 Aho-Corasick 自动机实现高效多模式匹配
    """

    # 实体类型映射
    ENTITY_TYPES = {
        "disease": "Disease",
        "drug": "Drug",
        "food": "Food",
        "check": "Check",
        "department": "Department",
        "producer": "Producer",
        "symptom": "Symptom",
        "deny": "Deny"  # 否定词/禁用词
    }

    # 实体类型优先级（数值越大优先级越高）
    # 当同一个词匹配到多个类型时，优先返回高优先级类型
    TYPE_PRIORITY = {
        "Drug": 10,        # 药品优先，因为药品名通常比较独特
        "Disease": 9,      # 疾病次之
        "Symptom": 8,      # 症状
        "Check": 7,        # 检查
        "Food": 6,         # 食物
        "Department": 5,   # 科室
        "Producer": 4,     # 生产商（优先级较低，避免干扰）
        "Deny": 1          # 否定词（最低）
    }

    def __init__(self, entity_dir: Optional[str] = None, use_aho: bool = True):
        """
        Args:
            entity_dir: 实体词典目录，默认使用项目 data/entity/
            use_aho: 是否使用 Aho-Corasick 自动机（需要安装 pyahocorasick）
        """
        self.entity_dir = entity_dir or self._get_default_entity_dir()
        self.use_aho = use_aho

        # 实体词典：{type: {entity_text: {"aliases": [...], "kg_id": ...}}}
        self.entity_dict: Dict[str, Dict[str, Dict]] = {}
        # 反向映射：{entity_text: [(type, original_name, is_alias)]}
        self.reverse_index: Dict[str, List[Tuple[str, str, bool]]] = {}

        # Aho-Corasick 自动机
        self.automaton = None
        # 纯 Python 回退方案：前缀树
        self.prefix_tree = {}

        # 预编译正则
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')

    def _get_default_entity_dir(self) -> str:
        """获取默认实体词典目录"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(project_root, "data", "entity")

    def load_entities(self):
        """加载所有实体词典"""
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

        logger.info(f"✅ 实体加载完成: {len(self.reverse_index)} 个唯一实体名")

        # 构建匹配引擎
        if self.use_aho:
            try:
                self._build_aho_automaton()
            except ImportError:
                logger.warning("⚠️  pyahocorasick 未安装，使用前缀树回退方案")
                self.use_aho = False
                self._build_prefix_tree()
        else:
            self._build_prefix_tree()

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

                # 存储主实体
                self.entity_dict[entity_type][main_name] = {
                    "aliases": aliases,
                    "kg_id": kg_id
                }

                # 反向索引主名
                if main_name not in self.reverse_index:
                    self.reverse_index[main_name] = []
                self.reverse_index[main_name].append((entity_type, main_name, False))

                # 反向索引别名
                for alias in aliases:
                    alias = alias.strip()
                    if alias and alias != main_name:
                        if alias not in self.reverse_index:
                            self.reverse_index[alias] = []
                        self.reverse_index[alias].append((entity_type, main_name, True))

                count += 1

        logger.info(f"   {entity_type}: {count} 条")

    def _build_aho_automaton(self):
        """构建 Aho-Corasick 自动机"""
        import ahocorasick
        self.automaton = ahocorasick.Automaton()

        for keyword in self.reverse_index.keys():
            if len(keyword) >= 1:  # 至少1个字符
                self.automaton.add_word(keyword, keyword)

        self.automaton.make_automaton()
        logger.info("✅ Aho-Corasick 自动机构建完成")

    def _build_prefix_tree(self):
        """构建前缀树（纯 Python 回退方案）"""
        self.prefix_tree = {}

        for keyword in self.reverse_index.keys():
            if not keyword:
                continue
            node = self.prefix_tree
            for char in keyword:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['__end__'] = keyword

        logger.info("✅ 前缀树构建完成")

    def match(self, text: str, use_longest: bool = True, use_priority: bool = False) -> List[EntityMatch]:
        """
        在文本中匹配实体

        Args:
            text: 输入文本
            use_longest: 是否使用最长匹配（过滤掉被包含的短匹配）
            use_priority: 是否使用类型优先级（同一位置只保留最高优先级类型），默认 False

        Returns:
            EntityMatch 列表
        """
        if not text:
            return []

        matches: List[EntityMatch] = []

        if self.use_aho and self.automaton:
            matches = self._match_aho(text)
        else:
            matches = self._match_prefix_tree(text)

        # 1. 过滤短匹配（保留最长）
        if use_longest:
            matches = self._filter_longest_matches(matches)

        # 2. 同一位置按类型优先级过滤（默认关闭）
        if use_priority:
            matches = self._filter_by_type_priority(matches)

        # 3. 过滤无意义的短匹配
        matches = self._filter_noise(matches)

        # 4. 按位置和类型优先级排序
        matches.sort(key=lambda x: (
            x.start,
            -x.end,
            -self.TYPE_PRIORITY.get(x.type, 0)
        ))
        return matches

    def _filter_noise(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """过滤无意义的匹配"""
        result = []
        for m in matches:
            # 过滤单个英文字母
            if len(m.text) == 1 and m.text.isascii() and not m.text.isdigit():
                continue
            # 过滤纯数字（除非是有意义的，这里暂时都过滤）
            if m.text.isdigit():
                continue
            result.append(m)
        return result

    def _filter_by_type_priority(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """同一位置只保留优先级最高的类型"""
        if not matches:
            return []

        # 按位置分组
        position_map: Dict[Tuple[int, int], List[EntityMatch]] = {}
        for m in matches:
            key = (m.start, m.end)
            if key not in position_map:
                position_map[key] = []
            position_map[key].append(m)

        result = []
        for key, group in position_map.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # 按优先级排序
                group.sort(
                    key=lambda x: (
                        -self.TYPE_PRIORITY.get(x.type, 0),
                        -len(x.text)  # 同优先级，长的优先
                    )
                )
                result.append(group[0])

        return result

    def _match_aho(self, text: str) -> List[EntityMatch]:
        """使用 Aho-Corasick 匹配"""
        matches = []
        for end_idx, keyword in self.automaton.iter(text):
            start_idx = end_idx - len(keyword) + 1

            # 检查边界：避免匹配到词中间
            if not self._is_valid_boundary(text, start_idx, end_idx + 1):
                continue

            # 查找对应的实体类型
            for entity_type, original_name, is_alias in self.reverse_index.get(keyword, []):
                # 对于单字实体需要更严格的检查
                if len(keyword) == 1:
                    if not self._is_single_char_valid(text, start_idx, end_idx + 1, entity_type):
                        continue

                # 获取 kg_id
                kg_id = None
                if original_name in self.entity_dict.get(entity_type, {}):
                    kg_id = self.entity_dict[entity_type][original_name].get("kg_id")

                match = EntityMatch(
                    text=keyword,
                    type=entity_type,
                    start=start_idx,
                    end=end_idx + 1,
                    kg_id=kg_id,
                    source="keyword_match",
                    confidence=1.0,
                    alias=original_name if is_alias else None
                )
                matches.append(match)

        return matches

    def _match_prefix_tree(self, text: str) -> List[EntityMatch]:
        """使用前缀树匹配（纯 Python）"""
        matches = []
        n = len(text)

        for i in range(n):
            node = self.prefix_tree
            for j in range(i, n):
                char = text[j]
                if char not in node:
                    break
                node = node[char]

                if '__end__' in node:
                    keyword = node['__end__']
                    start_idx = i
                    end_idx = j

                    # 检查边界
                    if not self._is_valid_boundary(text, start_idx, end_idx + 1):
                        continue

                    # 查找对应的实体类型
                    for entity_type, original_name, is_alias in self.reverse_index.get(keyword, []):
                        # 对于单字实体需要更严格的检查
                        if len(keyword) == 1:
                            if not self._is_single_char_valid(text, start_idx, end_idx + 1, entity_type):
                                continue

                        kg_id = None
                        if original_name in self.entity_dict.get(entity_type, {}):
                            kg_id = self.entity_dict[entity_type][original_name].get("kg_id")

                        match = EntityMatch(
                            text=keyword,
                            type=entity_type,
                            start=start_idx,
                            end=end_idx + 1,
                            kg_id=kg_id,
                            source="keyword_match",
                            confidence=1.0,
                            alias=original_name if is_alias else None
                        )
                        matches.append(match)

        return matches

    def _is_valid_boundary(self, text: str, start: int, end: int) -> bool:
        """检查匹配是否在有效边界（避免匹配到词中间）"""
        # 检查前边界
        if start > 0:
            prev_char = text[start - 1]
            # 如果前一个字符是中文，可能是词的一部分，需要谨慎
            if self.chinese_pattern.match(prev_char):
                # 对于中文，放宽限制，因为中文没有空格分隔
                pass

        # 检查后边界
        if end < len(text):
            next_char = text[end]
            if self.chinese_pattern.match(next_char):
                pass

        return True

    def _is_single_char_valid(self, text: str, start: int, end: int, entity_type: str) -> bool:
        """
        检查单字匹配是否有效
        对于单字（特别是 Deny 类型），需要更严格的检查
        """
        char = text[start:end]

        # Deny 类型的单字需要特别小心
        if entity_type == "Deny":
            # 检查是否是独立使用（周围有标点或空格）
            # 或者在句首/句尾
            has_left_boundary = (start == 0) or (not self.chinese_pattern.match(text[start - 1]))
            has_right_boundary = (end >= len(text)) or (not self.chinese_pattern.match(text[end]))

            # 如果被中文字符包围，很可能是词的一部分，拒绝匹配
            if not has_left_boundary and not has_right_boundary:
                return False

        # 其他类型的单字也需要谨慎
        else:
            # 太短的实体（单字）如果被其他中文字符包围，可能不是独立实体
            has_left_boundary = (start == 0) or (not self.chinese_pattern.match(text[start - 1]))
            has_right_boundary = (end >= len(text)) or (not self.chinese_pattern.match(text[end]))

            # 如果两边都是中文，可能是词的一部分
            if not has_left_boundary and not has_right_boundary:
                # 检查是否有更长的匹配也包含了这个位置（在后面的过滤中处理）
                pass

        return True

    def _filter_longest_matches(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """最长匹配过滤：保留最长的匹配，移除被包含的短匹配
        但相同位置、相同长度的不同类型匹配都会保留
        """
        if not matches:
            return []

        # 按起始位置升序，长度降序排序
        sorted_matches = sorted(matches, key=lambda x: (x.start, -(x.end - x.start)))

        result = []
        # 记录已保留的 (start, end) 范围
        kept_ranges = set()

        for match in sorted_matches:
            # 检查是否被已保留的长匹配包含
            # 注意：相同 (start, end) 的不算被包含
            is_contained = False
            for (s, e) in kept_ranges:
                # 只有当被严格包含（且不是相同范围）时才跳过
                if s <= match.start and match.end <= e and (s < match.start or e > match.end):
                    is_contained = True
                    break

            if not is_contained:
                result.append(match)
                # 记录这个范围（用于后续检测被包含的情况）
                kept_ranges.add((match.start, match.end))

        return result

    def to_dict(self, matches: List[EntityMatch]) -> List[Dict]:
        """将匹配结果转换为字典列表"""
        return [
            {
                "text": m.text,
                "type": m.type,
                "start": m.start,
                "end": m.end,
                "kg_id": m.kg_id,
                "source": m.source,
                "confidence": m.confidence,
                "alias": m.alias
            }
            for m in matches
        ]


# ================= 便捷函数 =================
def create_matcher(entity_dir: Optional[str] = None) -> KeywordMatcher:
    """创建并加载关键词匹配器"""
    matcher = KeywordMatcher(entity_dir=entity_dir)
    matcher.load_entities()
    return matcher


if __name__ == "__main__":
    # 简单测试
    matcher = create_matcher()

    test_texts = [
        "感冒可以吃对乙酰氨基酚吗？",
        "肺炎的常见症状有发热、咳嗽",
        "高血压患者应该低盐饮食",
        "建议做胸部CT检查"
    ]

    for text in test_texts:
        print(f"\n文本: {text}")
        matches = matcher.match(text)
        for m in matches:
            print(f"  [{m.type}] {m.text} @ {m.start}-{m.end}" +
                  (f" (alias of {m.alias})" if m.alias else ""))
