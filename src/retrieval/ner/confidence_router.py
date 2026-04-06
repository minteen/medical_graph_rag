# confidence_router.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .keyword_matcher import KeywordMatcher
from .ner_model import NERModel, NERPrediction
from .llm_extractor import LLMExtractor, LLMPrediction
from .entity_fuser import EntityFuser, MergedEntity

logger = logging.getLogger("ConfidenceRouter")


@dataclass
class ExtractionResult:
    """提取结果（带置信度信息）"""
    entities: List[Dict[str, Any]]
    avg_confidence: float
    coverage_score: float  # 文本覆盖率
    source: str


class ConfidenceRouter:
    """
    基于置信度的智能路由调度器

    调度策略:
    1. 先调用关键词匹配（置信度=1.0）
    2. 如果结果不足，调用 NER 模型
    3. 如果 NER 置信度低，调用 LLM
    4. 最终融合所有结果
    """

    def __init__(self,
                 entity_dir: Optional[str] = None,
                 use_aho: bool = True,
                 enable_ner: bool = True,
                 ner_model_id: Optional[str] = None,
                 ner_confidence_threshold: float = 0.1,
                 enable_llm: bool = True,
                 llm_model: Optional[str] = None,
                 llm_api_key: Optional[str] = None,
                 llm_base_url: Optional[str] = None,
                 llm_confidence: float = 0.85,
                 fusion_confidence_threshold: float = 0.1,
                 # 调度策略参数
                 min_entity_count: int = 2,  # 最少实体数量
                 long_text_threshold: int = 20,  # 长文本阈值
                 min_density_for_long_text: float = 0.05,  # 长文本的实体密度要求
                 min_avg_confidence: float = 0.6,  # 最小平均置信度
                 ner_confidence_threshold_for_llm: float = 0.3,  # 触发LLM的NER阈值
                 enable_rule_based: bool = True):  # 启用基于规则的预判
        """
        Args:
            ... (其他参数见 NERPipeline)
            min_entities_per_char: 触发NER的最小实体密度
            min_avg_confidence: 触发LLM的最小平均置信度
            min_coverage_ratio: 触发LLM的最小文本覆盖率
            ner_confidence_threshold_for_llm: NER平均置信度低于此值触发LLM
        """
        self.entity_dir = entity_dir
        self.use_aho = use_aho
        self.enable_ner = enable_ner
        self.ner_model_id = ner_model_id
        self.ner_confidence_threshold = ner_confidence_threshold
        self.enable_llm = enable_llm
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_confidence = llm_confidence
        self.fusion_confidence_threshold = fusion_confidence_threshold

        # 调度策略参数
        self.min_entity_count = min_entity_count
        self.long_text_threshold = long_text_threshold
        self.min_density_for_long_text = min_density_for_long_text
        self.min_avg_confidence = min_avg_confidence
        self.ner_confidence_threshold_for_llm = ner_confidence_threshold_for_llm
        self.enable_rule_based = enable_rule_based

        # 初始化各层
        self.keyword_matcher: Optional[KeywordMatcher] = None
        self.ner_model: Optional[NERModel] = None
        self.llm_extractor: Optional[LLMExtractor] = None
        self.fuser = EntityFuser(confidence_threshold=fusion_confidence_threshold)

        self._initialized = False

    def initialize(self):
        """初始化各层"""
        if self._initialized:
            return

        logger.info("🔧 初始化 ConfidenceRouter...")

        # Layer 1: 关键词匹配（必须）
        self.keyword_matcher = KeywordMatcher(
            entity_dir=self.entity_dir,
            use_aho=self.use_aho
        )
        self.keyword_matcher.load_entities()

        # Layer 2: NER 模型（可选）
        if self.enable_ner:
            self.ner_model = NERModel(
                model_id=self.ner_model_id,
                confidence_threshold=self.ner_confidence_threshold
            )
            self.ner_model.initialize()

        # Layer 3: LLM 提取（可选）
        if self.enable_llm:
            self.llm_extractor = LLMExtractor(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                model=self.llm_model,
                confidence=self.llm_confidence
            )
            self.llm_extractor._initialize_client()

        self._initialized = True
        logger.info("✅ ConfidenceRouter 初始化完成")

    def extract(self, text: str, return_dict: bool = True) -> List[Any]:
        """
        智能提取实体（基于置信度路由）

        Args:
            text: 输入文本
            return_dict: 是否返回字典

        Returns:
            提取的实体列表
        """
        if not self._initialized:
            self.initialize()

        if not text or not text.strip():
            return []

        logger.info(f"🔍 开始智能提取: {text[:30]}...")

        # Step 1: 关键词匹配（必选）
        layer1_result = self._extract_layer1(text)
        logger.info(f"Layer 1: {len(layer1_result.entities)} 个实体, "
                   f"avg_conf={layer1_result.avg_confidence:.3f}, "
                   f"coverage={layer1_result.coverage_score:.3f}")

        all_results = [layer1_result]

        # Step 2: 判断是否需要调用 NER 模型
        if self._should_call_ner(layer1_result, text):
            layer2_result = self._extract_layer2(text)
            if layer2_result:
                all_results.append(layer2_result)
                logger.info(f"Layer 2: {len(layer2_result.entities)} 个实体, "
                           f"avg_conf={layer2_result.avg_confidence:.3f}, "
                           f"coverage={layer2_result.coverage_score:.3f}")
        else:
            logger.info("⏭️  跳过 Layer 2 (关键词匹配结果充足)")

        # Step 3: 判断是否需要调用 LLM
        if self._should_call_llm(layer1_result, layer2_result if 'layer2_result' in locals() else None, text):
            layer3_result = self._extract_layer3(text)
            if layer3_result:
                all_results.append(layer3_result)
                logger.info(f"Layer 3: {len(layer3_result.entities)} 个实体, "
                           f"avg_conf={layer3_result.avg_confidence:.3f}, "
                           f"coverage={layer3_result.coverage_score:.3f}")
        else:
            logger.info("⏭️  跳过 Layer 3 (已有高置信度结果)")

        # 融合所有结果
        merged = self._fuse_results(all_results)

        if return_dict:
            return [asdict(e) for e in merged]
        return merged

    def _extract_layer1(self, text: str) -> ExtractionResult:
        """Layer 1: 关键词匹配"""
        matches = self.keyword_matcher.match(text)
        entities = self.keyword_matcher.to_dict(matches)

        # 计算覆盖率
        covered_chars = set()
        for e in entities:
            for i in range(e['start'], e['end']):
                covered_chars.add(i)
        coverage = len(covered_chars) / len(text) if text else 0

        return ExtractionResult(
            entities=entities,
            avg_confidence=1.0,  # 关键词匹配置信度固定为 1.0
            coverage_score=coverage,
            source="keyword_match"
        )

    def _extract_layer2(self, text: str) -> Optional[ExtractionResult]:
        """Layer 2: NER 模型"""
        if not self.ner_model:
            return None

        preds = self.ner_model.predict(text)
        if not preds:
            return None

        entities = self.ner_model.to_dict_list(preds)

        # 计算平均置信度
        avg_conf = sum(p.confidence for p in preds) / len(preds)

        # 计算覆盖率
        covered_chars = set()
        for p in preds:
            for i in range(p.start, p.end):
                covered_chars.add(i)
        coverage = len(covered_chars) / len(text) if text else 0

        return ExtractionResult(
            entities=entities,
            avg_confidence=avg_conf,
            coverage_score=coverage,
            source="ner_model"
        )

    def _extract_layer3(self, text: str) -> Optional[ExtractionResult]:
        """Layer 3: LLM 提取"""
        if not self.llm_extractor:
            return None

        preds = self.llm_extractor.extract(text)
        if not preds:
            return None

        entities = self.llm_extractor.to_dict_list(preds)

        # 计算覆盖率
        covered_chars = set()
        for p in preds:
            for i in range(p.start, p.end):
                covered_chars.add(i)
        coverage = len(covered_chars) / len(text) if text else 0

        return ExtractionResult(
            entities=entities,
            avg_confidence=self.llm_confidence,  # LLM 置信度固定
            coverage_score=coverage,
            source="llm_extract"
        )

    def _should_call_ner(self, layer1_result: ExtractionResult, text: str) -> bool:
        """判断是否需要调用 NER 模型（改进策略）"""
        # 如果未启用 NER，跳过
        if not self.enable_ner or not self.ner_model:
            return False

        entity_count = len(layer1_result.entities)

        # 策略 1: 绝对实体数量不足
        if entity_count < self.min_entity_count:
            logger.debug(f"  → 实体数量不足 ({entity_count} < {self.min_entity_count})")
            return True

        # 策略 2: 长文本的实体密度要求
        if (len(text) >= self.long_text_threshold and
            entity_count / len(text) < self.min_density_for_long_text):
            logger.debug(f"  → 长文本实体密度低 ({entity_count}/{len(text)}={entity_count/len(text):.3f})")
            return True

        # 策略 3: 基于规则的预判
        if self.enable_rule_based and self._rule_based_ner_prediction(text, layer1_result):
            logger.debug(f"  → 规则预判需要 NER")
            return True

        return False

    def _rule_based_ner_prediction(self, text: str, layer1_result: ExtractionResult) -> bool:
        """基于规则的 NER 调用预判"""
        # 规则 1: 包含常见未登录词模式
        unregistered_patterns = [
            "可以" in text and "吗" in text,  # "...可以...吗?"
            "应该" in text and "什么" in text,  # "应该...什么"
            "怎么" in text,  # "怎么..."
        ]
        if any(unregistered_patterns):
            return True

        # 规则 2: 包含复杂症状描述（但未被识别）
        symptom_indicators = ["感觉", "觉得", "有点", "总是", "经常", "有时候"]
        for indicator in symptom_indicators:
            if indicator in text:
                # 检查 indicator 附近是否有未识别的文本
                pos = text.find(indicator)
                if pos >= 0:
                    # 检查 indicator 前后 5 个字符
                    start = max(0, pos - 5)
                    end = min(len(text), pos + len(indicator) + 5)
                    context = text[start:end]

                    # 检查这个范围内是否没有实体
                    has_entity_in_context = False
                    for entity in layer1_result.entities:
                        if entity['start'] < end and entity['end'] > start:
                            has_entity_in_context = True
                            break

                    if not has_entity_in_context:
                        return True

        return False

    def _should_call_llm(self, layer1_result: ExtractionResult,
                        layer2_result: Optional[ExtractionResult], text: str) -> bool:
        """判断是否需要调用 LLM（改进策略）"""
        # 如果未启用 LLM，跳过
        if not self.enable_llm or not self.llm_extractor:
            return False

        # 检查是否有 API Key
        if not self.llm_extractor.api_key:
            return False

        # 策略 1: Layer 2 置信度太低
        if layer2_result and layer2_result.avg_confidence < self.ner_confidence_threshold_for_llm:
            logger.debug(f"  → Layer 2 置信度低 ({layer2_result.avg_confidence:.3f})")
            return True

        # 策略 2: 基于规则的 LLM 调用预判
        if self.enable_rule_based and self._rule_based_llm_prediction(text, layer1_result, layer2_result):
            logger.debug(f"  → 规则预判需要 LLM")
            return True

        # 策略 3: 复杂句式（兜底策略）
        if len(text) > 30 and (len(layer1_result.entities) + (len(layer2_result.entities) if layer2_result else 0)) < 3:
            logger.debug(f"  → 长文本实体少 (len={len(text)}, entities<3)")
            return True

        return False

    def _rule_based_llm_prediction(self, text: str,
                                 layer1_result: ExtractionResult,
                                 layer2_result: Optional[ExtractionResult]) -> bool:
        """基于规则的 LLM 调用预判"""
        all_entities = layer1_result.entities[:]
        if layer2_result:
            all_entities.extend(layer2_result.entities)

        # 规则 1: 包含复杂描述词但识别实体少
        complex_indicators = [
            "老是", "经常", "有时候", "偶尔",
            "有点", "比较", "特别", "非常",
            "好像", "似乎", "可能", "大概"
        ]

        for indicator in complex_indicators:
            if indicator in text:
                # 检查是否识别了足够的实体
                if len(all_entities) < 2:
                    return True

        # 规则 2: 包含多个疑问词
        question_words = ["吗", "呢", "什么", "怎么", "如何", "哪里", "为什么"]
        question_count = sum(1 for word in question_words if word in text)
        if question_count >= 2 and len(all_entities) < 2:
            return True

        # 规则 3: 口语化表达
        colloquial_patterns = [
            "不太舒服" in text,
            "感觉不对劲" in text,
            "不知道怎么回事" in text,
            "该怎么办" in text
        ]
        if any(colloquial_patterns) and len(all_entities) < 2:
            return True

        return False

    def _fuse_results(self, results: List[ExtractionResult]) -> List[MergedEntity]:
        """融合所有结果"""
        keyword_results = []
        ner_results = []
        llm_results = []

        for result in results:
            if result.source == "keyword_match":
                keyword_results = result.entities
            elif result.source == "ner_model":
                ner_results = result.entities
            elif result.source == "llm_extract":
                llm_results = result.entities

        return self.fuser.fuse(
            keyword_matches=keyword_results,
            ner_matches=ner_results,
            llm_matches=llm_results
        )

    def extract_batch(self, texts: List[str], return_dict: bool = True) -> List[List[Any]]:
        """批量提取"""
        return [self.extract(text, return_dict=return_dict) for text in texts]

    def __call__(self, text: str, return_dict: bool = True) -> List[Any]:
        """支持直接调用"""
        return self.extract(text, return_dict=return_dict)


# ================= 便捷函数 =================
def create_confidence_router(**kwargs) -> ConfidenceRouter:
    """创建置信度路由调度器"""
    router = ConfidenceRouter(**kwargs)
    router.initialize()
    return router


if __name__ == "__main__":
    # 简单测试
    router = create_confidence_router()

    test_cases = [
        "感冒可以吃对乙酰氨基酚吗？",
        "肺炎的常见症状有发热、咳嗽",
        "最近老是感觉头晕乎乎的，还有点恶心",
    ]

    for text in test_cases:
        print(f"\n{'='*60}")
        print(f"输入: {text}")
        print('-'*60)
        results = router.extract(text, return_dict=False)
        for r in results:
            sources = ",".join(r.sources)
            print(f"  [{r.type:10}] {r.text:20} (sources: {sources})")
