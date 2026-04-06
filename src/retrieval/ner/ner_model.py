# ner_model.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("NERModel")


@dataclass
class NERPrediction:
    """NER模型预测结果"""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    source: str = "ner_model"


class NERModel:
    """
    Layer 2: NER 模型层
    基于 ModelScope 的中文医学命名实体识别模型

    模型: iic/nlp_raner_named-entity-recognition_chinese-base-cmeee
    数据集: CMeEE (Chinese Medical Entity Extraction)
    """

    # CMeEE 实体类型映射到我们的标准类型
    CMEEE_TYPE_MAPPING = {
        "dis": "Disease",      # 疾病
        "sym": "Symptom",      # 症状
        "pro": "Check",        # 医疗程序
        "equ": "Check",        # 医疗器械
        "ite": "Check",        # 体检项目（新增）
        "dru": "Drug",         # 药品
        "bod": "Symptom",      # 肌体（身体部位）
        "dep": "Department",   # 科室
        "mic": "Disease",      # 微生物
        # 补充可能的其他类型
        "disease": "Disease",
        "drug": "Drug",
        "food": "Food",
        "check": "Check",
        "department": "Department",
        "producer": "Producer",
        "symptom": "Symptom"
    }

    # 标准实体类型（与 keyword_matcher 保持一致）
    STANDARD_TYPES = {
        "Disease", "Drug", "Food", "Check",
        "Department", "Producer", "Symptom", "Deny"
    }

    def __init__(self,
                 model_id: Optional[str] = None,
                 use_cuda: bool = False,
                 confidence_threshold: float = 0.1,  # 降低默认阈值，该模型输出的置信度普遍较低
                 batch_size: int = 8):
        """
        Args:
            model_id: ModelScope 模型 ID
            use_cuda: 是否使用 GPU
            confidence_threshold: 置信度阈值
            batch_size: 批处理大小
        """
        self.model_id = model_id or "iic/nlp_raner_named-entity-recognition_chinese-base-cmeee"
        self.use_cuda = use_cuda
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        self._model = None
        self._pipeline = None
        self._initialized = False

        # 检查依赖
        self._has_modelscope = self._check_dependency()

    def _check_dependency(self) -> bool:
        """检查依赖是否安装"""
        try:
            import modelscope
            return True
        except ImportError:
            logger.warning("⚠️  modelscope 未安装，NER 模型层将不可用")
            return False

    def initialize(self):
        """初始化模型（加载模型）"""
        if self._initialized:
            return

        if not self._has_modelscope:
            logger.error("❌ modelscope 未安装，无法初始化 NER 模型")
            return

        try:
            logger.info(f"🔧 加载 NER 模型: {self.model_id}")
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            self._pipeline = pipeline(
                task=Tasks.named_entity_recognition,
                model=self.model_id,
                model_revision=None,
                device='gpu' if self.use_cuda else 'cpu'
            )

            self._initialized = True
            logger.info("✅ NER 模型加载完成")

        except Exception as e:
            logger.error(f"❌ NER 模型加载失败: {e}")
            import traceback
            traceback.print_exc()

    def predict(self, text: str) -> List[NERPrediction]:
        """
        预测文本中的实体

        Args:
            text: 输入文本

        Returns:
            NERPrediction 列表
        """
        if not text or not text.strip():
            return []

        if not self._has_modelscope:
            return []

        if not self._initialized:
            self.initialize()

        if not self._pipeline:
            return []

        try:
            # ModelScope NER pipeline 输出格式通常是:
            # {
            #   "output": [
            #     {"type": "dis", "start": 0, "end": 2, "span": "感冒"},
            #     ...
            #   ]
            # }
            result = self._pipeline(text)

            predictions = []
            if isinstance(result, dict):
                outputs = result.get("output", [])
            elif isinstance(result, list):
                outputs = result
            else:
                outputs = []

            for item in outputs:
                pred = self._parse_item(item, text)
                if pred and pred.confidence >= self.confidence_threshold:
                    predictions.append(pred)

            # 后处理
            predictions = self._post_process(predictions, text)

            return predictions

        except Exception as e:
            logger.error(f"❌ NER 预测失败: {e}")
            return []

    def _parse_item(self, item: Dict, original_text: str) -> Optional[NERPrediction]:
        """解析单个预测项"""
        try:
            # 尝试不同的字段名
            # 模型输出: {'type': 'dis', 'start': np.int64(5), 'end': np.int64(7), 'prob': np.float32(0.423...), 'span': '甲亢'}
            entity_type = item.get("type") or item.get("entity_type") or item.get("tag")
            span = item.get("span") or item.get("text") or item.get("word")
            start = item.get("start") or item.get("start_pos")
            end = item.get("end") or item.get("end_pos")
            confidence = item.get("confidence") or item.get("prob") or item.get("score", 1.0)

            # 转换 numpy 类型为 Python 原生类型
            if hasattr(start, 'item'):
                start = start.item()
            if hasattr(end, 'item'):
                end = end.item()
            if hasattr(confidence, 'item'):
                confidence = confidence.item()

            # 如果没有 span 但有 start/end，从原文提取
            if not span and start is not None and end is not None:
                span = original_text[start:end]

            # 如果没有 start/end 但有 span，尝试在原文中查找
            if span and (start is None or end is None):
                pos = original_text.find(span)
                if pos >= 0:
                    start = pos
                    end = pos + len(span)

            if not span or start is None or end is None:
                return None

            # 映射类型
            standard_type = self.CMEEE_TYPE_MAPPING.get(entity_type, entity_type)
            if standard_type not in self.STANDARD_TYPES:
                # 尝试小写匹配
                standard_type = self.CMEEE_TYPE_MAPPING.get(entity_type.lower(), "Disease")

            return NERPrediction(
                text=span,
                type=standard_type,
                start=int(start),
                end=int(end),
                confidence=float(confidence),
                source="ner_model"
            )

        except Exception as e:
            logger.debug(f"解析预测项失败: {e}, item={item}")
            return None

    def _post_process(self, predictions: List[NERPrediction], text: str) -> List[NERPrediction]:
        """后处理预测结果"""
        if not predictions:
            return []

        # 1. 过滤无效的预测
        valid = []
        for pred in predictions:
            if pred.start < 0 or pred.end > len(text):
                continue
            if pred.start >= pred.end:
                continue
            if not pred.text or pred.text != text[pred.start:pred.end]:
                # 修正文本
                pred.text = text[pred.start:pred.end]
            valid.append(pred)

        predictions = valid

        # 2. 按位置排序
        predictions.sort(key=lambda x: (x.start, -x.end))

        # 3. 去重（相同位置只保留置信度最高的）
        result = []
        used_positions = {}

        for pred in predictions:
            key = (pred.start, pred.end)
            if key not in used_positions:
                used_positions[key] = pred
            else:
                existing = used_positions[key]
                if pred.confidence > existing.confidence:
                    used_positions[key] = pred

        result = list(used_positions.values())
        result.sort(key=lambda x: (x.start, -x.end))

        return result

    def predict_batch(self, texts: List[str]) -> List[List[NERPrediction]]:
        """批量预测"""
        return [self.predict(text) for text in texts]

    def to_dict_list(self, predictions: List[NERPrediction]) -> List[Dict]:
        """转换为字典列表"""
        return [
            {
                "text": p.text,
                "type": p.type,
                "start": p.start,
                "end": p.end,
                "confidence": p.confidence,
                "source": p.source,
                "kg_id": None  # NER 模型不直接提供 kg_id
            }
            for p in predictions
        ]

    def __call__(self, text: str) -> List[NERPrediction]:
        """支持直接调用"""
        return self.predict(text)


# ================= 便捷函数 =================
def create_ner_model(model_id: Optional[str] = None, **kwargs) -> NERModel:
    """创建 NER 模型"""
    model = NERModel(model_id=model_id, **kwargs)
    return model


if __name__ == "__main__":
    # 简单测试
    print("=" * 70)
    print("测试 NER 模型层")
    print("=" * 70)

    model = create_ner_model()

    test_texts = [
        "感冒可以吃对乙酰氨基酚吗？",
        "肺炎的常见症状有发热、咳嗽",
        "高血压患者应该低盐饮食",
        "建议做胸部CT检查"
    ]

    for text in test_texts:
        print(f"\n文本: {text}")
        preds = model.predict(text)
        if preds:
            for p in preds:
                print(f"  [{p.type}] {p.text} @ {p.start}-{p.end} (conf={p.confidence:.2f})")
        else:
            print("  (未预测到实体)")
