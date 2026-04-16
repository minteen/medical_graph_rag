
# intent_classification/pipeline.py
import os
import re
import logging
import time

logger = logging.getLogger("IntentClassificationPipeline")


INTENT_NAMES = {
    "symptom_inquiry": "症状查询",
    "disease_inquiry": "疾病查询",
    "medication_inquiry": "用药咨询",
    "examination_inquiry": "检查咨询",
    "diet_lifestyle": "饮食/生活",
    "other": "其他",
}

KUAKE_LABELS = sorted([
    "其他", "功效作用", "医疗费用", "后果表述", "就医建议",
    "指标解读", "治疗方案", "注意事项", "疾病表述", "病因分析", "病情诊断"
])

KUAKE_TO_MEDICALGRAPH = {
    "病情诊断": "symptom_inquiry",
    "疾病表述": "disease_inquiry",
    "病因分析": "disease_inquiry",
    "后果表述": "disease_inquiry",
    "治疗方案": "medication_inquiry",
    "功效作用": "medication_inquiry",
    "注意事项": "diet_lifestyle",
    "指标解读": "examination_inquiry",
    "就医建议": "other",
    "医疗费用": "other",
    "其他": "other",
}


class IntentResult:
    """意图识别结果"""
    def __init__(self, intent_type, confidence, source):
        self.intent_type = intent_type
        self.confidence = confidence
        self.source = source
        self.latency_ms = None
        self.kuake_label = None

    @property
    def intent_name(self):
        return INTENT_NAMES.get(self.intent_type, "其他")

    def to_dict(self):
        return {
            "intent_type": self.intent_type,
            "intent_name": self.intent_name,
            "confidence": self.confidence,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "kuake_label": self.kuake_label,
        }


class IntentClassificationPipeline:
    """
    医疗意图识别 Pipeline

    整合了 QueryIntentClassification 训练好的模型
    支持关键词匹配和模型分类

    使用示例:
        pipeline = IntentClassificationPipeline(
            checkpoint_path=r"E:\\Desktop\\QueryIntentClassification\\checkpoints\\medical_intent_classification\\best_model.ckpt",
            enable_model=True
        )
        result = pipeline.classify("感冒了怎么办？")
        print(result.intent_name, result.confidence)
    """

    def _load_dotenv(self):
        """加载 .env 文件"""
        try:
            from dotenv import load_dotenv
            env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                ".env"
            )
            if os.path.exists(env_path):
                load_dotenv(env_path)
                logger.info("已加载配置: %s", env_path)
        except ImportError:
            logger.debug("dotenv 未安装，跳过 .env 加载")
        except Exception as e:
            logger.debug("加载 .env 失败: %s", e)

    def __init__(self,
                 checkpoint_path=None,
                 model_name_or_path="medicalai/ClinicalBERT",
                 enable_model=True,
                 device=None):
        """
        Args:
            checkpoint_path: 训练好的模型 checkpoint 路径
            model_name_or_path: 预训练模型名称
            enable_model: 是否启用模型分类
            device: 计算设备
        """
        self.checkpoint_path = checkpoint_path
        self.model_name_or_path = model_name_or_path
        self.enable_model = enable_model and checkpoint_path is not None
        self.device = device

        self.model = None
        self.tokenizer = None

        self._initialized = False
        self._load_dotenv()

    def initialize(self):
        """初始化（加载模型等耗时操作）"""
        if self._initialized:
            return

        logger.info("初始化意图识别 Pipeline...")

        if self.enable_model:
            self._load_model()

        self._initialized = True
        logger.info("意图识别 Pipeline 初始化完成")

    def _load_model(self):
        """加载模型"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info("加载 tokenizer: %s", self.model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("使用设备: %s", self.device)

            logger.info("加载模型: %s", self.checkpoint_path)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]
                    new_state_dict[new_key] = value
                elif key.startswith("classifier."):
                    new_state_dict[key] = value

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=len(KUAKE_LABELS)
            )
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            logger.info("模型加载成功！")
        except Exception as e:
            logger.warning("模型加载失败，将仅使用关键词匹配: %s", e)
            self.enable_model = False

    def _predict_by_keywords(self, text):
        """基于关键词预测意图"""
        text_lower = text.lower()
        scores = {}
        scores["symptom_inquiry"] = 0
        scores["disease_inquiry"] = 0
        scores["medication_inquiry"] = 0
        scores["examination_inquiry"] = 0
        scores["diet_lifestyle"] = 0
        scores["other"] = 0

        keywords = {}
        keywords["symptom_inquiry"] = ["症状", "不舒服", "难受", "疼痛", "怎么办", "发热", "咳嗽", "头痛"]
        keywords["disease_inquiry"] = ["病", "疾病", "症", "炎", "是什么", "为什么", "感冒", "肺炎", "高血压"]
        keywords["medication_inquiry"] = ["药", "药品", "药物", "治疗", "吃什么", "能吃吗", "布洛芬", "阿莫西林"]
        keywords["examination_inquiry"] = ["检查", "化验", "体检", "血常规", "尿常规"]
        keywords["diet_lifestyle"] = ["吃", "饮食", "食物", "注意", "能吃吗", "低盐", "低脂"]
        keywords["other"] = ["科", "科室", "挂号", "医院"]

        for intent_type, kw_list in keywords.items():
            for kw in kw_list:
                if kw in text_lower:
                    scores[intent_type] = scores[intent_type] + 1

        best_intent = "other"
        best_score = 0

        for intent_type, score in scores.items():
            if score - best_score > 0:
                best_score = score
                best_intent = intent_type

        if best_score - 0 > 0:
            confidence = min(1.0, best_score / 5.0)
        else:
            confidence = 0.0
            best_intent = "other"

        return best_intent, confidence

    def _predict_by_model(self, text):
        """基于模型预测意图"""
        if not self.enable_model or self.model is None:
            return None, 0.0, None

        try:
            import torch
            import torch.nn.functional as F

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            )

            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_idx].item()

            kuake_label = KUAKE_LABELS[pred_idx]
            intent_type = KUAKE_TO_MEDICALGRAPH.get(kuake_label, "other")
            return intent_type, confidence, kuake_label

        except Exception as e:
            logger.warning("模型预测失败: %s", e)
            return None, 0.0, None

    def classify(self, text):
        """
        识别用户问题的意图

        Args:
            text: 用户问题

        Returns:
            IntentResult 包含意图类型、置信度等信息
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        kw_intent, kw_confidence = self._predict_by_keywords(text)
        result = None

        if self.enable_model:
            model_intent, model_confidence, kuake_label = self._predict_by_model(text)
            if model_intent and model_confidence - 0.6 >= 0:
                result = IntentResult(model_intent, model_confidence, "model")
                result.kuake_label = kuake_label

        if result is None:
            result = IntentResult(kw_intent, kw_confidence, "keyword")

        result.latency_ms = (time.time() - start_time) * 1000
        return result

    def __call__(self, text):
        return self.classify(text)


def create_pipeline(checkpoint_path=None, **kwargs):
    """创建并初始化意图识别 Pipeline"""
    pipeline = IntentClassificationPipeline(checkpoint_path=checkpoint_path, **kwargs)
    pipeline.initialize()
    return pipeline


if __name__ == "__main__":
    pipeline = create_pipeline(checkpoint_path=None, enable_model=False)

    test_texts = [
        "感冒了怎么办？",
        "高血压能吃什么药？",
        "肺炎需要做什么检查？",
        "糖尿病患者饮食要注意什么？",
        "看感冒挂什么科？",
    ]

    for i, text in enumerate(test_texts, 1):
        print("\n" + "="*60)
        print(f"Test {i}: {text}")
        print("-"*60)
        result = pipeline.classify(text)
        print(f"  意图: {result.intent_name}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  来源: {result.source}")
        print(f"  耗时: {result.latency_ms:.1f}ms")

