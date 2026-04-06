# llm_extractor.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("LLMExtractor")


@dataclass
class LLMPrediction:
    """LLM 提取结果"""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 0.85
    source: str = "llm_extract"
    note: Optional[str] = None


class LLMExtractor:
    """
    Layer 3: LLM 提取层
    使用 OpenAI 格式的 API 进行实体提取
    """

    # 支持的实体类型
    SUPPORTED_TYPES = {
        "Disease", "Drug", "Symptom", "Food",
        "Check", "Department", "Producer", "Deny"
    }

    # 类型描述（用于 Prompt）
    TYPE_DESCRIPTIONS = {
        "Disease": "疾病名称",
        "Drug": "药品名称",
        "Symptom": "症状描述",
        "Food": "食物名称",
        "Check": "检查项目或医疗设备",
        "Department": "科室名称",
        "Producer": "药品生产商",
    }

    # 默认系统提示词
    DEFAULT_SYSTEM_PROMPT = """你是专业的医学实体识别助手。请从用户输入的文本中提取医学实体。

支持的实体类型：
- Disease: 疾病名称
- Drug: 药品名称
- Symptom: 症状描述
- Food: 食物名称
- Check: 检查项目或医疗设备
- Department: 科室名称
- Producer: 药品生产商

输出要求：
1. 输出为 JSON 数组，每个元素包含：
   - text: 实体文本（必须与原文完全一致）
   - type: 实体类型（必须是上面列出的类型之一）
   - start: 起始位置（从0开始）
   - end: 结束位置（不包含该位置）
   - note: 备注说明（可选）

2. 只输出 JSON，不要其他文字说明。

3. 如果没有实体，输出空数组 []。

4. 确保 start 和 end 位置正确，实体文本必须等于 text[start:end]。"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.0,
                 max_retries: int = 3,
                 timeout: int = 60,
                 confidence: float = 0.85):
        """
        Args:
            api_key: API Key（默认从环境变量 OPENAI_API_KEY 读取）
            base_url: API Base URL（默认从环境变量 OPENAI_BASE_URL 读取）
            model: 模型名称
            system_prompt: 自定义系统提示词
            temperature: 温度参数（越低越稳定）
            max_retries: 最大重试次数
            timeout: 请求超时（秒）
            confidence: 默认置信度
        """

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("openai_model")
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.default_confidence = confidence

        self._client = None
        self._has_openai = self._check_dependency()

    def _check_dependency(self) -> bool:
        """检查依赖是否安装"""
        try:
            import openai
            return True
        except ImportError:
            logger.warning("⚠️  openai 未安装，LLM 提取层将不可用")
            return False

    def _initialize_client(self):
        """初始化 OpenAI 客户端"""
        if self._client:
            return

        if not self._has_openai:
            return

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info("✅ LLM 客户端初始化完成")

        except Exception as e:
            logger.error(f"❌ LLM 客户端初始化失败: {e}")

    def extract(self, text: str) -> List[LLMPrediction]:
        """
        从文本中提取实体

        Args:
            text: 输入文本

        Returns:
            LLMPrediction 列表
        """
        if not text or not text.strip():
            return []

        if not self._has_openai:
            return []

        if not self.api_key:
            logger.warning("⚠️  未设置 OPENAI_API_KEY，跳过 LLM 提取")
            return []

        self._initialize_client()
        if not self._client:
            return []

        # 调用 LLM
        response_text = self._call_llm(text)
        if not response_text:
            return []

        # 解析结果
        predictions = self._parse_response(response_text, text)
        return predictions

    def _call_llm(self, text: str) -> Optional[str]:
        """调用 LLM"""
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )

                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"⚠️  LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ LLM 调用最终失败: {e}")

        return None

    def _parse_response(self, response_text: str, original_text: str) -> List[LLMPrediction]:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON（可能有额外的文字）
            json_str = self._extract_json(response_text)
            if not json_str:
                return []

            data = json.loads(json_str)
            if not isinstance(data, list):
                return []

            predictions = []
            for item in data:
                pred = self._parse_item(item, original_text)
                if pred:
                    predictions.append(pred)

            return predictions

        except Exception as e:
            logger.warning(f"⚠️  解析 LLM 响应失败: {e}")
            logger.debug(f"响应内容: {response_text}")
            return []

    def _extract_json(self, text: str) -> Optional[str]:
        """从响应中提取 JSON 字符串"""
        text = text.strip()

        # 情况1: 直接就是 JSON 数组
        if text.startswith("["):
            # 找到匹配的 ]
            count = 0
            for i, c in enumerate(text):
                if c == "[":
                    count += 1
                elif c == "]":
                    count -= 1
                    if count == 0:
                        return text[:i + 1]
            return text

        # 情况2: 在 ```json ... ``` 中
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # 情况3: 在 ``` ... ``` 中
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                return parts[1].strip()

        return None

    def _parse_item(self, item: Dict, original_text: str) -> Optional[LLMPrediction]:
        """解析单个预测项"""
        try:
            # 提取字段
            entity_text = item.get("text") or item.get("entity") or item.get("span")
            entity_type = item.get("type") or item.get("entity_type")
            start = item.get("start")
            end = item.get("end")
            note = item.get("note")

            if not entity_text or entity_type is None:
                return None

            # 标准化类型
            entity_type = entity_type.capitalize()
            if entity_type not in self.SUPPORTED_TYPES:
                # 尝试映射
                type_mapping = {
                    "疾病": "Disease",
                    "药品": "Drug",
                    "药物": "Drug",
                    "症状": "Symptom",
                    "食物": "Food",
                    "检查": "Check",
                    "科室": "Department",
                    "生产商": "Producer",
                    "否定": "Deny",
                }
                entity_type = type_mapping.get(entity_type, entity_type)
                if entity_type not in self.SUPPORTED_TYPES:
                    return None

            # 清理实体文本（去除首尾空格）
            entity_text = entity_text.strip()
            if not entity_text:
                return None

            # LLM 经常返回错误的位置，优先通过文本查找正确位置
            found_positions = []
            search_text = entity_text
            pos = original_text.find(search_text)
            while pos >= 0:
                found_positions.append((pos, pos + len(search_text)))
                pos = original_text.find(search_text, pos + 1)

            # 如果 start/end 提供了，检查是否合理
            if start is not None and end is not None:
                try:
                    start = int(start)
                    end = int(end)
                    # 验证位置范围内的文本是否匹配
                    if 0 <= start < end <= len(original_text):
                        actual_text = original_text[start:end]
                        if actual_text == entity_text:
                            # 位置正确，直接使用
                            return LLMPrediction(
                                text=entity_text,
                                type=entity_type,
                                start=start,
                                end=end,
                                confidence=self.default_confidence,
                                source="llm_extract",
                                note=note
                            )
                        elif entity_text in actual_text:
                            # 实体是位置范围的子串，调整位置
                            sub_pos = actual_text.find(entity_text)
                            start = start + sub_pos
                            end = start + len(entity_text)
                            return LLMPrediction(
                                text=entity_text,
                                type=entity_type,
                                start=start,
                                end=end,
                                confidence=self.default_confidence,
                                source="llm_extract",
                                note=note
                            )
                except (ValueError, TypeError):
                    pass

            # 如果有找到的位置，使用第一个
            if found_positions:
                start, end = found_positions[0]
                return LLMPrediction(
                    text=entity_text,
                    type=entity_type,
                    start=start,
                    end=end,
                    confidence=self.default_confidence,
                    source="llm_extract",
                    note=note
                )

            # 无法确定位置，跳过
            logger.debug(f"无法确定实体位置: '{entity_text}'")
            return None

        except Exception as e:
            logger.debug(f"解析预测项失败: {e}, item={item}")
            return None

    def extract_batch(self, texts: List[str]) -> List[List[LLMPrediction]]:
        """批量提取"""
        return [self.extract(text) for text in texts]

    def to_dict_list(self, predictions: List[LLMPrediction]) -> List[Dict]:
        """转换为字典列表"""
        return [
            {
                "text": p.text,
                "type": p.type,
                "start": p.start,
                "end": p.end,
                "confidence": p.confidence,
                "source": p.source,
                "kg_id": None,
                "note": p.note
            }
            for p in predictions
        ]

    def __call__(self, text: str) -> List[LLMPrediction]:
        """支持直接调用"""
        return self.extract(text)


# ================= 便捷函数 =================
def create_llm_extractor(**kwargs) -> LLMExtractor:
    """创建 LLM 提取器"""
    return LLMExtractor(**kwargs)


if __name__ == "__main__":
    # 简单测试
    print("=" * 70)
    print("测试 LLM 提取层")
    print("=" * 70)

    extractor = create_llm_extractor()

    test_texts = [
        "感冒可以吃对乙酰氨基酚吗？",
        "肺炎的常见症状有发热、咳嗽",
    ]

    for text in test_texts:
        print(f"\n文本: {text}")
        if not extractor.api_key:
            print("  (未设置 API Key，跳过测试)")
            continue

        preds = extractor.extract(text)
        if preds:
            for p in preds:
                print(f"  [{p.type}] {p.text} @ {p.start}-{p.end}")
        else:
            print("  (未提取到实体)")
