# pipeline.py
import os
import logging
from typing import List, Dict, Any, Optional

from .keyword_matcher import KeywordMatcher, create_matcher
from .ner_model import NERModel, create_ner_model
from .llm_extractor import LLMExtractor, create_llm_extractor
from .entity_fuser import EntityFuser, MergedEntity

logger = logging.getLogger("NERPipeline")


class NERPipeline:
    """
    医疗 NER Pipeline（三层架构）

    使用示例:
        pipeline = NERPipeline(enable_ner=True, enable_llm=True)
        results = pipeline.extract("感冒可以吃对乙酰氨基酚吗？")
        for r in results:
            print(r.text, r.type)
    """

    def _load_dotenv(self):
        """加载 .env 文件"""
        try:
            from dotenv import load_dotenv
            # 从项目根目录加载
            env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                ".env"
            )
            if os.path.exists(env_path):
                load_dotenv(env_path)
                logger.info(f"✅ 已加载配置: {env_path}")
        except ImportError:
            logger.debug("dotenv 未安装，跳过 .env 加载")
        except Exception as e:
            logger.debug(f"加载 .env 失败: {e}")

    def __init__(self,
                 entity_dir: Optional[str] = None,
                 use_aho: bool = True,
                 enable_ner: bool = False,  # Layer 2: NER 模型
                 ner_model_id: Optional[str] = None,
                 ner_confidence_threshold: float = 0.1,  # 降低默认阈值，该模型输出的置信度普遍较低
                 fusion_confidence_threshold: float = 0.1,  # 融合时的置信度阈值也要降低
                 use_cuda: bool = False,
                 enable_llm: bool = False,  # Layer 3: LLM 提取
                 llm_model: Optional[str] = None,
                 llm_api_key: Optional[str] = None,
                 llm_base_url: Optional[str] = None,
                 llm_confidence: float = 0.85):
        """
        Args:
            entity_dir: 实体词典目录
            use_aho: 是否使用 Aho-Corasick（需要 pyahocorasick）
            enable_ner: 是否启用 NER 模型层
            ner_model_id: NER 模型 ID (ModelScope)
            ner_confidence_threshold: NER 模型置信度阈值
            use_cuda: 是否使用 GPU
            enable_llm: 是否启用 LLM 提取层（预留）
        """
        self.entity_dir = entity_dir
        self.use_aho = use_aho
        self.enable_ner = enable_ner
        self.ner_model_id = ner_model_id
        self.ner_confidence_threshold = ner_confidence_threshold
        self.use_cuda = use_cuda
        self.enable_llm = enable_llm
        self.fusion_confidence_threshold = fusion_confidence_threshold

        # LLM 配置
        self.llm_model = llm_model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_confidence = llm_confidence

        # 初始化各层
        self.keyword_matcher: Optional[KeywordMatcher] = None
        self.ner_model: Optional[NERModel] = None
        self.llm_extractor: Optional[LLMExtractor] = None
        self.fuser = EntityFuser(confidence_threshold=fusion_confidence_threshold)

        self._initialized = False

        # 尝试加载 .env 文件
        self._load_dotenv()

    def initialize(self):
        """初始化（加载模型等耗时操作）"""
        if self._initialized:
            return

        logger.info("🔧 初始化 NER Pipeline...")

        # Layer 1: 关键词匹配
        self.keyword_matcher = KeywordMatcher(
            entity_dir=self.entity_dir,
            use_aho=self.use_aho
        )
        self.keyword_matcher.load_entities()

        # Layer 2: NER 模型
        if self.enable_ner:
            logger.info("🔧 初始化 NER 模型层...")
            self.ner_model = NERModel(
                model_id=self.ner_model_id,
                use_cuda=self.use_cuda,
                confidence_threshold=self.ner_confidence_threshold
            )
            self.ner_model.initialize()

        # Layer 3: LLM 提取
        if self.enable_llm:
            logger.info("🔧 初始化 LLM 提取层...")
            self.llm_extractor = LLMExtractor(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                model=self.llm_model,
                confidence=self.llm_confidence
            )
            # LLM 不需要预先加载模型，只需要初始化客户端
            self.llm_extractor._initialize_client()

        self._initialized = True
        logger.info("✅ NER Pipeline 初始化完成")

    def extract(self, text: str, return_dict: bool = True) -> List[Any]:
        """
        从文本中提取实体

        Args:
            text: 输入文本
            return_dict: 是否返回字典列表（False 返回 MergedEntity 对象）

        Returns:
            提取结果列表
        """
        if not self._initialized:
            self.initialize()

        if not text or not text.strip():
            return []

        # Layer 1: 关键词匹配
        keyword_results = []
        if self.keyword_matcher:
            matches = self.keyword_matcher.match(text)
            keyword_results = self.keyword_matcher.to_dict(matches)

        # Layer 2: NER 模型
        ner_results = []
        if self.enable_ner and self.ner_model:
            preds = self.ner_model.predict(text)
            ner_results = self.ner_model.to_dict_list(preds)

        # Layer 3: LLM 提取
        llm_results = []
        if self.enable_llm and self.llm_extractor:
            preds = self.llm_extractor.extract(text)
            llm_results = self.llm_extractor.to_dict_list(preds)

        # 融合结果
        merged = self.fuser.fuse(
            keyword_matches=keyword_results,
            ner_matches=ner_results,
            llm_matches=llm_results
        )

        if return_dict:
            return self.fuser.to_dict_list(merged)
        return merged

    def extract_batch(self, texts: List[str], return_dict: bool = True) -> List[List[Any]]:
        """批量提取"""
        return [self.extract(text, return_dict=return_dict) for text in texts]

    def __call__(self, text: str, return_dict: bool = True) -> List[Any]:
        """支持直接调用"""
        return self.extract(text, return_dict=return_dict)


# ================= 便捷函数 =================
def create_pipeline(entity_dir: Optional[str] = None, **kwargs) -> NERPipeline:
    """创建并初始化 NER Pipeline"""
    pipeline = NERPipeline(entity_dir=entity_dir, **kwargs)
    pipeline.initialize()
    return pipeline


if __name__ == "__main__":
    # 测试
    pipeline = create_pipeline()

    test_texts = [
        "感冒可以吃对乙酰氨基酚吗？",
        "肺炎的常见症状有发热、咳嗽、咳痰",
        "高血压患者应该低盐饮食，避免吃腌制食品",
        "建议去心内科就诊，做心电图和胸部CT检查",
        "糖尿病可以吃二甲双胍或者注射胰岛素"
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {text}")
        print('-'*60)
        results = pipeline.extract(text, return_dict=False)
        for r in results:
            print(f"  [{r.type:10}] {r.text:20} @ {r.start}-{r.end}" +
                  (f" (kg_id={r.kg_id})" if r.kg_id else ""))
