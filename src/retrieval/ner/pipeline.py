# pipeline.py
import os
import logging
from typing import List, Dict, Any, Optional

from .keyword_matcher import KeywordMatcher, create_matcher
from .entity_fuser import EntityFuser, MergedEntity

logger = logging.getLogger("NERPipeline")


class NERPipeline:
    """
    医疗 NER Pipeline（当前仅 Layer 1: 关键词匹配）

    使用示例:
        pipeline = NERPipeline()
        results = pipeline.extract("感冒可以吃对乙酰氨基酚吗？")
        for r in results:
            print(r.text, r.type)
    """

    def __init__(self,
                 entity_dir: Optional[str] = None,
                 use_aho: bool = True,
                 enable_ner: bool = False,  # 预留：Layer 2
                 enable_llm: bool = False):  # 预留：Layer 3
        """
        Args:
            entity_dir: 实体词典目录
            use_aho: 是否使用 Aho-Corasick（需要 pyahocorasick）
            enable_ner: 是否启用 NER 模型层（预留）
            enable_llm: 是否启用 LLM 提取层（预留）
        """
        self.entity_dir = entity_dir
        self.use_aho = use_aho
        self.enable_ner = enable_ner
        self.enable_llm = enable_llm

        # 初始化各层
        self.keyword_matcher: Optional[KeywordMatcher] = None
        self.ner_model = None  # 预留
        self.llm_extractor = None  # 预留
        self.fuser = EntityFuser()

        self._initialized = False

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

        # TODO: Layer 2: NER 模型
        if self.enable_ner:
            logger.info("⚠️  NER 模型层尚未实现")

        # TODO: Layer 3: LLM 提取
        if self.enable_llm:
            logger.info("⚠️  LLM 提取层尚未实现")

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

        # Layer 2: NER 模型（预留）
        ner_results = []

        # Layer 3: LLM 提取（预留）
        llm_results = []

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
