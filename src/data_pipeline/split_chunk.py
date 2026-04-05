# vector_chunker.py
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# ================= 配置日志 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MedicalChunker")

# 章节标题到图谱关系的映射表（严格对齐你的 Schema）
SECTION_RELATION_MAP = {
    "分类路径": ["belongs_to"],
    "常见症状": ["has_symptom"],
    "并发疾病": ["acompany_with"],
    "推荐检查": ["need_check"],
    "治疗方案": ["cure_way"],
    "临床推荐用药": ["recommand_drug"],
    "常用辅助药物": ["common_drug"],
    "饮食宜吃": ["do_eat"],
    "饮食禁忌": ["no_eat"],
    "推荐食谱": ["recommand_eat"],
    "疾病简介": ["desc"],
    "发病原因": ["cause"],
    "预防措施": ["prevent"],
}


class MedicalChunker:
    """
    医学语义分块器：将 Graph-to-Text 长文本按医学逻辑切分，
    注入隐式上下文，绑定图谱元数据与安全标记，输出向量化友好格式。
    """

    def __init__(
            self,
            max_tokens: int = 256,
            overlap_sentences: int = 2,
            min_chunk_len: int = 15,
            output_dir: str = "./data/processed"
    ):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.min_chunk_len = min_chunk_len
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 预编译正则，提升批量处理性能
        self.section_pattern = re.compile(r'【(.*?)】\s*(.*?)(?=【|$)', re.DOTALL)
        self.sentence_delimiters = re.compile(r'(?<=[。；\n!?])')

    # ================= 内部工具方法 =================
    def _estimate_tokens(self, text: str) -> int:
        """轻量级中文 Token 估算（兼容 BGE-M3/MedCPT 等主流模型）"""
        # 经验公式：中文字符 + 标点 ≈ 1.2~1.5 token/字
        # 生产环境可替换为 transformers.AutoTokenizer.from_pretrained("BAAI/bge-m3")
        return int(len(text) * 0.65)

    def _parse_sections(self, raw_text: str) -> List[Dict[str, str]]:
        """按 【标题】 解析长文本为结构化段落"""
        sections = []
        for match in self.section_pattern.finditer(raw_text):
            title = match.group(1).strip()
            content = match.group(2).strip()
            # 过滤占位符/空内容
            if not content or content in ("暂无", "暂无相关信息", "（暂无）", "暂无相关"):
                continue
            sections.append({"title": title, "content": content})
        return sections

    def _split_sentences(self, text: str) -> List[str]:
        """按医学语义边界切句，保留终止标点"""
        # 使用正向后顾分割，保留句号/分号/换行在句末
        parts = self.sentence_delimiters.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _chunk_section(self, title: str, content: str, context_prefix: str) -> List[Dict]:
        """单段落智能分块（短段落保留，长段落滑动窗口+重叠）"""
        full_text = f"{context_prefix}{content}"
        token_len = self._estimate_tokens(full_text)

        # 1. 短段落直接保留
        if token_len <= self.max_tokens:
            return [{"text": full_text, "idx": 0}]

        # 2. 长段落按句子切分+滑动聚合
        sentences = self._split_sentences(content)
        chunks = []
        buffer = []
        buffer_tokens = 0
        idx = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if buffer_tokens + sent_tokens > self.max_tokens and buffer:
                # 达到上限，保存当前块
                chunk_text = f"{context_prefix}{''.join(buffer)}"
                chunks.append({"text": chunk_text, "idx": idx})
                idx += 1

                # 重叠策略：保留最后 overlap_sentences 句
                overlap = buffer[-self.overlap_sentences:] if len(buffer) > self.overlap_sentences else []
                buffer = overlap
                buffer_tokens = sum(self._estimate_tokens(s) for s in buffer)

            buffer.append(sent)
            buffer_tokens += sent_tokens

        if buffer and len(buffer) >= self.min_chunk_len:
            chunk_text = f"{context_prefix}{''.join(buffer)}"
            chunks.append({"text": chunk_text, "idx": idx})

        return chunks

    def _build_safety_flags(self, metadata: Dict[str, Any]) -> List[str]:
        """基于元数据提取安全合规标记"""
        flags = []
        # 治愈率解析
        prob_str = str(metadata.get("cured_prob", "")).strip()
        if prob_str and prob_str not in ("暂无", "未知"):
            # 提取数字部分：支持 "约40%", "85%", "0.00002%" 等
            nums = re.findall(r'(\d+(?:\.\d+)?)', prob_str)
            if nums:
                prob_val = float(nums[0])
                if prob_val < 50:
                    flags.append("low_cure_rate")
                elif prob_val < 10:
                    flags.append("rare_or_low_success")

        if str(metadata.get("yibao_status", "")).strip() == "否":
            flags.append("non_medical_insurance")

        return flags

    # ================= 核心处理流程 =================
    def process_document(self, doc: Dict) -> List[Dict]:
        """将单篇疾病长文本转换为 Chunk 列表"""
        kg_id = doc["kg_id"]
        entity_name = doc["metadata"].get("name", "未知疾病")
        sections = self._parse_sections(doc["raw_text"])
        safety_flags = self._build_safety_flags(doc["metadata"])

        chunks = []
        for sec in sections:
            title = sec["title"]
            content = sec["content"]
            relations = SECTION_RELATION_MAP.get(title, [])
            context_prefix = f"【{entity_name}】的{title}包括："

            sec_chunks = self._chunk_section(title, content, context_prefix)
            for c in sec_chunks:
                chunk = {
                    "chunk_id": f"{kg_id}__{title}__{c['idx']}",
                    "kg_id": kg_id,
                    "entity_type": doc["entity_type"],
                    "section_type": title,
                    "involved_relations": relations,
                    "text": c["text"],
                    "metadata": {
                        **doc["metadata"],
                        "safety_flags": safety_flags,
                        "source": "graph_linearization_chunked",
                        "version": doc["metadata"].get("version", "v1.0")
                    }
                }
                chunks.append(chunk)
        return chunks

    def run(self, input_file: str, output_file: Optional[str] = None):
        """批量处理 JSONL 长文本，输出分块结果"""
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_chunked.jsonl")

        if not os.path.exists(input_file):
            logger.error(f"❌ 输入文件不存在: {input_file}")
            return

        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        processed = 0
        total_chunks = 0

        logger.info(f"🚀 开始医学分块 | 输入: {input_file} | 输出: {output_file}")

        try:
            with open(input_file, 'r', encoding='utf-8') as fin, \
                    open(output_file, 'w', encoding='utf-8') as fout, \
                    tqdm(total=total_lines, desc="分块进度") as pbar:

                for line in fin:
                    line = line.strip()
                    if not line: continue

                    try:
                        doc = json.loads(line)
                        chunks = self.process_document(doc)

                        for chunk in chunks:
                            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            total_chunks += 1

                        processed += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.warning(f"⚠️ 处理跳过 (第 {processed + 1} 行): {e}")
                        pbar.update(1)

        except Exception as e:
            logger.error(f"❌ 分块中断: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"✅ 分块完成！处理 {processed} 篇文档，生成 {total_chunks} 个 Chunk。\n📁 输出路径: {output_file}")


# ================= CLI 入口 =================
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "disease_long_texts.jsonl")
    DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_PATH)
    INPUT_FILE = os.getenv("INPUT_FILE", DEFAULT_INPUT_FILE)
    chunker = MedicalChunker(
        max_tokens=256,
        overlap_sentences=2,
        min_chunk_len=15,
        output_dir=OUTPUT_DIR
    )
    chunker.run(INPUT_FILE)