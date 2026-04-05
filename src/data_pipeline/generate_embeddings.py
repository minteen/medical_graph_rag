# generate_embeddings.py
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# ================= 配置日志 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EmbeddingGenerator")


class EmbeddingGenerator:
    """
    医学文本 Embedding 生成器：为分块结果生成向量
    使用 HuggingFace 镜像站（hf-mirror.com），国内访问快
    支持分别生成 CPU 和 GPU 版本的向量
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-m3",
            batch_size: int = 32,
            device: Optional[str] = None,
            use_fp16: bool = False,
            hf_endpoint: str = "https://hf-mirror.com"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.hf_endpoint = hf_endpoint
        self.model = None
        self.tokenizer = None

        # 设置 HuggingFace 镜像站
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
            logger.info(f"🌐 使用 HuggingFace 镜像站: {self.hf_endpoint}")

    def _clear_model(self):
        """清除当前加载的模型，释放显存/内存"""
        if self.model is not None:
            import gc
            try:
                del self.model
            except:
                pass
            self.model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            logger.info("🧹 已清理模型资源")

    def _load_model(self, force_device: Optional[str] = None):
        """延迟加载模型，可强制指定设备"""
        current_device = force_device if force_device is not None else self.device

        if self.model is not None and getattr(self, '_loaded_device', None) == current_device:
            return

        self._clear_model()

        logger.info(f"📦 加载 Embedding 模型: {self.model_name}")

        # 自动选择设备或使用强制设备
        import torch
        if current_device is None:
            if torch.cuda.is_available():
                current_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                current_device = "mps"
            else:
                current_device = "cpu"

        self._loaded_device = current_device
        logger.info(f"💻 使用设备: {current_device}")

        # 加载模型 (使用 sentence-transformers)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_name,
                device=current_device
            )
            if self.use_fp16 and current_device == "cuda":
                self.model = self.model.half()
        except ImportError:
            logger.error("❌ 请安装 sentence-transformers: pip install sentence-transformers")
            raise

    def generate_embeddings(self, texts: List[str], force_device: Optional[str] = None) -> List[List[float]]:
        """批量生成 Embedding，可强制指定设备"""
        self._load_model(force_device=force_device)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True  # 归一化，方便 COSINE 距离计算
        )

        return embeddings.tolist()

    def _write_output(self, chunks: List[Dict], embeddings: List[List[float]],
                     output_file: str, device_name: str):
        """写入输出文件"""
        logger.info(f"✍️  写入 {device_name} 输出文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as fout:
            for chunk, emb in zip(chunks, embeddings):
                chunk_with_emb = chunk.copy()
                chunk_with_emb["embedding"] = emb
                chunk_with_emb["embedding_device"] = device_name
                chunk_with_emb["hf_endpoint"] = self.hf_endpoint
                fout.write(json.dumps(chunk_with_emb, ensure_ascii=False) + "\n")

    def run(self, input_file: str, output_file_cpu: Optional[str] = None,
            output_file_gpu: Optional[str] = None, use_gpu_fast_path: bool = True) -> Tuple[str, Optional[str]]:
        """
        为 JSONL 分块结果添加 Embedding

        Args:
            input_file: 输入 JSONL 文件
            output_file_cpu: CPU 版本输出文件（可选）
            output_file_gpu: GPU 版本输出文件（可选）
            use_gpu_fast_path: 是否使用 GPU 加速路径（用 GPU 生成，同时保存为 CPU/GPU 两个文件）

        Returns:
            (cpu_output_path, gpu_output_path)
        """
        import torch
        has_cuda = torch.cuda.is_available()

        if not os.path.exists(input_file):
            logger.error(f"❌ 输入文件不存在: {input_file}")
            return None, None

        # 读取所有数据
        logger.info(f"📖 读取输入文件: {input_file}")
        chunks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        total = len(chunks)
        logger.info(f"📋 共 {total} 个 Chunk 待处理")

        # 准备输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        input_dir = os.path.dirname(input_file)

        if output_file_cpu is None:
            output_file_cpu = os.path.join(input_dir, f"{base_name}_embeddings_cpu.jsonl")

        if output_file_gpu is None:
            output_file_gpu = os.path.join(input_dir, f"{base_name}_embeddings_gpu.jsonl")

        texts = [c["text"] for c in chunks]
        embeddings = []

        # 选择设备：优先 GPU 加速
        device = "cuda" if (has_cuda and use_gpu_fast_path) else "cpu"
        device_name = "GPU" if device == "cuda" else "CPU"

        logger.info(f"🚀 使用 {device_name} 生成 Embedding...")

        for i in tqdm(range(0, total, self.batch_size), desc=f"{device_name} Embedding 进度"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.generate_embeddings(batch_texts, force_device=device)
            embeddings.extend(batch_embeddings)

        # 写入两个文件（同一份向量，分别标记为 cpu 和 gpu）
        self._write_output(chunks, embeddings, output_file_cpu, "cpu")
        logger.info(f"✅ CPU 版本已保存: {output_file_cpu}")

        self._write_output(chunks, embeddings, output_file_gpu, "gpu")
        logger.info(f"✅ GPU 版本已保存: {output_file_gpu}")

        logger.info(f"💡 说明：两个文件的向量内容完全相同，均由 {device_name} 生成")

        self._clear_model()
        return output_file_cpu, output_file_gpu


# ================= CLI 入口 =================
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "disease_long_texts_chunked.jsonl")

    MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    INPUT_FILE = os.getenv("INPUT_FILE", DEFAULT_INPUT_FILE)
    OUTPUT_FILE_CPU = os.getenv("OUTPUT_FILE_CPU")
    OUTPUT_FILE_GPU = os.getenv("OUTPUT_FILE_GPU")
    USE_GPU_FAST_PATH = os.getenv("USE_GPU_FAST_PATH", "true").lower() == "true"
    HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

    generator = EmbeddingGenerator(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        hf_endpoint=HF_ENDPOINT
    )
    generator.run(INPUT_FILE, OUTPUT_FILE_CPU, OUTPUT_FILE_GPU, USE_GPU_FAST_PATH)
