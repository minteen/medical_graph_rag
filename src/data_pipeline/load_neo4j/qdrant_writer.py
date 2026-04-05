# qdrant_writer.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, FieldIndexParams
)
from tqdm import tqdm

# ================= 配置日志 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MedicalVectorWriter")


class MedicalVectorWriter:
    """
    医学向量库写入器：将分块结果高效、幂等地写入向量数据库
    核心能力：严格Schema校验、载荷索引预建、批量Upsert、失败重试与统计
    """
    # 需要建立过滤索引的 Payload 字段（严格对齐 Graph-RAG 检索需求）
    FILTERABLE_FIELDS = {
        "entity_type": PayloadSchemaType.KEYWORD,
        "section_type": PayloadSchemaType.KEYWORD,
        "kg_id": PayloadSchemaType.KEYWORD,
        "version": PayloadSchemaType.KEYWORD,
        "safety_flags": PayloadSchemaType.KEYWORD,  # 数组自动展开索引
        "involved_relations": PayloadSchemaType.KEYWORD  # 数组自动展开索引
    }

    def __init__(
            self,
            url: str = "http://localhost:6333",
            api_key: Optional[str] = None,
            collection_name: str = "medical_kg_chunks",
            vector_dim: int = 1024,  # BGE-M3 默认维度
            batch_size: int = 1000
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.failed_chunks = []

    # ================= 集合与索引管理 =================
    def ensure_collection(self, recreate: bool = False):
        """创建集合并预建载荷过滤索引（必须在批量写入前调用）"""
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️ 已删除旧集合: {self.collection_name}")

        if not self.client.collection_exists(self.collection_name):
            logger.info(f"📦 创建集合: {self.collection_name} | 维度: {self.vector_dim}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "default_segment_number": 4,
                    "memmap_threshold": 20000  # 超过2万条切换内存映射，节省RAM
                }
            )
        else:
            logger.info(f"✅ 集合已存在: {self.collection_name}，跳过创建")

        # 预建 Payload 索引（大幅提升后续带条件检索的速度）
        self._create_payload_indexes()

    def _create_payload_indexes(self):
        for field, schema_type in self.FILTERABLE_FIELDS.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=FieldIndexParams(type=schema_type)
                )
            except Exception:
                logger.debug(f"索引已存在或跳过: {field}")

    # ================= 数据校验与转换 =================
    def _validate_and_convert(self, chunk: Dict) -> Optional[PointStruct]:
        """校验字段完整性，转换为 Qdrant PointStruct"""
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            logger.warning("⚠️ 跳过无 chunk_id 的记录")
            return None

        emb = chunk.get("embedding")
        if not emb or not isinstance(emb, list):
            logger.warning(f"⚠️ {chunk_id} 缺失或格式错误的 embedding")
            return None
        if len(emb) != self.vector_dim:
            logger.error(f"❌ {chunk_id} embedding 维度不匹配: {len(emb)} != {self.vector_dim}")
            return None

        # 提取 Payload（剥离原始文本与大数组，仅保留检索必需元数据）
        payload = {
            "kg_id": chunk["kg_id"],
            "entity_type": chunk["entity_type"],
            "section_type": chunk["section_type"],
            "involved_relations": chunk.get("involved_relations", []),
            "safety_flags": chunk.get("safety_flags", []),
            "cured_prob": chunk["metadata"].get("cured_prob"),
            "yibao_status": chunk["metadata"].get("yibao_status"),
            "category_path": chunk["metadata"].get("category_path"),
            "version": chunk["metadata"].get("version", "v1.0"),
            "chunk_text": chunk["text"]  # 保留原文供检索后展示
        }

        return PointStruct(
            id=chunk_id,
            vector=emb,
            payload=payload
        )

    # ================= 核心写入流程 =================
    def run(self, input_file: str):
        if not os.path.exists(input_file):
            logger.error(f"❌ 输入文件不存在: {input_file}")
            return

        self.ensure_collection()

        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        processed = 0
        inserted = 0
        skipped = 0

        logger.info(f"🚀 开始向量入库 | 文件: {input_file} | 批次: {self.batch_size}")

        try:
            with open(input_file, 'r', encoding='utf-8') as f, \
                    tqdm(total=total_lines, desc="写入进度") as pbar:

                batch_points = []
                for line in f:
                    line = line.strip()
                    if not line:
                        pbar.update(1)
                        continue

                    try:
                        chunk = json.loads(line)
                        point = self._validate_and_convert(chunk)
                        if point:
                            batch_points.append(point)
                            inserted += 1
                        else:
                            skipped += 1
                    except json.JSONDecodeError:
                        logger.warning("⚠️ JSON 解析失败，跳过")
                        skipped += 1

                    processed += 1
                    pbar.update(1)

                    # 达到批次大小，执行 Upsert
                    if len(batch_points) >= self.batch_size:
                        self._upsert_batch(batch_points)
                        batch_points.clear()

                # 写入剩余批次
                if batch_points:
                    self._upsert_batch(batch_points)

        except Exception as e:
            logger.error(f"❌ 入库中断: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"✅ 入库完成 | 处理: {processed} | 成功: {inserted} | 跳过: {skipped}")
            if self.failed_chunks:
                logger.warning(f"⚠️ 失败 Chunk 列表已保存至 ./failed_chunks.jsonl")
                with open("./failed_chunks.jsonl", "w", encoding="utf-8") as wf:
                    for fc in self.failed_chunks:
                        wf.write(json.dumps(fc, ensure_ascii=False) + "\n")

    def _upsert_batch(self, points: List[PointStruct]):
        try:
            # Qdrant upsert 天然幂等（按 ID 覆盖），适合增量更新
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # 生产环境可改为 wait=False 提升吞吐，但需确认一致性策略
            )
        except Exception as e:
            logger.error(f"❌ 批次写入失败 ({len(points)} 条): {e}")
            self.failed_chunks.extend([{"id": p.id, "error": str(e)} for p in points])


# ================= CLI 入口 =================
if __name__ == "__main__":
    # 推荐通过 .env 加载
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_kg_chunks")
    INPUT_FILE = os.getenv("INPUT_FILE", "./data/processed/disease_long_texts_chunked.jsonl")
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

    writer = MedicalVectorWriter(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        vector_dim=VECTOR_DIM,
        batch_size=1500
    )
    writer.run(INPUT_FILE)