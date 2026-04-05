# milvus_writer.py
# import os
import json
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# ================= 配置日志 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MedicalMilvusWriter")


class MilvusWriter:
    """
    医学向量库写入器 (Milvus 版本)：将分块结果高效、幂等地写入 Milvus
    核心能力：Schema-first 设计、标量/向量索引预建、批量Upsert、失败重试与统计
    """

    # 需要建立标量索引的字段（严格对齐 Graph-RAG 检索需求）
    INDEXABLE_FIELDS = [
        "kg_id",
        "entity_type",
        "section_type",
        "version"
    ]

    def __init__(
            self,
            host: str = "localhost",
            port: int = 19530,
            token: Optional[str] = None,
            collection_name: str = "medical_kg_chunks",
            vector_dim: int = 1024,  # BGE-M3 默认维度
            batch_size: int = 500,
            index_type: str = "HNSW",  # HNSW 或 IVF_FLAT
            metric_type: str = "COSINE"
    ):
        self.host = host
        self.port = port
        self.token = token
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.index_type = index_type
        self.metric_type = metric_type
        self.collection = None
        self.failed_chunks = []

    def connect(self):
        """连接 Milvus"""
        from pymilvus import connections
        logger.info(f"🔌 连接 Milvus: {self.host}:{self.port}")
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            token=self.token
        )

    def disconnect(self):
        """断开 Milvus 连接"""
        from pymilvus import connections
        connections.disconnect("default")
        logger.info("🔌 已断开 Milvus 连接")

    def _build_schema(self):
        """构建 Milvus Collection Schema"""
        from pymilvus import CollectionSchema, FieldSchema, DataType

        fields = [
            # 主键：chunk_id (格式: {kg_id}__{section_type}__{idx})
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=512,
                description="Chunk 唯一标识"
            ),
            # 向量字段
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.vector_dim,
                description="文本向量 (BGE-M3)"
            ),
            # 图谱关联字段
            FieldSchema(
                name="kg_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="知识图谱节点 ID"
            ),
            FieldSchema(
                name="entity_type",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="实体类型 (如 Disease)"
            ),
            FieldSchema(
                name="section_type",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="章节类型 (如 常见症状)"
            ),
            # 数组字段用 JSON 存储
            FieldSchema(
                name="involved_relations",
                dtype=DataType.JSON,
                description="关联的图谱关系类型"
            ),
            FieldSchema(
                name="safety_flags",
                dtype=DataType.JSON,
                description="安全合规标记"
            ),
            # 元数据字段
            FieldSchema(
                name="cured_prob",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="治愈率",
                nullable=True
            ),
            FieldSchema(
                name="yibao_status",
                dtype=DataType.VARCHAR,
                max_length=32,
                description="医保状态",
                nullable=True
            ),
            FieldSchema(
                name="category_path",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="分类路径",
                nullable=True
            ),
            FieldSchema(
                name="version",
                dtype=DataType.VARCHAR,
                max_length=32,
                description="数据版本",
                default_value="v1.0"
            ),
            # 原文用于检索后展示
            FieldSchema(
                name="chunk_text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Chunk 原文"
            ),
            # 完整 metadata 备份
            FieldSchema(
                name="metadata_json",
                dtype=DataType.JSON,
                description="完整元数据 JSON"
            )
        ]

        return CollectionSchema(
            fields=fields,
            description="医学知识图谱 Chunk 向量库",
            enable_dynamic_field=False
        )

    def _get_vector_index_params(self) -> Dict:
        """获取向量索引参数"""
        if self.index_type == "HNSW":
            return {
                "metric_type": self.metric_type,
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
        elif self.index_type == "IVF_FLAT":
            return {
                "metric_type": self.metric_type,
                "index_type": "IVF_FLAT",
                "params": {
                    "nlist": 1024
                }
            }
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

    def ensure_collection(self, recreate: bool = False):
        """创建集合并预建索引（必须在批量写入前调用）"""
        from pymilvus import utility, Collection

        if recreate and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"🗑️ 已删除旧集合: {self.collection_name}")

        if not utility.has_collection(self.collection_name):
            logger.info(f"📦 创建集合: {self.collection_name} | 维度: {self.vector_dim} | 索引: {self.index_type}")
            schema = self._build_schema()
            self.collection = Collection(name=self.collection_name, schema=schema)

            # 创建向量索引
            vector_index_params = self._get_vector_index_params()
            self.collection.create_index(
                field_name="embedding",
                index_params=vector_index_params
            )
            logger.info(f"✅ 向量索引已创建: {vector_index_params}")

            # 创建标量索引
            for field in self.INDEXABLE_FIELDS:
                try:
                    self.collection.create_index(field_name=field)
                    logger.info(f"✅ 标量索引已创建: {field}")
                except Exception as e:
                    logger.debug(f"标量索引 {field} 已存在或跳过: {e}")
        else:
            logger.info(f"✅ 集合已存在: {self.collection_name}，跳过创建")
            self.collection = Collection(self.collection_name)

    def _validate_and_convert(self, chunk: Dict) -> Optional[Dict]:
        """校验字段完整性，转换为 Milvus 行格式"""
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

        metadata = chunk.get("metadata", {})

        return {
            "chunk_id": chunk_id,
            "embedding": emb,
            "kg_id": chunk["kg_id"],
            "entity_type": chunk["entity_type"],
            "section_type": chunk["section_type"],
            "involved_relations": chunk.get("involved_relations", []),
            "safety_flags": metadata.get("safety_flags", []),
            "cured_prob": metadata.get("cured_prob"),
            "yibao_status": metadata.get("yibao_status"),
            "category_path": metadata.get("category_path"),
            "version": metadata.get("version", "v1.0"),
            "chunk_text": chunk["text"],
            "metadata_json": metadata
        }

    def _upsert_batch(self, chunks: List[Dict]):
        """批量 Upsert（Milvus insert 天然是 upsert 语义：主键冲突则覆盖）"""
        try:
            # 转换为列格式 (Milvus 需要按列插入)
            column_data = {
                "chunk_id": [],
                "embedding": [],
                "kg_id": [],
                "entity_type": [],
                "section_type": [],
                "involved_relations": [],
                "safety_flags": [],
                "cured_prob": [],
                "yibao_status": [],
                "category_path": [],
                "version": [],
                "chunk_text": [],
                "metadata_json": []
            }

            for c in chunks:
                column_data["chunk_id"].append(c["chunk_id"])
                column_data["embedding"].append(c["embedding"])
                column_data["kg_id"].append(c["kg_id"])
                column_data["entity_type"].append(c["entity_type"])
                column_data["section_type"].append(c["section_type"])
                column_data["involved_relations"].append(c["involved_relations"])
                column_data["safety_flags"].append(c["safety_flags"])
                column_data["cured_prob"].append(c["cured_prob"])
                column_data["yibao_status"].append(c["yibao_status"])
                column_data["category_path"].append(c["category_path"])
                column_data["version"].append(c["version"])
                column_data["chunk_text"].append(c["chunk_text"])
                column_data["metadata_json"].append(c["metadata_json"])

            # 按 schema 字段顺序排列
            ordered_data = [column_data[field.name] for field in self.collection.schema.fields]

            # 执行插入
            self.collection.insert(ordered_data)
            logger.debug(f"批量 Upsert 完成: {len(chunks)} 条")

        except Exception as e:
            logger.error(f"❌ 批次写入失败 ({len(chunks)} 条): {e}")
            self.failed_chunks.extend([{"chunk_id": c["chunk_id"], "error": str(e)} for c in chunks])

    def run(self, input_file: str):
        """批量处理 JSONL 分块结果，写入 Milvus"""
        import os
        if not os.path.exists(input_file):
            logger.error(f"❌ 输入文件不存在: {input_file}")
            return

        # 连接并确保集合存在
        self.connect()
        import os
        recreate = os.getenv("RECREATE_COLLECTION", "false").lower() == "true"
        self.ensure_collection(recreate=recreate)

        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        processed = 0
        inserted = 0
        skipped = 0

        logger.info(f"🚀 开始向量入库 | 文件: {input_file} | 批次: {self.batch_size}")

        try:
            with open(input_file, 'r', encoding='utf-8') as f, \
                    tqdm(total=total_lines, desc="写入进度") as pbar:

                batch_chunks = []
                for line in f:
                    line = line.strip()
                    if not line:
                        pbar.update(1)
                        continue

                    try:
                        chunk = json.loads(line)
                        converted = self._validate_and_convert(chunk)
                        if converted:
                            batch_chunks.append(converted)
                            inserted += 1
                        else:
                            skipped += 1
                    except json.JSONDecodeError:
                        logger.warning("⚠️ JSON 解析失败，跳过")
                        skipped += 1

                    processed += 1
                    pbar.update(1)

                    # 达到批次大小，执行 Upsert
                    if len(batch_chunks) >= self.batch_size:
                        self._upsert_batch(batch_chunks)
                        batch_chunks.clear()

                # 写入剩余批次
                if batch_chunks:
                    self._upsert_batch(batch_chunks)

            # 最后 flush 确保数据持久化
            self.collection.flush()
            logger.info("💾 数据已刷新到磁盘")

        except Exception as e:
            logger.error(f"❌ 入库中断: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"✅ 入库完成 | 处理: {processed} | 成功: {inserted} | 跳过: {skipped}")
            if self.failed_chunks:
                failed_path = os.path.join(os.path.dirname(input_file), "milvus_failed_chunks.jsonl")
                logger.warning(f"⚠️ 失败 Chunk 列表已保存至 {failed_path}")
                with open(failed_path, "w", encoding="utf-8") as wf:
                    for fc in self.failed_chunks:
                        wf.write(json.dumps(fc, ensure_ascii=False) + "\n")
            self.disconnect()


# ================= CLI 入口 =================
if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "disease_long_texts_chunked_embeddings.jsonl")

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_kg_chunks")
    INPUT_FILE = os.getenv("INPUT_FILE", DEFAULT_INPUT_FILE)
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))
    INDEX_TYPE = os.getenv("INDEX_TYPE", "HNSW")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))

    writer = MilvusWriter(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        token=MILVUS_TOKEN,
        collection_name=COLLECTION_NAME,
        vector_dim=VECTOR_DIM,
        batch_size=BATCH_SIZE,
        index_type=INDEX_TYPE
    )
    writer.run(INPUT_FILE)
