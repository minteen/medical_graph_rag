# graph_to_text_builder.py
import os
import json
import logging
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase
from tqdm import tqdm



# ================= 配置日志 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GraphToTextBuilder")


class GraphToTextBuilder:
    """
    知识图谱线性化构建器：将图谱节点+1跳关系转换为结构化长文本
    当前优先支持 Disease 实体，架构设计支持平滑扩展至 Drug/Symptom/Food 等
    """

    # 医学知识卡片模板（严格对齐提供的 Schema）
    DISEASE_TEMPLATE = """【疾病档案】{name}
【分类路径】{category}
【疾病简介】{desc}
【发病原因】{cause}
【易感人群】{easy_get}
【预防措施】{prevent}
【治疗周期】{cure_lasttime} | 【临床治愈率】{cured_prob}
【医保状态】{yibao_status}

【常见症状】{symptoms}
【并发疾病】{acompany_diseases}
【推荐检查】{need_checks}
【治疗方案】{cure_ways}
【临床推荐用药】{recommand_drugs}
【常用辅助药物】{common_drugs}
【饮食宜吃】{do_eat_foods}
【饮食禁忌】{no_eat_foods}
【推荐食谱】{recommand_recipes}"""

    def __init__(self, uri: str, user: str, password: str, output_dir: str, batch_size: int = 500):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.output_dir = output_dir
        self.batch_size = batch_size
        os.makedirs(output_dir, exist_ok=True)
        self.template = self.DISEASE_TEMPLATE

    # ---------------- 内部工具方法 ----------------
    def _clean_text(self, value: Any, default: str = "暂无相关信息") -> str:
        """统一清洗字段：处理 None/空列表/换行/首尾空格"""
        if isinstance(value, list):
            clean_items = [str(v).strip().replace("\n", " ") for v in value if v and str(v).strip()]
            return "、".join(clean_items) if clean_items else default
        if isinstance(value, str):
            cleaned = value.strip().replace("\n", " ")
            return cleaned if cleaned else default
        if value is None:
            return default
        return str(value).strip()

    # ---------------- 图谱查询 ----------------
    def _get_total_count(self) -> int:
        with self.driver.session() as session:
            res = session.run("MATCH (d:Disease) RETURN count(d) as cnt")
            return res.single()["cnt"]

    def _fetch_all_disease_ids(self) -> List[str]:
        """先获取所有 Disease 节点的 ID，用于高效分页"""
        query = """
        MATCH (d:Disease)
        RETURN COALESCE(d.id, elementId(d)) AS kg_id
        ORDER BY kg_id
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record["kg_id"] for record in result]

    def _fetch_disease_batch(self, kg_ids: List[str]) -> List[Dict]:
        """根据 ID 列表批量拉取疾病属性及1跳关系（避免 SKIP 性能问题）"""
        query = """
        MATCH (d:Disease)
        WHERE COALESCE(d.id, elementId(d)) IN $kg_ids
        WITH d
        OPTIONAL MATCH (d)-[:has_symptom]->(s:Symptom)
        WITH d, collect(DISTINCT s.name) AS symptoms
        OPTIONAL MATCH (d)-[:acompany_with]->(a:Disease)
        WITH d, symptoms, collect(DISTINCT a.name) AS acompany_diseases
        OPTIONAL MATCH (d)-[:need_check]->(c:Check)
        WITH d, symptoms, acompany_diseases, collect(DISTINCT c.name) AS need_checks
        OPTIONAL MATCH (d)-[:cure_way]->(cu:Cure)
        WITH d, symptoms, acompany_diseases, need_checks, collect(DISTINCT cu.name) AS cure_ways
        OPTIONAL MATCH (d)-[:recommand_drug]->(rd:Drug)
        WITH d, symptoms, acompany_diseases, need_checks, cure_ways, collect(DISTINCT rd.name) AS recommand_drugs
        OPTIONAL MATCH (d)-[:common_drug]->(cd:Drug)
        WITH d, symptoms, acompany_diseases, need_checks, cure_ways, recommand_drugs, collect(DISTINCT cd.name) AS common_drugs
        OPTIONAL MATCH (d)-[:do_eat]->(ef:Food)
        WITH d, symptoms, acompany_diseases, need_checks, cure_ways, recommand_drugs, common_drugs, collect(DISTINCT ef.name) AS do_eat_foods
        OPTIONAL MATCH (d)-[:no_eat]->(nf:Food)
        WITH d, symptoms, acompany_diseases, need_checks, cure_ways, recommand_drugs, common_drugs, do_eat_foods, collect(DISTINCT nf.name) AS no_eat_foods
        OPTIONAL MATCH (d)-[:recommand_eat]->(re:Food)
        RETURN
            COALESCE(d.id, elementId(d)) AS kg_id,
            d.name AS name,
            d.desc AS desc,
            d.cause AS cause,
            d.prevent AS prevent,
            d.cure_lasttime AS cure_lasttime,
            d.cured_prob AS cured_prob,
            d.easy_get AS easy_get,
            d.yibao_status AS yibao_status,
            d.category AS category,
            symptoms,
            acompany_diseases,
            need_checks,
            cure_ways,
            recommand_drugs,
            common_drugs,
            do_eat_foods,
            no_eat_foods,
            collect(DISTINCT re.name) AS recommand_recipes
        """
        with self.driver.session() as session:
            return session.run(query, kg_ids=kg_ids).data()

    # ---------------- 文本生成 ----------------
    def build_document(self, record: Dict) -> str:
        """将单条图谱记录渲染为医学长文本"""
        ctx = {
            "name": self._clean_text(record.get("name")),
            "category": self._clean_text(record.get("category")),
            "desc": self._clean_text(record.get("desc")),
            "cause": self._clean_text(record.get("cause")),
            "easy_get": self._clean_text(record.get("easy_get")),
            "prevent": self._clean_text(record.get("prevent")),
            "cure_lasttime": self._clean_text(record.get("cure_lasttime")),
            "cured_prob": self._clean_text(record.get("cured_prob")),
            "yibao_status": self._clean_text(record.get("yibao_status", "未知")),
            "symptoms": self._clean_text(record.get("symptoms")),
            "acompany_diseases": self._clean_text(record.get("acompany_diseases")),
            "need_checks": self._clean_text(record.get("need_checks")),
            "cure_ways": self._clean_text(record.get("cure_ways")),
            "recommand_drugs": self._clean_text(record.get("recommand_drugs")),
            "common_drugs": self._clean_text(record.get("common_drugs")),
            "do_eat_foods": self._clean_text(record.get("do_eat_foods")),
            "no_eat_foods": self._clean_text(record.get("no_eat_foods")),
            "recommand_recipes": self._clean_text(record.get("recommand_recipes")),
        }
        return self.template.format(**ctx)

    # ---------------- 主流程 ----------------
    def run(self):
        total = self._get_total_count()
        if total == 0:
            logger.warning("图谱中未找到 Disease 节点，请检查数据导入状态。")
            return

        logger.info(f"📋 正在获取所有疾病节点 ID...")
        all_kg_ids = self._fetch_all_disease_ids()
        if len(all_kg_ids) != total:
            logger.warning(f"ID 数量 ({len(all_kg_ids)}) 与总数 ({total}) 不一致，使用 ID 数量")
            total = len(all_kg_ids)

        output_file = os.path.join(self.output_dir, "disease_long_texts.jsonl")
        processed = 0

        logger.info(f"🚀 开始线性化构建 | 总计 {total} 个疾病 | 批次大小 {self.batch_size}")

        try:
            with open(output_file, "w", encoding="utf-8") as f, tqdm(total=total, desc="生成进度") as pbar:
                for i in range(0, total, self.batch_size):
                    batch_ids = all_kg_ids[i:i + self.batch_size]
                    batch = self._fetch_disease_batch(batch_ids)

                    if not batch:
                        logger.warning(f"批次 {i//self.batch_size + 1} 未获取到数据，跳过")
                        continue

                    for record in batch:
                        doc_text = self.build_document(record)
                        kg_id = record.get("kg_id", f"disease_{processed}")

                        # 输出格式严格对齐下游 Chunking & Vectorization 管线
                        output_record = {
                            "kg_id": kg_id,
                            "entity_type": "Disease",
                            "raw_text": doc_text,
                            "metadata": {
                                "name": record.get("name"),
                                "category": record.get("category"),
                                "cured_prob": record.get("cured_prob"),
                                "yibao_status": record.get("yibao_status"),
                                "source": "graph_linearization",
                                "version": "v1.0"
                            }
                        }
                        f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                        processed += 1
                        pbar.update(1)

                    logger.debug(f"已写入 {processed}/{total} 条")

        except Exception as e:
            logger.error(f"❌ 构建中断: {e}", exc_info=True)
            raise
        finally:
            self.driver.close()
            logger.info(f"✅ 构建完成！共处理 {processed} 条疾病记录。\n📁 输出路径: {output_file}")


# ================= CLI 入口 =================
if __name__ == "__main__":
    # 可通过环境变量或 .env 文件加载，此处为示例硬编码
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

    # 动态获取项目根目录，确保输出到 data/processed
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

    builder = GraphToTextBuilder(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASS,
        output_dir=OUTPUT_DIR,
        batch_size=500  # 8808 条疾病，500 为较优批次
    )
    builder.run()