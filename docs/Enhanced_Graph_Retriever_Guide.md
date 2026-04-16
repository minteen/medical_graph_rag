
# 增强版图谱检索器使用指南

## 概述

增强版图谱检索器（EnhancedGraphRetriever）整合了 NER、意图识别和图谱检索功能，能够：
1. 自动通过 NER 识别实体
2. 通过意图识别确定查询模板
3. 执行图谱检索
4. 返回自然语言数据

## 文件结构

```
MedicalGraphRAG/
├── src/retrieval/graph_retriever/
│   ├── __init__.py              # 更新：导出增强版检索器
│   ├── base.py                  # 基础数据结构
│   ├── retriever.py             # 基础图谱检索器
│   └── enhanced_retriever.py    # 新增：增强版图谱检索器
└── example_enhanced_graph_retriever.py  # 新增：使用示例
```

## 快速开始

### 基础使用（直接提供实体和意图）

```python
from src.retrieval.graph_retriever import (
    create_graph_retriever,
    create_enhanced_graph_retriever
)

# 创建基础图谱检索器
graph_retriever = create_graph_retriever()

# 创建增强版检索器
enhanced_retriever = create_enhanced_graph_retriever(
    graph_retriever=graph_retriever,
    ner_pipeline=None,
    intent_classifier=None
)

# 直接提供实体和意图进行检索
entities = [
    {"kg_id": "感冒", "type": "Disease", "text": "感冒"},
    {"kg_id": "发热", "type": "Symptom", "text": "发热"}
]

result = enhanced_retriever.retrieve(
    query="感冒了怎么办？",
    entities=entities,
    intent_type="symptom_inquiry"
)

print(result["natural_language"])
```

### 完整流程（NER + 意图识别 + 图谱检索）

```python
from src.retrieval.graph_retriever import (
    create_graph_retriever,
    create_enhanced_graph_retriever
)
from src.retrieval.ner import create_pipeline as create_ner_pipeline
from src.retrieval.intent_classification import create_pipeline as create_intent_pipeline

# 创建各组件
graph_retriever = create_graph_retriever()
ner_pipeline = create_ner_pipeline(enable_ner=True)
intent_classifier = create_intent_pipeline(
    checkpoint_path=r"E:\Desktop\QueryIntentClassification\checkpoints\medical_intent_classification\best_model.ckpt",
    enable_model=True
)

# 创建增强版检索器
enhanced_retriever = create_enhanced_graph_retriever(
    graph_retriever=graph_retriever,
    ner_pipeline=ner_pipeline,
    intent_classifier=intent_classifier
)

# 执行完整检索流程
result = enhanced_retriever.retrieve(query="感冒了怎么办？")

# 查看结果
print(f"识别到的实体: {[e.get('text', '') for e in result['entities']]}")
print(f"识别到的意图: {result['intent_name']}")
print(f"自然语言结果:\n{result['natural_language']}")

# 转换为 LLM 友好的上下文
context = enhanced_retriever.to_text_context(result)
print(f"\nLLM 上下文:\n{context}")
```

## 意图查询模板

系统为每个意图预设了查询模板，配置在 `INTENT_QUERY_TEMPLATES` 中：

| 意图类型 | 关系类型 | 跳数 | 说明 |
|---------|---------|------|------|
| symptom_inquiry | has_symptom, accompany_with, recommand_drug, cure_way | 1 | 症状查询 |
| disease_inquiry | has_symptom, recommand_drug, common_drug, need_check, cure_way, do_eat, no_eat, accompany_with | 2 | 疾病查询 |
| medication_inquiry | recommand_drug, common_drug, drugs_of, cure_way | 1 | 用药咨询 |
| examination_inquiry | need_check | 1 | 检查咨询 |
| diet_lifestyle | do_eat, no_eat, recommand_eat | 1 | 饮食/生活 |
| other | (全部关系) | 1 | 其他 |

## 检索结果结构

`retrieve()` 方法返回的字典结构：

```python
{
    "query": "用户查询",
    "entities": [  # 识别到的实体列表
        {
            "text": "实体文本",
            "type": "实体类型",
            "kg_id": "图谱ID",
            # ... 其他 NER 结果字段
        }
    ],
    "intent_type": "symptom_inquiry",  # 意图类型
    "intent_name": "症状查询",         # 意图名称
    "subgraph": SubgraphResult,        # 子图结果对象
    "natural_language": "自然语言描述...",  # 自然语言结果
    "success": True                    # 是否成功
}
```

## 运行示例

```bash
# 查看意图查询模板和基础使用
python example_enhanced_graph_retriever.py
```

## 工作流程

```
用户查询
    ↓
[Step 1] NER 实体提取 (如果提供了 ner_pipeline)
    ↓
[Step 2] 意图识别 (如果提供了 intent_classifier)
    ↓
[Step 3] 获取意图查询模板
    ↓
[Step 4] 执行图谱检索
    ↓
[Step 5] 生成自然语言描述
    ↓
返回结果
```

## 自定义查询模板

可以自定义或修改查询模板：

```python
from src.retrieval.graph_retriever import (
    IntentQueryTemplate,
    INTENT_QUERY_TEMPLATES,
    RelationType
)

# 创建自定义模板
custom_template = IntentQueryTemplate(
    intent_type="custom_inquiry",
    description="自定义查询",
    relations=[
        RelationType.HAS_SYMPTOM.value,
        RelationType.RECOMMEND_DRUG.value
    ],
    max_hops=1,
    prompt_template="自定义模板：\n{knowledge}"
)

# 替换默认模板
INTENT_QUERY_TEMPLATES["custom_inquiry"] = custom_template
```

