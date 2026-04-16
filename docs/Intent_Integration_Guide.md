
# 意图识别模块整合指南

## 概述

已成功将 QueryIntentClassification 项目中训练好的意图识别模型整合到 MedicalGraphRAG 项目中！

## 整合内容

### 新增文件

```
MedicalGraphRAG/
├── src/retrieval/intent_classification/    # 新增：意图识别模块
│   ├── __init__.py
│   └── pipeline.py             # 核心实现（关键词匹配 + 模型分类）
├── example_intent_classification.py       # 新增：使用示例
├── test_intent.py                          # 新增：测试文件
├── requirements-intent.txt                 # 新增：依赖文件
└── docs/Intent_Integration_Guide.md        # 本文档
```

### 功能特性

1. **关键词快速匹配**（Layer 1）
   - 纯 Python 实现，无需额外依赖
   - 支持 6 类医疗意图识别

2. **ClinicalBERT 模型分类**（Layer 2，可选）
   - 整合 QueryIntentClassification 训练好的模型
   - 支持 11 类 KUAKE-QIC 意图，自动映射到 6 类 MedicalGraphRAG 意图
   - 自动 GPU/CPU 切换

### 意图类别映射

| KUAKE-QIC 11类 | MedicalGraphRAG 6类 |
|----------------|---------------------|
| 病情诊断 | 症状查询 |
| 疾病表述 | 疾病查询 |
| 病因分析 | 疾病查询 |
| 后果表述 | 疾病查询 |
| 治疗方案 | 用药咨询 |
| 功效作用 | 用药咨询 |
| 注意事项 | 饮食/生活 |
| 指标解读 | 检查咨询 |
| 就医建议 | 其他 |
| 医疗费用 | 其他 |
| 其他 | 其他 |

## 快速开始

### 1. 安装依赖

```bash
# 关键词匹配模式无需额外依赖
# 如需使用模型分类：
pip install -r requirements-intent.txt
```

### 2. 基础使用（仅关键词匹配）

```python
from src.retrieval.intent_classification import create_pipeline

# 创建流水线（不加载模型）
pipeline = create_pipeline(
    checkpoint_path=None,
    enable_model=False
)

# 识别意图
result = pipeline("感冒了怎么办？")
print(f"意图: {result.intent_name}")
print(f"置信度: {result.confidence:.2f}")
```

### 3. 完整使用（关键词 + 模型）

```python
from src.retrieval.intent_classification import create_pipeline

# 模型 checkpoint 路径
CHECKPOINT_PATH = r"E:\Desktop\QueryIntentClassification\checkpoints\medical_intent_classification\best_model.ckpt"

# 创建流水线（加载模型）
pipeline = create_pipeline(
    checkpoint_path=CHECKPOINT_PATH,
    enable_model=True
)

# 识别意图
result = pipeline("感冒了怎么办？")
print(f"意图: {result.intent_name}")
print(f"置信度: {result.confidence:.2f}")
print(f"来源: {result.source}")
print(f"KUAKE标签: {result.kuake_label}")
```

## 运行示例

```bash
# 运行测试
python test_intent.py

# 运行完整示例
python example_intent_classification.py
```

## 支持的 6 类意图

| 意图类型 | 显示名称 | 说明 |
|---------|---------|------|
| symptom_inquiry | 症状查询 | 询问症状相关问题 |
| disease_inquiry | 疾病查询 | 询问疾病相关问题 |
| medication_inquiry | 用药咨询 | 询问用药相关问题 |
| examination_inquiry | 检查咨询 | 询问检查相关问题 |
| diet_lifestyle | 饮食/生活 | 询问饮食或生活注意事项 |
| other | 其他 | 无法归类的问题 |

## 与 RAG 流水线集成

```
用户查询
    ↓
[意图识别] ← 新增模块
    ↓
[根据意图调整检索策略]
    ↓
[NER 实体提取]
    ↓
[实体链接]
    ↓
[知识图谱检索 + 向量检索]  ← 根据意图动态调整权重
    ↓
[RAG 结果融合]
    ↓
[LLM 生成回答]
```

## 模型文件

训练好的模型位于：
```
E:\Desktop\QueryIntentClassification\checkpoints\medical_intent_classification\best_model.ckpt
```

**建议**：将模型文件复制到 `MedicalGraphRAG/models/intent_classification/` 目录下管理。

## 测试问题示例

| 问题 | 预期意图 |
|------|---------|
| "感冒了怎么办？" | 症状查询 |
| "高血压能吃什么药？" | 用药咨询 |
| "肺炎需要做什么检查？" | 检查咨询 |
| "糖尿病患者饮食要注意什么？" | 饮食/生活 |
| "看感冒挂什么科？" | 其他 |

## 注意事项

1. **模型路径**: 确保 `checkpoint_path` 指向正确的模型文件路径
2. **设备**: 自动使用 CUDA（如果可用），否则使用 CPU
3. **内存**: 模型加载需要约 2-3GB 内存
4. **依赖**: 使用模型前请安装 `torch` 和 `transformers`

## 下一步

- [ ] 将模型文件复制到项目目录
- [ ] 测试意图识别准确率
- [ ] 集成到完整的 RAG 流水线中
- [ ] 根据意图动态调整检索策略

