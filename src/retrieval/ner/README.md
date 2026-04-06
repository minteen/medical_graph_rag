# NER 关键词匹配层 (Layer 1)

## 概述

这是三层 NER 方案的第一层：**关键词匹配层**，提供高准确率、高速度的实体识别。

## 文件结构

```
src/retrieval/
├── __init__.py          # 模块入口
├── keyword_matcher.py   # 关键词匹配核心
├── entity_fuser.py      # 结果融合（预留 Layer 2/3）
├── pipeline.py          # NER Pipeline 总入口
└── README.md            # 本文档
```

## 快速开始

### 方式一：使用 Pipeline（推荐）

```python
from src.retrieval import create_pipeline

# 初始化
pipeline = create_pipeline()

# 提取实体
results = pipeline.extract("感冒发烧了怎么办？")

for r in results:
    print(r["type"], r["text"])
```

### 方式二：直接使用 KeywordMatcher

```python
from src.retrieval import KeywordMatcher

matcher = KeywordMatcher()
matcher.load_entities()

matches = matcher.match("肺炎需要做哪些检查？")
for m in matches:
    print(f"[{m.type}] {m.text}")
```

## 已支持的实体类型

| 类型 | 说明 | 数量 |
|-----|------|------|
| Disease | 疾病 | 8807 |
| Drug | 药品 | 3828 |
| Food | 食物 | 4870 |
| Check | 检查项目 | 3353 |
| Department | 科室 | 54 |
| Symptom | 症状 | 5998 |
| Producer | 生产商 | 17201 |
| Deny | 否定词 | 37 |

## 核心特性

### 1. 双引擎支持
- **Aho-Corasick 自动机**（推荐，需安装 `pyahocorasick`）
- **前缀树**（纯 Python，无需额外依赖）

### 2. 智能冲突消解
- 最长匹配优先
- 类型优先级：Drug > Disease > Symptom > Check > Food > ...
- 同一位置只保留最高优先级结果

### 3. 噪声过滤
- 自动过滤单英文字符误匹配
- 自动过滤纯数字
- 单字实体（如否定词）严格边界检查

## 性能优化建议

### 安装 Aho-Corasick（可选，推荐）

```bash
pip install pyahocorasick
```

可获得 10-100x 速度提升。

### 示例

项目根目录下有示例脚本：

```bash
python example_ner.py          # 简单示例
python test/test_ner.py        # 完整测试
```

## 后续扩展

### Layer 2: NER 模型（待实现）
- 使用 BERT/ChineseBERT 微调医疗数据
- 处理同义词、缩写、组合词

### Layer 3: LLM 提取（待实现）
- 使用大模型处理复杂句式、歧义、口语化表达
- 降低调用频率（目标 < 10%）
