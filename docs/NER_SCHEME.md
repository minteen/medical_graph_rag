# 医疗NER技术方案：关键词匹配 + NER模型 + LLM提取

## 一、方案概述

### 1.1 设计目标
- **准确率优先**：医疗领域对实体识别准确率要求极高
- **分层处理**：简单问题快速处理，复杂问题深度处理
- **可扩展**：支持新增实体类型和新兴医学术语

### 1.2 三层架构
```
Layer 1: 关键词匹配 (速度层) → 覆盖80%常见查询
Layer 2: NER模型 (泛化层) → 处理15%变体表达
Layer 3: LLM提取 (智能层) → 解决5%复杂/歧义问题
```

---

## 二、各层详细设计

### Layer 1: 关键词匹配层

#### 技术选型
- **算法**：Aho-Corasick 自动机（多模式匹配）
- **实现**：`pyahocorasick` 或 `flash-text`
- **响应时间**：< 1ms

#### 实体词典构建
```python
# 实体分层结构
ENTITY_DICT = {
    "Disease": {
        "急性肺脓肿": {"id": "xxx", "alias": ["肺脓肿"]},
        ...
    },
    "Drug": {...},
    "Symptom": {...},
    ...
}
```

#### 匹配策略
1. **精确匹配**：完全命中词典
2. **前缀/后缀匹配**：如"新冠肺炎"匹配"新型冠状病毒肺炎"
3. **同义词匹配**：使用别名表

#### 输出格式
```python
{
    "text": "急性肺脓肿",
    "type": "Disease",
    "start": 0,
    "end": 5,
    "kg_id": "4:0",
    "source": "keyword_match",
    "confidence": 1.0
}
```

---

### Layer 2: NER模型层

#### 模型选型
| 模型 | 优点 | 缺点 |
|-----|------|------|
| **ChineseBERT-Med** | 医疗领域预训练 | 需微调 |
| **BERT-base-chinese** | 通用，易获取 | 需要医疗数据微调 |
| **BiLSTM-CRF** | 轻量，可控 | 特征工程多 |
| **ERNIE-Health** | 百度医疗预训练 | 闭源/API调用 |

#### 推荐方案
**首选**：使用 `shibing624/bert-base-chinese-ner` 微调医疗数据

#### 微调数据生成（利用现有图谱）
```python
# 远程监督生成训练数据
模板1：[Disease]的常见症状有[Symptom]
模板2：[Disease]推荐使用[Drug]治疗
模板3：[Symptom]可能是[Disease]的表现
```

#### 模型输入输出
```python
输入："感冒可以吃对乙酰氨基酚吗？"
输出：[
    {"entity": "感冒", "type": "Disease", "start": 0, "end": 2, "confidence": 0.95},
    {"entity": "对乙酰氨基酚", "type": "Drug", "start": 6, "end": 12, "confidence": 0.98}
]
```

---

### Layer 3: LLM提取层

#### 适用场景
1. **复杂句式**："我最近老是感觉头晕乎乎的，还有点恶心"
2. **歧义消解**："感冒灵"可能指药品也可能指保健品
3. **新型实体**：未收录的新型疾病/药物
4. **口语化表达**："三高"、"脑梗"

#### Prompt设计
```python
SYSTEM_PROMPT = """你是专业的医学实体识别助手。请从用户输入中提取以下类型的实体：
- Disease: 疾病名称
- Drug: 药品名称
- Symptom: 症状描述
- Food: 食物名称
- Check: 检查项目
- Department: 科室
- Cure: 治疗方法

输出格式：JSON数组，每个元素包含：text, type, start, end, note

只输出JSON，不要其他文字。"""
```

#### 模型选择
- **本地部署**：Qwen-7B-Chat / ChatGLM3-6B
- **API调用**：Doubao / GPT-4 / Claude
- **轻量级**：使用 API 更划算

#### 成本控制
- 设置置信度门限：Layer 2 置信度 > 0.9 不调用 LLM
- 缓存机制：相同问题直接复用结果
- 输入截断：超长文本只送关键片段

---

## 三、结果融合策略

### 3.1 融合流程
```
各层结果 → 去重 → 冲突消解 → 置信度校准 → 图谱链接 → 最终输出
```

### 3.2 冲突消解规则
| 情况 | 处理方式 |
|-----|---------|
| 同位置不同类型 | 优先 Layer1 > Layer2 > Layer3 |
| 重叠实体 | 保留最长匹配 |
| 置信度差异 > 0.3 | 取高置信度结果 |

### 3.3 置信度校准
```python
def calibrate_confidence(result):
    base_score = {
        "keyword_match": 1.0,
        "ner_model": result["confidence"],
        "llm_extract": 0.85
    }
    # 如果能链接到知识图谱，加分
    if has_kg_link(result):
        return min(base_score[result["source"]] + 0.1, 1.0)
    return base_score[result["source"]]
```

---

## 四、实现路线图

### Phase 1: 关键词匹配层（1-2天）
- [ ] 加载实体词典，构建AC自动机
- [ ] 实现基础匹配和结果格式化
- [ ] 单元测试：准确率 90%+

### Phase 2: NER模型层（3-5天）
- [ ] 准备微调数据（远程监督生成）
- [ ] 微调医疗NER模型
- [ ] 集成到 pipeline

### Phase 3: LLM提取层（2-3天）
- [ ] 设计Prompt模板
- [ ] 封装LLM调用接口
- [ ] 实现结果解析

### Phase 4: 融合与优化（2-3天）
- [ ] 实现结果融合逻辑
- [ ] 批量测试与调优
- [ ] 性能优化

---

## 五、预期效果

| 指标 | 目标值 |
|-----|-------|
| **整体准确率** | 95%+ |
| **平均响应时间** | < 100ms (不调用LLM) / < 1s (调用LLM) |
| **LLM调用率** | < 10% |
| **覆盖实体类型** | 8类全支持 |

---

## 六、文件结构建议

```
src/ner/
├── __init__.py
├── keyword_matcher.py      # Layer1: 关键词匹配
├── ner_model.py            # Layer2: NER模型
├── llm_extractor.py        # Layer3: LLM提取
├── entity_fuser.py         # 结果融合
├── kg_linker.py            # 图谱链接
└── pipeline.py             # 总入口
```
