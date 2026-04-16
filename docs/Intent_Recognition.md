# 医疗问答意图识别技术方案

## 概述

本文档描述医疗问答场景下的意图识别技术方案，采用**三层混合架构**：
- **Layer 1**: 关键词快速匹配
- **Layer 2**: 文本分类模型
- **Layer 3**: LLM 兜底

---

## 一、意图分类体系

### 1.1 意图定义

| 意图编号 | 意图名称 | 占比估计 | 示例问题 |
|---------|---------|---------|---------|
| 1 | 症状查询 | 35% | "这是什么症状？""发热可能是什么病？""咳嗽怎么办？" |
| 2 | 疾病查询 | 25% | "感冒是什么？""为什么会得肺炎？""感冒能治好吗？" |
| 3 | 用药咨询 | 20% | "感冒吃什么药？""高血压能吃这个药吗？""对乙酰氨基酚怎么吃？" |
| 4 | 检查咨询 | 10% | "肺炎需要做什么检查？""这个检查结果正常吗？" |
| 5 | 饮食/生活 | 8% | "感冒能吃什么？""高血压患者要注意什么？" |
| 6 | 其他 | 2% | "看感冒挂什么科？""需要去医院吗？" |

### 1.2 意图详细描述

#### 症状查询 (symptom_inquiry)
**目标**: 用户询问症状相关信息

**子类型**:
- 症状识别: 询问某个症状是什么
- 症状关联: 询问症状可能对应的疾病
- 症状缓解: 询问如何缓解症状

**触发词**:
- 症状: 症状、不舒服、难受、疼痛
- 疑问: 怎么办、怎么缓解、是什么、可能是

---

#### 疾病查询 (disease_inquiry)
**目标**: 用户询问疾病相关信息

**子类型**:
- 疾病信息: 询问疾病是什么
- 疾病原因: 询问疾病成因
- 疾病预后: 询问疾病能否治好、多久能好

**触发词**:
- 疾病: 病、疾病、症、炎
- 疑问: 是什么、为什么、能治好吗、多久好

---

#### 用药咨询 (medication_inquiry)
**目标**: 用户询问用药相关问题

**子类型**:
- 用药推荐: 询问该吃什么药
- 用药禁忌: 询问能不能吃某个药
- 用法用量: 询问药该怎么吃、吃多少

**触发词**:
- 药物: 药、药品、药物、胶囊、片、颗粒
- 疑问: 吃什么、能吃吗、怎么吃、吃多少

---

#### 检查咨询 (examination_inquiry)
**目标**: 用户询问检查相关问题

**子类型**:
- 检查推荐: 询问需要做什么检查
- 检查解读: 询问检查结果是否正常

**触发词**:
- 检查: 检查、化验、体检、拍片、CT
- 疑问: 需要做吗、结果正常吗、怎么看

---

#### 饮食/生活 (diet_lifestyle)
**目标**: 用户询问饮食或生活注意事项

**子类型**:
- 饮食建议: 询问能吃什么、不能吃什么
- 生活注意: 询问需要注意什么

**触发词**:
- 饮食: 吃、饮食、食物、忌口
- 生活: 注意、休息、运动、作息
- 疑问: 能吃吗、要注意什么

---

#### 其他 (other)
**目标**: 其他无法归类的问题

**子类型**:
- 科室推荐
- 转诊建议
- 其他

---

## 二、混合方案架构

### 2.1 系统架构图

```
用户问题
    ↓
[Layer 1] 关键词快速匹配
    ↓ (命中且置信度 > 0.9)
    ↓ (未命中或置信度低)
[Layer 2] 文本分类模型
    ↓ (置信度 > 0.7)
    ↓ (置信度 ≤ 0.7)
[Layer 3] LLM 兜底
    ↓
最终意图结果
```

### 2.2 分层策略

| 层级 | 技术方案 | 响应时间 | 准确率 | 成本 | 使用场景 |
|------|---------|---------|--------|------|---------|
| Layer 1 | 关键词匹配 | <10ms | 85% | 低 | 高频问题，精确匹配 |
| Layer 2 | 文本分类 | <100ms | 92% | 中 | 中频问题，泛化 |
| Layer 3 | LLM | <1s | 95% | 高 | 低频/复杂问题 |

---

## 三、Layer 1: 关键词快速匹配

### 3.1 关键词库设计

#### 意图触发词库

```python
INTENT_KEYWORDS = {
    "symptom_inquiry": {
        "primary": ["症状", "不舒服", "难受", "疼痛", "痒", "麻", "肿", "酸", "胀"],
        "secondary": ["怎么办", "怎么缓解", "是什么", "可能是", "会不会是", "要注意"],
        "boost": ["发热", "咳嗽", "头痛", "腹痛", "腹泻", "呕吐", "皮疹", "头晕"]
    },
    "disease_inquiry": {
        "primary": ["病", "疾病", "症", "炎", "综合征"],
        "secondary": ["是什么", "为什么", "怎么得的", "能治好吗", "多久好", "严重吗"],
        "boost": ["感冒", "肺炎", "高血压", "糖尿病", "冠心病", "胃炎"]
    },
    "medication_inquiry": {
        "primary": ["药", "药品", "药物", "胶囊", "片", "颗粒", "口服液", "注射液"],
        "secondary": ["吃什么", "能吃吗", "怎么吃", "吃多少", "有副作用吗", "效果好吗"],
        "boost": ["对乙酰氨基酚", "布洛芬", "阿莫西林", "头孢", "降压药", "降糖药"]
    },
    "examination_inquiry": {
        "primary": ["检查", "化验", "体检", "拍片", "CT", "B超", "核磁", "验血", "尿检"],
        "secondary": ["需要做吗", "结果正常吗", "怎么看", "多少钱", "注意事项"],
        "boost": ["血常规", "尿常规", "肝肾功能", "胸片", "心电图"]
    },
    "diet_lifestyle": {
        "primary": ["吃", "饮食", "食物", "忌口", "营养", "休息", "运动", "作息"],
        "secondary": ["能吃吗", "不能吃吗", "要注意", "怎么办", "怎么调理"],
        "boost": ["低盐", "低脂", "清淡", "烟酒", "熬夜", "锻炼"]
    },
    "other": {
        "primary": ["科", "科室", "挂号", "医院", "医生", "看病"],
        "secondary": ["挂什么科", "看哪个科", "需要去医院吗"],
        "boost": []
    }
}
```

### 3.2 匹配算法

#### 算法流程

```
输入: 用户问题 text
输出: (intent_id, confidence)

1. 预处理
   - 文本转小写
   - 去标点符号
   - 中文分词 (jieba)

2. 关键词匹配
   for each intent in INTENT_KEYWORDS:
       primary_count = 匹配 primary 关键词数量
       secondary_count = 匹配 secondary 关键词数量
       boost_count = 匹配 boost 关键词数量

       score = (primary_count * 3) + (secondary_count * 1) + (boost_count * 2)

3. 计算置信度
   if 最高分数 > 阈值 (5):
       confidence = min(1.0, score / 15)
   else:
       confidence = 0.0

4. 返回结果
   return (intent_with_highest_score, confidence)
```

#### 阈值设置

| 阈值 | 说明 | 动作 |
|------|------|------|
| 0.9 | 高置信度 | 直接返回，跳过后续层 |
| 0.5-0.9 | 中置信度 | 继续 Layer 2 验证 |
| <0.5 | 低置信度 | 跳过，直接到 Layer 2 |

---

## 四、Layer 2: 文本分类模型

### 4.1 模型选择

#### 推荐模型

| 模型 | 说明 | 优势 |
|------|------|------|
| **hfl/chinese-roberta-wwm-ext** | 中文 RoBERTa | 通用场景表现好 |
| **nghuyong/ernie-3.0-base-zh** | 百度 ERNIE | 中文理解能力强 |
| **Med-BERT** | 医疗预训练 | 医疗领域适配 |

**推荐**: `hfl/chinese-roberta-wwm-ext` (平衡性能和易用性)

### 4.2 模型架构

```
输入: [CLS] 用户问题 [SEP]
    ↓
Embedding Layer
    ↓
Transformer Layers × 12
    ↓
<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>  token (取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>  输出)
    ↓
Linear Layer (768 → 6)
    ↓
Softmax
    ↓
输出: 6个意图的概率分布
```

### 4.3 训练数据准备

#### 数据格式

```json
{
  "text": "感冒可以吃对乙酰氨基酚吗？",
  "intent": "medication_inquiry",
  "intent_id": 3
}
```

#### 数据增强策略

| 策略 | 说明 | 示例 |
|------|------|------|
| 同义词替换 | 替换实体和关键词 | "感冒" → "伤风" |
| 回译 | 中文→英文→中文 | 增加多样性 |
| 上下文扰动 | 轻微调整语序 | 保留意图 |
| 实体泛化 | 替换具体实体 | "阿莫西林" → "[药品]" |

#### 数据规模建议

| 阶段 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| MVP | 500 | 100 | 100 |
| 基础版 | 2000 | 300 | 300 |
| 完整版 | 10000+ | 1000 | 1000 |

### 4.4 训练配置

```python
TrainingConfig:
  model_name: "hfl/chinese-roberta-wwm-ext"
  max_length: 64
  batch_size: 32
  learning_rate: 2e-5
  epochs: 5
  warmup_ratio: 0.1
  weight_decay: 0.01
  label_smoothing: 0.1
```

### 4.5 置信度校准

```
原始概率 → 温度缩放 → Platt 缩放 → 最终置信度

温度缩放:
  calibrated_logits = logits / T

Platt 缩放 (可选):
  使用验证集拟合 sigmoid 函数
```

---

## 五、Layer 3: LLM 兜底

### 5.1 提示词设计

```python
SYSTEM_PROMPT = """你是一个专业的医疗问答意图识别助手。

请将用户的问题分类为以下 6 个类别之一：
1. 症状查询 - 询问症状相关问题，如"这是什么症状？"、"咳嗽怎么办？"
2. 疾病查询 - 询问疾病相关问题，如"感冒是什么？"、"肺炎能治好吗？"
3. 用药咨询 - 询问用药相关问题，如"感冒吃什么药？"、"能吃这个药吗？"
4. 检查咨询 - 询问检查相关问题，如"需要做什么检查？"、"结果正常吗？"
5. 饮食/生活 - 询问饮食或生活注意事项，如"能吃什么？"、"要注意什么？"
6. 其他 - 无法归类到上述类别的问题

请按照以下 JSON 格式输出，不要添加任何其他内容：
{
  "intent_id": 类别编号 (1-6),
  "intent_name": "类别名称",
  "confidence": 置信度 (0.0-1.0),
  "reasoning": "简要说明判断理由"
}
"""
```

### 5.2 调用策略

| 策略 | 说明 |
|------|------|
| 模型选择 | GPT-3.5-turbo / Claude 3 Haiku / 通义千问 |
| Temperature | 0.1 (低温度，确定性输出) |
| Max Tokens | 200 (短输出) |
| 超时时间 | 5s |
| 重试次数 | 2次 |

### 5.3 结果解析

```python
def parse_llm_response(response: str) -> Dict:
    """解析 LLM 响应，提取结构化结果"""
    try:
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "intent_id": result["intent_id"],
                "intent_name": result["intent_name"],
                "confidence": result["confidence"],
                "success": True
            }
    except Exception as e:
        logger.error(f"LLM 响应解析失败: {e}")

    # 兜底返回
    return {
        "intent_id": 6,
        "intent_name": "其他",
        "confidence": 0.5,
        "success": False
    }
```

---

## 六、结果融合与决策

### 6.1 决策逻辑

```
function determine_final_intent(layer1_result, layer2_result, layer3_result):

    # Layer 1 高置信度 → 直接返回
    if layer1_result.confidence >= 0.9:
        return layer1_result

    # Layer 2 结果
    if layer2_result:
        # Layer 1 和 Layer 2 一致 → 返回
        if layer1_result.intent_id == layer2_result.intent_id:
            return {
                "intent_id": layer2_result.intent_id,
                "intent_name": layer2_result.intent_name,
                "confidence": max(layer1_result.confidence, layer2_result.confidence) * 1.1,
                "source": "layer2_agreement"
            }

        # Layer 2 高置信度 → 返回 Layer 2
        if layer2_result.confidence >= 0.8:
            return {
                "intent_id": layer2_result.intent_id,
                "intent_name": layer2_result.intent_name,
                "confidence": layer2_result.confidence,
                "source": "layer2_high_confidence"
            }

    # Layer 3 兜底
    return {
        "intent_id": layer3_result.intent_id,
        "intent_name": layer3_result.intent_name,
        "confidence": layer3_result.confidence,
        "source": "layer3_llm"
    }
```

### 6.2 与下游系统集成

#### 意图 → 检索策略映射

```python
INTENT_RETRIEVAL_STRATEGY = {
    "symptom_inquiry": {
        "graph_enabled": True,
        "vector_enabled": True,
        "graph_weight": 0.6,
        "vector_weight": 0.4,
        "graph_relations": ["has_symptom", "acompany_with"],
        "vector_filter": ["Symptom", "Disease"]
    },
    "disease_inquiry": {
        "graph_enabled": True,
        "vector_enabled": True,
        "graph_weight": 0.5,
        "vector_weight": 0.5,
        "graph_relations": ["has_symptom", "recommand_drug", "need_check", "cure_way"],
        "vector_filter": ["Disease"]
    },
    "medication_inquiry": {
        "graph_enabled": True,
        "vector_enabled": True,
        "graph_weight": 0.7,
        "vector_weight": 0.3,
        "graph_relations": ["recommand_drug", "common_drug", "drugs_of"],
        "vector_filter": ["Drug", "Disease"]
    },
    "examination_inquiry": {
        "graph_enabled": True,
        "vector_enabled": True,
        "graph_weight": 0.4,
        "vector_weight": 0.6,
        "graph_relations": ["need_check"],
        "vector_filter": ["Check", "Disease"]
    },
    "diet_lifestyle": {
        "graph_enabled": True,
        "vector_enabled": True,
        "graph_weight": 0.3,
        "vector_weight": 0.7,
        "graph_relations": ["do_eat", "no_eat", "recommand_eat"],
        "vector_filter": ["Food", "Disease"]
    },
    "other": {
        "graph_enabled": True,
        "vector_enabled": True,
        "graph_weight": 0.5,
        "vector_weight": 0.5,
        "graph_relations": [],
        "vector_filter": []
    }
}
```

---

## 七、评估指标

### 7.1 核心指标

| 指标 | 公式 | 目标值 |
|------|------|--------|
| 准确率 (Accuracy) | 正确预测数 / 总预测数 | ≥ 90% |
| 宏 F1 (Macro-F1) | 各类别 F1 的平均值 | ≥ 88% |
| 加权 F1 (Weighted-F1) | 按类别权重加权的 F1 | ≥ 90% |
| 平均响应时间 | 所有请求的平均耗时 | < 100ms |
| P99 响应时间 | 99分位耗时 | < 500ms |
| Layer 3 使用率 | Layer 3 调用占比 | < 10% |

### 7.2 混淆矩阵

```
预测 →
真实 ↓
        症状  疾病  用药  检查  饮食  其他
症状      TP    FN    FN    FN    FN    FN
疾病      FP    TP    FN    FN    FN    FN
用药      FP    FP    TP    FN    FN    FN
检查      FP    FP    FP    TP    FN    FN
饮食      FP    FP    FP    FP    TP    FN
其他      FP    FP    FP    FP    FP    TP
```

### 7.3 分层性能

| 层级 | 召回率 | 说明 |
|------|--------|------|
| Layer 1 | ≥ 60% | 60% 的问题在第一层解决 |
| Layer 2 | ≥ 30% | 30% 的问题在第二层解决 |
| Layer 3 | ≤ 10% | 10% 的问题需要 LLM |

---

## 八、实施路线图

### 阶段 1: MVP (1-2 周)

**目标**: 快速上线，验证意图分类的合理性

**任务**:
- [ ] 建立关键词库
- [ ] 实现 Layer 1 关键词匹配
- [ ] 收集 1000 条真实用户问题
- [ ] 标注 500 条数据
- [ ] A/B 测试，验证意图分布

**交付物**: Layer 1 关键词匹配系统

---

### 阶段 2: 基础版 (2-4 周)

**目标**: 引入模型，提升准确率

**任务**:
- [ ] 标注 2000 条数据
- [ ] 数据增强
- [ ] 微调 RoBERTa 模型
- [ ] 实现 Layer 2 分类
- [ ] 实现分层决策逻辑
- [ ] 离线评估，准确率 ≥ 85%

**交付物**: 完整的两层系统

---

### 阶段 3: 完整版 (4-6 周)

**目标**: LLM 兜底，覆盖长尾问题

**任务**:
- [ ] 收集 500 条疑难问题
- [ ] 设计 LLM 提示词
- [ ] 实现 Layer 3 LLM 兜底
- [ ] 完整系统集成测试
- [ ] 在线评估，准确率 ≥ 90%
- [ ] 建立反馈闭环

**交付物**: 完整的三层混合系统

---

### 阶段 4: 优化 (持续)

**目标**: 持续优化，降低成本

**任务**:
- [ ] 持续收集用户反馈
- [ ] 定期重训模型
- [ ] 优化关键词库
- [ ] 降低 LLM 使用率 (< 5%)
- [ ] 性能优化，P99 < 200ms

---

## 九、监控与维护

### 9.1 监控指标

```
实时监控:
- QPS (每秒查询数)
- 各层调用比例
- 平均/ P99 响应时间
- 意图分布变化
- 错误率

离线监控:
- 每日准确率抽样
- 混淆矩阵分析
- 难例收集
- 用户反馈分析
```

### 9.2 日志记录

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "uuid",
  "text": "感冒可以吃对乙酰氨基酚吗？",
  "layer1_result": {
    "intent_id": 3,
    "intent_name": "medication_inquiry",
    "confidence": 0.95
  },
  "layer2_result": null,
  "layer3_result": null,
  "final_result": {
    "intent_id": 3,
    "intent_name": "medication_inquiry",
    "confidence": 0.95,
    "source": "layer1_high_confidence"
  },
  "latency_ms": 5,
  "user_feedback": null
}
```

---

## 十、成本分析

### 10.1 计算资源

| 资源 | 说明 | 配置 |
|------|------|------|
| Layer 1 | CPU 计算 | 1vCPU 足够 |
| Layer 2 | GPU 推理 | T4 / 3090 / 云服务 |
| Layer 3 | API 调用 | OpenAI / 通义千问 API |

### 10.2 API 成本估算

假设 QPS = 1，Layer 3 使用率 = 10%:

```
每日请求: 86400
Layer 3 调用: 8640
每次调用 cost: ¥0.002
每日成本: ¥17.28
每月成本: ¥518.4
```

---

## 附录

### A. 数据标注规范

### B. 模型训练脚本示例

### C. 部署配置文件示例

### D. 常见问题 FAQ
