# modol-pytorch

PyTorch rewrite of the original TensorFlow TFT pipeline.

## 项目结构

```
modol-pytorch/
├── base_model.py          # PyTorch TFT 模型定义
├── data_processor.py      # 数据预处理和序列生成
├── train.py               # 训练逻辑
├── evaluation.py          # 评估和可视化
├── main.py                # 主入口程序
├── nlp/                   # NLP 情感分析模块（新增）
│   ├── __init__.py
│   ├── sentiment_analyzer.py  # 碳交易新闻情感分析器
│   └── feature_processor.py   # NLP 特征处理器
└── requirements.txt       # 依赖配置
```

## Included modules:
- `base_model.py`: PyTorch TFT-style model and `QuantileLoss`
- `data_processor.py`: preprocessing, sequence building, synthetic data generator, split helpers
- `train.py`: `CustomDataset`, `TFTTrainer`, and end-to-end training utility
- `evaluation.py`: metrics evaluator and plotting helpers
- `main.py`: full training/evaluation entrypoint
- `nlp/`: NLP-based sentiment analysis for carbon trading news
- `requirements.txt`: dependencies for this folder

## NLP 情感分析模块

新增的 NLP 模块提供以下功能：

### 功能特性
- **情感分析**: 基于 FinBERT2 的中文金融情感分析
- **政策强度**: 基于关键词密度的政策强度分析
- **市场信心**: 综合情感和关键词的市场信心评分
- **ESG 主题分类**: 多维度 ESG 主题识别
- **碳价信号**: 综合多维度预测信号

### 快速开始

```python
from nlp import CarbonSentimentAnalyzer, NLPFeatureProcessor

# 方式 1: 直接使用分析器
analyzer = CarbonSentimentAnalyzer()
texts = ["新闻 1", "新闻 2"]
results = analyzer.process_texts(texts)

# 方式 2: 使用特征处理器（推荐用于 TFT 集成）
processor = NLPFeatureProcessor(
    text_columns=["title", "content"],  # 文本列
    date_column="date"                   # 日期列
)
df_with_features = processor.process(df)
```

### 输出特征说明

| 特征名 | 说明 | 范围 |
|--------|------|------|
| `_nlp_sentiment` | 情感分数 | [-1, 1] |
| `_nlp_policy_intensity` | 政策强度 | [0, 1] |
| `_nlp_uncertainty` | 不确定性 | [0, 1] |
| `_nlp_market_confidence` | 市场信心 | [0, 1] |
| `_nlp_carbon_signal` | 碳价预测信号 | [-1, 1] |
| `_nlp_esg_score` | ESG 主题分数 | [0, 1] |

### 依赖安装

```bash
pip install transformers>=4.0.0
```

Quick start:

```bash
cd modol-pytorch
python main.py
```
