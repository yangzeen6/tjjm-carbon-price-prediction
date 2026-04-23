"""NLP 情感分析模块 / NLP sentiment analysis package for carbon trading news.

提供以下功能：
- 基于 FinBERT2 的中文金融情感分析
- ESG 主题分类
- 政策强度分析
- 市场信心分析
- 碳价信号预测
- 碳市场新闻数据集加载

English summary:
- FinBERT-style sentiment analysis for carbon-related news.
- ESG topic extraction and downstream TFT feature generation.
- Carbon news dataset loading utilities.
"""

from .sentiment_analyzer import CarbonSentimentAnalyzer, ModelConfig, CarbonConfig
from .feature_processor import NLPFeatureProcessor, load_carbon_news_csv

__all__ = [
    "CarbonSentimentAnalyzer",
    "ModelConfig",
    "CarbonConfig",
    "NLPFeatureProcessor",
    "load_carbon_news_csv",
]
