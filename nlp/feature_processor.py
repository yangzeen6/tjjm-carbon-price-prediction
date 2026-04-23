"""NLP 特征处理器 / NLP feature processor for the TFT pipeline.

本模块提供 NLP 情感分析特征与 TFT 模型输入格式的对接口：
- 将新闻文本转换为情感分数
- 将情感分数对齐到时间序列
- 生成 TFT 可用的特征 DataFrame

English summary:
- Convert raw news text into derived sentiment features.
- Aggregate features by date and align them for TFT consumption.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .sentiment_analyzer import CarbonConfig, CarbonSentimentAnalyzer, ModelConfig


class NLPFeatureProcessor:
    """NLP 特征处理器 / Convert news text into TFT-ready time-series features.

    功能：
    1. 接收带时间戳的新闻数据
    2. 使用 CarbonSentimentAnalyzer 分析情感
    3. 将情感分数聚合到日度频率
    4. 生成与 TFT 输入兼容的特征 DataFrame
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        carbon_config: Optional[CarbonConfig] = None,
        date_column: str = "date",
        text_columns: Optional[List[str]] = None,
        aggregation_method: str = "mean",
    ):
        """初始化 NLP 特征处理器

        Args:
            model_config: 模型配置
            carbon_config: 碳交易配置
            date_column: 日期列名
            text_columns: 文本列名列表，默认适配碳市场新闻数据集 ['title', 'content']
            aggregation_method: 同日新闻聚合方法 ('mean', 'max', 'min')
        """
        self.model_config = model_config or ModelConfig()
        self.carbon_config = carbon_config or CarbonConfig()
        self.date_column = date_column
        # 默认使用碳市场新闻数据集的列名：title, content
        # 数据集结构：date, title, url, content, category
        self.text_columns = text_columns or ["title", "content"]
        self.aggregation_method = aggregation_method

        # 初始化分析器（延迟加载）
        self._analyzer: Optional[CarbonSentimentAnalyzer] = None

    @property
    def analyzer(self) -> CarbonSentimentAnalyzer:
        """懒加载分析器 / Lazily instantiate the analyzer."""
        if self._analyzer is None:
            self._analyzer = CarbonSentimentAnalyzer(
                model_config=self.model_config, carbon_config=self.carbon_config
            )
        return self._analyzer

    def set_analyzer(self, analyzer: CarbonSentimentAnalyzer):
        """设置外部分析器实例 / Inject an existing analyzer instance."""
        self._analyzer = analyzer

    def _build_full_texts(self, df: pd.DataFrame) -> List[str]:
        """拼接多列文本 / Concatenate text columns into one field."""
        texts = []
        for _, row in df.iterrows():
            parts = []
            for col in self.text_columns:
                if col in row and pd.notna(row[col]):
                    parts.append(str(row[col]))
            texts.append(" ".join(parts))
        return texts

    def process(
        self, df: pd.DataFrame, return_details: bool = True
    ) -> pd.DataFrame:
        """处理 DataFrame 并添加 NLP 派生特征 / Enrich a dataframe with NLP features.

        Args:
            df: 输入 DataFrame，需包含日期列和文本列
            return_details: 是否返回详细特征

        Returns:
            包含原始数据和 NLP 特征的 DataFrame
        """
        # 先把 title/content 等多列融合成单条文本，保证后续分析输入一致。
        full_texts = self._build_full_texts(df)

        # 批量分析各类派生特征。
        sentiment_scores = self.analyzer.predict_batch(full_texts)
        policy_intensities = self.analyzer.analyze_policy_intensity_batch(full_texts)
        uncertainty_scores = self.analyzer.analyze_uncertainty_batch(full_texts)
        market_confidences = self.analyzer.analyze_market_confidence_batch(
            full_texts, sentiment_scores
        )
        esg_topics = self.analyzer.classify_esg_topic_batch(full_texts)
        carbon_signals = self.analyzer.calculate_carbon_signal_batch(
            sentiment_scores, policy_intensities, market_confidences, uncertainty_scores
        )

        # 将特征写回原始 DataFrame，便于直接 join 到 TFT 训练表。
        result = df.copy()
        result["_nlp_sentiment"] = sentiment_scores
        result["_nlp_policy_intensity"] = policy_intensities
        result["_nlp_uncertainty"] = uncertainty_scores
        result["_nlp_market_confidence"] = market_confidences
        result["_nlp_carbon_signal"] = carbon_signals

        # ESG 主题特征：保留 primary 和 score，便于后续 one-hot 或统计聚合。
        esg_primary = []
        esg_scores = []
        for esg in esg_topics:
            if esg:
                sorted_topics = sorted(esg.items(), key=lambda x: x[1], reverse=True)
                esg_primary.append(sorted_topics[0][0] if sorted_topics else "unknown")
                esg_scores.append(sorted_topics[0][1] if sorted_topics else 0)
            else:
                esg_primary.append("unknown")
                esg_scores.append(0)

        result["_nlp_esg_primary"] = esg_primary
        result["_nlp_esg_score"] = esg_scores

        return result

    def aggregate_by_date(
        self, df: pd.DataFrame, date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """按日期聚合 NLP 特征 / Aggregate NLP features by date.

        Args:
            df: 包含 NLP 特征的 DataFrame
            date_col: 日期列名

        Returns:
            日度聚合特征 DataFrame
        """
        date_col = date_col or self.date_column
        nlp_cols = [
            "_nlp_sentiment",
            "_nlp_policy_intensity",
            "_nlp_uncertainty",
            "_nlp_market_confidence",
            "_nlp_carbon_signal",
            "_nlp_esg_score",
        ]

        agg_dict = {}
        for col in nlp_cols:
            if col in df.columns:
                agg_dict[col] = self.aggregation_method

        # 也计算新闻数量，作为当天信息密度的辅助特征。
        daily = df.groupby(df[date_col].dt.date if hasattr(df[date_col], "dt") else df[date_col]).agg(
            {**{col: self.aggregation_method for col in nlp_cols if col in df.columns}, "news_count": lambda x: len(x)}
        )

        return daily

    def create_time_series_features(
        self,
        df: pd.DataFrame,
        target_date_col: Optional[str] = None,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """创建与 TFT 模型兼容的时间序列特征 / Create TFT-compatible time-series features.

        将 NLP 特征对齐到目标日期索引，并处理缺失日期

        Args:
            df: 原始 DataFrame（含日期列）
            target_date_col: 目标日期列名
            fill_method: 缺失值填充方法 ('ffill', 'bfill', 'zero')

        Returns:
            包含 NLP 特征的时间序列 DataFrame
        """
        target_date_col = target_date_col or self.date_column

        # 确保日期列是 datetime 类型，避免 groupby/排序时出现类型不一致。
        df = df.copy()
        df[target_date_col] = pd.to_datetime(df[target_date_col])

        # 按日期排序后再做处理，确保时间顺序稳定。
        df = df.sort_values(target_date_col)

        # 处理 NLP 特征
        result = self.process(df)

        return result


def load_carbon_news_csv(csv_path: str, date_column: str = "date") -> pd.DataFrame:
    """加载碳市场新闻 CSV 数据集 / Load carbon news dataset.

    数据集结构:
    - date: 日期 (格式：2014-1-14 14:41)
    - title: 新闻标题
    - url: 新闻链接
    - content: 新闻内容
    - category: 类别 (如 "碳市场", "碳金融")

    Args:
        csv_path: CSV 文件路径
        date_column: 日期列名

    Returns:
        加载后的 DataFrame
    """
    df = pd.read_csv(csv_path)
    df[date_column] = pd.to_datetime(df[date_column])
    return df

    def get_feature_columns(self) -> List[str]:
        """获取 NLP 特征列名列表 / Return the NLP feature column names."""
        return [
            "_nlp_sentiment",
            "_nlp_policy_intensity",
            "_nlp_uncertainty",
            "_nlp_market_confidence",
            "_nlp_carbon_signal",
            "_nlp_esg_score",
        ]

    def get_feature_descriptions(self) -> Dict[str, str]:
        """获取特征描述 / Return human-readable feature descriptions."""
        return {
            "_nlp_sentiment": "新闻情感分数，范围 [-1, 1]，正值表示积极情绪",
            "_nlp_policy_intensity": "政策强度分数，范围 [0, 1]，基于政策关键词密度",
            "_nlp_uncertainty": "不确定性分数，范围 [0, 1]，基于不确定性词汇密度",
            "_nlp_market_confidence": "市场信心分数，范围 [0, 1]，综合情感和关键词",
            "_nlp_carbon_signal": "碳价预测信号，范围 [-1, 1]，综合多维度分析",
            "_nlp_esg_score": "ESG 主题显著性分数",
        }


def prepare_nlp_features_for_tft(
    df: pd.DataFrame,
    date_column: str = "date",
    text_columns: Optional[List[str]] = None,
    model_config: Optional[ModelConfig] = None,
) -> pd.DataFrame:
    """便捷函数：准备 NLP 特征用于 TFT 模型 / Convenience wrapper for TFT feature prep.

    Args:
        df: 输入 DataFrame
        date_column: 日期列名
        text_columns: 文本列名列表
        model_config: 模型配置

    Returns:
        包含 NLP 特征的 DataFrame
    """
    processor = NLPFeatureProcessor(
        model_config=model_config,
        date_column=date_column,
        text_columns=text_columns,
    )
    return processor.process(df)
