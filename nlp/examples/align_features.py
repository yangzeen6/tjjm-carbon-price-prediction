#!/usr/bin/env python
"""NLP 特征与碳市场数据对齐示例 / Align NLP features with carbon market data.

本脚本演示完整流程:
1. 加载碳市场新闻数据
2. 提取 NLP 情感特征
3. 按日期聚合
4. 与碳市场交易数据对齐
5. 生成最终训练特征表

使用方法:
    python -m nlp.examples.align_features
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from nlp.feature_processor import NLPFeatureProcessor, load_carbon_news_csv


def create_sample_carbon_price_data(news_df: pd.DataFrame) -> pd.DataFrame:
    """创建示例碳价格数据 / Create sample carbon price data."""
    dates = pd.date_range(
        start=news_df["date"].min(),
        end=news_df["date"].max() + pd.Timedelta(days=30),
        freq="D"
    )
    # 生成模拟的碳价格数据
    np.random.seed(42)
    price = np.cumsum(np.random.randn(len(dates))) + 50
    volume = np.random.randint(100, 1000, len(dates))

    price_df = pd.DataFrame({
        "date": dates,
        "carbon_price": price,
        "trading_volume": volume
    })
    return price_df


def align_features_with_price(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    date_col: str = "date"
) -> pd.DataFrame:
    """将 NLP 特征与价格数据对齐 / Align NLP features with price data.

    Args:
        news_df: 包含 NLP 特征的新闻数据
        price_df: 价格/交易数据
        date_col: 日期列名

    Returns:
        对齐后的 DataFrame
    """
    # 按日期聚合新闻特征
    daily = news_df.groupby(news_df[date_col].dt.date if hasattr(news_df[date_col], "dt") else news_df[date_col]).agg({
        "_nlp_sentiment": "mean",
        "_nlp_policy_intensity": "mean",
        "_nlp_uncertainty": "mean",
        "_nlp_market_confidence": "mean",
        "_nlp_carbon_signal": "mean",
        "_nlp_esg_score": "mean",
        "news_count": lambda x: len(x)
    }).reset_index()

    daily.columns = ["date"] + [f"_{col}" if not col.startswith("_") else col for col in daily.columns[1:]]

    # 与价格数据合并
    merged = price_df.merge(daily, on="date", how="left")

    # 填充缺失的 NLP 特征（无新闻的日期）
    nlp_cols = [c for c in merged.columns if c.startswith("_")]
    merged[nlp_cols] = merged[nlp_cols].fillna(0)

    return merged


def main():
    print("=" * 60)
    print("NLP 特征与碳市场数据对齐示例")
    print("=" * 60)

    # 1. 加载数据
    data_dir = Path(__file__).parent.parent.parent.parent / "cyq-data-4.12" / "data"
    csv_path = data_dir / "carbon_news_full_with_content.csv"

    print(f"\n[1] 加载新闻数据：{csv_path}")
    news_df = load_carbon_news_csv(str(csv_path))
    print(f"新闻数据：{len(news_df)} 条")

    # 2. 提取 NLP 特征
    print("\n[2] 提取 NLP 特征...")
    processor = NLPFeatureProcessor(
        date_column="date",
        text_columns=["title", "content"],
        aggregation_method="mean",
    )

    # 处理前 20 条作为演示
    sample_news = news_df.head(20).copy()
    result = processor.process(sample_news)
    print(f"提取的特征列：{processor.get_feature_columns()}")

    # 3. 生成模拟价格数据（实际使用请用真实价格数据）
    print("\n[3] 生成示例价格数据...")
    price_df = create_sample_carbon_price_data(news_df)

    # 4. 对齐特征
    print("\n[4] 对齐 NLP 特征与价格数据...")
    aligned_df = align_features_with_price(result, price_df)
    print(f"对齐后数据形状：{aligned_df.shape}")
    print(f"\n数据预览:")
    print(aligned_df.head(10))

    # 5. 保存到 CSV
    output_path = data_dir / "carbon_news_with_nlp_features.csv"
    aligned_df.to_csv(output_path, index=False)
    print(f"\n[5] 已保存特征数据到：{output_path}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
