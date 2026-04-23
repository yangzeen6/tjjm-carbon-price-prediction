#!/usr/bin/env python
"""处理碳市场新闻数据示例 / Example script for processing carbon news data.

本脚本演示如何使用 NLPFeatureProcessor 处理碳市场新闻数据集：
1. 加载 CSV 格式的新闻数据
2. 使用 NLPFeatureProcessor 提取情感特征
3. 将特征合并到时间序列数据
4. 生成可用于 TFT 模型训练的特征表

使用方法:
    python -m nlp.examples.process_carbon_news

数据集结构:
- date: 日期 (格式：2014-1-14 14:41)
- title: 新闻标题
- url: 新闻链接
- content: 新闻内容
- category: 类别 (如 "碳市场", "碳金融")
"""

import sys
from pathlib import Path

# 添加父目录到路径，确保可以导入 nlp 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from nlp.feature_processor import NLPFeatureProcessor, load_carbon_news_csv


def main():
    # 配置数据路径
    data_dir = Path(__file__).parent.parent.parent.parent / "cyq-data-4.12" / "data"
    csv_path = data_dir / "carbon_news_full_with_content.csv"

    print("=" * 60)
    print("碳市场新闻 NLP 特征处理示例")
    print("=" * 60)

    # 1. 加载碳市场新闻数据
    print(f"\n[1] 加载数据：{csv_path}")
    df = load_carbon_news_csv(str(csv_path))
    print(f"数据形状：{df.shape}")
    print(f"列名：{list(df.columns)}")
    print(f"日期范围：{df['date'].min()} 到 {df['date'].max()}")
    print(f"类别分布:\n{df['category'].value_counts()}")

    # 2. 初始化 NLP 特征处理器
    # 默认配置已适配碳市场新闻数据集 (title, content 列)
    print("\n[2] 初始化 NLP 特征处理器")
    processor = NLPFeatureProcessor(
        date_column="date",
        text_columns=["title", "content"],  # 适配数据集列名
        aggregation_method="mean",
    )

    # 3. 处理新闻数据，提取 NLP 特征
    print("\n[3] 提取 NLP 特征...")
    print("特征包括:")
    print("  - _nlp_sentiment: 新闻情感分数 [-1, 1]")
    print("  - _nlp_policy_intensity: 政策强度分数 [0, 1]")
    print("  - _nlp_uncertainty: 不确定性分数 [0, 1]")
    print("  - _nlp_market_confidence: 市场信心分数 [0, 1]")
    print("  - _nlp_carbon_signal: 碳价预测信号 [-1, 1]")
    print("  - _nlp_esg_primary: 主要 ESG 主题")
    print("  - _nlp_esg_score: ESG 主题显著性分数")

    # 处理前 10 条数据作为示例（完整处理可去掉限制）
    sample_df = df.head(10).copy()
    result = processor.process(sample_df)

    print("\n[4] 处理结果示例 (前 10 条):")
    display_cols = ["date", "category", "_nlp_sentiment", "_nlp_policy_intensity", "_nlp_carbon_signal"]
    print(result[display_cols].to_string())

    # 5. 按日期聚合特征
    print("\n[5] 按日期聚合特征...")
    daily_features = processor.aggregate_by_date(result)
    print(f"聚合后形状：{daily_features.shape}")
    print(daily_features.head())

    print("\n" + "=" * 60)
    print("处理完成！特征已准备好用于 TFT 模型训练")
    print("=" * 60)


if __name__ == "__main__":
    main()
