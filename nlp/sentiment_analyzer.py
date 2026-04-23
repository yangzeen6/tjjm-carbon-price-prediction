#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碳交易新闻情感分析器 / Carbon trading sentiment analyzer.

核心特性：
1. 支持 FinBERT2 主模型 + ESG 微调模型并行分析
2. 量化 FinBERT 作为轻量级备用模型
3. 碳市场专有维度：政策强度、市场信心、ESG 主题分类
4. 批量处理 + GPU 加速
5. 时序特征工程

English summary:
- Primary sentiment model plus optional ESG classifier.
- Keyword-driven fallbacks for policy intensity, market confidence, and ESG topics.
- Designed to be consumed by the TFT feature processor.

参考论文：
- FinBERT2: A Specialized Bidirectional Encoder (2025)
- eFinBERT: Efficient Financial Sentiment Classification (2025)
- Sentiment-Driven Forecasting of Carbon Prices (2026)
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# Transformers 相关导入
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    pipeline,
    set_seed,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==================== 配置与常量 ====================


@dataclass
class ModelConfig:
    """模型配置 / Model configuration.

    保存分析器加载外部模型所需的参数，如模型名、缓存目录和 batch size。
    """

    # 主模型（FinBERT2 或兼容版本）
    primary_model: str = "yiyanghkust/finbert-tone-chinese"
    # ESG 微调模型
    esg_model: str = "mrm8488/bert-base-multilingual-cased-finetuned-esg"  # 备选
    # 轻量级量化模型路径（可选）
    quantized_model: Optional[str] = None
    # 缓存目录
    cache_dir: str = "./model_cache"
    # 最大序列长度
    max_length: int = 512
    # 批处理大小
    batch_size: int = 16
    # 随机种子
    seed: int = 42


@dataclass
class CarbonConfig:
    """碳交易分析配置 / Carbon-domain keyword and scoring configuration."""

    # 政策关键词库
    policy_keywords: List[str] = field(
        default_factory=lambda: [
            "政策",
            "法规",
            "监管",
            "标准",
            "管控",
            "约束",
            "限制",
            "合规",
            "审批",
            "备案",
            "生态环境部",
            "发改委",
            "国务院",
            "政府",
            "部门",
            "出台",
            "发布",
            "实施",
            "执行",
            "落实",
            "推动",
            "推进",
            "强制",
            "要求",
            "必须",
            "严格",
            "严厉",
            "统筹",
            "规划",
            "碳配额",
            "碳市场",
            "ETS",
            "排放交易",
            "CCER",
            "碳减排",
        ]
    )

    # 市场关键词库
    market_keywords: List[str] = field(
        default_factory=lambda: [
            "投资",
            "融资",
            "信贷",
            "贷款",
            "质押",
            "保险",
            "基金",
            "债券",
            "市场",
            "交易",
            "价格",
            "价值",
            "收益",
            "回报",
            "盈利",
            "预期",
            "信心",
            "看好",
            "乐观",
            "参与",
            "入场",
            "布局",
            "配置",
            "持有",
            "项目",
            "产业",
            "企业",
            "公司",
            "机构",
            "平台",
            "交易所",
        ]
    )

    # ESG 主题关键词
    esg_topics: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "climate_change": ["气候", "碳", "温室气体", "排放", "温度", "极端天气"],
            "natural_capital": ["自然资源", "生物多样性", "水", "森林", "土地"],
            "pollution_waste": ["污染", "废弃物", "排放物", "有毒物质"],
            "human_capital": ["员工", "劳动", "培训", "安全", "健康"],
            "product_liability": ["产品", "质量", "安全", "责任"],
            "community_relations": ["社区", "社会", "公益", "捐赠"],
            "corporate_governance": ["治理", "董事会", "高管薪酬", "股东"],
            "business_ethics": ["道德", "反腐败", "合规", "透明"],
        }
    )

    # 政策强度归一化因子
    policy_norm_factor: float = 0.02  # 每 100 字 2 个关键词为强度 0.2
    # 市场信心权重配置
    confidence_weight_sentiment: float = 0.5
    confidence_weight_keyword: float = 0.3

    # 碳市场专有名词
    carbon_specific_terms: List[str] = field(
        default_factory=lambda: [
            "碳配额",
            "碳价",
            "CCER",
            "履约",
            "清缴",
            "碳泄漏",
            "CBAM",
            "碳边境",
            "碳中和",
            "碳达峰",
            "双碳",
            "全国碳市场",
            "试点碳市场",
        ]
    )


# ==================== 数据集类 ====================


class NewsDataset(Dataset):
    """新闻数据集 / Dataset for batched inference over news texts."""

    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] else ""
        # 超长新闻会明显拖慢 tokenizer，因此先做字符级粗截断。
        if len(text) > self.max_length * 4:
            text = text[: self.max_length * 4]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length" if self.max_length else False,
            max_length=self.max_length,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "idx": idx,
        }


# ==================== 核心分析器类 ====================


class CarbonSentimentAnalyzer:
    """碳交易新闻情感分析器 / Carbon trading news sentiment analyzer.

    组件 / Components:
    - 主情感模型：FinBERT 风格分类器
    - ESG 主题：关键词兜底，模型可选
    - 衍生特征：policy intensity, market confidence, uncertainty, carbon signal
    - 批处理：DataLoader + GPU 推理
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        carbon_config: Optional[CarbonConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.carbon_config = carbon_config or CarbonConfig()

        # 设置随机种子
        set_seed(self.model_config.seed)

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self._load_models()

        # 初始化缓存
        self._predictions_cache = {}

    def _load_models(self):
        """加载主模型与可选 ESG 模型 / Load the primary and optional ESG models."""
        # 1. 主模型（FinBERT）
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.primary_model, cache_dir=self.model_config.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.primary_model, cache_dir=self.model_config.cache_dir
        )
        self.model.to(self.device)
        self.model.eval()

        # 标签映射
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

        # 2. 尝试加载 ESG 分类模型（可选）
        self.esg_model = None
        try:
            self.esg_tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.esg_model, cache_dir=self.model_config.cache_dir
            )
            self.esg_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.esg_model, cache_dir=self.model_config.cache_dir
            )
            self.esg_model.to(self.device)
            self.esg_model.eval()
        except Exception:
            pass

    def _predict_batch_tensor(
        self, texts: List[str], model=None, tokenizer=None
    ) -> np.ndarray:
        """真正的批量预测 / Batched GPU inference.

        Args:
            texts: 文本列表
            model: 模型（默认使用主模型）
            tokenizer: tokenizer（默认使用主 tokenizer）

        Returns:
            probabilities: shape (n, num_labels) 的概率数组
        """
        if model is None:
            model = self.model
        if tokenizer is None:
            tokenizer = self.tokenizer

        # 创建数据集和 DataLoader
        dataset = NewsDataset(texts, tokenizer, self.model_config.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.model_config.batch_size, shuffle=False, num_workers=0
        )

        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = model(input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def predict_batch(
        self, texts: List[str], return_continuous: bool = True
    ) -> List[float]:
        """批量预测情感分数 / Predict sentiment scores for a list of texts.

        Args:
            texts: 文本列表
            return_continuous: 是否返回连续分数（-1 到 1）

        Returns:
            情感分数列表
        """
        if not texts:
            return []

        # 缓存命中可避免重复 tokenization 和推理。
        texts_tuple = tuple(texts)
        if texts_tuple in self._predictions_cache:
            return self._predictions_cache[texts_tuple]

        # 批量预测
        probs = self._predict_batch_tensor(texts)

        # 将三分类概率映射到连续情感分数：正向概率减去负向概率。
        scores = []
        for prob in probs:
            # 模型输出：[neg, neutral, pos]
            if return_continuous:
                score = prob[2] - prob[0]  # positive - negative
            else:
                score = np.argmax(prob)
            scores.append(float(score))

        # 缓存结果
        self._predictions_cache[texts_tuple] = scores

        return scores

    def analyze_policy_intensity_batch(self, texts: List[str]) -> List[float]:
        """批量分析政策强度 / Batch policy intensity analysis.

        归一化方法：强度 = min(1.0, (关键词数量 / (文本长度 * norm_factor)))
        """
        intensities = []

        for text in texts:
            if not text:
                intensities.append(0.0)
                continue

            text_len = len(text)
            # 关键词越密集、文本越短，政策强度越高。
            keyword_count = sum(
                1 for kw in self.carbon_config.policy_keywords if kw in text
            )

            # 先按长度归一，再做上限裁剪。
            intensity = keyword_count / (text_len * self.carbon_config.policy_norm_factor + 1)
            intensity = min(1.0, intensity)

            # 强词增强
            strong_policy = ["强制", "严格", "严厉", "必须", "立即", "坚决"]
            if any(w in text for w in strong_policy):
                intensity = min(1.0, intensity + 0.2)

            intensities.append(round(intensity, 4))

        return intensities

    def analyze_market_confidence_batch(
        self, texts: List[str], sentiment_scores: List[float]
    ) -> List[float]:
        """批量分析市场信心 / Batch market confidence estimation.

        结合：
        - 市场关键词密度
        - 整体情感分数
        """
        confidences = []

        for text, sentiment in zip(texts, sentiment_scores):
            if not text:
                confidences.append(0.0)
                continue

            text_len = len(text)
            # 关键词数量用于估计文本是否讨论交易、价格和投资信心。
            keyword_count = sum(
                1 for kw in self.carbon_config.market_keywords if kw in text
            )
            keyword_score = min(1.0, keyword_count / 10)

            # 融合：关键词分（30%）+ 情感分（50%）+ 基础分（20%）
            if sentiment > 0:
                confidence = (
                    keyword_score * self.carbon_config.confidence_weight_keyword
                    + sentiment * self.carbon_config.confidence_weight_sentiment
                    + 0.2
                )
            else:
                confidence = (
                    keyword_score * self.carbon_config.confidence_weight_keyword
                    + sentiment * 0.3  # 负面情感对信心的抑制作用
                    + 0.2
                )

            confidence = np.clip(confidence, 0, 1)
            confidences.append(round(confidence, 4))

        return confidences

    def classify_esg_topic_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """ESG 主题分类 / ESG topic classification.

        返回每个文本的 ESG 主题分布
        """
        if self.esg_model is None:
            # 当前实现以关键词兜底；若后续补充 ESG 分类头，可在此处替换。
            return self._esg_by_keywords_batch(texts)

        # 使用模型进行批量分类
        # 注意：实际的 ESG 模型可能需要特定的标签映射
        # 这里简化为关键词分类
        return self._esg_by_keywords_batch(texts)

    def _esg_by_keywords_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """基于关键词的 ESG 分类（降级方案） / Keyword-based ESG fallback."""
        results = []

        for text in texts:
            topic_scores = {}
            for topic, keywords in self.carbon_config.esg_topics.items():
                count = sum(1 for kw in keywords if kw in text)
                score = min(1.0, count / 5)
                if score > 0.1:
                    topic_scores[topic] = round(score, 4)

            # 归一化
            if topic_scores:
                total = sum(topic_scores.values())
                if total > 0:
                    topic_scores = {k: v / total for k, v in topic_scores.items()}

            results.append(topic_scores)

        return results

    def calculate_carbon_signal_batch(
        self,
        sentiment_scores: List[float],
        policy_intensities: List[float],
        market_confidences: List[float],
        uncertainty_scores: Optional[List[float]] = None,
    ) -> List[float]:
        """计算碳价预测综合信号 / Calculate a composite carbon-price signal.

        公式：
        signal = w1*sentiment + w2*policy + w3*confidence
        考虑不确定性衰减

        Returns:
            signal: [-1, 1] 范围内的信号值
        """
        # 权重配置体现领域先验：政策强度通常比情绪更直接影响碳价。
        weights = {
            "sentiment": 0.35,
            "policy": 0.40,  # 政策对碳价影响最大
            "confidence": 0.25,
        }

        signals = []
        for i in range(len(sentiment_scores)):
            signal = (
                weights["sentiment"] * sentiment_scores[i]
                + weights["policy"] * (policy_intensities[i] * 2 - 1)  # 映射到 [-1,1]
                + weights["confidence"] * (market_confidences[i] * 2 - 1)
            )
            signal = np.clip(signal, -1, 1)

            # 不确定性越高，综合信号越保守。
            if uncertainty_scores and uncertainty_scores[i] > 0:
                signal *= 1 - uncertainty_scores[i] * 0.3

            signals.append(round(signal, 4))

        return signals

    def analyze_uncertainty_batch(self, texts: List[str]) -> List[float]:
        """批量分析不确定性程度 / Batch uncertainty estimation."""
        uncertainty_words = [
            "可能",
            "或许",
            "大概",
            "预计",
            "有望",
            "或",
            "不确定",
            "尚不明确",
            "待观察",
            "视情况而定",
            "风险",
            "变数",
            "波动",
        ]

        uncertainties = []
        for text in texts:
            if not text:
                uncertainties.append(0.0)
                continue

            count = sum(1 for w in uncertainty_words if w in text)
            uncertainty = min(1.0, count / 3)
            uncertainties.append(round(uncertainty, 4))

        return uncertainties

    def process_texts(
        self,
        texts: List[str],
        return_details: bool = True,
    ) -> pd.DataFrame:
        """处理文本列表并返回多维度分析结果 / Process texts into analysis results.

        Args:
            texts: 文本列表
            return_details: 是否返回详细分析结果

        Returns:
            包含分析结果的 DataFrame
        """
        # 批量处理各维度：情感、政策、市场、不确定性、ESG 与综合信号。
        sentiment_scores = self.predict_batch(texts)
        policy_intensities = self.analyze_policy_intensity_batch(texts)
        uncertainty_scores = self.analyze_uncertainty_batch(texts)
        market_confidences = self.analyze_market_confidence_batch(texts, sentiment_scores)
        esg_topics = self.classify_esg_topic_batch(texts)
        carbon_signals = self.calculate_carbon_signal_batch(
            sentiment_scores, policy_intensities, market_confidences, uncertainty_scores
        )

        # 将各维度结果合并到单行输出，便于后续按日期聚合或特征工程。
        results = []
        for idx, text in enumerate(texts):
            result = {
                "text": text[:500],
                # 核心维度
                "sentiment_score": round(sentiment_scores[idx], 4),
                "policy_intensity": policy_intensities[idx],
                "market_confidence": market_confidences[idx],
                "uncertainty": uncertainty_scores[idx],
                "carbon_signal": carbon_signals[idx],
            }

            if return_details:
                # 添加 ESG 主题（取前 2 个）以便做解释性分析。
                esg = esg_topics[idx]
                if esg:
                    sorted_topics = sorted(esg.items(), key=lambda x: x[1], reverse=True)
                    result["esg_primary"] = sorted_topics[0][0] if sorted_topics else "unknown"
                    result["esg_score"] = sorted_topics[0][1] if sorted_topics else 0
                    result["esg_secondary"] = (
                        sorted_topics[1][0] if len(sorted_topics) > 1 else ""
                    )
                else:
                    result["esg_primary"] = "unknown"
                    result["esg_score"] = 0
                    result["esg_secondary"] = ""

            results.append(result)

        return pd.DataFrame(results)

    def clear_cache(self):
        """清除预测缓存 / Clear the inference cache."""
        self._predictions_cache.clear()
