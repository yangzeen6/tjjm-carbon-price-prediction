"""PyTorch TFT 数据处理工具 / Data processing utilities for the TFT pipeline.

本模块负责：
1. 对连续特征进行标准化，对类别特征进行编码；
2. 基于时间窗口构造 encoder/prediction 序列；
3. 提供按 id 或按时间的切分工具；
4. 生成可用于调试和 smoke test 的合成面板时间序列数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TFTDataProcessor:
    """预处理 DataFrame 并构建模型输入序列 / Preprocess dataframe into model-ready sequences.

    说明 / Notes:
    - target 与连续特征使用 StandardScaler；
    - 类别特征使用 LabelEncoder；
    - create_sequences 会保留每个滑动窗口对应的原始片段，便于后续调试；
    - prepare_model_inputs 会把 DataFrame 转为 NumPy 数组字典，适配训练器。
    """

    def __init__(
        self,
        num_encoder_steps: int = 168,
        num_prediction_steps: int = 24,
        static_columns: Optional[List[str]] = None,
        time_varying_known_categorical: Optional[List[str]] = None,
        time_varying_known_continuous: Optional[List[str]] = None,
        time_varying_observed_categorical: Optional[List[str]] = None,
        time_varying_observed_continuous: Optional[List[str]] = None,
        target_column: str = "target",
        time_idx_column: str = "time_idx",
        id_column: str = "id",
    ):
        self.num_encoder_steps = num_encoder_steps
        self.num_prediction_steps = num_prediction_steps
        self.static_columns = static_columns or []
        self.time_varying_known_categorical = time_varying_known_categorical or []
        self.time_varying_known_continuous = time_varying_known_continuous or []
        self.time_varying_observed_categorical = time_varying_observed_categorical or []
        self.time_varying_observed_continuous = time_varying_observed_continuous or []
        self.target_column = target_column
        self.time_idx_column = time_idx_column
        self.id_column = id_column

        self.target_scaler = StandardScaler()
        self.continuous_scalers: Dict[str, StandardScaler] = {}
        self.categorical_encoders: Dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合所有 scaler/encoder 并转换训练集 / Fit transformers on training data."""
        out = df.copy()

        if self.target_column in out.columns:
            out[self.target_column] = self.target_scaler.fit_transform(out[[self.target_column]]).flatten()

        for col in self.time_varying_known_continuous + self.time_varying_observed_continuous:
            if col in out.columns and col != self.target_column:
                scaler = StandardScaler()
                out[col] = scaler.fit_transform(out[[col]]).flatten()
                self.continuous_scalers[col] = scaler

        for col in self.time_varying_known_categorical + self.time_varying_observed_categorical:
            if col in out.columns:
                encoder = LabelEncoder()
                out[col] = encoder.fit_transform(out[col].astype(str))
                self.categorical_encoders[col] = encoder

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的 scaler/encoder 转换验证集或测试集 / Transform held-out data."""
        out = df.copy()

        if self.target_column in out.columns:
            out[self.target_column] = self.target_scaler.transform(out[[self.target_column]]).flatten()

        for col, scaler in self.continuous_scalers.items():
            if col in out.columns:
                out[col] = scaler.transform(out[[col]]).flatten()

        for col, encoder in self.categorical_encoders.items():
            if col in out.columns:
                out[col] = out[col].astype(str)
                known = set(encoder.classes_)
                out.loc[~out[col].isin(known), col] = encoder.classes_[0]
                out[col] = encoder.transform(out[col])

        return out

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """把标准化后的 target 反变换回原始尺度 / Map normalized target back to original scale."""
        arr = np.asarray(values)
        shape = arr.shape
        if arr.size == 0:
            return arr
        flat = arr.reshape(-1, 1)
        inv = self.target_scaler.inverse_transform(flat)
        return inv.reshape(shape)

    def create_sequences(self, df: pd.DataFrame) -> Dict[str, Any]:
        """构建滑动窗口序列 / Build sliding-window sequences.

        每个样本由 encoder + prediction 长度的连续片段组成，
        targets 取窗口最后 num_prediction_steps 个 time step。
        """
        total_steps = self.num_encoder_steps + self.num_prediction_steps
        sequences: List[pd.DataFrame] = []
        targets: List[np.ndarray] = []

        if self.id_column in df.columns:
            groups = df.groupby(self.id_column)
        else:
            groups = [(None, df)]

        for _, group in groups:
            group = group.sort_values(self.time_idx_column)
            for i in range(len(group) - total_steps + 1):
                seq = group.iloc[i : i + total_steps]
                sequences.append(seq)
                targets.append(seq[self.target_column].iloc[-self.num_prediction_steps :].to_numpy(dtype=np.float32))

        return {"sequences": sequences, "targets": np.asarray(targets, dtype=np.float32)}

    def prepare_model_inputs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """将序列 DataFrame 打包成模型输入字典 / Pack sequences into model input tensors.

        返回结构与 BaseTemporalFusionTransformer.forward 对齐，
        便于直接交给 CustomDataset 和 trainer。
        """
        total_steps = self.num_encoder_steps + self.num_prediction_steps
        seq_dict = self.create_sequences(df)
        sequences: List[pd.DataFrame] = seq_dict["sequences"]
        targets = seq_dict["targets"]

        if len(sequences) == 0:
            raise ValueError(
                "No sequences built. Check id grouping and encoder/prediction lengths."
                f" Required minimum per id: {total_steps}"
            )

        inputs: Dict[str, Any] = {"target": targets}

        if self.static_columns:
            inputs["static"] = np.stack(
                [seq[self.static_columns].iloc[0].to_numpy(dtype=np.float32) for seq in sequences], axis=0
            )

        if self.time_varying_known_categorical:
            cols = [c for c in self.time_varying_known_categorical if c in sequences[0].columns]
            if cols:
                # [N, C, T]
                inputs["time_varying_known_categorical"] = np.stack(
                    [np.stack([seq[col].to_numpy(dtype=np.int64) for col in cols], axis=0) for seq in sequences],
                    axis=0,
                )

        if self.time_varying_known_continuous:
            cols = [c for c in self.time_varying_known_continuous if c in sequences[0].columns]
            if cols:
                # [N, T, D]
                inputs["time_varying_known_continuous"] = np.stack(
                    [seq[cols].to_numpy(dtype=np.float32) for seq in sequences], axis=0
                )

        if self.time_varying_observed_categorical:
            cols = [c for c in self.time_varying_observed_categorical if c in sequences[0].columns]
            if cols:
                inputs["time_varying_observed_categorical"] = np.stack(
                    [np.stack([seq[col].to_numpy(dtype=np.int64) for col in cols], axis=0) for seq in sequences],
                    axis=0,
                )

        if self.time_varying_observed_continuous:
            cols = [c for c in self.time_varying_observed_continuous if c in sequences[0].columns]
            if cols:
                inputs["time_varying_observed_continuous"] = np.stack(
                    [seq[cols].to_numpy(dtype=np.float32) for seq in sequences], axis=0
                )

        return inputs


def _stable_int(x: float) -> int:
    """稳定的向下取整 / Stable floor-to-int helper."""
    if x <= 0:
        return 0
    return int(np.floor(x))


def train_test_split_by_id(
    df: pd.DataFrame,
    test_size: float = 0.2,
    id_column: str = "id",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 id 分组后切分训练/测试集 / Split train-test by panel id."""
    if id_column not in df.columns:
        raise ValueError(f"id column {id_column!r} is required")
    ids = df[id_column].dropna().unique()
    if len(ids) == 0:
        raise ValueError("No valid ids found")

    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_test = _stable_int(n * float(test_size))
    if test_size > 0 and n_test == 0:
        n_test = 1
    if n_test >= n:
        n_test = max(1, n - 1)

    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])

    return df[df[id_column].isin(train_ids)].copy(), df[df[id_column].isin(test_ids)].copy()


def train_val_test_split_by_id(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2,
    id_column: str = "id",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 id 切分 train/val/test / Split train, validation, and test sets by id."""
    train_df, test_df = train_test_split_by_id(df, test_size=test_size, id_column=id_column, seed=seed)
    train_df, val_df = train_test_split_by_id(train_df, test_size=val_size, id_column=id_column, seed=seed + 1)
    return train_df, val_df, test_df


def train_test_split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_idx_column: str = "time_idx",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分 / Split a single time series chronologically."""
    if time_idx_column not in df.columns:
        raise ValueError(f"time index column {time_idx_column!r} not found")
    out = df.sort_values(time_idx_column)
    split = int(len(out) * (1 - test_size))
    return out.iloc[:split].copy(), out.iloc[split:].copy()


class GeneratedDataset:
    """生成合成面板时间序列数据 / Generate synthetic panel time-series data.

    用途 / Use cases:
    - 快速 smoke test
    - 训练流程联调
    - 没有真实数据时的示例运行
    """

    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 192,
        num_features: int = 5,
        noise_level: float = 0.1,
        trend_prob: float = 0.2,
        seasonality_prob: float = 0.3,
        random_walk_prob: float = 0.2,
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.noise_level = noise_level
        self.trend_prob = trend_prob
        self.seasonality_prob = seasonality_prob
        self.random_walk_prob = random_walk_prob

    def generate(self) -> pd.DataFrame:
        """生成合成数据表 / Generate a synthetic dataframe."""
        rng = np.random.default_rng(42)
        rows: List[Dict[str, Any]] = []

        for sample_id in range(self.num_samples):
            base = rng.normal(0.0, 1.0)
            values = np.full(self.sequence_length, base, dtype=np.float32)

            if rng.random() < self.trend_prob:
                values += np.linspace(0, rng.uniform(-2.0, 2.0), self.sequence_length)

            if rng.random() < self.seasonality_prob:
                period = rng.integers(12, 48)
                amp = rng.uniform(0.5, 2.0)
                values += amp * np.sin(np.arange(self.sequence_length) * (2 * np.pi / period))

            if rng.random() < self.random_walk_prob:
                values += np.cumsum(rng.normal(0, 0.05, self.sequence_length))

            values += rng.normal(0, self.noise_level, self.sequence_length)

            feature_mat = []
            for _ in range(self.num_features):
                feature_mat.append(values + rng.normal(0, 0.2, self.sequence_length))
            feature_mat = np.stack(feature_mat, axis=1)

            for t in range(self.sequence_length):
                row = {
                    "id": sample_id,
                    "time_idx": t,
                    "target": float(values[t]),
                    "day_of_week": int(t % 7),
                    "month": int((t % 12) + 1),
                }
                for f in range(self.num_features):
                    row[f"feature_{f}"] = float(feature_mat[t, f])
                rows.append(row)

        return pd.DataFrame(rows)
