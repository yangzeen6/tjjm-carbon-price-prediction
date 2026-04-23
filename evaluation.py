"""PyTorch TFT 评估与可视化工具 / Evaluation and visualization utilities.

本模块提供两类能力：
1. TFTMetrics: 常用回归和区间预测指标的 NumPy 实现；
2. TFTEvaluator / TFTVisualizer: 模型推理与可视化。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TFTMetrics:
    """评估指标集合 / Metric collection for forecasting."""

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """平均绝对百分比误差 / Mean absolute percentage error."""
        return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """对称 MAPE / Symmetric MAPE."""
        return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

    @staticmethod
    def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """单分位数 pinball loss / Pinball loss for one quantile."""
        errors = y_true - y_pred
        return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantiles: List[float]) -> Dict[float, float]:
        """多个分位数的 pinball loss / Pinball loss for multiple quantiles."""
        losses: Dict[float, float] = {}
        for i, q in enumerate(quantiles):
            losses[q] = TFTMetrics.quantile_loss(y_true, y_pred[:, :, i], q)
        return losses

    @staticmethod
    def coverage(y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
        """区间覆盖率 / Prediction interval coverage."""
        return float(np.mean((y_true >= lower_bound) & (y_true <= upper_bound)))


class TFTEvaluator:
    """评估器 / Evaluator with an API close to the TensorFlow version.

    输入通常是模型已经准备好的 NumPy 字典，内部会自动转换为 torch.Tensor。
    若提供 data_processor，则会把标准化空间中的预测还原到原始尺度。
    """

    def __init__(self, model: torch.nn.Module, data_processor=None, device: Optional[str] = None):
        self.model = model
        self.data_processor = data_processor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.metrics = TFTMetrics()

    def predict_from_inputs(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """从 NumPy 输入字典执行推理 / Run inference from NumPy inputs."""
        self.model.eval()
        torch_inputs: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if "categorical" in key:
                torch_inputs[key] = torch.tensor(value, dtype=torch.long, device=self.device)
            else:
                torch_inputs[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_pred = self.model(torch_inputs)
        return y_pred.detach().cpu().numpy()

    def evaluate(
        self,
        inputs: Dict[str, np.ndarray],
        targets: np.ndarray,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> Dict[str, float]:
        """计算常见回归指标与区间指标 / Compute regression and interval metrics."""
        preds = self.predict_from_inputs(inputs)
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        median_pred = preds[:, :, median_idx]

        y_true = targets
        if self.data_processor is not None:
            y_true = self.data_processor.inverse_transform_target(targets)
            preds_inv = np.zeros_like(preds)
            for i in range(preds.shape[-1]):
                preds_inv[:, :, i] = self.data_processor.inverse_transform_target(preds[:, :, i])
            preds = preds_inv
            median_pred = preds[:, :, median_idx]

        result = {
            "mae": float(mean_absolute_error(y_true.reshape(-1), median_pred.reshape(-1))),
            "rmse": float(np.sqrt(mean_squared_error(y_true.reshape(-1), median_pred.reshape(-1)))),
            "mape": self.metrics.mape(y_true, median_pred),
            "smape": self.metrics.smape(y_true, median_pred),
        }

        pinballs = self.metrics.pinball_loss(y_true, preds, quantiles)
        result["pinball_loss"] = float(np.mean(list(pinballs.values())))
        lower_idx = quantiles.index(min(quantiles))
        upper_idx = quantiles.index(max(quantiles))
        result["coverage"] = self.metrics.coverage(y_true, preds[:, :, lower_idx], preds[:, :, upper_idx])
        return result


class TFTVisualizer:
    """可视化助手 / Visualization helper for TFT outputs."""

    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        import os

        os.makedirs(save_dir, exist_ok=True)

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        num_samples: int = 5,
        title: str = "Predictions vs True Values",
        save_path: Optional[str] = None,
    ):
        """绘制预测与真实值曲线 / Plot prediction curves against ground truth."""
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for idx, ax in enumerate(axes):
            if idx >= len(y_true):
                break
            steps = np.arange(len(y_true[idx]))
            ax.plot(steps, y_true[idx], "k-", label="True", linewidth=2)

            if y_pred.ndim == 3:
                m_idx = quantiles.index(0.5) if 0.5 in quantiles else y_pred.shape[-1] // 2
                lo_idx = quantiles.index(min(quantiles))
                hi_idx = quantiles.index(max(quantiles))
                ax.plot(steps, y_pred[idx, :, m_idx], "b-", label="Predicted", linewidth=1.5)
                ax.fill_between(steps, y_pred[idx, :, lo_idx], y_pred[idx, :, hi_idx], alpha=0.25, color="blue")
            else:
                ax.plot(steps, y_pred[idx], "b-", label="Predicted", linewidth=1.5)

            ax.legend(loc="upper left")
            ax.grid(True)
            ax.set_title(f"Sample {idx + 1}")

        plt.suptitle(title)
        plt.tight_layout()
        if save_path is None:
            save_path = f"{self.save_dir}/predictions.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_horizon_metrics(self, horizon_df: pd.DataFrame, save_path: Optional[str] = None):
        """按预测步长绘制指标 / Plot metrics by forecast horizon."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        metrics = ["mae", "rmse", "mape", "smape"]
        for ax, metric in zip(axes.flatten(), metrics):
            if metric in horizon_df.columns:
                ax.plot(horizon_df["horizon"], horizon_df[metric], marker="o")
                ax.set_title(f"{metric.upper()} by Horizon")
                ax.set_xlabel("Horizon")
                ax.grid(True)
        plt.tight_layout()
        if save_path is None:
            save_path = f"{self.save_dir}/horizon_metrics.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
