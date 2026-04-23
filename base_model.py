"""PyTorch Temporal Fusion Transformer (TFT) 实现。

本模块提供一个轻量但接口兼容的 TFT-style 网络，用于多步时间序列预测。
核心设计目标：
1. 将静态特征、已知时间特征、观测时间特征统一映射到同一隐藏空间。
2. 通过 LSTM 编码局部时序依赖，再用 causal self-attention 建模长依赖。
3. 输出多个分位数预测，以支持不确定性估计与区间预测。

Module goals:
- unify heterogeneous tabular time-series features into a shared hidden space;
- encode temporal dynamics with LSTM + causal attention;
- predict multiple quantiles for robust forecasting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn


TensorLike = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]


class QuantileLoss(nn.Module):
    """分位数回归损失 / Quantile regression loss.

    对于每个分位数 q，损失定义为：
    L_q(y, \hat{y}) = max((q-1)(y-\hat{y}), q(y-\hat{y}))

    这使模型可以同时学习低位、中位和高位预测，适合碳价这类波动较强的序列。
    Tensor shapes:
    - y_true: [B, T]
    - y_pred: [B, T, Q]
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles cannot be empty")
        self.quantiles = quantiles

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """计算分位数损失 / Compute quantile loss for all horizons.

        Args:
            y_true: 真实值，shape = [B, T]
            y_pred: 分位数预测，shape = [B, T, Q]

        Returns:
            标量损失值 / Scalar loss tensor.
        """
        if y_true.dim() != 2:
            raise ValueError(f"y_true must be [B, T], got shape={tuple(y_true.shape)}")
        if y_pred.dim() != 3:
            raise ValueError(f"y_pred must be [B, T, Q], got shape={tuple(y_pred.shape)}")
        if y_pred.size(-1) != len(self.quantiles):
            raise ValueError("last dimension of y_pred must match number of quantiles")

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - y_pred[:, :, i]
            losses.append(torch.maximum((q - 1.0) * errors, q * errors).unsqueeze(-1))
        return torch.mean(torch.cat(losses, dim=-1))


class BaseTemporalFusionTransformer(nn.Module):
    """紧凑版 TFT-style 架构 / Compact Temporal Fusion Transformer-style network.

    设计说明 / Design notes:
    - 输入使用字典形式，兼容静态、已知时间、观测时间三类特征。
    - 类别特征使用 Embedding，连续特征使用 Linear 投影到 hidden_size。
    - LSTM 负责局部时序编码，MultiheadAttention 负责跨时间依赖。
    - 输出层按分位数维度预测，默认 [0.1, 0.5, 0.9]。
    """

    def __init__(
        self,
        hidden_size: int = 160,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_lstm_layers: int = 1,
        num_lstm_steps: int = 1,
        static_input_dim: int = 0,
        time_varying_known_categorical_dims: Optional[List[int]] = None,
        time_varying_known_continuous_dims: int = 0,
        time_varying_observed_categorical_dims: Optional[List[int]] = None,
        time_varying_observed_continuous_dims: int = 0,
        num_encoder_steps: int = 168,
        num_prediction_steps: int = 24,
        quantiles: Optional[List[float]] = None,
        **_: Dict,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_steps = num_lstm_steps
        self.static_input_dim = static_input_dim
        self.time_varying_known_categorical_dims = time_varying_known_categorical_dims or []
        self.time_varying_known_continuous_dims = time_varying_known_continuous_dims
        self.time_varying_observed_categorical_dims = time_varying_observed_categorical_dims or []
        self.time_varying_observed_continuous_dims = time_varying_observed_continuous_dims
        self.num_encoder_steps = num_encoder_steps
        self.num_prediction_steps = num_prediction_steps
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        self.known_cat_embeddings = nn.ModuleList(
            [nn.Embedding(dim, hidden_size) for dim in self.time_varying_known_categorical_dims]
        )
        self.obs_cat_embeddings = nn.ModuleList(
            [nn.Embedding(dim, hidden_size) for dim in self.time_varying_observed_categorical_dims]
        )

        self.known_cont_linears = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(self.time_varying_known_continuous_dims)]
        )
        self.obs_cont_linears = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(self.time_varying_observed_continuous_dims)]
        )

        self.static_linear = nn.Linear(static_input_dim, hidden_size) if static_input_dim > 0 else None

        lstm_dropout = dropout if num_lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, len(self.quantiles))

    @staticmethod
    def _split_cat_inputs(x: TensorLike) -> List[torch.Tensor]:
        """统一分类输入格式 / Normalize categorical inputs.

        支持三种输入：
        - list/tuple of tensors
        - [B, T] 的单个分类张量
        - [B, C, T] 的多分类张量
        """
        if isinstance(x, (list, tuple)):
            return [t.long() for t in x]
        if isinstance(x, torch.Tensor):
            if x.dim() == 2:
                return [x.long()]
            if x.dim() == 3:
                # [B, C, T] -> C x [B, T]
                return [x[:, i, :].long() for i in range(x.size(1))]
        raise ValueError("categorical input must be list/tuple or tensor [B,T]/[B,C,T]")

    def _encode_temporal_features(self, inputs: Dict[str, TensorLike]) -> torch.Tensor:
        """将异构时间特征编码到统一隐藏表示。

        输出 shape:
        - temporal: [B, T, H]

        处理顺序：
        1. 编码 known categorical / continuous;
        2. 编码 observed categorical / continuous;
        3. 将所有特征在特征维上平均，得到统一的时序表示。
        """
        feats: List[torch.Tensor] = []

        known_cat = inputs.get("time_varying_known_categorical")
        if known_cat is not None:
            known_cat_list = self._split_cat_inputs(known_cat)
            for emb, cat in zip(self.known_cat_embeddings, known_cat_list):
                feats.append(emb(cat))

        known_cont = inputs.get("time_varying_known_continuous")
        if known_cont is not None:
            if not isinstance(known_cont, torch.Tensor) or known_cont.dim() != 3:
                raise ValueError("time_varying_known_continuous must be tensor [B,T,D]")
            for i, linear in enumerate(self.known_cont_linears):
                if i < known_cont.size(-1):
                    feats.append(linear(known_cont[:, :, i : i + 1]))

        obs_cat = inputs.get("time_varying_observed_categorical")
        if obs_cat is not None:
            obs_cat_list = self._split_cat_inputs(obs_cat)
            for emb, cat in zip(self.obs_cat_embeddings, obs_cat_list):
                feats.append(emb(cat))

        obs_cont = inputs.get("time_varying_observed_continuous")
        if obs_cont is not None:
            if not isinstance(obs_cont, torch.Tensor) or obs_cont.dim() != 3:
                raise ValueError("time_varying_observed_continuous must be tensor [B,T,D]")
            for i, linear in enumerate(self.obs_cont_linears):
                if i < obs_cont.size(-1):
                    feats.append(linear(obs_cont[:, :, i : i + 1]))

        if not feats:
            raise ValueError("No valid temporal features in inputs")

        # 将每一类输入都投影到 [B, T, H] 后再平均，保持实现简单且可解释。
        temporal = torch.stack(feats, dim=0).mean(dim=0)

        static_x = inputs.get("static")
        if self.static_linear is not None and isinstance(static_x, torch.Tensor):
            temporal = temporal + self.static_linear(static_x).unsqueeze(1)

        return temporal

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """构建 causal mask / Build autoregressive attention mask.

        True 表示该位置被屏蔽，确保每个时间步只能看到自己及过去信息。
        """
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        inputs: Dict[str, TensorLike],
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """前向传播 / Forward pass.

        流程：
        1. 编码多源时间特征 -> [B, T, H]
        2. LSTM 建模局部时序上下文 -> [B, T, H]
        3. causal self-attention 建模长依赖 -> [B, T, H]
        4. FFN 进一步变换并输出 quantile 预测 -> [B, T_pred, Q]
        """
        temporal = self._encode_temporal_features(inputs)
        lstm_out, _ = self.lstm(temporal)

        seq_len = lstm_out.size(1)
        attn_out, attn_weights = self.attention(
            lstm_out,
            lstm_out,
            lstm_out,
            attn_mask=self._causal_mask(seq_len, lstm_out.device),
            need_weights=True,
        )
        x = self.attn_norm(lstm_out + self.dropout(attn_out))

        ff = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ff))

        pred = x[:, self.num_encoder_steps :, :]
        out = self.output_layer(pred)

        if return_attention_weights:
            # attention_weights 对后续解释性分析和可视化很有帮助。
            return out, {"attention_weights": attn_weights}
        return out

    def get_attention_weights(self, inputs: Dict[str, TensorLike]) -> Dict[str, torch.Tensor]:
        """仅返回注意力权重 / Return attention weights only."""
        _, weights = self.forward(inputs, return_attention_weights=True)
        return weights
