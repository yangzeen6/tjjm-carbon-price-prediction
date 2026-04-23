"""PyTorch TFT 端到端入口 / End-to-end entrypoint for the TFT workflow.

执行顺序 / Pipeline:
1. load config -> 2. create directories -> 3. load or generate data
4. preprocess and build DataLoader -> 5. train model -> 6. evaluate and visualize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from base_model import BaseTemporalFusionTransformer
from data_processor import (
    GeneratedDataset,
    TFTDataProcessor,
    train_test_split_time_series,
    train_val_test_split_by_id,
)
from evaluation import TFTEvaluator, TFTVisualizer
from train import CustomDataset, TFTTrainer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """加载 JSON 配置 / Load a JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict, save_path: str):
    """保存配置 / Save config to disk."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def setup_directories(config: Dict) -> Dict[str, str]:
    """创建实验目录结构 / Create the experiment directory tree."""
    base_dir = config.get("base_dir", "output")
    experiment_name = config.get("experiment_name", f"tft_torch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    dirs = {
        "base": base_dir,
        "experiment": os.path.join(base_dir, experiment_name),
        "models": os.path.join(base_dir, experiment_name, "models"),
        "logs": os.path.join(base_dir, experiment_name, "logs"),
        "plots": os.path.join(base_dir, experiment_name, "plots"),
        "results": os.path.join(base_dir, experiment_name, "results"),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs


def create_default_config() -> Dict:
    """创建默认配置 / Build a default runnable configuration."""
    return {
        "experiment_name": "tft_torch_default",
        "base_dir": "output",
        "seed": 42,
        "model": {
            "hidden_size": 160,
            "dropout": 0.1,
            "num_heads": 4,
            "num_lstm_layers": 1,
            "num_encoder_steps": 168,
            "num_prediction_steps": 24,
            "quantiles": [0.1, 0.5, 0.9],
        },
        "data": {
            "use_generated": True,
            "data_path": None,
            "split_method": "auto",
            "num_samples": 1000,
            "sequence_length": 192,
            "num_features": 5,
            "static_columns": [],
            "time_varying_known_categorical": ["day_of_week", "month"],
            "time_varying_known_continuous": [],
            "time_varying_observed_categorical": [],
            "time_varying_observed_continuous": [
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
            ],
            "target_column": "target",
            "time_idx_column": "time_idx",
            "id_column": "id",
        },
        "train": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "patience": 15,
            "reduce_lr_patience": 5,
            "validation_split": 0.1,
            "test_split": 0.2,
        },
    }


def set_random_seeds(seed: int = 42):
    """设置随机种子 / Set RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_or_generate_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载真实数据或生成合成数据，并完成 train/val/test 切分。"""
    data_cfg = config["data"]

    if data_cfg["use_generated"]:
        generator = GeneratedDataset(
            num_samples=data_cfg["num_samples"],
            sequence_length=data_cfg["sequence_length"],
            num_features=data_cfg["num_features"],
        )
        data = generator.generate()
    else:
        path = data_cfg["data_path"]
        if path.endswith(".csv"):
            data = pd.read_csv(path)
        elif path.endswith(".parquet"):
            data = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported data format: {path}")

    split_method = (data_cfg.get("split_method") or "auto").lower()
    use_id_split = split_method == "id" or (split_method == "auto" and data_cfg.get("use_generated", False))

    if use_id_split:
        train_df, val_df, test_df = train_val_test_split_by_id(
            data,
            val_size=config["train"]["validation_split"],
            test_size=config["train"]["test_split"],
            id_column=data_cfg["id_column"],
            seed=config.get("seed", 42),
        )
    else:
        train_df, test_df = train_test_split_time_series(
            data,
            test_size=config["train"]["test_split"],
            time_idx_column=data_cfg["time_idx_column"],
        )
        train_df, val_df = train_test_split_time_series(
            train_df,
            test_size=config["train"]["validation_split"],
            time_idx_column=data_cfg["time_idx_column"],
        )

    return train_df, val_df, test_df


def prepare_data(config: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """把原始 DataFrame 转成训练器可直接消费的 DataLoader。"""
    data_cfg = config["data"]
    train_cfg = config["train"]

    processor = TFTDataProcessor(
        num_encoder_steps=config["model"]["num_encoder_steps"],
        num_prediction_steps=config["model"]["num_prediction_steps"],
        static_columns=data_cfg["static_columns"],
        time_varying_known_categorical=data_cfg["time_varying_known_categorical"],
        time_varying_known_continuous=data_cfg["time_varying_known_continuous"],
        time_varying_observed_categorical=data_cfg["time_varying_observed_categorical"],
        time_varying_observed_continuous=data_cfg["time_varying_observed_continuous"],
        target_column=data_cfg["target_column"],
        time_idx_column=data_cfg["time_idx_column"],
        id_column=data_cfg["id_column"],
    )

    train_p = processor.fit_transform(train_df)
    val_p = processor.transform(val_df)
    test_p = processor.transform(test_df)

    train_loader = CustomDataset(train_p, processor, batch_size=train_cfg["batch_size"], shuffle=True).to_dataloader()
    val_loader = CustomDataset(val_p, processor, batch_size=train_cfg["batch_size"], shuffle=False).to_dataloader()
    test_loader = CustomDataset(test_p, processor, batch_size=train_cfg["batch_size"], shuffle=False).to_dataloader()
    return processor, train_loader, val_loader, test_loader


def build_model(config: Dict) -> BaseTemporalFusionTransformer:
    """根据配置构建 TFT 模型 / Build the TFT model from config."""
    model_cfg = config["model"].copy()
    data_cfg = config["data"]

    model_cfg["time_varying_known_categorical_dims"] = [7, 12] if data_cfg["time_varying_known_categorical"] else []
    model_cfg["time_varying_known_continuous_dims"] = len(data_cfg["time_varying_known_continuous"])
    model_cfg["time_varying_observed_categorical_dims"] = []
    model_cfg["time_varying_observed_continuous_dims"] = len(data_cfg["time_varying_observed_continuous"])
    model_cfg["static_input_dim"] = len(data_cfg["static_columns"])

    return BaseTemporalFusionTransformer(**model_cfg)


def run_pipeline(config: Dict) -> Dict[str, float]:
    """执行完整训练与评估流水线 / Run the full train-eval pipeline."""
    set_random_seeds(config.get("seed", 42))
    dirs = setup_directories(config)

    config_path = os.path.join(dirs["experiment"], "config.json")
    save_config(config, config_path)

    train_df, val_df, test_df = load_or_generate_data(config)
    processor, train_loader, val_loader, test_loader = prepare_data(config, train_df, val_df, test_df)

    # 这里先单独构建模型，再交给 trainer，方便后续扩展到预训练权重或替换子模块。
    model = build_model(config)
    trainer = TFTTrainer(config["model"], model_dir=dirs["models"], experiment_name=config["experiment_name"])
    trainer.model = model.to(trainer.device)

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=config["train"]["epochs"],
        learning_rate=config["train"]["learning_rate"],
        patience=config["train"]["patience"],
        reduce_lr_patience=config["train"]["reduce_lr_patience"],
    )

    metrics = trainer.evaluate(test_loader)
    trainer.save_model(os.path.join(dirs["models"], "final_model.pt"))

    evaluator = TFTEvaluator(trainer.model, data_processor=processor, device=str(trainer.device))
    first_batch_inputs, first_batch_targets = next(iter(test_loader))
    np_inputs = {k: v.numpy() for k, v in first_batch_inputs.items()}
    np_targets = first_batch_targets.numpy()
    eval_metrics = evaluator.evaluate(np_inputs, np_targets, config["model"]["quantiles"])

    metrics_path = os.path.join(dirs["results"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    visualizer = TFTVisualizer(save_dir=dirs["plots"])
    preds = evaluator.predict_from_inputs(np_inputs)
    if processor is not None:
        # 预测结果通常停留在标准化空间，这里统一反变换回原始价格尺度。
        y_true = processor.inverse_transform_target(np_targets)
        preds_inv = np.zeros_like(preds)
        for i in range(preds.shape[-1]):
            preds_inv[:, :, i] = processor.inverse_transform_target(preds[:, :, i])
        preds = preds_inv
    else:
        y_true = np_targets

    visualizer.plot_predictions(
        y_true=y_true,
        y_pred=preds,
        quantiles=config["model"]["quantiles"],
        num_samples=min(5, len(y_true)),
        save_path=os.path.join(dirs["plots"], "predictions.png"),
    )

    logger.info("Evaluation metrics: %s", eval_metrics)
    return eval_metrics


def main():
    """CLI 入口 / Command-line entrypoint."""
    parser = argparse.ArgumentParser(description="PyTorch TFT pipeline")
    parser.add_argument("--config", type=str, default="", help="Path to JSON config")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()

    run_pipeline(config)


if __name__ == "__main__":
    main()
