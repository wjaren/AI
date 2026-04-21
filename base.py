# =========================
# Python 标准库
# =========================
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import seaborn as sns
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from collections import defaultdict
import time
import math
import copy                                            # 深拷贝对象
import gc                                              # 垃圾回收与显存/内存清理
import json                                            # JSON 读写
import math                                            # 数学函数
import os                                              # 操作系统与路径/环境变量
import random                                          # 随机数工具
import time                                            # 时间统计与计时
from collections import Counter, defaultdict           # 计数器与默认字典
from pathlib import Path                               # 面向对象路径处理
from typing import Any, Dict, List, Optional, Tuple    # 类型注解

# =========================
# 科学计算与可视化
# =========================
import cv2                                             # OpenCV 图像处理
import numpy as np                                     # 数值计算
import pandas as pd                                    # 表格数据处理
import seaborn as sns                                  # 统计可视化
from matplotlib import pyplot as plt                   # 画图
from PIL import Image                                  # PIL 图像对象
from tqdm import tqdm                                  # 进度条

# =========================
# 机器学习工具
# =========================
from sklearn.datasets import fetch_california_housing, fetch_openml  # 示例数据集
from sklearn.metrics import (
    accuracy_score,                                    # 分类准确率
    confusion_matrix,                                  # 混淆矩阵
    f1_score,                                          # F1 指标
    mean_squared_error,                                # 均方误差 MSE
    precision_score,                                   # 精确率
    recall_score,                                      # 召回率
    roc_auc_score,                                     # ROC-AUC
    classification_report,                             # 分类报告
    r2_score,                                          # 回归 R2
    mean_absolute_error,                               # 平均绝对误差 MAE
    mean_absolute_percentage_error,                    # 平均绝对百分比误差 MAPE
    root_mean_squared_error,                           # 均方根误差 RMSE
    top_k_accuracy_score,                              # Top-K 准确率
    log_loss,                                          # 对数损失
    hinge_loss,                                        # Hinge 损失
    matthews_corrcoef,                                 # MCC 指标
    balanced_accuracy_score,                           # 平衡准确率
    average_precision_score,                           # 平均精确率 AP
    cohen_kappa_score,                                 # Cohen's Kappa
    jaccard_score,                                     # Jaccard 相似度
    brier_score_loss,                                  # Brier 分数损失
    d2_absolute_error_score,                           # D2（absolute-error）
    d2_pinball_score,                                  # D2（pinball）
    fbeta_score,                                       # F-beta 指标
    hamming_loss,                                      # 汉明损失
    ndcg_score,                                        # NDCG 排序指标
    rand_score,                                        # Rand 聚类指标
    adjusted_rand_score,                               # 调整 Rand 指标
    v_measure_score,                                   # V-measure
    completeness_score,                                # 完整性得分
    homogeneity_score,                                 # 同质性得分
    normalized_mutual_info_score,                      # 归一化互信息 NMI
    silhouette_score,                                  # 轮廓系数
    davies_bouldin_score,                              # Davies-Bouldin 指数
    calinski_harabasz_score,                           # Calinski-Harabasz 指数
    pair_confusion_matrix,                             # 成对混淆矩阵
    multilabel_confusion_matrix                        # 多标签混淆矩阵
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # 交叉验证与划分
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler   # 预处理器

# =========================
# PyTorch 核心
# =========================
import torch                                           # PyTorch 主库
import torch.nn as nn                                  # 神经网络模块
import torch.nn.functional as F                        # 函数式层/激活
import torch.optim as optim                            # 优化器
from torch import autocast                             # 自动混合精度上下文
from torch.cuda.amp import GradScaler                  # 混合精度梯度缩放
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,                                 # 余弦退火学习率
    ExponentialLR,                                     # 指数衰减学习率
    OneCycleLR,                                        # OneCycle 策略
    ReduceLROnPlateau,                                 # 指标停滞时降学习率
    StepLR                                             # 阶梯式学习率
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, TensorDataset, random_split  # 数据加载与采样
from torch.utils.tensorboard import SummaryWriter      # TensorBoard 日志

# =========================
# TorchVision（CV）
# =========================
from torchvision import datasets, models, transforms   # 视觉数据集/模型/变换
from torchvision.transforms import (
    Compose,                                           # 组合多个变换
    ConvertImageDtype,                                 # 图像 dtype 转换
    Normalize,                                         # 标准化
    RandomHorizontalFlip,                              # 随机水平翻转
    RandomResizedCrop,                                 # 随机裁剪并缩放
    Resize,                                            # 调整尺寸
    ToTensor                                           # 转为张量
)

# =========================
# TensorFlow/Keras
# =========================
import tensorflow as tf                                # TensorFlow 主库

# =========================
# 训练常用基础设置
# =========================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)

    plt.rcParams['font.sans-serif'] = ['SimHei']           # 设置中文字体为 SimHei
    plt.rcParams['axes.unicode_minus'] = False             # 解决负号显示问题

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Torch version:', torch.__version__)

# =========================
# PyTorch 通用训练器：早停、训练曲线、断点续训、评估与结果分析
# =========================

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class TorchTrainer:
    """适用于常见 PyTorch 分类/回归任务的通用训练器。"""

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        scheduler: Optional[Any] = None,
        task: str = "auto",
        checkpoint_dir: Union[str, Path] = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        grad_clip: Optional[float] = None,
        amp: bool = False,
        accumulation_steps: int = 1,
        scheduler_step: str = "epoch",
        save_best: bool = True,
        save_last: bool = True,
        writer: Optional[Any] = None,
    ) -> None:
        # 保存训练所需的核心对象：模型、损失函数、优化器和调度器。
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # 如果外部没有指定设备，则优先使用 CUDA，否则回退到 CPU。
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        # 统一把 checkpoint_dir 转为 Path，便于跨平台拼接路径。
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        # 混合精度只在 CUDA 上启用，CPU 环境下自动关闭。
        self.amp = bool(amp and self.device.type == "cuda")
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.scheduler_step = scheduler_step
        self.save_best = save_best
        self.save_last = save_last
        self.writer = writer

        # 将模型移动到目标设备，并初始化早停、历史记录和断点状态。
        self.model.to(self.device)
        self.scaler = self._make_grad_scaler(self.amp)
        self.best_score = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self.no_improve_epochs = 0
        self.start_epoch = 1
        self.history = defaultdict(list)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        resume_from: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        # 如果传入已有 checkpoint，则从该断点继续训练。
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        # 主训练循环：每轮训练、验证、记录指标、保存断点并判断早停。
        stop_training = False
        for epoch in range(self.start_epoch, epochs + 1):
            started_at = time.time()
            # 训练一个 epoch，返回平均 loss 和自动计算出的任务指标。
            train_logs = self._run_one_epoch(train_loader, train=True)
            logs = {f"train_{k}": v for k, v in train_logs.items()}

            # 如果提供验证集，则额外跑验证流程，用于早停和模型选择。
            if val_loader is not None:
                val_logs = self._run_one_epoch(val_loader, train=False)
                logs.update({f"val_{k}": v for k, v in val_logs.items()})

            # 学习率调度器支持三种更新节奏：epoch、batch、plateau。
            if self.scheduler is not None and self.scheduler_step == "epoch":
                self.scheduler.step()
            if self.scheduler is not None and self.scheduler_step == "plateau":
                metric = logs.get(self.monitor)
                if metric is not None:
                    self.scheduler.step(metric)

            # 记录当前学习率和单轮耗时，方便后续画训练曲线。
            logs["lr"] = self._current_lr()
            logs["epoch_seconds"] = time.time() - started_at
            for key, value in logs.items():
                self.history[key].append(float(value))
                if self.writer is not None:
                    self.writer.add_scalar(key, float(value), epoch)

            # last.pt 始终保存最近一轮，适合中断后继续训练。
            if self.save_last:
                self.save_checkpoint(
                    self.checkpoint_dir / "last.pt", epoch, is_best=False)

            # 根据 monitor 判断是否刷新最佳模型，并累计无改善轮数。
            if self.monitor in logs:
                improved = self._is_improved(logs[self.monitor])
                if improved:
                    self.best_score = float(logs[self.monitor])
                    self.best_epoch = epoch
                    self.no_improve_epochs = 0
                    if self.save_best:
                        self.save_checkpoint(
                            self.checkpoint_dir / "best.pt", epoch, is_best=True)
                else:
                    self.no_improve_epochs += 1
                    stop_training = self.no_improve_epochs >= self.patience

            if verbose:
                print(self._format_epoch_log(epoch, epochs, logs))

            if stop_training:
                if verbose:
                    print(
                        f"早停触发：{self.monitor} 连续 {self.patience} 轮没有改善，最佳 epoch = {self.best_epoch}")
                break

        if self.writer is not None:
            self.writer.flush()
        return dict(self.history)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        return self._run_one_epoch(loader, train=False)

    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 预测阶段固定为 eval 模式，并关闭梯度计算。
        self.model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack_batch(batch)
                outputs = self.model(inputs)
                y_true.append(targets.detach().cpu())
                y_score.append(outputs.detach().cpu())
                y_pred.append(self._to_prediction(outputs).detach().cpu())
        return (
            torch.cat(y_true).numpy(),
            torch.cat(y_pred).numpy(),
            torch.cat(y_score).numpy(),
        )

    def plot_history(self, metrics: Optional[Sequence[str]] = None, figsize: Tuple[int, int] = (14, 4)) -> None:
        # 绘制训练/验证指标曲线，例如 loss、acc、f1、rmse、r2。
        history = dict(self.history)
        if not history:
            raise ValueError("history 为空，请先调用 fit()。")

        metrics = list(metrics or self._infer_plot_metrics(history))
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        axes = np.atleast_1d(axes)
        x = np.arange(1, len(next(iter(history.values()))) + 1)

        for ax, metric in zip(axes, metrics):
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"
            if train_key in history:
                ax.plot(x, history[train_key], label=train_key)
            if val_key in history:
                ax.plot(x, history[val_key], label=val_key)
            if metric in history:
                ax.plot(x, history[metric], label=metric)
            ax.set_title(metric)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(
        self,
        loader: DataLoader,
        class_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (6, 5),
    ) -> None:
        # 分类任务分析：根据预测类别和真实类别绘制混淆矩阵。
        y_true, y_pred, _ = self.predict(loader)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("预测类别")
        plt.ylabel("真实类别")
        plt.title("混淆矩阵")
        plt.gcf().set_size_inches(figsize)
        plt.tight_layout()
        plt.show()

    def plot_regression_analysis(self, loader: DataLoader, figsize: Tuple[int, int] = (12, 4)) -> None:
        # 回归任务分析：生成拟合图、残差图和残差分布图。
        y_true, y_pred, _ = self.predict(loader)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        # 残差 = 真实值 - 预测值，用来观察模型是否存在系统性偏差。
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        low, high = min(y_true.min(), y_pred.min()), max(
            y_true.max(), y_pred.max())
        axes[0].plot([low, high], [low, high], "r--")
        axes[0].set_title("真实值 vs 预测值")
        axes[0].set_xlabel("真实值")
        axes[0].set_ylabel("预测值")

        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(0, color="r", linestyle="--")
        axes[1].set_title("残差图")
        axes[1].set_xlabel("预测值")
        axes[1].set_ylabel("残差")

        sns.histplot(residuals, kde=True, ax=axes[2])
        axes[2].set_title("残差分布")
        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, path: Union[str, Path], epoch: int, is_best: bool = False) -> None:
        # checkpoint 中保存完整训练现场：模型、优化器、调度器、AMP、历史指标和早停状态。
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "history": dict(self.history),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "no_improve_epochs": self.no_improve_epochs,
            "monitor": self.monitor,
            "mode": self.mode,
            "is_best": is_best,
        }
        torch.save(checkpoint, Path(path))

    def load_checkpoint(self, path: Union[str, Path], map_location: Optional[Any] = None) -> Dict[str, Any]:
        # 恢复 checkpoint 后，下一次 fit 会从 epoch + 1 开始。
        checkpoint = torch.load(
            Path(path), map_location=map_location or self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.history = defaultdict(list, checkpoint.get("history", {}))
        self.best_score = checkpoint.get("best_score", self.best_score)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.no_improve_epochs = checkpoint.get("no_improve_epochs", 0)
        self.start_epoch = int(checkpoint["epoch"]) + 1
        return checkpoint

    def _make_grad_scaler(self, enabled: bool) -> Any:
        return GradScaler("cuda", enabled=enabled)

    def _run_one_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        # train=True 时启用训练模式；train=False 时启用评估模式。
        self.model.train(train)
        total_loss, total_samples = 0.0, 0
        all_targets, all_outputs = [], []
        iterator = tqdm(loader, leave=False, desc="train" if train else "eval")

        # 开始训练前清空梯度，set_to_none=True 通常更省显存。
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(iterator, start=1):
            # 支持 batch 为 (x, y) 或字典格式，并自动移动到 device。
            inputs, targets = self._unpack_batch(batch)
            batch_size = self._batch_size(targets)

            # 训练时开启梯度，验证/测试时关闭梯度，减少内存开销。
            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    # 梯度累积时要把 loss 平均到每个累积步，避免梯度被放大。
                    loss_for_backward = loss / self.accumulation_steps

                if train:
                    self.scaler.scale(loss_for_backward).backward()
                    # 达到累积步数或最后一个 batch 时，才真正更新一次参数。
                    should_step = step % self.accumulation_steps == 0 or step == len(
                        loader)
                    if should_step:
                        if self.grad_clip is not None:
                            self.scaler.unscale_(self.optimizer)
                            # 梯度裁剪可以缓解梯度爆炸，常用于 RNN 或不稳定训练。
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler is not None and self.scheduler_step == "batch":
                            self.scheduler.step()

            # 按样本数加权累计 loss，避免最后一个小 batch 影响均值。
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            all_targets.append(targets.detach().cpu())
            all_outputs.append(outputs.detach().cpu())
            iterator.set_postfix(loss=total_loss / max(total_samples, 1))

        # 汇总整轮输出后统一计算指标，比逐 batch 平均更准确。
        targets_np = torch.cat(all_targets).numpy()
        outputs_tensor = torch.cat(all_outputs)
        logs = {"loss": total_loss / max(total_samples, 1)}
        logs.update(self._compute_metrics(targets_np, outputs_tensor))
        return logs

    def _unpack_batch(self, batch: Any) -> Tuple[Any, torch.Tensor]:
        # 兼容字典 batch：常见键包括 x/inputs/image/features 和 y/target/label/labels。
        if isinstance(batch, dict):
            inputs = batch.get("x", batch.get(
                "inputs", batch.get("image", batch.get("features"))))
            targets = batch.get("y", batch.get(
                "target", batch.get("label", batch.get("labels"))))
        else:
            inputs, targets = batch[0], batch[1]
        return self._to_device(inputs), self._to_device(targets)

    def _to_device(self, data: Any) -> Any:
        # 递归处理 tensor、dict、list、tuple，适配多输入模型。
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._to_device(v) for v in data)
        return data

    def _batch_size(self, targets: torch.Tensor) -> int:
        return int(targets.shape[0]) if hasattr(targets, "shape") and targets.ndim > 0 else 1

    def _compute_metrics(self, targets_np: np.ndarray, outputs_tensor: torch.Tensor) -> Dict[str, float]:
        # 自动识别分类/回归任务，也可以通过 task 参数手动指定。
        task = self._resolve_task(targets_np, outputs_tensor)
        if task == "classification":
            # 分类指标：多分类取 argmax，二分类默认 sigmoid 后以 0.5 为阈值。
            preds = self._to_prediction(outputs_tensor).numpy().reshape(-1)
            targets = targets_np.reshape(-1)
            if not np.issubdtype(targets.dtype, np.integer):
                targets = (targets >= 0.5).astype(int)
            metrics = {
                "acc": accuracy_score(targets, preds),
                "balanced_acc": balanced_accuracy_score(targets, preds),
                "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
            }
            if outputs_tensor.ndim == 2 and outputs_tensor.shape[1] > 1:
                prob = torch.softmax(outputs_tensor, dim=1).numpy()
                if prob.shape[1] == 2:
                    metrics["auc"] = roc_auc_score(targets, prob[:, 1])
            elif len(np.unique(targets)) == 2:
                prob = torch.sigmoid(outputs_tensor.reshape(-1)).numpy()
                metrics["auc"] = roc_auc_score(targets, prob)
            return metrics

        # 回归指标：MAE、MSE、RMSE 和 R2。
        preds = outputs_tensor.numpy().reshape(-1)
        targets = targets_np.reshape(-1)
        return {
            "mae": mean_absolute_error(targets, preds),
            "mse": mean_squared_error(targets, preds),
            "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
            "r2": r2_score(targets, preds),
        }

    def _resolve_task(self, targets_np: np.ndarray, outputs_tensor: torch.Tensor) -> str:
        # task=auto 时，根据输出形状和标签类型推断任务类型。
        if self.task in {"classification", "regression"}:
            return self.task
        if outputs_tensor.ndim == 2 and outputs_tensor.shape[1] > 1:
            return "classification"
        unique_targets = np.unique(targets_np)
        if np.issubdtype(targets_np.dtype, np.integer) and len(unique_targets) <= 50:
            return "classification"
        if len(unique_targets) <= 2 and set(unique_targets.tolist()).issubset({0, 1, 0.0, 1.0}):
            return "classification"
        return "regression"

    def _to_prediction(self, outputs: torch.Tensor) -> torch.Tensor:
        # 将模型原始输出转换为最终预测结果。
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            return outputs.argmax(dim=1)
        if self.task == "classification":
            return (torch.sigmoid(outputs).reshape(-1) >= 0.5).long()
        return outputs.reshape(-1)

    def _is_improved(self, value: float) -> bool:
        # mode=min 表示指标越小越好，例如 val_loss；mode=max 表示越大越好，例如 val_acc。
        if self.mode == "min":
            return value < self.best_score - self.min_delta
        return value > self.best_score + self.min_delta

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _format_epoch_log(self, epoch: int, epochs: int, logs: Dict[str, float]) -> str:
        pieces = [f"Epoch {epoch}/{epochs}"]
        for key, value in logs.items():
            pieces.append(f"{key}={value:.4f}")
        return " | ".join(pieces)

    def _infer_plot_metrics(self, history: Dict[str, List[float]]) -> List[str]:
        candidates = ["loss", "acc", "f1_macro", "mae", "rmse", "r2", "lr"]
        return [m for m in candidates if m in history or f"train_{m}" in history or f"val_{m}" in history]


def test():
    train_ds = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    val_ds = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    mean = train_ds.data.float().mean() / 255
    std = train_ds.data.float().std() / 255
    transforms = nn.Sequential(
        Normalize(mean, std)
    )

    class NeuralNetwork(nn.Module):
        def __init__(self, layers_num=3, transforms=None, activate=nn.ReLU(), dropout=0):
            super(NeuralNetwork, self).__init__()
            self.transform = transforms
            self.flatten = nn.Flatten()
            self.layers = nn.Sequential()

            self.layers.add_module('input', nn.Linear(28 * 28, 64))

            self.layers.add_module('batchnorm_0', nn.BatchNorm1d(64))
            self.layers.add_module('relu_0', activate)
            if dropout > 0:
                self.layers.add_module('dropout_0', nn.Dropout(dropout))

            for i in range(1, layers_num):
                self.layers.add_module(f'linear_{i}', nn.Linear(64, 64))
                self.layers.add_module(f'batchnorm_{i}', nn.BatchNorm1d(64))
                self.layers.add_module(
                    f'{activate.__class__.__name__.lower()}_{i}', activate)
                if dropout > 0:
                    self.layers.add_module(f'dropout_{i}', nn.Dropout(dropout))

            self.layers.add_module('output', nn.Linear(64, 10))
            self.init_weights()

        def init_weights(self):
            for m in self.layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            x = self.transform(x) if self.transform else x
            x = self.flatten(x)
            logits = self.layers(x)
            return logits

    model = NeuralNetwork(
        layers_num=3, transforms=transforms,
        activate=nn.ReLU(), dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step="plateau",
        monitor="val_loss",
        mode="min",
        patience=5,
        device=device,
    )

    trainer.fit(train_loader, val_loader, epochs=50)

    trainer.plot_history()
    trainer.plot_confusion_matrix(
        val_loader, class_names=train_ds.classes, normalize=True)

    print("测试通过！")


if __name__ == "__main__":
    test()
