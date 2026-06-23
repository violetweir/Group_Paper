#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asym_GroupedMamba_UNet + DINOv3 decoder1 语义蒸馏 + 稳定版 GAM 训练脚本。

核心设计：
1. 学生网络在训练阶段返回 decoder1 后、up1 前的 H/32 深层解码语义特征。
2. DINOv3 教师参数完全冻结，仅在有效蒸馏权重大于 0 时执行前向。
3. 蒸馏损失采用 L2 归一化后的逐位置余弦距离。
4. 支持纯分割预热、蒸馏线性 warm-up 和延迟启动 GAM。
5. GAM 在梯度裁剪前统计 decoder1 梯度占全模型梯度的比例。
6. GAM 单个 Epoch 的权重变化默认限制在 0.8～1.25 倍，并设置绝对上下限。
7. 同时报告标准 sigmoid 阈值指标和旧版逐图 Min-Max 指标，便于公平对比。
8. 每个 Run 独立初始化随机种子、GAM 状态和蒸馏权重。
"""

import argparse
import copy
import csv
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# 同时兼容：
# 1) 脚本放在项目 tools/ 目录；
# 2) 脚本与网络文件放在同一目录。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for path in (CURRENT_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from networks.Asym_GroupedMamba_UNet_DSI_GAM_v3 import (
        Asym_GroupedMamba_UNet_T,
        Asym_GroupedMamba_UNet_S,
        Asym_GroupedMamba_UNet,
        Asym_GroupedMamba_UNet_M,
        Asym_GroupedMamba_UNet_L,
        Asym_GroupedMamba_UNet_XL,
    )
except ModuleNotFoundError:
    from Asym_GroupedMamba_UNet_DSI_GAM import (
        Asym_GroupedMamba_UNet_T,
        Asym_GroupedMamba_UNet_S,
        Asym_GroupedMamba_UNet,
        Asym_GroupedMamba_UNet_M,
        Asym_GroupedMamba_UNet_L,
        Asym_GroupedMamba_UNet_XL,
    )

from teacher.dinov3_teacher import MKUNetDINOv3Teacher
from utils.dataloader_polyp import get_loader
from utils.utils import AvgMeter, cal_params_flops, clip_gradient


NET_MODELS = {
    "Tiny": Asym_GroupedMamba_UNet_T,
    "Small": Asym_GroupedMamba_UNet_S,
    "Base": Asym_GroupedMamba_UNet,
    "Medium": Asym_GroupedMamba_UNet_M,
    "Large": Asym_GroupedMamba_UNet_L,
    "XLarge": Asym_GroupedMamba_UNet_XL,
}


@dataclass
class GAMState:
    enabled: bool
    rho: float
    delta: float
    current_weight: float
    default_weight: float
    start_epoch: int
    stop_epoch: int
    min_weight: float
    max_weight: float
    step_down_factor: float
    step_up_factor: float


@dataclass
class BestState:
    best_val_dice: float = 0.0
    test_dice_at_best_val: float = 0.0
    best_epoch: int = 0


class ModelEMA:
    """
    轻量 EMA 封装：训练时更新原模型，验证/保存时可使用 EMA 权重。

    设计参考常见检测/分割训练流程：
    - module 保存滑动平均后的模型副本；
    - decay 控制平滑强度；
    - warmups 用于前期逐步增大 decay，避免训练初期 EMA 过慢。
    """
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        warmups: int = 1000,
        start: int = 0,
    ) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

        self.decay = float(decay)
        self.warmups = int(warmups)
        self.start = int(start)
        self.updates = 0

    def _decay(self) -> float:
        if self.updates <= self.start:
            return 0.0
        if self.warmups > 0:
            return self.decay * (1.0 - math.exp(-(self.updates - self.start) / self.warmups))
        return self.decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = self._decay()
        model_state = model.state_dict()

        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach().to(device=ema_value.device)
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value.to(dtype=ema_value.dtype))


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_float_list(value: str) -> List[float]:
    rates = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not rates:
        raise argparse.ArgumentTypeError("size_rates 不能为空")
    if any(rate <= 0 for rate in rates):
        raise argparse.ArgumentTypeError("size_rates 必须全部大于 0")
    return rates


def parse_target_prefixes(value: str) -> Tuple[str, ...]:
    prefixes: List[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        prefixes.append(item if item.endswith(".") else item + ".")
    if not prefixes:
        raise ValueError("GAM 目标模块不能为空")
    return tuple(prefixes)


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def structure_loss(pred: torch.Tensor, mask: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(1e-8)

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (weight * (wbce + wiou)).mean()


def dice_coefficient(predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


def iou_score(predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def unwrap_feature_tensor(output: Any) -> torch.Tensor:
    """从教师模型可能返回的 Tensor/list/tuple/dict 中提取特征 Tensor。"""
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        preferred_keys = (
            "feature",
            "features",
            "out",
            "last_hidden_state",
            "x_norm_patchtokens",
        )
        for key in preferred_keys:
            if key in output:
                try:
                    return unwrap_feature_tensor(output[key])
                except (TypeError, ValueError):
                    pass
        for value in output.values():
            try:
                return unwrap_feature_tensor(value)
            except (TypeError, ValueError):
                continue

    if isinstance(output, (list, tuple)):
        for value in reversed(output):
            try:
                return unwrap_feature_tensor(value)
            except (TypeError, ValueError):
                continue

    raise TypeError(f"无法从教师输出中提取 Tensor，实际类型: {type(output)}")


def compute_distill_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    if student_features.ndim != 4 or teacher_features.ndim != 4:
        raise ValueError(
            "蒸馏特征必须为 B×C×H×W，"
            f"学生={tuple(student_features.shape)}, 教师={tuple(teacher_features.shape)}"
        )

    if teacher_features.shape[-2:] != student_features.shape[-2:]:
        teacher_features = F.interpolate(
            teacher_features,
            size=student_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    teacher_features = teacher_features.to(
        device=student_features.device,
        dtype=student_features.dtype,
    )

    if teacher_features.shape[1] != student_features.shape[1]:
        raise RuntimeError(
            "教师与学生投影后的通道数不一致："
            f"student={student_features.shape[1]}, teacher={teacher_features.shape[1]}。"
            "请检查 --teacher_dim、--dinov3_model_name 和教师权重是否匹配。"
        )

    student_tokens = student_features.flatten(2).transpose(1, 2)
    teacher_tokens = teacher_features.flatten(2).transpose(1, 2).detach()

    student_tokens = F.normalize(student_tokens, p=2, dim=-1, eps=1e-6)
    teacher_tokens = F.normalize(teacher_tokens, p=2, dim=-1, eps=1e-6)

    cosine_similarity = F.cosine_similarity(
        student_tokens,
        teacher_tokens,
        dim=-1,
        eps=1e-6,
    )
    return (1.0 - cosine_similarity).mean()


def get_effective_distill_weight(
    base_weight: float,
    epoch: int,
    start_epoch: int,
    warmup_epochs: int,
    stop_epoch: int = 0,
) -> Tuple[float, float]:
    """
    返回当前 Epoch 实际使用的蒸馏权重与 warm-up 比例。

    例如 start_epoch=10, warmup_epochs=20：
    - Epoch 1~9: scale=0，纯分割训练；
    - Epoch 10~29: scale 从 1/20 线性增加到 1；
    - Epoch >=30: scale=1。
    """
    if base_weight <= 0.0 or epoch < start_epoch:
        return 0.0, 0.0

    # stop_epoch > 0 时，从该 Epoch 开始真正关闭 DINOv3 蒸馏。
    if stop_epoch > 0 and epoch >= stop_epoch:
        return 0.0, 0.0

    if warmup_epochs <= 0:
        return float(base_weight), 1.0

    scale = (epoch - start_epoch + 1) / float(warmup_epochs)
    scale = min(max(scale, 0.0), 1.0)
    return float(base_weight) * scale, scale


def compute_target_grad_percentage(
    model: torch.nn.Module,
    target_prefixes: Sequence[str],
) -> float:
    """
    分子：指定目标模块的梯度 L1 范数。
    分母：整个学生网络所有有效参数梯度的 L1 范数。

    必须在梯度裁剪之前调用。
    """
    total_l1 = 0.0
    target_l1 = 0.0

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        clean_name = name[len("module."):] if name.startswith("module.") else name
        grad_l1 = parameter.grad.detach().abs().sum().item()
        if not math.isfinite(grad_l1):
            continue

        total_l1 += grad_l1
        if any(clean_name.startswith(prefix) for prefix in target_prefixes):
            target_l1 += grad_l1

    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * target_l1 / total_l1


def update_gam_weight(
    state: GAMState,
    avg_percentage: float,
    epoch: int,
) -> Tuple[float, float, str]:
    """在 Epoch 结束后更新下一 Epoch 的基础蒸馏权重。"""
    old_weight = float(state.current_weight)
    new_weight = old_weight
    reason = "unchanged_in_range"

    if not state.enabled:
        return old_weight, old_weight, "gam_disabled"

    if epoch < state.start_epoch:
        return old_weight, old_weight, "before_gam_start"

    if avg_percentage <= 1e-6 or not math.isfinite(avg_percentage):
        new_weight = state.default_weight
        reason = "reset_to_default_invalid_grad"
    elif state.stop_epoch > 0 and epoch >= state.stop_epoch:
        new_weight = state.default_weight
        reason = "fixed_default_weight_phase"
    else:
        lower_bound = state.rho - state.delta
        upper_bound = state.rho + state.delta

        if avg_percentage < lower_bound or avg_percentage > upper_bound:
            target_percentage = (
                upper_bound if avg_percentage < lower_bound else lower_bound
            )

            p_current = min(max(avg_percentage / 100.0, 1e-9), 1.0 - 1e-9)
            p_target = min(max(target_percentage / 100.0, 1e-9), 1.0 - 1e-9)

            numerator = p_target * (1.0 - p_current)
            denominator = p_current * (1.0 - p_target)

            if abs(denominator) >= 1e-12:
                raw_ratio = numerator / denominator
                base_weight = (
                    old_weight if old_weight > 1e-12
                    else max(state.default_weight, state.min_weight)
                )

                # 比原始 0.1~10 倍更保守，防止几轮内权重爆炸或坍缩。
                bounded_ratio = min(
                    max(raw_ratio, state.step_down_factor),
                    state.step_up_factor,
                )
                new_weight = base_weight * bounded_ratio
                reason = (
                    f"adjusted_to_{target_percentage:.4f}%_"
                    f"ratio_{bounded_ratio:.4f}"
                )
            else:
                new_weight = state.default_weight
                reason = "reset_to_default_bad_denominator"

    new_weight = min(max(float(new_weight), state.min_weight), state.max_weight)
    state.current_weight = new_weight
    return old_weight, new_weight, reason

def evaluate(
    model: torch.nn.Module,
    root_path: str,
    dataset: str,
    opt: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    """
    同时计算两套指标：
    1. standard：sigmoid 后直接固定阈值；
    2. minmax：复现旧脚本的逐图 Min-Max 后固定阈值。

    返回的 dice/iou 由 --eval_minmax 决定，用于保存最佳模型。
    """
    data_path = os.path.join(root_path, dataset)
    image_root = os.path.join(data_path, "images") + os.sep
    gt_root = os.path.join(data_path, "masks") + os.sep

    model.eval()
    loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        split="test",
        color_image=opt.color_image,
    )

    standard_dice_sum = 0.0
    standard_iou_sum = 0.0
    minmax_dice_sum = 0.0
    minmax_iou_sum = 0.0
    raw_prob_min_sum = 0.0
    raw_prob_max_sum = 0.0
    raw_prob_mean_sum = 0.0
    standard_fg_ratio_sum = 0.0
    minmax_fg_ratio_sum = 0.0
    total_images = 0

    with torch.no_grad():
        for pack in loader:
            images, gts, original_shapes, _ = pack
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, dtype=torch.float32, non_blocking=True)

            outputs = model(images)
            predictions = outputs[0] if isinstance(outputs, list) else outputs

            for index in range(images.shape[0]):
                h_orig = int(original_shapes[0][index])
                w_orig = int(original_shapes[1][index])

                raw_prob = predictions[index:index + 1]
                raw_prob = F.interpolate(
                    raw_prob,
                    size=(h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid().squeeze()

                gt = gts[index:index + 1]
                gt = F.interpolate(
                    gt,
                    size=(h_orig, w_orig),
                    mode="nearest",
                ).squeeze()
                target_binary = (gt >= opt.gt_threshold).float()

                pred_standard = (raw_prob >= opt.pred_threshold).float()
                normalized_prob = (raw_prob - raw_prob.min()) / (
                    raw_prob.max() - raw_prob.min() + 1e-8
                )
                pred_minmax = (normalized_prob >= opt.pred_threshold).float()

                standard_dice_sum += dice_coefficient(
                    pred_standard, target_binary
                ).item()
                standard_iou_sum += iou_score(
                    pred_standard, target_binary
                ).item()
                minmax_dice_sum += dice_coefficient(
                    pred_minmax, target_binary
                ).item()
                minmax_iou_sum += iou_score(
                    pred_minmax, target_binary
                ).item()

                raw_prob_min_sum += raw_prob.min().item()
                raw_prob_max_sum += raw_prob.max().item()
                raw_prob_mean_sum += raw_prob.mean().item()
                standard_fg_ratio_sum += pred_standard.mean().item()
                minmax_fg_ratio_sum += pred_minmax.mean().item()
                total_images += 1

    if total_images == 0:
        raise RuntimeError(f"数据集 {dataset} 中没有可评估图像")

    standard_dice = standard_dice_sum / total_images
    standard_iou = standard_iou_sum / total_images
    minmax_dice = minmax_dice_sum / total_images
    minmax_iou = minmax_iou_sum / total_images

    if opt.eval_minmax:
        primary_dice, primary_iou = minmax_dice, minmax_iou
    else:
        primary_dice, primary_iou = standard_dice, standard_iou

    return {
        "dice": primary_dice,
        "iou": primary_iou,
        "n": float(total_images),
        "standard_dice": standard_dice,
        "standard_iou": standard_iou,
        "minmax_dice": minmax_dice,
        "minmax_iou": minmax_iou,
        "prob_min": raw_prob_min_sum / total_images,
        "prob_max": raw_prob_max_sum / total_images,
        "prob_mean": raw_prob_mean_sum / total_images,
        "standard_fg_ratio": standard_fg_ratio_sum / total_images,
        "minmax_fg_ratio": minmax_fg_ratio_sum / total_images,
    }

def train_one_epoch(
    train_loader: Iterable,
    model: torch.nn.Module,
    teacher_model: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    opt: argparse.Namespace,
    gam_state: GAMState,
    target_prefixes: Sequence[str],
    device: torch.device,
    ema_model: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    effective_weight, warmup_scale = get_effective_distill_weight(
        base_weight=gam_state.current_weight,
        epoch=epoch,
        start_epoch=opt.distill_start_epoch,
        warmup_epochs=opt.distill_warmup_epochs,
        stop_epoch=opt.distill_stop_epoch,
    )
    distill_active = bool(opt.use_dinov3) and teacher_model is not None and effective_weight > 0.0

    total_loss_meter = AvgMeter()
    seg_loss_meter = AvgMeter()
    distill_loss_meter = AvgMeter()
    weighted_distill_meter = AvgMeter()
    grad_percentages: List[float] = []

    total_step = len(train_loader)
    epoch_start = time.time()

    for step, (base_images, base_gts) in enumerate(train_loader, start=1):
        base_images = base_images.to(device, non_blocking=True)
        base_gts = base_gts.to(device, dtype=torch.float32, non_blocking=True)

        for rate in opt.size_rates:
            optimizer.zero_grad(set_to_none=True)

            if abs(rate - 1.0) > 1e-9:
                train_size = int(round(opt.img_size * rate / 32.0) * 32)
                images = F.interpolate(
                    base_images,
                    size=(train_size, train_size),
                    mode="bilinear",
                    align_corners=True,
                )
                gts = F.interpolate(
                    base_gts,
                    size=(train_size, train_size),
                    mode="nearest",
                )
            else:
                images = base_images
                gts = base_gts

            # 纯分割预热阶段不执行教师前向，节省显存和时间。
            teacher_features = None
            if distill_active:
                with torch.no_grad():
                    teacher_output = teacher_model(images)
                    teacher_features = unwrap_feature_tensor(teacher_output)

            student_output = model(images)
            if isinstance(student_output, tuple):
                if len(student_output) != 2:
                    raise RuntimeError(
                        "学生网络训练阶段返回 tuple 时必须为 "
                        "([segmentation], distill_feature)。"
                    )
                segmentation_outputs, student_features = student_output
            else:
                segmentation_outputs = student_output
                student_features = None

            pred = (
                segmentation_outputs[0]
                if isinstance(segmentation_outputs, list)
                else segmentation_outputs
            )

            loss_seg = structure_loss(pred, gts)
            if distill_active:
                if student_features is None:
                    raise RuntimeError(
                        "当前启用了 DINOv3 蒸馏，但学生网络没有返回 distill_feature。"
                        "请确认网络实例 use_distill_feature=True。"
                    )
                loss_distill = compute_distill_loss(
                    student_features=student_features,
                    teacher_features=teacher_features,
                )
            else:
                loss_distill = loss_seg.new_zeros(())

            weighted_distill = effective_weight * loss_distill
            loss_total = loss_seg + weighted_distill

            if not torch.isfinite(loss_total):
                raise FloatingPointError(
                    f"检测到非有限损失: total={loss_total.item()}, "
                    f"seg={loss_seg.item()}, distill={loss_distill.item()}, "
                    f"effective_weight={effective_weight}"
                )

            loss_total.backward()

            # GAM 必须读取未经裁剪的梯度。
            grad_pct = compute_target_grad_percentage(
                model=model,
                target_prefixes=target_prefixes,
            )
            grad_percentages.append(grad_pct)

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if ema_model is not None:
                ema_model.update(model)

            if abs(rate - 1.0) <= 1e-9:
                total_loss_meter.update(loss_total.detach(), opt.batchsize)
                seg_loss_meter.update(loss_seg.detach(), opt.batchsize)
                distill_loss_meter.update(loss_distill.detach(), opt.batchsize)
                weighted_distill_meter.update(
                    weighted_distill.detach(), opt.batchsize
                )

        if step % opt.print_interval == 0 or step == total_step:
            seg_value = float(seg_loss_meter.show())
            kd_value = float(weighted_distill_meter.show())
            kd_to_seg = kd_value / max(seg_value, 1e-8)
            message = (
                f"{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], "
                f"Step [{step:04d}/{total_step:04d}], "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}, "
                f"Loss: {float(total_loss_meter.show()):.4f}, "
                f"Seg: {seg_value:.4f}, "
                f"Distill: {float(distill_loss_meter.show()):.4f}, "
                f"Base-W: {gam_state.current_weight:.6f}, "
                f"Effective-W: {effective_weight:.6f}, "
                f"Warmup: {warmup_scale:.3f}, "
                f"KD/Seg: {kd_to_seg:.3f}"
            )
            print(message)
            logging.info(message)

    finite_grad_percentages = [
        value for value in grad_percentages if math.isfinite(value)
    ]
    avg_grad_percentage = (
        float(np.mean(finite_grad_percentages))
        if finite_grad_percentages
        else 0.0
    )

    return {
        "train_seconds": time.time() - epoch_start,
        "loss": float(total_loss_meter.show()),
        "loss_seg": float(seg_loss_meter.show()),
        "loss_distill": float(distill_loss_meter.show()),
        "weighted_distill_loss": float(weighted_distill_meter.show()),
        "avg_grad_percentage": avg_grad_percentage,
        "base_distill_weight": float(gam_state.current_weight),
        "effective_distill_weight": float(effective_weight),
        "distill_warmup_scale": float(warmup_scale),
    }

def append_metrics_csv(csv_path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_full_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    gam_state: GAMState,
    best_state: BestState,
    opt: argparse.Namespace,
    ema_model: Optional[ModelEMA] = None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "gam_state": asdict(gam_state),
            "best_state": asdict(best_state),
            "ema": ema_model.module.state_dict() if ema_model is not None else None,
            "ema_updates": ema_model.updates if ema_model is not None else 0,
            "args": vars(opt),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GroupedMamba-UNet + DINOv3 decoder1 蒸馏 + "
            "warm-up + 稳定版 RT-DETRv4-style GAM"
        )
    )

    parser.add_argument("--dataset", type=str, default="ClinicDB")
    parser.add_argument(
        "--network",
        type=str,
        default="Base",
        choices=list(NET_MODELS.keys()),
    )
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--test_batchsize", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--color_image", type=str2bool, default=True)
    parser.add_argument("--augmentation", type=str2bool, default=True)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument(
        "--size_rates",
        type=parse_float_list,
        default=parse_float_list("0.75,1.0,1.25"),
        help="逗号分隔，例如 0.75,1.0,1.25",
    )

    parser.add_argument(
        "--train_path",
        type=str,
        default="",
        help="为空时自动使用 ./data/polyp/target/<dataset>/train/",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="",
        help="为空时自动使用 ./data/polyp/target/<dataset>/",
    )
    parser.add_argument("--train_save", type=str, default="./model_pth")

    # DINOv3 教师。
    parser.add_argument(
        "--dinov3_repo",
        type=str,
        default="/Data_8TB/lht/MK-UNet/dinov3",
    )
    parser.add_argument(
        "--dinov3_weights",
        type=str,
        default=(
            "/Data_8TB/lht/MK-UNet/teacher/"
            "dinov3_vitb16_pretrain_lvd1689m.pth"
        ),
    )
    parser.add_argument(
        "--dinov3_model_name",
        type=str,
        default="dinov3_vitb16",
    )
    parser.add_argument("--dinov3_patch_size", type=int, default=16)
    parser.add_argument("--teacher_dim", type=int, default=768)
    parser.add_argument("--teacher_downsample_factor", type=int, default=32)
    parser.add_argument(
        "--use_dinov3",
        type=str2bool,
        default=True,
        help="false 时完全跳过 DINOv3 教师、蒸馏损失和 GAM 调权，用于 No-DINO 消融",
    )

    # 蒸馏调度：先纯分割，再线性增大蒸馏权重。
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=0.1,
        help="warm-up 完成后使用的初始/基础蒸馏权重",
    )
    parser.add_argument(
        "--distill_default_weight",
        type=float,
        default=0.1,
        help="无效梯度或后期固定阶段回退的基础权重",
    )
    parser.add_argument("--distill_min_weight", type=float, default=0.01)
    parser.add_argument("--distill_max_weight", type=float, default=2.0)
    parser.add_argument(
        "--distill_start_epoch",
        type=int,
        default=10,
        help="该 Epoch 开始引入 DINOv3 蒸馏；此前仅训练分割任务",
    )
    parser.add_argument(
        "--distill_warmup_epochs",
        type=int,
        default=20,
        help="蒸馏权重从 0 线性增加到基础权重所需 Epoch 数",
    )
    parser.add_argument(
        "--distill_stop_epoch",
        type=int,
        default=0,
        help=">0 时从该 Epoch 开始真正关闭 DINOv3 蒸馏；0 表示不关闭",
    )

    # GAM。
    parser.add_argument("--gam_enabled", type=str2bool, default=True)
    parser.add_argument("--gam_rho", type=float, default=10.0)
    parser.add_argument("--gam_delta", type=float, default=1.0)
    parser.add_argument(
        "--gam_start_epoch",
        type=int,
        default=30,
        help="建议设置在蒸馏 warm-up 完成之后",
    )
    parser.add_argument(
        "--gam_stop_epoch",
        type=int,
        default=360,
        help="达到该 Epoch 后恢复 default_weight；设为 0 表示不启用",
    )
    parser.add_argument(
        "--gam_target_modules",
        type=str,
        default="decoder1",
        help="当前网络的 DINOv3 对齐点在 decoder1 后，因此默认监控 decoder1",
    )
    parser.add_argument(
        "--gam_step_down",
        type=float,
        default=0.8,
        help="单个 Epoch 权重最多缩小到上一轮的该倍数",
    )
    parser.add_argument(
        "--gam_step_up",
        type=float,
        default=1.25,
        help="单个 Epoch 权重最多放大到上一轮的该倍数",
    )

    # EMA，可选；默认关闭，避免改变你当前实验基线。
    parser.add_argument("--use_ema", type=str2bool, default=False)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_warmups", type=int, default=1000)
    parser.add_argument("--ema_start", type=int, default=0)

    # 可复现性与评估。
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    parser.add_argument("--gt_threshold", type=float, default=0.2)
    parser.add_argument(
        "--eval_minmax",
        type=str2bool,
        default=True,
        help=(
            "true：使用旧脚本逐图 Min-Max 指标选择最佳模型；"
            "false：使用标准 sigmoid 固定阈值指标。两套指标都会被记录。"
        ),
    )
    parser.add_argument("--save_interval", type=int, default=20)

    return parser

def main() -> None:
    parser = build_parser()
    opt = parser.parse_args()

    if not opt.train_path:
        opt.train_path = f"./data/polyp/target/{opt.dataset}/train/"
    if not opt.test_path:
        opt.test_path = f"./data/polyp/target/{opt.dataset}/"

    if not opt.use_dinov3:
        # No-DINO 消融：彻底关闭教师、蒸馏损失和 GAM 调权。
        opt.distill_weight = 0.0
        opt.distill_default_weight = 0.0
        opt.gam_enabled = False
        opt.distill_stop_epoch = 1

    if opt.teacher_dim <= 0:
        raise ValueError("teacher_dim 必须大于 0")
    if opt.gam_delta < 0:
        raise ValueError("gam_delta 不能小于 0")
    if opt.gam_rho <= opt.gam_delta:
        raise ValueError("gam_rho 必须大于 gam_delta，保证下界为正")
    if opt.distill_weight < 0 or opt.distill_default_weight < 0:
        raise ValueError("蒸馏权重不能小于 0")
    if opt.distill_min_weight <= 0:
        raise ValueError("distill_min_weight 必须大于 0")
    if opt.distill_max_weight < opt.distill_min_weight:
        raise ValueError("distill_max_weight 必须不小于 distill_min_weight")
    if opt.distill_start_epoch < 1:
        raise ValueError("distill_start_epoch 必须大于等于 1")
    if opt.distill_warmup_epochs < 0:
        raise ValueError("distill_warmup_epochs 不能小于 0")
    if opt.distill_stop_epoch < 0:
        raise ValueError("distill_stop_epoch 不能小于 0")
    if (
        opt.use_dinov3
        and opt.distill_stop_epoch > 0
        and opt.distill_stop_epoch <= opt.distill_start_epoch
    ):
        raise ValueError("distill_stop_epoch 必须大于 distill_start_epoch，或设为 0")
    if opt.gam_start_epoch < 1:
        raise ValueError("gam_start_epoch 必须大于等于 1")
    if not (0.0 < opt.gam_step_down <= 1.0):
        raise ValueError("gam_step_down 必须位于 (0, 1] 内")
    if opt.gam_step_up < 1.0:
        raise ValueError("gam_step_up 必须大于等于 1")

    warmup_finished_epoch = (
        opt.distill_start_epoch
        if opt.distill_warmup_epochs <= 0
        else opt.distill_start_epoch + opt.distill_warmup_epochs - 1
    )
    earliest_safe_gam_epoch = warmup_finished_epoch + 1
    if opt.use_dinov3 and opt.gam_enabled and opt.gam_start_epoch < earliest_safe_gam_epoch:
        print(
            f"[Config Warning] gam_start_epoch={opt.gam_start_epoch} 早于蒸馏 "
            f"warm-up 完成后的安全时间点 {earliest_safe_gam_epoch}，"
            f"已自动调整为 {earliest_safe_gam_epoch}。"
        )
        opt.gam_start_epoch = earliest_safe_gam_epoch

    if (
        opt.gam_enabled
        and opt.gam_stop_epoch > 0
        and opt.gam_stop_epoch <= opt.gam_start_epoch
    ):
        raise ValueError("gam_stop_epoch 必须大于 gam_start_epoch，或设为 0")

    target_prefixes = parse_target_prefixes(opt.gam_target_modules)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.train_save, exist_ok=True)

    # 教师只加载一次，多次 Run 共享同一冻结教师；No-DINO 消融时不加载，节省显存和时间。
    teacher_model: Optional[torch.nn.Module] = None
    if opt.use_dinov3:
        print("\n[Teacher] 正在加载 DINOv3 教师模型...")
        teacher_model = MKUNetDINOv3Teacher(
            dinov3_repo_path=opt.dinov3_repo,
            model_name=opt.dinov3_model_name,
            dinov3_weights_path=opt.dinov3_weights,
            patch_size=opt.dinov3_patch_size,
            target_downsample_factor=opt.teacher_downsample_factor,
        ).to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        print(
            f"[Teacher] 加载完成 | model={opt.dinov3_model_name} | "
            f"teacher_dim={opt.teacher_dim} | downsample={opt.teacher_downsample_factor}\n"
        )
    else:
        print("\n[Teacher] use_dinov3=False，跳过 DINOv3 教师加载，执行 No-DINO 消融。\n")

    for run in range(1, opt.runs + 1):
        run_seed = opt.seed + run - 1
        set_random_seed(run_seed, deterministic=opt.deterministic)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        distill_tag = "DINOv3_GAM" if opt.use_dinov3 else "NoDINO"
        ema_tag = "EMA" if opt.use_ema else "NoEMA"
        run_id = (
            f"{opt.dataset}_GroupedMamba_{distill_tag}_{ema_tag}_{opt.network}_"
            f"bs{opt.batchsize}_lr{opt.lr}_e{opt.epoch}_"
            f"aug{opt.augmentation}_seed{run_seed}_run{run}_t{timestamp}"
        )
        save_dir = os.path.join(opt.train_save, run_id)
        os.makedirs(save_dir, exist_ok=True)

        log_path = os.path.join(save_dir, f"train_log_{run_id}.log")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            force=True,
        )
        logging.info("Arguments: %s", vars(opt))
        logging.info("GAM target prefixes: %s", target_prefixes)

        print("=" * 80)
        print(
            f"Run {run}/{opt.runs} | Network={opt.network} | Seed={run_seed} | "
            f"Save={save_dir}"
        )
        print("=" * 80)

        model_class = NET_MODELS[opt.network]
        model = model_class(
            num_classes=opt.num_classes,
            in_channels=opt.in_channels,
            teacher_dim=opt.teacher_dim,
            use_distill_feature=opt.use_dinov3,
        ).to(device)

        matched_target_parameters = [
            name
            for name, _ in model.named_parameters()
            if any(
                (name[len("module."):] if name.startswith("module.") else name)
                .startswith(prefix)
                for prefix in target_prefixes
            )
        ]
        if not matched_target_parameters:
            raise ValueError(
                f"GAM 目标模块 {target_prefixes} 没有匹配到任何模型参数。"
            )
        target_param_count = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if any(
                (name[len("module."):] if name.startswith("module.") else name)
                .startswith(prefix)
                for prefix in target_prefixes
            )
        )
        target_message = (
            f"[GAM Target] modules={opt.gam_target_modules} | "
            f"matched_tensors={len(matched_target_parameters)} | "
            f"params={target_param_count / 1e3:.3f}K"
        )
        print(target_message)
        logging.info(target_message)

        # FLOPs 统计必须使用 eval 接口，否则训练接口会额外返回蒸馏特征。
        try:
            model.eval()
            cal_params_flops(model, opt.img_size, logging)
        except Exception as error:
            print(f"FLOPs calculation skipped: {error}")
            logging.warning("FLOPs calculation skipped: %s", error)
        finally:
            model.train()

        ema_model: Optional[ModelEMA] = None
        if opt.use_ema:
            ema_model = ModelEMA(
                model=model,
                decay=opt.ema_decay,
                warmups=opt.ema_warmups,
                start=opt.ema_start,
            )
            ema_message = (
                f"[EMA] enabled=True | decay={opt.ema_decay} | "
                f"warmups={opt.ema_warmups} | start={opt.ema_start}"
            )
            print(ema_message)
            logging.info(ema_message)
        else:
            logging.info("[EMA] enabled=False")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=opt.epoch,
            eta_min=1e-6,
        )

        train_loader = get_loader(
            image_root=os.path.join(opt.train_path, "images") + os.sep,
            gt_root=os.path.join(opt.train_path, "masks") + os.sep,
            batchsize=opt.batchsize,
            trainsize=opt.img_size,
            shuffle=True,
            augmentation=opt.augmentation,
            split="train",
            color_image=opt.color_image,
        )

        gam_state = GAMState(
            enabled=opt.gam_enabled,
            rho=opt.gam_rho,
            delta=opt.gam_delta,
            current_weight=min(
                max(opt.distill_weight, opt.distill_min_weight),
                opt.distill_max_weight,
            ) if opt.distill_weight > 0 else 0.0,
            default_weight=min(
                max(opt.distill_default_weight, opt.distill_min_weight),
                opt.distill_max_weight,
            ) if opt.distill_default_weight > 0 else 0.0,
            start_epoch=opt.gam_start_epoch,
            stop_epoch=opt.gam_stop_epoch,
            min_weight=opt.distill_min_weight,
            max_weight=opt.distill_max_weight,
            step_down_factor=opt.gam_step_down,
            step_up_factor=opt.gam_step_up,
        )
        best_state = BestState()
        total_train_seconds = 0.0
        metrics_csv = os.path.join(save_dir, "epoch_metrics.csv")

        for epoch in range(1, opt.epoch + 1):
            # 后期固定阶段从本 Epoch 开始就使用 default_weight，
            # 而不是训练完该 Epoch 后才切换。
            if (
                gam_state.enabled
                and gam_state.stop_epoch > 0
                and epoch >= gam_state.stop_epoch
            ):
                gam_state.current_weight = gam_state.default_weight

            epoch_stats = train_one_epoch(
                train_loader=train_loader,
                model=model,
                teacher_model=teacher_model,
                optimizer=optimizer,
                epoch=epoch,
                opt=opt,
                gam_state=gam_state,
                target_prefixes=target_prefixes,
                device=device,
                ema_model=ema_model,
            )
            total_train_seconds += epoch_stats["train_seconds"]

            weight_before, weight_after, gam_reason = update_gam_weight(
                state=gam_state,
                avg_percentage=epoch_stats["avg_grad_percentage"],
                epoch=epoch,
            )

            gam_message = (
                f"[GAM] Epoch {epoch:03d} | "
                f"Grad={epoch_stats['avg_grad_percentage']:.6f}% | "
                f"Range=[{gam_state.rho - gam_state.delta:.6f}%, "
                f"{gam_state.rho + gam_state.delta:.6f}%] | "
                f"Base-W={weight_before:.8f}->{weight_after:.8f} | "
                f"Effective-W={epoch_stats['effective_distill_weight']:.8f} | "
                f"Warmup={epoch_stats['distill_warmup_scale']:.4f} | "
                f"Reason={gam_reason}"
            )
            print("\n" + gam_message + "\n")
            logging.info(gam_message)

            # 当前 Epoch 训练完成后进行评估。
            eval_model = ema_model.module if ema_model is not None else model
            epoch_results: Dict[str, Dict[str, float]] = {}
            for dataset_split in ("test", "val"):
                metrics = evaluate(
                    model=eval_model,
                    root_path=opt.test_path,
                    dataset=dataset_split,
                    opt=opt,
                    device=device,
                )
                epoch_results[dataset_split] = metrics
                primary_name = "MinMax" if opt.eval_minmax else "Standard"
                result_message = (
                    f"Epoch: {epoch}, Dataset: {dataset_split}, "
                    f"Primary({primary_name}) Dice: {metrics['dice']:.4f}, "
                    f"IoU: {metrics['iou']:.4f}, N: {int(metrics['n'])} | "
                    f"Standard Dice/IoU: {metrics['standard_dice']:.4f}/"
                    f"{metrics['standard_iou']:.4f} | "
                    f"MinMax Dice/IoU: {metrics['minmax_dice']:.4f}/"
                    f"{metrics['minmax_iou']:.4f} | "
                    f"Prob[min/max/mean]: {metrics['prob_min']:.4f}/"
                    f"{metrics['prob_max']:.4f}/{metrics['prob_mean']:.4f} | "
                    f"FG(std/mm): {metrics['standard_fg_ratio']:.4f}/"
                    f"{metrics['minmax_fg_ratio']:.4f}"
                )
                print(result_message)
                logging.info(result_message)

            val_dice = epoch_results["val"]["dice"]
            test_dice = epoch_results["test"]["dice"]
            if val_dice > best_state.best_val_dice:
                old_best = best_state.best_val_dice
                best_state.best_val_dice = val_dice
                best_state.test_dice_at_best_val = test_dice
                best_state.best_epoch = epoch

                torch.save(
                    eval_model.state_dict(),
                    os.path.join(save_dir, f"{run_id}-best.pth"),
                )
                if ema_model is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_dir, f"{run_id}-best-raw.pth"),
                    )
                best_message = (
                    f"### Best Model Saved: Val Dice {old_best:.4f} -> "
                    f"{val_dice:.4f}, Test Dice={test_dice:.4f}, Epoch={epoch}, "
                    f"Weights={'EMA' if ema_model is not None else 'Raw'} ###"
                )
                print(best_message)
                logging.info(best_message)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            torch.save(
                eval_model.state_dict(),
                os.path.join(save_dir, f"{run_id}-last.pth"),
            )
            if ema_model is not None:
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"{run_id}-last-raw.pth"),
                )
            save_full_checkpoint(
                path=os.path.join(save_dir, f"{run_id}-last-checkpoint.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                gam_state=gam_state,
                best_state=best_state,
                opt=opt,
                ema_model=ema_model,
            )

            if opt.save_interval > 0 and epoch % opt.save_interval == 0:
                torch.save(
                    eval_model.state_dict(),
                    os.path.join(save_dir, f"{run_id}-epoch_{epoch}.pth"),
                )

            append_metrics_csv(
                metrics_csv,
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "use_dinov3": opt.use_dinov3,
                    "use_ema": opt.use_ema,
                    "ema_updates": ema_model.updates if ema_model is not None else 0,
                    "loss": epoch_stats["loss"],
                    "loss_seg": epoch_stats["loss_seg"],
                    "loss_distill": epoch_stats["loss_distill"],
                    "weighted_distill_loss": epoch_stats["weighted_distill_loss"],
                    "target_grad_percentage": epoch_stats["avg_grad_percentage"],
                    "distill_base_weight_used": weight_before,
                    "distill_effective_weight_used": epoch_stats["effective_distill_weight"],
                    "distill_warmup_scale": epoch_stats["distill_warmup_scale"],
                    "distill_weight_next": weight_after,
                    "gam_reason": gam_reason,
                    "test_dice": epoch_results["test"]["dice"],
                    "test_iou": epoch_results["test"]["iou"],
                    "test_standard_dice": epoch_results["test"]["standard_dice"],
                    "test_standard_iou": epoch_results["test"]["standard_iou"],
                    "test_minmax_dice": epoch_results["test"]["minmax_dice"],
                    "test_minmax_iou": epoch_results["test"]["minmax_iou"],
                    "test_prob_min": epoch_results["test"]["prob_min"],
                    "test_prob_max": epoch_results["test"]["prob_max"],
                    "test_prob_mean": epoch_results["test"]["prob_mean"],
                    "test_standard_fg_ratio": epoch_results["test"]["standard_fg_ratio"],
                    "test_minmax_fg_ratio": epoch_results["test"]["minmax_fg_ratio"],
                    "val_dice": epoch_results["val"]["dice"],
                    "val_iou": epoch_results["val"]["iou"],
                    "val_standard_dice": epoch_results["val"]["standard_dice"],
                    "val_standard_iou": epoch_results["val"]["standard_iou"],
                    "val_minmax_dice": epoch_results["val"]["minmax_dice"],
                    "val_minmax_iou": epoch_results["val"]["minmax_iou"],
                    "val_prob_min": epoch_results["val"]["prob_min"],
                    "val_prob_max": epoch_results["val"]["prob_max"],
                    "val_prob_mean": epoch_results["val"]["prob_mean"],
                    "val_standard_fg_ratio": epoch_results["val"]["standard_fg_ratio"],
                    "val_minmax_fg_ratio": epoch_results["val"]["minmax_fg_ratio"],
                    "best_val_dice": best_state.best_val_dice,
                    "test_dice_at_best_val": best_state.test_dice_at_best_val,
                    "train_seconds": epoch_stats["train_seconds"],
                },
            )

        summary = (
            f"\n{'=' * 50}\n"
            f"FINAL RESULTS: {run_id}\n"
            f"Best Epoch: {best_state.best_epoch}\n"
            f"Best Val Dice: {best_state.best_val_dice:.4f}\n"
            f"Test Dice at Best Val: {best_state.test_dice_at_best_val:.4f}\n"
            f"Final Distill Weight: {gam_state.current_weight:.8f}\n"
            f"Use DINOv3: {opt.use_dinov3} | Use EMA: {opt.use_ema}\n"
            f"Total Train Time: {total_train_seconds:.2f}s\n"
            f"{'=' * 50}"
        )
        print(summary)
        logging.info(summary)


if __name__ == "__main__":
    main()
