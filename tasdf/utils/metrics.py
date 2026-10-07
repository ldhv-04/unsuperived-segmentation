from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn

DEFAULT_CLASS_NAMES = {
    0: "Background",
    1: "Zona Pellucida (ZP)",
    2: "Trophectoderm (TE)",
    3: "Inner Cell Mass (ICM)",
    4: "Blastocyst Cavity (BC)",
}

DEFAULT_PALETTE = {
    0: (0, 0, 0),        # 0: Background - Đen
    1: (0, 114, 178),    # 1: ZP - Xanh lam
    2: (0, 158, 115),    # 2: TE - Xanh lục
    3: (213, 94, 0),     # 3: ICM - Cam đỏ
    4: (240, 228, 66),   # 4: BC - Vàng
}


def compute_multiclass_dice(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 5
) -> float:
    """Tính mDice trên batch tensor (PyTorch)."""
    pred_classes = torch.argmax(preds, dim=1)
    dice_scores = []
    for c in range(1, num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append((2.0 * intersection / (union + 1e-6)).item())
    return float(np.mean(dice_scores)) if dice_scores else 0.0


def compute_sample_metrics(
    pred_mask: np.ndarray, target_mask: np.ndarray, num_classes: int = 5
) -> dict[int, dict[str, float]]:
    """Tính toán chi tiết Dice và IoU cho từng lớp trên một mặt nạ numpy 2D."""
    metrics = {}
    for c in range(1, num_classes):
        p = pred_mask == c
        t = target_mask == c
        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        total = p.sum() + t.sum()

        dice = (2.0 * intersection / (total + 1e-6)) if total > 0 else 1.0
        iou = (intersection / (union + 1e-6)) if union > 0 else 1.0

        metrics[c] = {"dice": float(dice), "iou": float(iou)}
    return metrics


def colorize_mask(
    mask: np.ndarray, palette: dict[int, tuple[int, int, int]] | None = None
) -> np.ndarray:
    """Tô màu mặt nạ nhãn 2D theo bảng màu chuẩn RGB."""
    if palette is None:
        palette = DEFAULT_PALETTE
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        color_mask[mask == class_id] = color
    return color_mask


def decompose_te_and_blastocoel(
    mask: np.ndarray,
    te_thickness_ratio: float = 0.06,
    min_thickness: int = 4,
    max_thickness: int = 20,
) -> np.ndarray:
    """
    Phân rã khối nội phôi (Class 2) thành 2 cấu trúc giải phẫu sinh học:
    - Vành biểu mô ngoài: Trophectoderm (TE - Class 2)
    - Khoang dịch rỗng bên trong: Blastocoel (BC - Class 4)
    Bảo toàn khối tế bào nụ ICM (Class 3) và màng ZP (Class 1).
    """
    decomposed = mask.copy()
    inner_mask = (mask == 2).astype(np.uint8)

    area = int(np.sum(inner_mask))
    if area < 100:
        return decomposed

    # Bán kính tương đương R = sqrt(Area / pi)
    equiv_radius = np.sqrt(area / np.pi)
    thickness = int(np.clip(equiv_radius * te_thickness_ratio, min_thickness, max_thickness))

    # Co hình thái học (Erosion) bằng phần tử elip
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1))
    cavity_mask = cv2.erode(inner_mask, kernel)

    # Gán lõi trung tâm thành Blastocoel (Class 4), loại trừ ICM (Class 3)
    is_cavity = (cavity_mask > 0) & (mask != 3)
    decomposed[is_cavity] = 4

    return decomposed


def get_model_parameter_stats(model: nn.Module) -> dict[str, Any]:
    """
    Tính toán chi tiết các chỉ số tham số mô hình:
    - Tổng số tham số (total_params)
    - Tham số có thể huấn luyện (trainable_params)
    - Tham số đóng băng / non-trainable (non_trainable_params)
    - Tỷ lệ tham số huấn luyện (trainable_ratio)
    - Kích thước bộ nhớ ước tính (size_mb) ở định dạng FP32
    - Thống kê chi tiết theo từng nhánh / module con (submodules)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    trainable_ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    size_mb = (total_params * 4) / (1024 * 1024)

    submodules = {}
    for name, child in model.named_children():
        c_total = sum(p.numel() for p in child.parameters())
        c_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        c_size_mb = (c_total * 4) / (1024 * 1024)
        submodules[name] = {
            "total_params": c_total,
            "trainable_params": c_trainable,
            "non_trainable_params": c_total - c_trainable,
            "size_mb": c_size_mb,
        }

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable,
        "trainable_ratio": trainable_ratio,
        "size_mb": size_mb,
        "submodules": submodules,
    }


def format_parameter_summary(stats: dict[str, Any]) -> str:
    """Định dạng kết quả kiểm toán tham số thành bảng ASCII trực quan."""
    lines = [
        "=" * 78,
        f"{'BẢNG THỐNG KÊ THAM SỐ MÔ HÌNH (MODEL PARAMETER AUDIT)':^78}",
        "=" * 78,
        f"  {'Thành phần / Module':<30} {'Tổng tham số':>15} {'Trainable':>15} {'Kích thước':>12}",
        "-" * 78,
    ]
    for name, sub in stats.get("submodules", {}).items():
        lines.append(
            f"  {name:<30} {sub['total_params']:>15,d} {sub['trainable_params']:>15,d} {sub['size_mb']:>9.2f} MB"
        )
    if stats.get("submodules"):
        lines.append("-" * 78)
    lines.append(
        f"  {'TỔNG CỘNG (TOTAL)':<30} {stats['total_params']:>15,d} {stats['trainable_params']:>15,d} {stats['size_mb']:>9.2f} MB"
    )
    lines.append(f"  * Tỷ lệ tham số huấn luyện: {stats.get('trainable_ratio', 0.0):.2f}%")
    lines.append(f"  * Ước tính dung lượng bộ nhớ (FP32 weights): {stats.get('size_mb', 0.0):.2f} MB")
    lines.append("=" * 78)
    return "\n".join(lines)

