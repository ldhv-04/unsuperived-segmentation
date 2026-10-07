from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tasdf.utils.metrics import compute_sample_metrics, DEFAULT_CLASS_NAMES


class Validator:
    """
    Động cơ đánh giá độc lập mô hình trên tập kiểm tra/kiểm định:
    Tính toán chi tiết độ đo Dice và IoU cho từng lớp giải phẫu và trung bình toàn diện.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        device: torch.device | None = None,
        class_names: dict[int, str] | None = None,
        num_classes: int = 5,
        branch: str = "net1",
    ):
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.num_classes = num_classes
        self.branch = branch

    def validate(self, model: nn.Module) -> dict[str, Any]:
        """Thực hiện đánh giá trên toàn bộ val_loader."""
        model.eval()
        accum = {c: {"dice": [], "iou": []} for c in range(1, self.num_classes)}

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                if self.branch == "ensemble":
                    (p1, _), (p2, _) = model(x, branch="both")
                    preds = (torch.softmax(p1, dim=1) + torch.softmax(p2, dim=1)) / 2.0
                else:
                    p, _ = model(x, branch=self.branch)
                    preds = torch.softmax(p, dim=1)

                pred_masks = torch.argmax(preds, dim=1).cpu().numpy()
                target_masks = y.numpy()

                for i in range(len(x)):
                    sample_m = compute_sample_metrics(
                        pred_masks[i], target_masks[i], num_classes=self.num_classes
                    )
                    for c in range(1, self.num_classes):
                        accum[c]["dice"].append(sample_m[c]["dice"])
                        accum[c]["iou"].append(sample_m[c]["iou"])

        per_class_dice = {}
        per_class_iou = {}
        for c in range(1, self.num_classes):
            name = self.class_names.get(c, f"Class {c}")
            per_class_dice[name] = float(np.mean(accum[c]["dice"])) if accum[c]["dice"] else 0.0
            per_class_iou[name] = float(np.mean(accum[c]["iou"])) if accum[c]["iou"] else 0.0

        mean_dice = float(np.mean(list(per_class_dice.values()))) if per_class_dice else 0.0
        mean_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0

        return {
            "mDice": mean_dice,
            "mIoU": mean_iou,
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
        }

    def print_results(self, results: dict[str, Any], title: str = "VAL"):
        """In bảng kết quả đánh giá chuẩn khoa học."""
        print("\n" + "=" * 65)
        print(f"  KẾT QUẢ ĐÁNH GIÁ ĐỘ LẬP | TẬP: [{title.upper()}] | NHÁNH: [{self.branch}]")
        print("=" * 65)
        print(f"{'Cấu trúc Phôi học':<28} | {'Dice (%)':<12} | {'IoU / Jaccard (%)':<18}")
        print("-" * 65)
        for name, dice in results["per_class_dice"].items():
            iou = results["per_class_iou"].get(name, 0.0)
            print(f"{name:<28} | {dice * 100:<12.2f} | {iou * 100:<18.2f}")
        print("-" * 65)
        print(
            f"{'TRUNG BÌNH (mDice / mIoU)':<28} | {results['mDice'] * 100:<12.2f} | {results['mIoU'] * 100:<18.2f}"
        )
        print("=" * 65 + "\n")
