from datetime import datetime
import json
from pathlib import Path
from typing import Any
import torch


class DualLogger:
    """
    Hệ thống ghi nhận kép:
    1. Text stream: Xuất bảng biểu ASCII ra màn hình (Notebook cell) và file train.log.
    2. JSON stream: Xuất mảng đối tượng số có cấu trúc ra metrics.json phục vụ vẽ đồ thị.
    """

    def __init__(self, output_dir: str | Path, exp_name: str = "train"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "train.log"
        self.json_file = self.output_dir / "metrics.json"

        self.history: list[dict[str, Any]] = []

        # Khởi tạo log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 85}\n")
            f.write(f">> PHIÊN HUẤN LUYỆN TASDF: {exp_name.upper()} | KHỞI TẠO: {datetime.now().isoformat()}\n")
            f.write(f"{'=' * 85}\n")

    def log_header(self):
        header = (
            f"{'Epoch':<8} | {'GPU_Mem':<9} | {'L_total':<9} | {'L_sup':<9} | "
            f"{'L_dist':<9} | {'Val_Dice':<10} | {'ZP_Dice':<9} | {'TE_Dice':<9} | "
            f"{'ICM_Dice':<9} | {'BC_Dice':<9}"
        )
        separator = "-" * len(header)
        print("\n" + separator)
        print(header)
        print(separator)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(separator + "\n" + header + "\n" + separator + "\n")

    def log_epoch(self, metrics: dict[str, Any]):
        """Ghi nhận thông số sau mỗi epoch vào cả 2 luồng."""
        epoch = metrics.get("epoch", 0)
        total_epochs = metrics.get("total_epochs", 0)
        epoch_str = f"{epoch:02d}/{total_epochs:02d}" if total_epochs else f"{epoch:02d}"

        # Lấy GPU mem nếu có CUDA
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_str = f"{gpu_mem:.2f}G"

        l_total = f"{metrics.get('train_total_loss', 0.0):.4f}"
        l_sup = f"{metrics.get('train_loss_sup', 0.0):.4f}"
        l_dist = f"{metrics.get('train_loss_distill', 0.0):.4f}"
        val_dice = f"{metrics.get('val_mDice', 0.0) * 100:.2f}%"

        per_class = metrics.get("per_class_dice", {})
        zp = f"{per_class.get('Zona Pellucida (ZP)', 0.0) * 100:.1f}%"
        te = f"{per_class.get('Trophectoderm (TE)', 0.0) * 100:.1f}%"
        icm = f"{per_class.get('Inner Cell Mass (ICM)', 0.0) * 100:.1f}%"
        bc = f"{per_class.get('Blastocyst Cavity (BC)', 0.0) * 100:.1f}%"

        row = (
            f"{epoch_str:<8} | {mem_str:<9} | {l_total:<9} | {l_sup:<9} | "
            f"{l_dist:<9} | {val_dice:<10} | {zp:<9} | {te:<9} | "
            f"{icm:<9} | {bc:<9}"
        )
        print(row)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(row + "\n")

        # Lưu dữ liệu có cấu trúc vào metrics.json
        record = metrics.copy()
        record["timestamp"] = datetime.now().isoformat()
        record["gpu_mem_gb"] = gpu_mem
        self.history.append(record)

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def log_message(self, message: str):
        """In thông báo tùy chỉnh ra màn hình và file log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
