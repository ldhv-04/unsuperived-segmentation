from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.final_model import TASDFNet
from src.datasets.embryo_dataset import EmbryoDataset


def main():
    print("=" * 75)
    print(" KIỂM TRA MÔ HÌNH TASDFNET TRÊN ẢNH PHÔI THẬT & XUẤT KẾT QUẢ DỰ ĐOÁN ")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Thiết bị tính toán: {device}")

    # 1. Nạp ảnh phôi thật qua EmbryoDataset
    data_dir = "/home/terminator/workspace/Attech/Research/dataset/ICM Blastocyst"
    dataset = EmbryoDataset(data_dir=data_dir, size=256, is_labeled=False)
    print(f">> Đã tìm thấy {len(dataset)} ảnh phôi trong '{data_dir}'")

    img_tensor = dataset[0].unsqueeze(0).to(device)  # [1, 1, 256, 256]
    print(f">> Kích thước tensor đầu vào: {list(img_tensor.shape)}")

    # 2. Khởi tạo mô hình TASDFNet
    model = TASDFNet(
        in_channels=1, base_channels=64, num_classes=5, k_iters=10
    ).to(device)
    model.eval()

    # 3. Chạy suy luận (Inference qua net1)
    print(">> Đang chạy qua mạng TASDFNet (Backbone -> DualHeads: Pixel & Topo)...")
    with torch.no_grad():
        p1, t1 = model(img_tensor, branch="net1")
        probs = F.softmax(p1, dim=1)  # Phân phối xác suất 5 lớp

    print(f">> Logits Pixel: {list(p1.shape)} | Khung xương ITGM: {list(t1.shape)}")
    print("=" * 75)
    print(">> Đang trích xuất và vẽ biểu đồ kết quả ra file visualize/img/model_output_sanity.png...")

    # 4. Trực quan hóa kết quả phân đoạn & khung xương
    img_np = img_tensor.squeeze().cpu().numpy()
    skeleton_np = t1.squeeze().cpu().numpy()

    # Trích xuất bản đồ xác suất của các thành phần phôi
    prob_zp = probs[0, 1].cpu().numpy()   # Kênh 1: Màng ZP
    prob_te = probs[0, 2].cpu().numpy()   # Kênh 2: Tế bào TE
    prob_icm = probs[0, 3].cpu().numpy()  # Kênh 3: Khối nụ ICM
    prob_bc = probs[0, 4].cpu().numpy()   # Kênh 4: Khoang rỗng BC

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    # Panel 1: Ảnh phôi gốc
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("1. Ảnh Phôi Thật\n[1, 256, 256]", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Bản đồ màng ZP
    axes[1].imshow(prob_zp, cmap="magma")
    axes[1].set_title("2. Dự đoán Màng ZP\n(Zona Pellucida)", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Bản đồ Tế bào TE
    axes[2].imshow(prob_te, cmap="viridis")
    axes[2].set_title("3. Dự đoán Tế bào TE\n(Trophectoderm)", fontsize=11)
    axes[2].axis("off")

    # Panel 4: Bản đồ Khối ICM
    axes[3].imshow(prob_icm, cmap="plasma")
    axes[3].set_title("4. Dự đoán Khối ICM\n(Inner Cell Mass)", fontsize=11)
    axes[3].axis("off")

    # Panel 5: Khung xương mềm ITGM
    axes[4].imshow(skeleton_np, cmap="inferno")
    axes[4].set_title("5. Khung xương Tô-pô ITGM\n(Soft-Skeleton T)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    output_path = "visualize/img/model_output_sanity.png"
    plt.savefig(output_path, dpi=150)
    print(f">> THÀNH CÔNG! Đã lưu ảnh kết quả tại: {output_path}")


if __name__ == "__main__":
    main()