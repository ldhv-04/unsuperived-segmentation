import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tasdf.models.itgm import ITGM


def load_real_embryo(image_path: str, size: int = 256) -> torch.Tensor:
    """Nạp ảnh phôi thật từ đĩa, chuẩn hóa và chuyển thành Tensor PyTorch (1, 1, H,

    W)
    """
    # 1. Đọc ảnh và chuyển sang ảnh xám 1 channel ('L' = Luminance/Grayscale)
    img = Image.open(image_path).convert("L")

    # 2. Resize về kích thước 256x256 chuẩn của bài báo TASDF
    img = img.resize((size, size))

    # 3. Chuẩn hóa pixel từ [0, 255] về số thực [0.0, 1.0]
    img_np = np.array(img, dtype=np.float32) / 255.0

    # 4. Thêm chiều Batch (B=1) và chiều Channel (C=1)
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    return tensor


def main():
    # Đường dẫn đến file ảnh phôi thật trên máy của bạn
    image_path = "/home/terminator/workspace/Attech/Research/dataset/ICM Blastocyst/Blast_PCRM_R13-0073A.png"

    print("=" * 65)
    print(f">> Đang nạp ảnh phôi thật từ: {image_path}")
    embryo_tensor = load_real_embryo(image_path, size=256)
    print(f">> Tensor kích thước: {embryo_tensor.shape}, Kiểu: {embryo_tensor.dtype}")
    print("=" * 65)

    # Khởi tạo mô hình ITGM
    model = ITGM(iterations=10, kernel_size=3)

    # Chạy forward pass
    final_skeleton, history = model(embryo_tensor, return_intermediates=True)

    # In thông số định lượng qua từng vòng lặp
    print(
        f"{'Bước':<8} | {'Số pixel tế bào (Eroded)':<25} | {'Số pixel Xương tích lũy':<25}"
    )
    print("-" * 65)
    for item in history:
        step = item["step"]
        eroded_pixels = (item["f_eroded"] > 0.1).sum().item()
        skel_pixels = (item["accumulated_skeleton"] > 0.1).sum().item()
        print(f"Bước {step:<3} | {eroded_pixels:<25} | {skel_pixels:<25}")

    print("=" * 65)
    print(">> Đang lưu biểu đồ trực quan ra 'real_embryo_itgm_steps.png'...")

    # Vẽ biểu đồ 3 hàng x 4 cột hiển thị quá trình gọt giũa ảnh thật
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Hình 0: Ảnh kính hiển vi gốc
    axes[0].imshow(embryo_tensor.squeeze().numpy(), cmap="gray")
    axes[0].set_title(
        "Ảnh Kính Hiển Vi Thật (HMC)", fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    # Các hình tiếp theo: Khung xương tích lũy qua từng bước (Step 0 đến 10)
    for i, item in enumerate(history):
        ax = axes[i + 1]
        skel_img = item["accumulated_skeleton"].squeeze().numpy()
        ax.imshow(skel_img, cmap="magma")
        ax.set_title(f"Khung xương Bước {item['step']}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("visualize/img/real_embryo_itgm_steps.png", dpi=150)
    print(
        ">> THÀNH CÔNG! Hãy mở file 'real_embryo_itgm_steps.png' để xem kết quả!"
    )


if __name__ == "__main__":
    main()