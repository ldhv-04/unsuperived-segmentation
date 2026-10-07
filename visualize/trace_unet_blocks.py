from pathlib import Path
import sys

# Định vị thư mục gốc để import module chuẩn xác
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tasdf.models.unet_blocks import DoubleConv, Down, Up
import torch


def load_real_embryo(image_path: str, size: int = 256) -> torch.Tensor:
  img = Image.open(image_path).convert('L').resize((size, size))
  img_np = np.array(img, dtype=np.float32) / 255.0
  return torch.tensor(img_np).unsqueeze(0).unsqueeze(0)


def main():
  image_path = '/home/terminator/workspace/Attech/Research/dataset/ICM Blastocyst/Blast_PCRM_R13-0073A.png'

  print('=' * 75)
  print(' KIỂM TRA DÒNG CHẢY DỮ LIỆU CỦA ẢNH PHÔI THẬT QUA CÁC KHỐI U-NET ')
  print('=' * 75)

  # 1. Nạp ảnh thật
  x0 = load_real_embryo(image_path, size=256)
  print(
      f'0. Ảnh Phôi Gốc       | Shape: {list(x0.shape)} |'
      ' [Batch=1, Kênh=1, 256x256]'
  )

  # 2. Khởi tạo 3 khối
  conv1 = DoubleConv(in_channels=1, out_channels=64)
  down1 = Down(in_channels=64, out_channels=128)
  up1 = Up(in_channels=128, out_channels=64)

  # 3. Chạy qua DoubleConv
  x1 = conv1(x0)
  print(
      f'1. Sau DoubleConv(1->64)| Shape: {list(x1.shape)} | Giữ nguyên 256x256,'
      ' trích 64 kênh đặc trưng'
  )

  # 4. Chạy qua Down (Co hẹp 1/2)
  x2 = down1(x1)
  print(
      f'2. Sau Down(64->128)   | Shape: {list(x2.shape)} | Thu nhỏ 1/2 thành'
      ' 128x128, tăng lên 128 kênh'
  )

  # 5. Chạy qua Up (Giãn x2 và ghép với x1)
  x_up = up1(x2, x1)
  print(
      f'3. Sau Up(128->64)     | Shape: {list(x_up.shape)} | Khôi phục 256x256'
      ' nhờ Skip Connection!'
  )

  print('=' * 75)
  print('>> Đang vẽ biểu đồ trực quan hóa đặc trưng ra file unet_trace.png...')

  # Vẽ minh họa dòng chảy thị giác
  fig, axes = plt.subplots(1, 4, figsize=(18, 5))

  # Ảnh gốc
  axes[0].imshow(x0.squeeze().detach().numpy(), cmap='gray')
  axes[0].set_title(
      f'Ảnh Gốc\nShape: {list(x0.shape)[1:]}', fontsize=11, fontweight='bold'
  )
  axes[0].axis('off')

  # 1 kênh bất kỳ sau DoubleConv (xem mạng trích viền màng ZP thế nào)
  feature_conv1 = x1[0, 0].detach().numpy()
  axes[1].imshow(feature_conv1, cmap='viridis')
  axes[1].set_title(
      f'Sau DoubleConv (Kênh 0)\nShape: {list(x1.shape)[1:]}', fontsize=11
  )
  axes[1].axis('off')

  # 1 kênh sau Down (ảnh bị nén 1/2 độ phân giải)
  feature_down1 = x2[0, 0].detach().numpy()
  axes[2].imshow(feature_down1, cmap='plasma')
  axes[2].set_title(
      f'Sau Down (Thu nhỏ 1/2)\nShape: {list(x2.shape)[1:]}', fontsize=11
  )
  axes[2].axis('off')

  # 1 kênh sau Up (đã khôi phục lại độ phân giải 256x256)
  feature_up1 = x_up[0, 0].detach().numpy()
  axes[3].imshow(feature_up1, cmap='inferno')
  axes[3].set_title(f'Sau Up (Phóng to 256)\nShape: {list(x_up.shape)[1:]}', fontsize=11)
  axes[3].axis('off')

  plt.tight_layout()
  plt.savefig('visualize/img/unet_trace.png', dpi=150)
  print(">> XONG! Mở file 'unet_trace.png' để tận mắt chiêm ngưỡng!")


if __name__ == '__main__':
  main()