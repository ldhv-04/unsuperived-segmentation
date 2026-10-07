# Quickstart Guide: TASDF Engine in Jupyter Notebook

## 1. Cài đặt chế độ phát triển (Editable Install)
Trong terminal hoặc trong cell đầu tiên của Jupyter Notebook:
```bash
!pip install -e .
```

## 2. Huấn luyện Mô hình trong 3 Dòng Lệnh
```python
from tasdf import TASDF

# Khởi tạo mô hình
model = TASDF()

# Chạy huấn luyện
results = model.train(
    data="configs/coco_blastocyst.yaml", 
    epochs=30, 
    batch=4,
    device="cuda"
)
```

## 3. Xem Kết quả Nhật ký & Đồ thị
```python
import pandas as pd
import matplotlib.pyplot as plt

# Đọc file metrics.json trực tiếp từ thư mục chạy
metrics_df = pd.read_json(results["metrics_path"])

# Vẽ đồ thị tiến độ
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
metrics_df.plot(x="epoch", y=["train_total_loss", "train_loss_sup"], ax=ax[0], title="Loss Curves")
metrics_df.plot(x="epoch", y=["val_mDice", "ZP_Dice", "TE_Dice", "ICM_Dice"], ax=ax[1], title="Validation Dice")
plt.tight_layout()
plt.show()
```

## 4. Dự đoán và Xuất Ảnh Mặt nạ
```python
# Tải mô hình tốt nhất vừa huấn luyện
trained_model = TASDF(results["best_checkpoint"])

# Dự đoán thư mục ảnh phôi
predictions = trained_model.predict(
    source="/path/to/ICM Blastocyst",
    save=True,
    output_dir="runs/predict/test1"
)
```
