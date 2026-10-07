# Data Model & Entity Specifications: Ultralytics Engine

**Feature**: `001-ultralytics-style-engine`  
**Created**: 2026-10-07  

---

## 1. Core Entities

### 1.1. `DataConfig`
Thực thể định nghĩa thông tin tập dữ liệu:
```python
class DataConfig:
    path: Path             # Thư mục gốc chứa dữ liệu
    format: str            # 'coco', 'folder', 'embryo'
    train: str             # Đường dẫn file/thư mục train tương đối hoặc tuyệt đối
    val: str               # Đường dẫn file/thư mục val tương đối hoặc tuyệt đối
    test: str              # Đường dẫn file/thư mục test tương đối hoặc tuyệt đối
    labeled_ratio: float   # Tỷ lệ nhãn phục vụ bán giám sát (0.0 < r <= 1.0)
    image_size: int        # Kích thước resize ảnh (mặc định 256)
    names: dict[int, str]  # Ánh xạ class_id sang tên giải phẫu học
```

### 1.2. `TrainingConfig`
Thực thể định nghĩa siêu tham số huấn luyện:
```python
class TrainingConfig:
    epochs: int            # Tổng số epoch huấn luyện
    batch_size: int        # Kích thước lô nạp mẫu
    learning_rate: float   # Tốc độ học (Adam)
    weight_decay: float    # Hệ số suy giảm trọng số
    device: str            # 'cuda' hoặc 'cpu'
    amp: bool              # Sử dụng FP16 Automatic Mixed Precision
    accum_steps: int       # Số bước tích lũy gradient
    checkpoint_dir: Path   # Thư mục lưu checkpoint
    save_period: int       # Tần suất lưu checkpoint (theo epoch)
    resume: bool           # Tiếp tục từ checkpoint cuối cùng
```

### 1.3. `EpochMetrics`
Thực thể dữ liệu thống kê sau mỗi epoch:
```python
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_loss_sup: float
    train_loss_distill: float
    val_mDice: float
    val_mIoU: float
    per_class_dice: dict[str, float]
    per_class_iou: dict[str, float]
    gpu_memory_used_mb: float
    timestamp: str
```

### 1.4. `CheckpointState`
Cấu trúc lưu trạng thái toàn vẹn trên đĩa:
```python
{
    "epoch": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "best_val_dice": float,
    "config": dict,
    "tasdf_version": str,
}
```
