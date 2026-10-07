# Contract: Data Configuration Schema (`data.yaml`)

```yaml
# Đường dẫn thư mục gốc dữ liệu (hỗ trợ tương đối hoặc tuyệt đối)
path: /path/to/dataset_root

# Định dạng dữ liệu: 'coco' hoặc 'folder'
format: coco

# Đường dẫn đến các tập con (tương đối so với 'path' hoặc tuyệt đối)
train: train
val: valid
test: test

# Kích thước chuẩn hóa ảnh (vuông)
image_size: 256

# Tỷ lệ gán nhãn trong tập huấn luyện (phục vụ bài toán bán giám sát TASDF)
labeled_ratio: 0.2

# Danh mục các lớp giải phẫu phôi học
names:
  0: Background
  1: Zona Pellucida (ZP)
  2: Trophectoderm (TE)
  3: Inner Cell Mass (ICM)
  4: Blastocoel Cavity (BC)
```
