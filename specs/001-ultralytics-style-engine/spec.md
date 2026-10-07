# Feature Specification: Ultralytics-Inspired Decoupled Training & Inference Engine

**Feature ID**: `001-ultralytics-style-engine`  
**Created**: 2026-10-07  
**Status**: DRAFT (Awaiting User Lock-In)  

---

## 1. Executive Summary & Problem Framing

Hiện tại, mã nguồn mô hình TASDF đang trong trạng thái nguyên mẫu nghiên cứu (Research Prototype):
- Các thiết lập đường dẫn và cấu hình gắn chặt vào môi trường máy trạm cục bộ (`/home/terminator/...`).
- Quy trình huấn luyện phụ thuộc vào các script dòng lệnh rời rạc (`train.py`, `eval.py`, `inference.py`) với các tham số phân mảnh.
- Không thể khởi chạy hoặc điều khiển trực tiếp một cách thanh thoát trong môi trường Jupyter Notebook.
- Thiếu lớp trừu tượng hóa bộ nạp dữ liệu (Dataset Abstraction Layer), gây khó khăn khi đối mặt với các tập dữ liệu phôi học có cấu trúc thư mục, định dạng nhãn hoặc quy mô khác nhau.
- Nhật ký thực nghiệm phụ thuộc vào màn hình console thô, thiếu file lưu trữ có cấu trúc rõ ràng.

**Mục tiêu cốt lõi**:
Tái cấu trúc kiến trúc phần mềm theo triết lý thiết kế hướng người dùng tương tự như thư viện Ultralytics (YOLO):
1. Cung cấp **High-Level Python API** cho phép khởi tạo, huấn luyện, đánh giá và suy luận chỉ với vài dòng lệnh (`from tasdf import TASDF; model = TASDF(); model.train(data=...)`).
2. Tương thích tự nhiên với **Jupyter Notebook trên máy trạm** lẫn giao diện dòng lệnh (CLI).
3. **Phi cục bộ hóa dữ liệu**: Quản lý tập dữ liệu thông qua cấu hình `data.yaml` độc lập, hỗ trợ linh hoạt các định dạng khác nhau (COCO, Mask Folder, v.v.).
4. **Hệ thống giám sát tối giản & chi tiết**: Tự động lưu trữ song song `train.log` (bảng text ASCII trực quan) và `metrics.json` (dữ liệu số cấu trúc hóa theo từng epoch).

---

## 2. User Scenarios & Workflows

### Scenario 1: Huấn luyện trực tiếp trong Jupyter Notebook trên máy trạm (Primary Workflow)
- **Người dùng**: Nhà nghiên cứu / Kỹ sư AI làm việc trên Jupyter Notebook.
- **Quy trình tương tác**:
  ```python
  from tasdf import TASDF

  # 1. Khởi tạo mô hình (từ kiến trúc mặc định hoặc từ checkpoint đã có)
  model = TASDF("tasdf.yaml")  # hoặc TASDF("checkpoints/best.pt")

  # 2. Huấn luyện với cấu hình dữ liệu bất kỳ
  results = model.train(
      data="configs/dataset_custom.yaml",
      epochs=50,
      batch=4,
      lr=1e-4,
      device="cuda"
  )

  # 3. Đánh giá độc lập
  metrics = model.val(data="configs/dataset_custom.yaml")

  # 4. Suy luận trực quan
  model.predict(source="path/to/embryo_images", save=True)
  ```
- **Kỳ vọng**: Kernel không bị treo, tiến trình hiển thị rõ ràng, kết quả trả về dưới dạng đối tượng Python chứa metrics.

### Scenario 2: Cấu hình tập dữ liệu độc lập và linh hoạt (Data-Agnostic Specification)
- **Người dùng**: Mang mô hình sang huấn luyện một tập dữ liệu mới với cấu trúc thư mục tùy ý.
- **Quy trình tương tác**:
  Tạo một file định nghĩa dữ liệu `data.yaml`:
  ```yaml
  path: /absolute/or/relative/path/to/data_root
  format: coco           # hoặc 'mask_folder', 'image_pairs'
  train: images/train    # hoặc file json tương ứng
  val: images/val
  test: images/test
  labeled_ratio: 0.2
  names:
    0: Background
    1: Zona Pellucida (ZP)
    2: Trophectoderm (TE)
    3: Inner Cell Mass (ICM)
    4: Blastocoel Cavity (BC)
  ```
- **Kỳ vọng**: Động cơ dữ liệu tự động nhận diện `format` và nạp mẫu hợp lệ, không phụ thuộc vào bất kỳ đường dẫn máy cá nhân nào.

### Scenario 3: Tiếp tục huấn luyện an toàn (Fault-Tolerant Resumption)
- **Người dùng**: Buổi huấn luyện dài hạn bị ngắt quãng do tắt máy trạm hoặc kernel khởi động lại.
- **Quy trình tương tác**:
  ```python
  model = TASDF("runs/train/exp1/checkpoints/last.pt")
  model.train(resume=True)
  ```
- **Kỳ vọng**: Hệ thống khôi phục nguyên vẹn trạng thái model, optimizer, scheduler và epoch hiện tại, tiếp tục huấn luyện mà không mất mát dữ liệu trước đó.

### Scenario 4: Giám sát tối giản với Text Log và JSON Metrics
- **Người dùng**: Cần theo dõi tiến độ huấn luyện và vẽ biểu đồ phân tích sau phiên chạy.
- **Quy trình**:
  Mỗi lần huấn luyện, một thư mục đầu ra `runs/train/expN/` được tạo tự động chứa:
  1. `train.log`: Ghi nhận bảng biểu ASCII hiển thị tiến độ theo từng epoch (Epoch, Mem, Sup Loss, Distill Loss, Val Dice, Per-Class Dice).
  2. `metrics.json`: Mảng JSON ghi nhận chi tiết toàn bộ các đại lượng toán học ở từng epoch, dễ dàng nạp vào `pandas.DataFrame` để vẽ biểu đồ loss/dice trong 1 dòng code.

---

## 3. Functional Requirements

- **FR-1 (Unified High-Level Interface)**:
  - Cung cấp lớp đối tượng trung tâm `TASDF` đóng gói toàn bộ logic khởi tạo, huấn luyện, kiểm định và dự đoán.
  - Tách rời hoàn toàn lớp trình diễn (Notebook/CLI) khỏi lớp thuật toán lõi (`TASDFNet`, `ITGM`, `MainLoss`).

- **FR-2 (Dataset Contract & Factory Decoupling)**:
  - Chuẩn hóa giao diện dữ liệu (Dataset SPI). Mọi bộ nạp dữ liệu (COCO format, mask image format) phải triển khai một giao diện chung và xuất ra tensor hợp lệ.
  - Phân tích cú pháp đường dẫn linh hoạt: tự động nhận diện đường dẫn tương đối so với file YAML hoặc đường dẫn tuyệt đối.

- **FR-3 (Zero Hardcoded Configuration)**:
  - Không chứa bất kỳ đường dẫn máy cá nhân (`/home/terminator/...`) trong mã nguồn gốc hay trong bộ kiểm thử `pytest`.
  - Toàn bộ tham số cấu hình có thể ghi đè linh hoạt qua Python API arguments hoặc CLI arguments.

- **FR-4 (Self-Contained Automated Test Fixtures)**:
  - Bộ kiểm thử tự động `pytest` sử dụng cơ chế sinh dữ liệu phôi giả lập (Synthetic Embryo In-Memory Fixtures) để kiểm chứng tính toàn vẹn của forward/backward, loss, scheduler, và data pipeline mà không đòi hỏi sự tồn tại của file dữ liệu thực tế trên đĩa.

- **FR-5 (Lightweight Dual Telemetry)**:
  - Module ghi nhận nhật ký nhẹ (`DualLogger`) độc lập, ghi đồng thời ra stdout, `train.log` và `metrics.json`.
  - Không bắt buộc cài đặt hay phụ thuộc vào các dịch vụ đám mây bên thứ ba (như wandb, comet) nhưng vẫn bảo đảm tính toàn vẹn và chi tiết của thông số.

- **FR-6 (Fault-Tolerant Checkpoint Serialization)**:
  - Checkpoint lưu trữ đầy đủ: trọng số mô hình, trọng số optimizer, learning rate scheduler state, epoch index, best metric tracker.

---

## 4. Success Criteria

- **SC-1 (Tối giản khởi chạy)**: Nhà nghiên cứu có thể khởi chạy một phiên huấn luyện đầy đủ từ Jupyter Notebook chỉ với tối đa **3 dòng code**.
- **SC-2 (Tính khả chuyển tuyệt đối - Portability)**: Mã nguồn có thể được sao chép sang bất kỳ máy trạm, server hoặc notebook nào khác và chạy `pytest` thành công 100% ngay lập tức mà không cần chỉnh sửa đường dẫn.
- **SC-3 (Tính liên tục của nhật ký)**: `train.log` và `metrics.json` được cập nhật đồng bộ sau mỗi epoch với độ trễ I/O $< 50\text{ms}$.
- **SC-4 (Tính khôi phục)**: Khi kích hoạt `resume=True`, mô hình tiếp tục huấn luyện chính xác từ epoch cuối cùng đã lưu mà không bị nhảy bước học (learning rate jump) hay rò rỉ bộ nhớ.
- **SC-5 (Độc lập bộ nhớ trên Notebook)**: Kernel Jupyter không bị hiện tượng rò rỉ VRAM (memory leak) qua các epoch liên tiếp.

---

## 5. Non-Functional Constraints & Boundaries

- **Nguyên tắc Clean Architecture**: Tuân thủ nghiêm ngặt Rule 2.1 (Files $\le 300$ dòng, functions $\le 40$ dòng).
- **Nguyên tắc Anti-Overengineering (Rule 2.13)**: Không tạo các lớp bao bọc trừu tượng dư thừa (YAGNI). Thiết kế API bám sát các mẫu thiết kế Pythonic tiêu chuẩn.
- **Giữ nguyên thuật toán lõi**: Không làm thay đổi cơ chế toán học của mô hình TASDF (dual U-Net, ITGM soft skeleton, dynamic sigmoid ramp-up loss).
