# Architecture & Technical Research: Ultralytics-Inspired Training Engine

**Feature**: `001-ultralytics-style-engine`  
**Created**: 2026-10-07  

---

## 1. Decision 1: High-Level API Design Pattern (Ultralytics Paradigm)

- **Quyết định**: Áp dụng mẫu thiết kế **Facade Pattern** kết hợp **Delegation**:
  - Class `TASDF` là bề mặt tương tác duy nhất.
  - Khi gọi `model.train(...)`, nó ủy quyền cho `Trainer(model, cfg).train()`.
  - Khi gọi `model.val(...)`, nó ủy quyền cho `Validator(model, cfg).validate()`.
  - Khi gọi `model.predict(...)`, nó ủy quyền cho `Predictor(model, cfg).predict()`.
- **Lý do**:
  - Tách bạch hoàn toàn logic điều phối (orchestration) khỏi logic thuật toán và vòng lặp tính toán.
  - Người dùng trên Jupyter Notebook chỉ cần nhớ 1 class duy nhất `TASDF` mà không cần biết hàng chục tham số phức tạp bên dưới.
- **Giải pháp thay thế đã đánh giá**:
  - *Viết script đơn thuần với `argparse`*: Bất tiện khi dùng trong Notebook vì phải gọi `subprocess` hoặc truyền danh sách chuỗi `sys.argv`.

---

## 2. Decision 2: Config System & Dynamic Overrides

- **Quyết định**: Hệ thống cấu hình dựa trên `dict` phân cấp với khả năng ghi đè qua `**kwargs`:
  ```python
  model.train(data="configs/coco.yaml", epochs=30, batch_size=4, lr=1e-4)
  ```
  Các tham số truyền trực tiếp trong hàm sẽ ghi đè lên giá trị mặc định được định nghĩa trong file YAML.
- **Lý do**:
  - Cực kỳ linh hoạt, không yêu cầu người dùng phải chỉnh sửa file đĩa cứng. Phù hợp tuyệt đối với môi trường Notebook nơi tham số thường được thử nghiệm nhanh trong các cell.
- **Giải pháp thay thế đã đánh giá**:
  - *Hydra / OmegaConf*: Quá nặng nề và phức tạp cho nhu cầu huấn luyện của dự án, sinh ra nhiều overhead không cần thiết (vi phạm Rule 2.13).

---

## 3. Decision 3: Dual Telemetry Logging Strategy

- **Quyết định**: Xây dựng module tự trị `DualLogger`:
  - Luồng 1 (Human-Facing): Định dạng text ASCII với header cố định, ghi ra stdout (Notebook output) và append vào `train.log`.
  - Luồng 2 (Machine-Facing): File `metrics.json` lưu mảng các đối tượng JSON đại diện cho từng epoch.
- **Lý do**:
  - File JSON cho phép tải trực tiếp vào `pandas` để vẽ đồ thị trong Notebook chỉ với 2 dòng lệnh.
  - Zero dependencies: Không ép người dùng cài đặt hay đăng ký tài khoản các dịch vụ đám mây bên thứ ba.

---

## 4. Decision 4: Self-Contained Testing via In-Memory Synthetic Fixtures

- **Quyết định**: Trong `tests/conftest.py`, định nghĩa các fixture sinh dữ liệu ảnh và nhãn giả lập trên bộ nhớ (RAM / temp dir):
  - Sinh ảnh giả lập: `torch.rand(2, 1, 64, 64)`.
  - Sinh nhãn giả lập: `torch.randint(0, 5, (2, 64, 64))`.
  - Sinh file JSON COCO giả lập trong thư mục `tmp_path`.
- **Lý do**:
  - Đảm bảo kiểm thử tự động độc lập 100% với môi trường. Khi bạn clone sang máy trạm mới, Google Colab hay Kaggle, lệnh `pytest` luôn luôn chạy thành công mà không đòi hỏi sự hiện diện của dataset thực tế.
