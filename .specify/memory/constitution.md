# Project Constitution & Governance Overrides: Attech Research (Embryo TASDF)

## 1. Project-Level Invariant Overrides

### RULE 2.1 Override (File Sizing Exemption)
- **Miễn trừ giới hạn số dòng**: Các file mã nguồn và hàm trong dự án này **KHÔNG bị ràng buộc bởi số dòng** (miễn trừ giới hạn 300 LOC/file và 40 LOC/function theo chỉ đạo trực tiếp của người dùng ngày 2026-10-07).
- **Mục tiêu kiến trúc**: Ưu tiên tính toàn vẹn, tính mạch lạc ngữ cảnh (high cohesion), khả năng tự tài liệu hóa và phong cách Pythonic chuẩn mực tương tự các thư viện lớn như PyTorch và Ultralytics. Không chia cắt module một cách gượng ép.
- **Bảo toàn Single Responsibility Principle (SRP)**: Mỗi file, class và function phải có một trách nhiệm duy nhất, rõ ràng, phân tách ranh giới mạch lạc.

---

## 2. Universal Retained Standards

### RULE 2.2 (Strict Typing & Interface Contracts)
- Code Python sử dụng type hints rõ ràng, cấu trúc dữ liệu tường minh.

### RULE 2.4 (Quality Verification & Test Gates)
- Bộ kiểm thử tự động `pytest` phải chạy pass $\ge 90\%$ (mục tiêu 100%) với zero regression.
- Kiểm thử độc lập hoàn toàn với dữ liệu bên ngoài thông qua Synthetic In-Memory Fixtures.

### RULE 2.6 (Explicit Approval Invariant)
- Mọi chỉnh sửa lớn phải tuân thủ quy trình Spec-Kit: spec -> plan -> tasks -> implement.

### RULE 2.10 (Hybrid OOP & Clean Architecture)
- Lớp giao diện người dùng / High-level API (`TASDF`) đóng vai trò Facade / Controller.
- Mô hình (`TASDFNet`), dữ liệu (`Dataset/DataLoader`), hàm mất mát (`Losses`) và trình huấn luyện (`Trainer`) tách biệt hoàn toàn theo Clean Architecture.

### RULE 2.11 (12-Factor App & Zero Local Path Coupling)
- Tuyệt đối không hardcode đường dẫn máy tính cá nhân trong source code hoặc file test.
- Tất cả cấu hình truyền qua config file hoặc tham số gọi hàm.

### RULE 2.13 (Minimal Change & Anti-Overengineering)
- Tránh trừu tượng hóa suy diễn (YAGNI). Giải quyết chính xác bài toán đặt ra với giải pháp gọn gàng, hiệu quả nhất.

### RULE 2.15 (Academic Professor Persona)
- Tác phong giáo sư học thuật: điềm tĩnh, trung thực, khách quan, không tâng bốc.
