# Technical Implementation Plan: Ultralytics-Inspired Decoupled Training Engine

**Feature**: `001-ultralytics-style-engine`  
**Spec Reference**: [`spec.md`](./spec.md)  
**Created**: 2026-10-07  
**Status**: APPROVED & IN PROGRESS  

---

## 1. Technical Context & Scope

### 1.1. Core Architectural Strategy
Tái cấu trúc mã nguồn hiện tại từ dạng monolithic scripts thành một thư viện Python chuẩn mực (`tasdf`), tuân theo mẫu thiết kế Facade + Clean Architecture tương tự Ultralytics:
1. **Lớp Giao diện (High-Level API - `TASDF`)**:
   - Đóng vai trò là entrypoint chính cho người dùng trong Jupyter Notebook và script Python.
   - Quản lý vòng đời mô hình: `train()`, `val()`, `predict()`.
2. **Lớp Động cơ Thực thi (Execution Engine - `tasdf/engine/`)**:
   - `Trainer`: Quản lý vòng lặp epoch, tối ưu hóa (AMP FP16, Adam, GradScaler), checkpointing và phục hồi trạng thái (`resume`).
   - `Validator`: Đo lường chi tiết đa lớp (Dice, IoU, per-class breakdown).
   - `Predictor`: Suy luận trên ảnh đơn/thư mục, hỗ trợ tạo ảnh phủ màu (overlay) và trích xuất mặt nạ.
   - `DualLogger`: Ghi nhận đồng thời bảng text ASCII đẹp mắt trên màn hình/file `train.log` và dữ liệu chuỗi số trong `metrics.json`.
3. **Lớp Hợp đồng Dữ liệu (Data Contract Layer - `tasdf/data/`)**:
   - Trừu tượng hóa nguồn dữ liệu qua `BaseEmbryoDataset`.
   - Hỗ trợ các định dạng khác nhau thông qua file cấu hình `data.yaml`.
   - Xóa bỏ 100% các hardcoded paths trong mã nguồn và bộ test.
4. **Lớp Mô hình & Hàm Mất mát (Core Science Layer - `tasdf/models/` & `tasdf/losses/`)**:
   - Giữ nguyên vẹn 100% thuật toán toán học cốt lõi: dual U-Net `TASDFNet`, module khung xương mềm `ITGM`, và hàm mất mát `MainLoss`.
   - Đóng gói thành các module chuẩn mực, import tự nhiên mà không cần `sys.path.insert`.

---

## 2. Constitution & Governance Alignment

- **RULE 2.1 Override (File Sizing Exemption)**: Áp dụng miễn trừ giới hạn dòng theo chỉ thị của người dùng. Các module được viết liền mạch, đầy đủ tài liệu hóa, ưu tiên sự rõ ràng và tự nhiên của Python.
- **RULE 2.4 (Automated Quality Gate)**: Bộ kiểm thử `pytest` đạt $\ge 90\%$ (mục tiêu 100%) dựa trên `conftest.py` sinh synthetic fixtures trên RAM.
- **RULE 2.10 (Clean Architecture & DIP)**: `Trainer` phụ thuộc vào interface trừu tượng của `Dataset` và `nn.Module`, không phụ thuộc vào đường dẫn đĩa cứng.
- **RULE 2.11 (12-Factor App)**: Không hardcode bất kỳ đường dẫn nào; toàn bộ tham số nhận qua cấu hình YAML hoặc đối số hàm.
- **RULE 2.13 (Minimal Change / Anti-Overengineering)**: Không thêm thư viện wrapper nặng nề, chỉ sử dụng PyTorch thuần và thư viện chuẩn của Python.

---

## 3. Phase Breakdown & Execution Sequence

### Phase 0: Nghiên cứu Kiến trúc & Giải tỏa Điểm chưa rõ (`research.md`)
- Xác định mô hình config parsing (hỗ trợ nested YAML + kwargs overrides).
- Xác định cấu trúc lưu checkpoint (`latest.pt`, `best.pt`) đảm bảo tính toàn vẹn khi `resume`.
- Thiết kế định dạng file `metrics.json` và `train.log`.

### Phase 1: Hợp đồng Giao diện & Mô hình Dữ liệu (`data-model.md`, `contracts/`, `quickstart.md`)
- `data-model.md`: Cấu trúc thực thể `TASDFModelConfig`, `DatasetConfig`, `TrainingState`, `MetricRecord`.
- `contracts/api.md`: Hợp đồng chữ ký hàm của class `TASDF`.
- `contracts/data_config.md`: Schema định dạng của file `data.yaml`.
- `quickstart.md`: Hướng dẫn kiểm chứng nhanh 3 dòng trên Jupyter Notebook.

### Phase 2: Phân rã Công việc & Triển khai Tuần tự (`tasks.md` $\to$ `/speckit-implement`)
- Tái cấu trúc thư viện `src/tasdf/`.
- Chuyển giao các module core model và loss sang package mới.
- Viết `DualLogger` và `BaseDataset` abstraction.
- Viết `Trainer`, `Validator`, `Predictor` và facade `TASDF`.
- Viết lại toàn bộ bộ test `tests/` với in-memory fixtures.
- Kiểm thử toàn diện 100% pass.
