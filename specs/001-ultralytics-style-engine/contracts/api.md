# API Contract: TASDF Facade

```python
class TASDF:
    """
    Điểm truy cập chính cho việc huấn luyện, đánh giá và suy luận mô hình TASDF.
    Thiết kế theo chuẩn tương thích Ultralytics.
    """

    def __init__(
        self,
        model: str | Path | None = None,
        task: str = "segment",
    ) -> None:
        """
        Khởi tạo đối tượng TASDF.
        
        Args:
            model: Đường dẫn tới file cấu hình yaml (ví dụ: 'tasdf.yaml') 
                   hoặc file trọng số checkpoint đã huấn luyện (ví dụ: 'best.pt').
            task: Tác vụ (mặc định 'segment').
        """
        ...

    def train(
        self,
        data: str | Path | dict,
        epochs: int = 50,
        batch: int = 4,
        imgsz: int = 256,
        lr: float = 1e-4,
        device: str | None = None,
        amp: bool = True,
        accum_steps: int = 1,
        project: str = "runs/train",
        name: str = "exp",
        resume: bool = False,
        **kwargs,
    ) -> dict[str, any]:
        """
        Bắt đầu vòng lặp huấn luyện bán giám sát.
        
        Returns:
            Dictionary chứa lịch sử metrics và đường dẫn file checkpoint tốt nhất.
        """
        ...

    def val(
        self,
        data: str | Path | dict | None = None,
        split: str = "val",
        batch: int = 4,
        device: str | None = None,
        **kwargs,
    ) -> dict[str, any]:
        """
        Thực hiện đánh giá độc lập mô hình trên tập kiểm tra.
        
        Returns:
            Dictionary chứa mDice, mIoU và chỉ số chi tiết từng lớp giải phẫu.
        """
        ...

    def predict(
        self,
        source: str | Path | list[Path] | np.ndarray,
        save: bool = True,
        output_dir: str | Path = "runs/predict",
        device: str | None = None,
        **kwargs,
    ) -> list[dict[str, any]]:
        """
        Chạy suy luận và xuất kết quả trực quan (mặt nạ, overlay).
        
        Returns:
            Danh sách kết quả dự đoán kèm theo mảng mặt nạ và thống kê lớp.
        """
        ...
```
