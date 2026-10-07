#!/usr/bin/env python3
"""
create_low_resource_coco.py
Công cụ tách lọc và tạo bộ dữ liệu COCO bán giám sát độc lập (Frozen Low-Resource Dataset).
- Giữ nguyên 100% ảnh đầu vào trong tập train.
- Giữ lại đúng 10% nhãn cho các ảnh đạt độ hoàn thiện cao nhất (ưu tiên đủ 3 cấu trúc ZP, TE, ICM).
- Tước bỏ toàn bộ nhãn của 90% ảnh còn lại để tạo dữ liệu không nhãn thuần túy.
- Giữ nguyên 100% tập valid và test.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil


def link_or_copy(src: Path, dst: Path, mode: str = "hardlink"):
    """Tạo hardlink hoặc copy file an toàn."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Frozen Low-Resource COCO Dataset")
    parser.add_argument(
        "--src",
        type=str,
        default="/home/terminator/workspace/Attech/Research/dataset/Blast Segment.v15i.coco-segmentation",
        help="Source COCO dataset directory",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/home/terminator/workspace/Attech/Research/dataset/Blast_Segment_10p_coco",
        help="Destination directory for the filtered dataset",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.10,
        help="Ratio of images to keep labeled (default: 0.10 for 10 percent)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified selection",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy"],
        help="File transfer mode for images (hardlink or copy)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {src_dir}")

    print("=" * 75)
    print(" KHỞI TẠO BỘ DỮ LIỆU BÁN GIÁM SÁT 10% NHÃN (TASDF LOW-LABEL DATASET) ")
    print("=" * 75)
    print(f">> Nguồn:      {src_dir}")
    print(f">> Đích:       {dst_dir}")
    print(f">> Tỷ lệ nhãn: {args.ratio * 100:.1f}%")
    print(f">> Seed:       {args.seed}")
    print(f">> Chế độ:     {args.mode}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Xử lý tập TRAIN
    train_src = src_dir / "train"
    train_dst = dst_dir / "train"
    train_dst.mkdir(parents=True, exist_ok=True)

    ann_src_file = train_src / "_annotations.coco.json"
    if not ann_src_file.exists():
        ann_src_file = train_src / "annotations.json"
    if not ann_src_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file annotation trong {train_src}")

    with open(ann_src_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    total_images = len(images)
    num_labeled = max(1, round(total_images * args.ratio))
    print(f"\n[+] Phân tích tập train: Tổng số ảnh = {total_images} | Số ảnh giữ nhãn = {num_labeled}")

    # Nhóm danh mục nhãn theo 3 cấu trúc sinh học chính:
    # ZP: 7, 8, 9 | TE: 4, 5, 6 | ICM: 1, 2, 3
    img_to_structs = {}
    img_to_anns = {}
    for ann in annotations:
        iid = ann["image_id"]
        cid = ann["category_id"]
        img_to_anns.setdefault(iid, []).append(ann)

        struct = None
        if cid in (7, 8, 9):
            struct = "ZP"
        elif cid in (4, 5, 6):
            struct = "TE"
        elif cid in (1, 2, 3):
            struct = "ICM"
        if struct:
            img_to_structs.setdefault(iid, set()).add(struct)

    # Phân tầng ưu tiên ảnh có đủ cả 3 cấu trúc
    tier_full = [img["id"] for img in images if img_to_structs.get(img["id"], set()) == {"ZP", "TE", "ICM"}]
    tier_partial = [img["id"] for img in images if img["id"] not in tier_full and len(img_to_structs.get(img["id"], set())) > 0]
    tier_empty = [img["id"] for img in images if img["id"] not in tier_full and img["id"] not in tier_partial]

    print(f"    - Ảnh có đủ cả 3 cấu trúc (ZP + TE + ICM): {len(tier_full)}")
    print(f"    - Ảnh có một phần cấu trúc:                 {len(tier_partial)}")
    print(f"    - Ảnh không có cấu trúc nào:               {len(tier_empty)}")

    # Lựa chọn xác định theo seed
    rng = random.Random(args.seed)
    # Sắp xếp trước khi shuffle để đảm bảo tính tất định trên mọi hệ điều hành
    tier_full.sort()
    rng.shuffle(tier_full)

    if len(tier_full) >= num_labeled:
        selected_labeled_ids = set(tier_full[:num_labeled])
    else:
        tier_partial.sort()
        rng.shuffle(tier_partial)
        needed = num_labeled - len(tier_full)
        selected_labeled_ids = set(tier_full + tier_partial[:needed])

    # Lọc lại annotations: Chỉ giữ annotations của ảnh được chọn
    filtered_annotations = [ann for ann in annotations if ann["image_id"] in selected_labeled_ids]

    # Kiểm toán annotations giữ lại
    from collections import Counter
    kept_cat_counts = Counter(ann["category_id"] for ann in filtered_annotations)
    cats_by_id = {c["id"]: c["name"] for c in categories}

    print(f"\n[+] Đã chọn lọc thành công {len(selected_labeled_ids)} ảnh có nhãn:")
    for cid in sorted(kept_cat_counts.keys()):
        print(f"    - Lớp {cid} ({cats_by_id.get(cid, 'Unknown')}): {kept_cat_counts[cid]} annotations")

    # Tạo tệp _annotations.coco.json mới cho tập train
    new_coco_train = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": categories,
        "images": images,  # Giữ nguyên 100% ảnh
        "annotations": filtered_annotations,  # Chỉ còn nhãn của 10% ảnh
    }

    train_dst_ann = train_dst / "_annotations.coco.json"
    with open(train_dst_ann, "w", encoding="utf-8") as f:
        json.dump(new_coco_train, f, indent=2)

    # Chuyển ảnh train
    print(f"\n[+] Sao chép/liên kết {len(images)} ảnh tập train sang {train_dst}...")
    for img in images:
        f_name = img["file_name"]
        src_img = train_src / f_name
        dst_img = train_dst / f_name
        if src_img.exists():
            link_or_copy(src_img, dst_img, mode=args.mode)

    # 2. Xử lý tập VALID và TEST (Giữ nguyên 100% để đánh giá khách quan)
    for split in ["valid", "test"]:
        split_src = src_dir / split
        split_dst = dst_dir / split
        if not split_src.exists():
            continue
        split_dst.mkdir(parents=True, exist_ok=True)
        print(f"\n[+] Đồng bộ nguyên vẹn tập '{split}' (100% nhãn để làm chuẩn đối chứng)...")

        # Copy file annotation
        for ann_name in ["_annotations.coco.json", "annotations.json"]:
            if (split_src / ann_name).exists():
                shutil.copy2(split_src / ann_name, split_dst / ann_name)

        # Chuyển ảnh
        for f in split_src.glob("*"):
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                link_or_copy(f, split_dst / f.name, mode=args.mode)

    print("\n" + "=" * 75)
    print(" BÁO CÁO HOÀN TẤT BỘ DỮ LIỆU ĐỘC LẬP ")
    print("=" * 75)
    print(f">> Tổng số ảnh Train:       {len(images)}")
    print(f">> Số ảnh Có Nhãn (10%):    {len(selected_labeled_ids)} ({len(filtered_annotations)} annotations)")
    print(f">> Số ảnh Không Nhãn (90%): {len(images) - len(selected_labeled_ids)} (0 annotations - zero leakage)")
    print(f">> Thư mục sẵn sàng tại:    {dst_dir.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
