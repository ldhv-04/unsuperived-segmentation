"""
inference.py - CLI entrypoint for inference and mask visualization on embryo images.
Thin wrapper delegating directly to the high-level TASDF facade API.
"""
import argparse
from pathlib import Path
from tasdf import TASDF


def parse_args():
    parser = argparse.ArgumentParser(description="TASDF Inference CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="runs/predict", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=256, help="Inference image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--branch", type=str, default="net1", choices=["net1", "net2", "ensemble"], help="Inference branch")
    parser.add_argument("--decompose", action="store_true", help="Enable topological decomposition of TE and blastocoel")
    parser.add_argument("--te_thickness_ratio", type=float, default=0.06, help="Thickness ratio for TE annular ring")
    return parser.parse_args()


def main():
    args = parse_args()
    model = TASDF(model=args.checkpoint)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        device=args.device,
        branch=args.branch,
        decompose=args.decompose,
        te_thickness_ratio=args.te_thickness_ratio,
        save=True,
    )
    print(f"\n[+] Đã hoàn thành suy luận trên {len(results)} ảnh! Kết quả lưu tại: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
