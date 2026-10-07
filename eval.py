"""
eval.py - CLI entrypoint for evaluating TASDF models.
Thin wrapper delegating directly to the high-level TASDF facade API.
"""
import argparse
import json
from pathlib import Path
from tasdf import TASDF


def parse_args():
    parser = argparse.ArgumentParser(description="TASDF Evaluation CLI")
    parser.add_argument("--config", type=str, default="configs/coco_blastocyst.yaml", help="Path to YAML config")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "valid", "test", "train"], help="Split to evaluate")
    parser.add_argument("--branch", type=str, default="net1", choices=["net1", "net2", "ensemble"], help="Evaluation branch")
    parser.add_argument("--imgsz", type=int, default=256, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save metrics JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    model = TASDF(model=args.weights)
    results = model.val(
        data=args.config,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        branch=args.branch,
    )
    if args.save_json:
        out_p = Path(args.save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f">> Chỉ số đánh giá đã lưu tại: {out_p.resolve()}")


if __name__ == "__main__":
    main()
