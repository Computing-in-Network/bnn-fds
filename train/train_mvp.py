#!/usr/bin/env python3
import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基线训练入口 MVP（占位）")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/train_mvp.json"),
        help="训练配置文件",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        # 减去表头
        return max(0, sum(1 for _ in reader) - 1)


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    run_name = cfg.get("run_name", "mvp_run")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    cases_csv = Path(cfg["input"]["cases_csv"])
    manifest_csv = Path(cfg["input"]["manifest_csv"])
    artifact_dir = Path(cfg["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    num_cases = count_rows(cases_csv)
    num_jobs = count_rows(manifest_csv)

    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["learning_rate"])
    model_name = cfg["model"]["name"]

    # 占位训练日志：模拟逐 epoch 损失下降
    train_log = []
    loss = 1.0
    for epoch in range(1, epochs + 1):
        loss = loss * random.uniform(0.75, 0.92)
        train_log.append({"epoch": epoch, "loss": round(loss, 6)})

    model_meta = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "seed": seed,
        "inputs": {
            "cases_csv": str(cases_csv),
            "manifest_csv": str(manifest_csv),
            "num_cases": num_cases,
            "num_jobs": num_jobs,
        },
        "train_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
        },
        "status": "completed_placeholder_train",
    }

    with (artifact_dir / "model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)
    with (artifact_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)

    print(f"训练占位流程完成，产物目录: {artifact_dir}")
    print(f"- model_meta.json")
    print(f"- train_log.json")


if __name__ == "__main__":
    main()
