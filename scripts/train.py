"""YOLO 학습 CLI. 기본은 실행 명령만 출력(사람이 tmux/별도 창에서 돌린다). --run이면 직접 실행."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/dataset/dataset.yaml")
    ap.add_argument("--out", default="runs/train")
    ap.add_argument("--run", action="store_true", help="명령 출력 대신 직접 학습 실행")
    ap.add_argument("--workers", type=int, default=2,
                    help="데이터로더 워커 수 (Windows에서 8이면 pin memory 스레드가 죽는 사례)")
    ap.add_argument("--model", default=None, help="모델 오버라이드 (기본: config detector.model)")
    ap.add_argument("--name", default="crater", help="run 이름 (runs/train/<name>)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    det = cfg["detector"]
    model_name = args.model or det["model"]
    kwargs = dict(
        data=args.data, imgsz=int(det["imgsz"]), epochs=int(det["epochs"]),
        batch=int(det["batch"]), seed=args.seed, project=args.out, name=args.name,
        exist_ok=True, deterministic=True, workers=args.workers,
    )
    cli = (
        f'.venv\\Scripts\\python -m ultralytics.cfg train model={model_name} '
        + " ".join(f"{k}={v}" for k, v in kwargs.items())
    )
    cmd = (
        f'.venv\\Scripts\\python scripts\\train.py --config {args.config} --seed {args.seed} '
        f"--data {args.data} --out {args.out} --run "
        f"> logs\\train.log 2>&1"
    )
    if not args.run:
        print("다음 명령을 별도 창(tmux)에서 실행해라:")
        print(f"  {cmd}")
        print(f"(내부적으로 ultralytics train: model={model_name}, {kwargs})")
        print(f"참고 CLI 동등 명령: {cli}")
        return

    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(**kwargs)
    print(f"best: {Path(args.out) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
