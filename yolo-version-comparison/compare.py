"""
YOLO Version Benchmark: v8 vs v11 vs v26
=========================================
사용법:
  python compare.py                          # 샘플 이미지로 실행
  python compare.py --image path/to/img.jpg  # 직접 이미지 지정
  python compare.py --runs 10                # 반복 횟수 조정
  python compare.py --models v8,v26          # 특정 버전만 비교
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


# ── 패키지 자동 설치 ─────────────────────────────────────────────────────────
def ensure_packages():
    try:
        import ultralytics  # noqa
    except ImportError:
        print("[setup] ultralytics 설치 중...")
        os.system("pip install ultralytics --quiet")

ensure_packages()

from ultralytics import YOLO  # noqa: E402


# ── 모델 정의 ────────────────────────────────────────────────────────────────
ALL_MODELS = {
    "v8":  {"label": "YOLOv8n",  "weight": "yolov8n.pt"},
    "v11": {"label": "YOLOv11n", "weight": "yolo11n.pt"},
    "v26": {"label": "YOLOv26n", "weight": "yolo26n.pt"},
}

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ── 단일 모델 벤치마크 ────────────────────────────────────────────────────────
def benchmark(cfg: dict, source, runs: int) -> dict:
    label  = cfg["label"]
    weight = cfg["weight"]

    print(f"\n{'─'*54}")
    print(f"  [{label}]  ({weight})")
    print(f"{'─'*54}")

    # 1. 모델 로딩 시간 측정
    t0 = time.perf_counter()
    model = YOLO(weight)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  ✓ 로딩     : {load_ms:.1f} ms")

    # 2. 워밍업 (JIT / 커널 캐시)
    model(source, verbose=False)
    print(f"  ✓ 워밍업   : 완료")

    # 3. 반복 추론
    times = []
    last_results = None
    for i in range(runs):
        t1 = time.perf_counter()
        last_results = model(source, verbose=False)
        elapsed = (time.perf_counter() - t1) * 1000
        times.append(elapsed)
        print(f"  Run {i+1:>2}/{runs}  : {elapsed:.1f} ms")

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps    = 1000.0 / avg_ms

    # 4. 검출 결과 파싱
    result         = last_results[0]
    boxes          = result.boxes
    detections     = []
    class_counts   = {}

    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = model.names[cls_id]
            xyxy   = [round(v, 1) for v in box.xyxy[0].tolist()]
            detections.append({"class": name, "confidence": round(conf, 4), "bbox": xyxy})
            class_counts[name] = class_counts.get(name, 0) + 1

    # 5. 파라미터 수 (M단위)
    params_M = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)

    # 6. 어노테이션 이미지 저장
    ann_path = str(OUT_DIR / f"{label}_annotated.jpg")
    result.save(filename=ann_path)

    record = {
        "model":        label,
        "weight":       weight,
        "load_ms":      round(load_ms, 1),
        "avg_ms":       round(avg_ms, 1),
        "std_ms":       round(std_ms, 1),
        "fps":          round(fps, 1),
        "params_M":     params_M,
        "num_detect":   len(detections),
        "class_counts": class_counts,
        "detections":   detections[:15],
    }

    print(f"\n  ── 요약 ────────────────────────────────────────")
    print(f"  추론 평균  : {avg_ms:.1f} ± {std_ms:.1f} ms  |  FPS: {fps:.1f}")
    print(f"  파라미터   : {params_M} M")
    print(f"  검출 결과  : {len(detections)} 개  {class_counts}")
    return record


# ── 터미널 비교표 ─────────────────────────────────────────────────────────────
def print_table(records: list):
    print("\n" + "═" * 74)
    print("  최종 비교 결과")
    print("═" * 74)
    header = f"  {'모델':<12} {'로딩(ms)':>10} {'추론(ms)':>10} {'±std':>7} {'FPS':>8} {'파라미터':>10} {'검출수':>8}"
    print(header)
    print("  " + "─" * 70)

    for r in records:
        print(f"  {r['model']:<12} {r['load_ms']:>10.1f} {r['avg_ms']:>10.1f} "
              f"{r['std_ms']:>7.1f} {r['fps']:>8.1f} {r['params_M']:>9.2f}M {r['num_detect']:>8}")

    print("═" * 74)

    fast   = max(records, key=lambda x: x["fps"])
    most   = max(records, key=lambda x: x["num_detect"])
    light  = min(records, key=lambda x: x["params_M"])
    print(f"\n  🚀  최고 속도     : {fast['model']}  ({fast['fps']} FPS)")
    print(f"  🎯  최다 검출     : {most['model']}  ({most['num_detect']}개)")
    print(f"  🪶  최경량        : {light['model']}  ({light['params_M']} M params)")
    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLO v8 / v11 / v26 벤치마크")
    parser.add_argument("--image",  "-i", default=None,
                        help="비교할 이미지 경로 (없으면 ultralytics 샘플 이미지)")
    parser.add_argument("--runs",   "-r", type=int, default=5,
                        help="추론 반복 횟수 (기본 5)")
    parser.add_argument("--models", "-m", default="v8,v11,v26",
                        help="비교할 버전 쉼표 구분 (기본 v8,v11,v26)")
    args = parser.parse_args()

    # 모델 선택
    selected_keys = [k.strip() for k in args.models.split(",")]
    model_list = [ALL_MODELS[k] for k in selected_keys if k in ALL_MODELS]
    if not model_list:
        print(f"[error] 유효한 모델 키 없음. v8 / v11 / v26 중 선택하세요.")
        return

    # 소스 이미지 결정
    if args.image and Path(args.image).exists():
        source = args.image
    else:
        try:
            import ultralytics
            sample = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
            source = str(sample) if sample.exists() else "https://ultralytics.com/images/bus.jpg"
        except Exception:
            source = "https://ultralytics.com/images/bus.jpg"

    print(f"\n{'█' * 54}")
    print(f"  YOLO Benchmark  ·  v8 vs v11 vs v26")
    print(f"{'█' * 54}")
    print(f"  이미지 : {source}")
    print(f"  반복   : {args.runs}회")
    print(f"  모델   : {[m['label'] for m in model_list]}")

    records = []
    for cfg in model_list:
        rec = benchmark(cfg, source, args.runs)
        records.append(rec)

    # JSON 저장
    out_path = OUT_DIR / "benchmark.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\n  💾  결과 JSON → {out_path}")

    print_table(records)
    print(f"  📊  대시보드 → dashboard.html 을 브라우저에서 열고 benchmark.json 로드\n")


if __name__ == "__main__":
    main()
