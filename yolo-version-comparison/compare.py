"""
YOLO Version Benchmark: v8 vs v11 vs v26 (n/s 사이즈 x pt/onnx/engine 포맷)
=========================================
사용법:
  python compare.py                              # 샘플 이미지, 전체 6모델 x pt+onnx (engine은 가능하면 포함)
  python compare.py --image path/to/img.jpg      # 직접 이미지 지정
  python compare.py --runs 10                    # 반복 횟수 조정
  python compare.py --models v8n,v26n             # 특정 모델만 비교
  python compare.py --formats pt,onnx             # TensorRT 제외하고 비교

TensorRT(engine) export는 CUDA + TensorRT가 설치된 환경(예: Colab GPU)에서만 동작합니다.
미설치 환경에서는 자동으로 스킵되고 pt/onnx 결과만 남습니다.
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


# ── 모델 정의 (버전 3 x 사이즈 2 = 6개) ────────────────────────────────────────
ALL_MODELS = {
    "v8n":  {"label": "YOLOv8n",  "weight": "yolov8n.pt"},
    "v8s":  {"label": "YOLOv8s",  "weight": "yolov8s.pt"},
    "v11n": {"label": "YOLOv11n", "weight": "yolo11n.pt"},
    "v11s": {"label": "YOLOv11s", "weight": "yolo11s.pt"},
    "v26n": {"label": "YOLOv26n", "weight": "yolo26n.pt"},
    "v26s": {"label": "YOLOv26s", "weight": "yolo26s.pt"},
}

ALL_FORMATS = ["pt", "onnx", "engine"]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
EXPORT_DIR = OUT_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)


# ── 포맷별 export (결과 캐시) ─────────────────────────────────────────────────
def export_model(weight: str, label: str, fmt: str):
    """주어진 포맷의 weight 경로를 반환. pt는 원본 그대로, onnx/engine은 export 후 캐시."""
    if fmt == "pt":
        return weight

    ext = "onnx" if fmt == "onnx" else "engine"
    fmt_dir = EXPORT_DIR / fmt
    fmt_dir.mkdir(exist_ok=True)
    cached = fmt_dir / f"{label}.{ext}"
    if cached.exists():
        return str(cached)

    try:
        model = YOLO(weight)
        exported = model.export(format=fmt, imgsz=640, half=(fmt == "engine"))
        exported_path = Path(exported)
        exported_path.rename(cached) if exported_path != cached else None
        return str(cached) if cached.exists() else str(exported_path)
    except Exception as e:
        print(f"  [skip] {label} → {fmt} export 불가 ({e.__class__.__name__}: {e})")
        return None


# ── 단일 모델 벤치마크 ────────────────────────────────────────────────────────
def benchmark(cfg: dict, model_path: str, fmt: str, source, runs: int, params_hint: float = None) -> dict:
    label = cfg["label"]

    print(f"\n{'─'*54}")
    print(f"  [{label}]  format={fmt}  ({model_path})")
    print(f"{'─'*54}")

    # 1. 모델 로딩 시간 측정
    t0 = time.perf_counter()
    model = YOLO(model_path)
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

    # 5. 파라미터 수 (M단위) — pt만 직접 계산, onnx/engine은 pt 결과값 재사용
    try:
        params_M = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    except Exception:
        params_M = params_hint

    # 6. 어노테이션 이미지 저장
    ann_path = str(OUT_DIR / f"{label}_{fmt}_annotated.jpg")
    result.save(filename=ann_path)

    record = {
        "model":        label,
        "format":       fmt,
        "weight":       model_path,
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
    print("\n" + "═" * 82)
    print("  최종 비교 결과 (모델 x 포맷)")
    print("═" * 82)
    header = f"  {'모델':<12} {'포맷':>8} {'로딩(ms)':>10} {'추론(ms)':>10} {'±std':>7} {'FPS':>8} {'파라미터':>10} {'검출수':>8}"
    print(header)
    print("  " + "─" * 78)

    for r in records:
        params = f"{r['params_M']:.2f}M" if r["params_M"] is not None else "-"
        print(f"  {r['model']:<12} {r['format']:>8} {r['load_ms']:>10.1f} {r['avg_ms']:>10.1f} "
              f"{r['std_ms']:>7.1f} {r['fps']:>8.1f} {params:>10} {r['num_detect']:>8}")

    print("═" * 82)

    fast   = max(records, key=lambda x: x["fps"])
    most   = max(records, key=lambda x: x["num_detect"])
    withp  = [r for r in records if r["params_M"] is not None]
    print(f"\n  🚀  최고 속도     : {fast['model']} ({fast['format']})  ({fast['fps']} FPS)")
    print(f"  🎯  최다 검출     : {most['model']} ({most['format']})  ({most['num_detect']}개)")
    if withp:
        light = min(withp, key=lambda x: x["params_M"])
        print(f"  🪶  최경량        : {light['model']}  ({light['params_M']} M params)")
    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLO v8 / v11 / v26 (n/s) 벤치마크")
    parser.add_argument("--image",  "-i", default=None,
                        help="비교할 이미지 경로 (없으면 ultralytics 샘플 이미지)")
    parser.add_argument("--runs",   "-r", type=int, default=5,
                        help="추론 반복 횟수 (기본 5)")
    parser.add_argument("--models", "-m", default="v8n,v8s,v11n,v11s,v26n,v26s",
                        help="비교할 모델 쉼표 구분 (기본 6개 전체: v8n,v8s,v11n,v11s,v26n,v26s)")
    parser.add_argument("--formats", "-f", default="pt,onnx,engine",
                        help="비교할 포맷 쉼표 구분 (기본 pt,onnx,engine · engine=TensorRT는 미설치 시 자동 스킵)")
    args = parser.parse_args()

    # 모델 선택
    selected_keys = [k.strip() for k in args.models.split(",")]
    model_list = [ALL_MODELS[k] for k in selected_keys if k in ALL_MODELS]
    if not model_list:
        print(f"[error] 유효한 모델 키 없음. {', '.join(ALL_MODELS)} 중 선택하세요.")
        return

    # 포맷 선택
    selected_formats = [f.strip() for f in args.formats.split(",") if f.strip() in ALL_FORMATS]
    if not selected_formats:
        print(f"[error] 유효한 포맷 없음. {', '.join(ALL_FORMATS)} 중 선택하세요.")
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
    print(f"  YOLO Benchmark  ·  v8 vs v11 vs v26 (n/s)")
    print(f"{'█' * 54}")
    print(f"  이미지 : {source}")
    print(f"  반복   : {args.runs}회")
    print(f"  모델   : {[m['label'] for m in model_list]}")
    print(f"  포맷   : {selected_formats}")

    records = []
    for cfg in model_list:
        pt_params_hint = None
        for fmt in selected_formats:
            model_path = export_model(cfg["weight"], cfg["label"], fmt)
            if model_path is None:
                continue
            rec = benchmark(cfg, model_path, fmt, source, args.runs, params_hint=pt_params_hint)
            if fmt == "pt":
                pt_params_hint = rec["params_M"]
            records.append(rec)

    # JSON 저장
    out_path = OUT_DIR / "benchmark.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\n  💾  결과 JSON → {out_path}")

    print_table(records)
    print(f"  📊  대시보드 → dashboard.html 을 브라우저에서 열고 benchmark.json 로드\n")


if __name__ == "__main__":
    main()
