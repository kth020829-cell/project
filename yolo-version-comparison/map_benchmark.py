# -*- coding: utf-8 -*-
"""
YOLO v8 vs v11 vs v26 — NEU-DET Fine-tuning & mAP Benchmark
=============================================================
Google Colab에서 순서대로 실행하세요.
Runtime > Change runtime type > GPU 설정 필수
"""

# ────────────────────────────────────────────────────────────
# CELL 1 | 패키지 설치
# ────────────────────────────────────────────────────────────
!pip install ultralytics roboflow -q


# ────────────────────────────────────────────────────────────
# CELL 2 | NEU-DET 데이터셋 다운로드
# ────────────────────────────────────────────────────────────
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")   # ← Roboflow 로그인 후 본인 키로 교체
project = rf.workspace("defectdatasets").project("neu-det-fquva")
version = project.version(1)
dataset = version.download("yolov8")


# ────────────────────────────────────────────────────────────
# CELL 3 | 데이터 확인
# ────────────────────────────────────────────────────────────
import os

BASE = "/content/NEU-DET-1"

for split in ["train", "valid", "test"]:
    imgs = len(os.listdir(f"{BASE}/{split}/images"))
    lbls = len(os.listdir(f"{BASE}/{split}/labels"))
    print(f"{split:6s} | images: {imgs:4d}장  labels: {lbls:4d}개")


# ────────────────────────────────────────────────────────────
# CELL 4 | YAML 경로 수정 (Roboflow 상대경로 → 절대경로)
# ────────────────────────────────────────────────────────────
YAML_CONTENT = """
path: /content/NEU-DET-1
train: train/images
val:   valid/images
test:  test/images

nc: 6
names:
  - crazing
  - inclusion
  - patches
  - pitted_surface
  - rolled-in_scale
  - scratches
"""

YAML_PATH = "/content/neu_det_fixed.yaml"
with open(YAML_PATH, "w") as f:
    f.write(YAML_CONTENT)

print("✅ yaml 저장 완료")
print(open(YAML_PATH).read())


# ────────────────────────────────────────────────────────────
# CELL 5 | 파인튜닝 + mAP 측정 (메인 실험 / 약 1시간 소요)
# ────────────────────────────────────────────────────────────
from ultralytics import YOLO
import json

MODELS = {
    "YOLOv8n":  "yolov8n.pt",
    "YOLOv11n": "yolo11n.pt",
    "YOLOv26n": "yolo26n.pt",
}

CLASS_NAMES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]

all_results = {}

for name, weight in MODELS.items():
    print(f"\n{'='*50}")
    print(f"  [{name}]  학습 시작")
    print(f"{'='*50}")

    model = YOLO(weight)

    # 파인튜닝
    model.train(
        data=YAML_PATH,
        epochs=50,          # 빠른 실험용 (정식 비교는 100~200 권장)
        imgsz=640,
        batch=16,
        device=0,
        verbose=False,
        project="/content/runs",
        name=name,
        exist_ok=True,      # 재실행 시 덮어쓰기 허용
    )

    # best.pt로 검증
    best_pt = f"/content/runs/{name}/weights/best.pt"
    trained = YOLO(best_pt)
    metrics = trained.val(data=YAML_PATH, imgsz=640, device=0, verbose=False)

    # 클래스별 AP 추출
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        try:
            per_class[cls] = round(float(metrics.box.ap[i]), 4)
        except Exception:
            per_class[cls] = 0.0

    all_results[name] = {
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50_95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
        "per_class": per_class,
    }

    print(f"\n  mAP@50    : {all_results[name]['mAP50']}")
    print(f"  mAP@50-95 : {all_results[name]['mAP50_95']}")
    print(f"  Per-class : {per_class}")

# 결과 저장
with open("/content/map_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n\n✅ 전체 완료 → map_results.json 저장됨")


# ────────────────────────────────────────────────────────────
# CELL 6 | 결과 요약 출력
# ────────────────────────────────────────────────────────────
import json

with open("/content/map_results.json") as f:
    results = json.load(f)

print(f"\n{'─'*60}")
print(f"  {'모델':<12} {'mAP@50':>8} {'mAP@50-95':>10} {'Precision':>10} {'Recall':>8}")
print(f"{'─'*60}")
for model, r in results.items():
    print(f"  {model:<12} {r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
          f"{r['precision']:>10.4f} {r['recall']:>8.4f}")
print(f"{'─'*60}")


# ────────────────────────────────────────────────────────────
# CELL 7 | 결과 파일 다운로드
# ────────────────────────────────────────────────────────────
from google.colab import files
files.download("/content/map_results.json")
print("✅ map_results.json 다운로드 완료")
