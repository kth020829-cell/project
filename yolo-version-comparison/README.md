# YOLO v8 vs v11 vs v26 — 벤치마크 비교 프로젝트

버전(v8/v11/v26) x 사이즈(n/s) x 배포 포맷(PyTorch/ONNX/TensorRT) 3개 축으로 속도·정확도
트레이드오프를 비교하는 프로젝트입니다.

## 구조

```
yolo-version-comparison/
├── compare.py          ← 로컬 속도 벤치마크 (6모델 x pt/onnx/engine)
├── map_benchmark.py    ← Colab NEU-DET 파인튜닝 + mAP 벤치마크 (6모델)
├── dashboard.html      ← 결과 시각화 대시보드
├── requirements.txt
└── results/            ← 자동 생성
    ├── benchmark.json
    ├── map_results.json
    ├── exports/         ← onnx/engine export 캐시
    │   ├── onnx/*.onnx
    │   └── engine/*.engine
    └── *_annotated.jpg
```

## 설치

```bash
pip install -r requirements.txt
```

TensorRT(`engine` 포맷)는 NVIDIA GPU + CUDA + TensorRT가 설치된 환경(Colab GPU 등)에서만
동작합니다. 미설치 환경에서는 `compare.py`가 자동으로 건너뜁니다.

## 실행 방법 — 속도 벤치마크 (compare.py)

```bash
# 기본 실행: 6모델(v8n,v8s,v11n,v11s,v26n,v26s) x pt+onnx+engine
python compare.py

# 직접 이미지 지정
python compare.py --image path/to/your/image.jpg

# 반복 횟수 조정 (더 정확한 평균)
python compare.py --runs 10

# 특정 모델만 비교
python compare.py --models v8n,v26n

# TensorRT 없이 pt/onnx만 비교 (로컬 환경 권장)
python compare.py --formats pt,onnx
```

`onnx`/`engine` export는 `results/exports/`에 캐시되어 이후 실행부터는 재사용됩니다.
TensorRT 최초 export는 엔진 컴파일 때문에 모델당 수 분이 추가로 걸릴 수 있습니다.

## 실행 방법 — 정확도 벤치마크 (map_benchmark.py, Colab)

Google Colab에서 GPU 런타임으로 셀을 순서대로 실행하세요. NEU-DET 데이터셋으로 6개 모델을
각각 50 epoch 파인튜닝 후 mAP를 측정합니다 (약 1~2시간, Colab 무료 GPU 기준).

## 대시보드 보기

1. `compare.py` → `results/benchmark.json`, Colab `map_benchmark.py` → `map_results.json` 생성
2. `dashboard.html` 을 브라우저에서 열기
3. 두 JSON 파일을 각각 로드 (또는 **"실제 결과로 보기"** 로 예시 데이터 확인)

### 대시보드 섹션

| 섹션 | 설명 |
|------|------|
| Speed vs Accuracy Tradeoff (Pareto) | FPS-mAP@50 산점도, 버전별 n→s 연결선으로 사이즈 스케일링 표시 |
| Size Scaling | 사이즈(n/s)에 따른 mAP@50 / FPS 변화 라인 차트 |
| Export Format Speedup | PyTorch/ONNX/TensorRT 포맷별 FPS 비교 (2개 이상 포맷 데이터가 있을 때만 표시) |
| Per-Class AP Heatmap / Radar | 결함 유형별 성능 — N/S 토글로 사이즈 전환 |

## 비교 지표

| 지표 | 설명 |
|------|------|
| 로딩 시간 (ms) | 모델 파일 로드 시간 |
| 추론 시간 (ms) | 이미지 1장 처리 평균 시간 |
| FPS | 초당 처리 프레임 수 |
| 파라미터 수 (M) | 모델 크기 (낮을수록 가벼움, onnx/engine은 pt 값 재사용) |
| 검출 수 | 이미지에서 검출한 객체 수 |
| 표준편차 | 추론 안정성 (낮을수록 안정적) |
| mAP@50 / mAP@50-95 | NEU-DET 파인튜닝 후 검출 정확도 |

## 사용 모델

| 모델 | 가중치 | 사이즈 |
|------|--------|------|
| YOLOv8n  | yolov8n.pt  | nano |
| YOLOv8s  | yolov8s.pt  | small |
| YOLOv11n | yolo11n.pt  | nano |
| YOLOv11s | yolo11s.pt  | small |
| YOLOv26n | yolo26n.pt  | nano |
| YOLOv26s | yolo26s.pt  | small |

모델 가중치는 최초 실행 시 자동 다운로드됩니다.
