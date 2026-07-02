# YOLO v8 vs v11 vs v26 — 벤치마크 비교 프로젝트

버전(v8/v11/v26) x 사이즈(n/s) x 배포 포맷(PyTorch/ONNX/TensorRT) 3개 축으로 속도·정확도
트레이드오프를 비교하는 프로젝트입니다.

## 실험 설계

버전(YOLOv8/v11/v26) x 사이즈(n/s) x 배포 포맷(PyTorch/ONNX/TensorRT) 3개 축으로
속도-정확도 트레이드오프를 정량 분석했습니다. 총 6개 모델(v8n, v8s, v11n, v11s, v26n, v26s)을
NEU-DET에 동일 조건(50 epochs, imgsz 640)으로 파인튜닝하고, mAP는 Colab GPU(T4)에서,
추론 속도는 로컬 CPU 환경에서 각각 측정했습니다. 속도와 정확도를 서로 다른 하드웨어에서
측정했기 때문에 FPS 수치를 GPU 배포 시나리오의 성능으로 해석하면 안 되며, "환경이 같다면
버전·사이즈·포맷 간 상대적 차이가 어떻게 나타나는가"를 보는 데 초점을 맞췄습니다.

## Dataset

[NEU-DET](https://universe.roboflow.com/defectdatasets/neu-det-fquva) — 철강 표면 결함 탐지
데이터셋 (1,799 images, 6 defect classes: crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches). Roboflow를 통해 YOLO 포맷으로 다운로드하며, `map_benchmark.py`가
train/valid/test 분할 및 `data.yaml` 생성까지 처리합니다.

## 구조

```
yolo-version-comparison/
├── compare.py          ← 로컬 속도 벤치마크 (6모델 x pt/onnx/engine)
├── map_benchmark.py    ← Colab NEU-DET 파인튜닝 + mAP 벤치마크 (6모델)
├── merge_results.py    ← nano/small 세션에서 나온 map_*.json을 하나로 병합
├── dashboard.html      ← 결과 시각화 대시보드
├── requirements.txt
└── results/            ← 자동 생성
    ├── benchmark.json
    ├── map_results.json
    ├── exports/         ← onnx/engine export 캐시 (.gitignore 대상)
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
각각 50 epoch 파인튜닝 후 mAP를 측정합니다 (약 1~2시간, Colab 무료 GPU 기준). nano/small을
서로 다른 세션에서 나눠 돌린 경우 `merge_results.py`로 `map_nano.json` + `map_small.json`을
`map_results.json`으로 병합할 수 있습니다.

## 대시보드 보기

1. `compare.py` → `results/benchmark.json`, Colab `map_benchmark.py` → `map_results.json` 생성
2. `dashboard.html` 을 브라우저에서 열기
3. 두 JSON 파일을 각각 로드 (**"실제 결과로 보기"** 버튼은 실측치가 아닌 예시/데모 데이터를
   보여주는 용도이니 실제 분석에는 반드시 직접 만든 JSON을 로드해서 확인하세요)

### 대시보드 섹션

| 섹션 | 설명 |
|------|------|
| Speed vs Accuracy Tradeoff (Pareto) | FPS-mAP@50 산점도, 버전별 n→s 연결선으로 사이즈 스케일링 표시 |
| Size Scaling | 사이즈(n/s)에 따른 mAP@50 / FPS 변화 라인 차트 |
| Export Format Speedup | PyTorch/ONNX/TensorRT 포맷별 FPS 비교 (2개 이상 포맷 데이터가 있을 때만 표시) |
| Per-Class AP Heatmap / Radar | 결함 유형별 성능 — N/S 토글로 사이즈 전환 |

## 결과

아래는 이번 실행에서 실측한 값입니다 (`results/map_results.json`, `results/benchmark.json`
pt 포맷 기준). mAP는 Colab GPU, FPS/추론시간은 로컬 CPU에서 측정했습니다.

| 모델 | mAP@50 | mAP@50-95 | Precision | Recall | FPS (CPU, pt) | 추론(ms) | 파라미터 |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 73.5% | 42.8% | 66.3% | 70.9% | 1.7 | 571.5ms | 3.15M |
| YOLOv8s | **75.6%** ★ | **43.6%** ★ | 70.1% | 72.7% | 1.4 | 713.1ms | 11.16M |
| YOLOv11n | 74.6% | 42.2% | 70.4% | 69.5% | **5.1** ★ | **195.8ms** ★ | 2.62M |
| YOLOv11s | 74.6% | 42.0% | 66.2% | **73.8%** ★ | 0.9 | 1118.9ms | 9.44M |
| YOLOv26n | 71.3% | 41.4% | **70.6%** ★ | 67.8% | 3.4 | 292.9ms | **2.41M** ★ |
| YOLOv26s | 73.3% | 42.7% | 69.0% | 70.2% | 2.1 | 480.3ms | 9.50M |

전체 mAP@50 1위는 YOLOv8s(75.6%)지만, Precision은 YOLOv26n(70.6%), Recall은 YOLOv11s(73.8%),
CPU 추론 속도는 YOLOv11n(5.1 FPS)이 각각 앞서 지표별로 최적 모델이 갈립니다.

## 모델 사이즈에 따른 성능 변화 (n → s)

| 버전 | mAP@50 변화 | FPS 변화 (CPU, pt) | 파라미터 변화 |
|---|---|---|---|
| YOLOv8  | 73.5% → 75.6% (+2.1%p) | 1.7 → 1.4 (-17.6%) | 3.15M → 11.16M |
| YOLOv11 | 74.6% → 74.6% (-0.1%p, 사실상 변화 없음) | 5.1 → 0.9 (-82.4%) | 2.62M → 9.44M |
| YOLOv26 | 71.3% → 73.3% (+2.0%p) | 3.4 → 2.1 (-38.2%) | 2.41M → 9.50M |

v8·v26은 사이즈를 키우면 mAP가 2%p 안팎 오르고 속도는 완만하게(-18~38%) 떨어지는 전형적인
트레이드오프를 보였지만, v11은 정확도 향상이 거의 없는 채로(-0.1%p) 속도만 82% 급락해
이번 실험에서는 사이즈 확장의 비용 대비 효과가 가장 낮았습니다. "사이즈를 키우면 일정 폭의
정확도가 오른다"는 가설은 v8·v26에서는 대체로 성립했지만 v11에서는 반례가 나타났습니다 —
사이즈 확장의 효과가 모든 버전(아키텍처)에 균일하게 재현되지는 않는다는 뜻입니다.

## 배포 포맷별 속도 비교

| 모델 | PyTorch (ms) | ONNX (ms) | 속도 비율 (pt/onnx) |
|---|---|---|---|
| YOLOv8n | 571.5 | 291.4 | 1.96x |
| YOLOv8s | 713.1 | 412.7 | 1.73x |
| YOLOv11n | 195.8 | 231.5 | 0.85x (오히려 느려짐) |
| YOLOv11s | 1118.9 | 484.9 | 2.31x |
| YOLOv26n | 292.9 | 192.8 | 1.52x |
| YOLOv26s | 480.3 | 294.3 | 1.63x |

로컬 CPU 환경 기준, ONNX export는 6개 모델 중 5개에서 1.5~2.3배 속도 향상을 보였습니다.
유일한 예외는 YOLOv11n으로, ONNX가 PyTorch eager 모드보다 오히려 15% 느렸고 표준편차(89.1ms)도
다른 모델 대비 커서 측정 노이즈일 가능성이 있습니다 — 반복 측정으로 재확인이 필요합니다.

**한계**: TensorRT(engine) 포맷은 이번 실행 환경(로컬, CUDA/TensorRT 미설치)에서 export 자체가
전부 스킵되어 6개 모델 어디에서도 실측 데이터를 얻지 못했습니다. n/s 사이즈 문제가 아니라
환경 자체의 제약이며, NVIDIA GPU + TensorRT가 설치된 환경(Colab GPU 등)에서 재실행이 필요합니다.

## 핵심 인사이트

1. **nano 기준 정확도 1위는 v11n, 그러나 Recall은 v8n이 가장 높다** — mAP@50 기준
   v11n(74.6%) > v8n(73.5%) > v26n(71.3%)이지만, Recall은 v8n(70.9%)이 가장 높습니다.
   결함을 "놓치지 않는 것"이 중요한 현장에서는 v8n이 더 유리할 수 있습니다.
2. **v26n은 Scratches에서만 단독 1위** — Pitted Surface는 v11n(56.2%)이 가장 높고,
   Scratches는 v26n(54.8%)이 가장 높습니다. 전반적 mAP는 낮지만 특정 결함 탐지가 목적이면
   v26n이 선택지가 될 수 있습니다.
3. **Crazing은 모든 nano 모델에서 가장 낮은 AP를 보였다** — 3개 nano 모델 모두 AP 15~16%
   수준으로 6개 결함 클래스 중 최하위입니다. 미세 균열은 nano 모델 공통의 한계로 보입니다.
4. **사이즈 확장(n→s) 효과는 버전마다 다르다** — v8/v26은 mAP가 2%p 안팎 오르지만, v11은
   사실상 변화가 없으면서(-0.1%p) FPS만 82% 급락합니다. "크기를 키우면 항상 좋아진다"는
   가정이 모든 아키텍처에 균일하게 적용되지는 않습니다.
5. **결함 클래스별 개선 여부는 일관되지 않는다** — Crazing은 v8·v26에서는 소폭 개선
   (+7.8~13.1%)됐지만 v11에서는 오히려 하락(-6.5%)했습니다. 같은 v11에서 Pitted Surface는
   더 큰 폭(-11.0%)으로 떨어져, Crazing만의 특수한 문제라기보다 단일 실행 결과의 클래스별
   변동성으로 보는 것이 더 정확합니다.
6. **버전이 높다고 항상 좋은 것은 아니다** — 전체 mAP는 v8s(75.6%)가 가장 높고, Precision은
   v26n(70.6%), Recall은 v11s(73.8%)가 각각 1위입니다. 지표와 목적에 따라 최적 모델이
   달라집니다.

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

## 실험의 한계

- **TensorRT export 미완성**: 로컬 실행 환경에 CUDA/TensorRT가 없어 6개 모델 전부 engine
  export가 스킵됐습니다. n/s 사이즈 구분과 무관하게 TensorRT 비교 자체를 하지 못한 것이 한계입니다.
- **속도 벤치마크가 GPU가 아닌 로컬 CPU에서 수행됨**: FPS 0.9~5.1 수준으로, 실제 GPU 배포
  환경의 속도차와는 다를 수 있습니다.
- **단일 실행(seed 1개)**: 반복 학습·측정 없이 1회씩만 수행해 mAP·per-class AP의 1~2%p
  차이는 측정 노이즈일 가능성이 있습니다.
- **병렬 실험 환경 차이**: nano/small 모델 그룹을 서로 다른 Colab 세션(계정)에서 병렬로
  학습시켜, 완전히 동일한 하드웨어 상태였다고 보장할 수 없습니다.

## 향후 개선 방향

- GPU 환경에서 속도 벤치마크 재측정 (`compare.py`를 CUDA 환경에서 실행)
- CUDA/TensorRT 환경을 확보해 engine export 실측 비교 완성
- 동일 설정으로 2~3회 반복 학습(다른 seed)해 클래스별 AP 변동폭을 신뢰구간으로 제시
- m/l 사이즈까지 확장해 Pareto 곡선을 더 촘촘히 채우기
- Crazing처럼 경계가 불명확한 클래스에 대해 세그멘테이션 기반 접근과 비교

## 결론

1. **버전이 높다고 항상 좋은 것은 아니다** — 전체 mAP는 v8s가 가장 높고, Precision·Recall
   1위는 각각 v26n, v11s로 지표마다 다른 모델이 앞섭니다.
2. **사이즈 확장의 효과는 버전마다 다르다** — v8·v26은 n→s에서 mAP가 소폭 오르지만, v11은
   정확도 이득 없이 속도만 크게 잃습니다.
3. **클래스별 개선 여부는 노이즈가 커서 단정하기 어렵다** — Crazing을 포함한 일부 클래스는
   버전에 따라 사이즈 확장의 효과가 갈리며, 단일 실행 결과라 반복 검증이 필요합니다.
4. **배포 포맷 최적화는 실측 가능한 범위(ONNX)에서 확인됐다** — ONNX export가 6개 모델 중
   5개에서 1.5~2.3배 속도 향상을 보였지만, TensorRT는 환경 제약으로 검증하지 못해 추후
   과제로 남습니다.

## 참고

- 데이터셋: [NEU-DET (Roboflow, defectdatasets/neu-det-fquva)](https://universe.roboflow.com/defectdatasets/neu-det-fquva)
- 실측 결과 원본: `results/map_results.json`, `results/benchmark.json`
