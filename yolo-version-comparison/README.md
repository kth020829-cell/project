# YOLO v8 vs v11 vs v26 — Industrial Defect Detection Benchmark

> YOLOv8n · YOLOv11n · YOLOv26n을 산업 결함 탐지 도메인(NEU-DET)에서 동일 조건으로 파인튜닝하고,  
> 속도·정확도·결함 유형별 성능을 비교 분석한 개인 프로젝트입니다.

<br>

## 프로젝트 배경

YOLO 버전이 업데이트될수록 항상 더 나은 것인가? 라는 질문에서 출발했습니다.  
단순한 속도 비교를 넘어, **산업 현장에서 실제로 의미 있는 지표**인 결함 유형별 탐지 정확도(per-class AP)까지 분석하여  
"어떤 조건에서 어떤 버전이 유리한가"에 대한 실질적인 인사이트를 도출하는 것이 목표입니다.

<br>

## 데이터셋

**NEU-DET** (Northeastern University Surface Defect Dataset)

| 항목 | 내용 |
|---|---|
| 출처 | Roboflow Universe (원본: 동북대학교 공개 데이터셋) |
| 총 이미지 수 | 1,799장 (train 1,259 / val 360 / test 180) |
| 결함 클래스 수 | 6개 |
| 이미지 크기 | 200×200 px (학습 시 640으로 리사이즈) |
| 증강 여부 | 없음 (원본 그대로 사용, 논문 수치와 비교 가능하도록) |

**결함 클래스**

| 클래스 | 설명 |
|---|---|
| Crazing | 미세 균열 — 표면 전체에 퍼진 실금 |
| Inclusion | 내포물 — 이물질이 박힌 형태 |
| Patches | 패치 — 얼룩처럼 나타나는 결함 |
| Pitted Surface | 피팅 — 표면의 작은 구멍들 |
| Rolled-in Scale | 롤드인 스케일 — 압연 중 이물질이 눌린 형태 |
| Scratches | 스크래치 — 표면 긁힘 |

<br>

## 실험 환경

| 항목 | 내용 |
|---|---|
| GPU | Tesla T4 (Google Colab) |
| Framework | Ultralytics 8.4.46 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Epochs | 50 |
| Image size | 640×640 |
| Batch size | 16 |
| Optimizer | 기본값 (AdamW) |

<br>

## 모델 사양

| 모델 | 가중치 | 파라미터 수 | GFLOPs |
|---|---|---|---|
| YOLOv8n | yolov8n.pt | 3.15M | 8.7 |
| YOLOv11n | yolo11n.pt | 2.62M | 6.5 |
| YOLOv26n | yolo26n.pt | 2.41M | 5.4 |

모두 nano 사이즈로 통일하여 공정한 비교를 구성했습니다.

<br>

## 결과

### 전체 성능 비교

| 모델 | mAP@50 | mAP@50-95 | Precision | Recall | FPS | 추론(ms) |
|---|---|---|---|---|---|---|
| YOLOv8n | 73.5% | 42.8% | 66.3% | **70.9%** | **69.4** | **14.4** |
| **YOLOv11n** | **74.9%** | **43.0%** | **70.7%** | 68.8% | 64.7 | 15.5 |
| YOLOv26n | 71.1% | 42.2% | 69.5% | 67.1% | 58.7 | 17.0 |

### 결함 유형별 AP@50

| 결함 | YOLOv8n | YOLOv11n | YOLOv26n |
|---|---|---|---|
| Crazing | 16.2% | 16.6% | 15.3% |
| Inclusion | 44.5% | **47.2%** | 45.0% |
| Patches | **60.3%** | 59.8% | 57.7% |
| Pitted Surface | 55.2% | 53.7% | **56.1%** ★ |
| Rolled-in Scale | **28.8%** | 28.3% | 23.8% |
| Scratches | 52.0% | 52.4% | **55.4%** ★ |

<br>

## 핵심 인사이트

**1. v11이 전반적 mAP 1위지만, Recall은 v8이 가장 높다**

mAP@50 기준으로는 v11(74.9%)이 1위이지만, Recall은 v8(70.9%)이 가장 높습니다.  
산업 현장에서 결함을 "놓치는 것"이 더 큰 손실이라면, 단순히 mAP만으로 모델을 선택해서는 안 된다는 것을 시사합니다.

**2. v26은 특정 결함 유형에서 역전된다**

전체 mAP는 최하위인 v26이 Pitted Surface와 Scratches에서 단독 1위를 기록합니다.  
버전이 높다고 모든 상황에서 우세한 것이 아니며, 탐지 목표 결함에 따라 모델 선택 전략이 달라져야 합니다.

**3. Crazing(미세 균열)은 nano 모델의 공통 한계**

3모델 모두 Crazing AP가 15~17% 수준으로, 나머지 결함과 비교해 극단적으로 낮습니다.  
미세 균열은 경계가 불명확하고 표면 전체에 분산되는 특성상, nano 모델의 수용 영역(receptive field) 한계가 드러나는 영역입니다.  
Medium 이상 모델이나 Multi-scale feature 강화 기법을 적용하면 유의미한 개선이 가능할 것으로 예상됩니다.

<br>

## 실험의 한계

이 프로젝트의 결과를 해석할 때 다음 한계를 고려해야 합니다.

- **Epochs 50**: 관련 논문들은 100~300 epochs를 사용합니다. 완전히 수렴된 모델 비교가 아닐 수 있어 절대값보다는 상대적 순위에 집중해야 합니다.
- **단일 실험**: Random seed를 고정하지 않고 1회만 학습했습니다. 통계적 유의미성을 확보하려면 3회 이상 반복 실험 후 평균을 내야 합니다.
- **소규모 데이터**: 학습 이미지 1,259장은 실제 산업 환경 대비 매우 작습니다. 결과가 데이터 분할 방식에 민감할 수 있습니다.
- **측정 환경 분리**: mAP는 파인튜닝 세션에서, FPS/추론 시간은 별도 세션에서 측정하여 완전히 동일한 환경이 아닙니다.

<br>

## 실행 방법

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

[Roboflow Universe - NEU-DET](https://universe.roboflow.com/defectdatasets/neu-det-fquva)에서  
YOLOv8 포맷으로 다운로드 후 `data/NEU-DET-1/` 폴더에 위치시킵니다.

```
data/NEU-DET-1/
├── train/images/   (1,259장)
├── train/labels/
├── valid/images/   (360장)
├── valid/labels/
├── test/images/    (180장)
└── test/labels/
```

### 3. 속도 벤치마크 (pretrained 모델 추론 속도)

```bash
python compare.py
# 결과: results/benchmark.json
```

### 4. 파인튜닝 + mAP 측정 (Google Colab 권장)

```python
# Colab에서 실행
!python map_benchmark.py
# 결과: results/map_results.json
```

### 5. 대시보드 확인

`dashboard.html`을 브라우저에서 열고 두 JSON 파일을 로드합니다.

<br>

## 파일 구조

```
yolo-compare/
├── compare.py          # 속도 벤치마크 스크립트
├── map_benchmark.py    # 파인튜닝 + mAP 측정 스크립트
├── dashboard.html      # 결과 시각화 대시보드
├── requirements.txt    # 의존성
├── results/
│   ├── benchmark.json      # 속도 측정 결과
│   └── map_results.json    # mAP 측정 결과
└── README.md
```

<br>

## 향후 개선 방향

- [ ] Epochs 100~200으로 재실험하여 수렴된 모델 비교
- [ ] 3회 반복 실험으로 평균 및 표준편차 제시
- [ ] nano → small → medium 사이즈 확장 비교
- [ ] Crazing 특화 — Multi-scale feature 강화 또는 더 큰 모델 적용
- [ ] GC10-DET 등 다른 산업 결함 데이터셋으로 일반화 검증

<br>

## 참고

- [Ultralytics Docs](https://docs.ultralytics.com)
- [NEU-DET Dataset](https://universe.roboflow.com/defectdatasets/neu-det-fquva)
- YOLO11s 기반 NEU-DET 논문 baseline: mAP@50 74.69%, mAP@50-95 42.66% (GRACE, 2025)
