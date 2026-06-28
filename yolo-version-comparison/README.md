# YOLO v8 vs v11 vs v26 — 벤치마크 비교 프로젝트

## 구조

```
yolo_compare/
├── compare.py        ← 벤치마크 실행 스크립트
├── dashboard.html    ← 결과 시각화 대시보드
├── requirements.txt
└── results/          ← 자동 생성
    ├── benchmark.json
    ├── YOLOv8n_annotated.jpg
    ├── YOLOv11n_annotated.jpg
    └── YOLOv26n_annotated.jpg
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
# 기본 실행 (ultralytics 샘플 이미지 bus.jpg 사용)
python compare.py

# 직접 이미지 지정
python compare.py --image path/to/your/image.jpg

# 반복 횟수 조정 (더 정확한 평균)
python compare.py --runs 10

# 특정 버전만 비교
python compare.py --models v8,v26
python compare.py --models v8,v11
```

## 대시보드 보기

1. `python compare.py` 실행 → `results/benchmark.json` 생성
2. `dashboard.html` 을 브라우저에서 열기
3. **"benchmark.json 로드"** 버튼 클릭 → 결과 파일 선택
4. 또는 **"데모 데이터 보기"** 로 미리 확인

## 비교 지표

| 지표 | 설명 |
|------|------|
| 로딩 시간 (ms) | 모델 파일 로드 시간 |
| 추론 시간 (ms) | 이미지 1장 처리 평균 시간 |
| FPS | 초당 처리 프레임 수 |
| 파라미터 수 (M) | 모델 크기 (낮을수록 가벼움) |
| 검출 수 | 이미지에서 검출한 객체 수 |
| 표준편차 | 추론 안정성 (낮을수록 안정적) |

## 사용 모델

| 모델 | 가중치 | 설명 |
|------|--------|------|
| YOLOv8n | yolov8n.pt | Ultralytics YOLOv8 nano |
| YOLOv11n | yolo11n.pt | Ultralytics YOLOv11 nano |
| YOLOv26n | yolo26n.pt | Ultralytics YOLOv26 nano |

모델 가중치는 최초 실행 시 자동 다운로드됩니다.
