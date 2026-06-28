# 트윗 감정분석 - NLP 모델 비교 실험

Kaggle Sentiment Analysis 데이터셋(Tweet Sentiment Extraction 형식)을 사용하여
RNN, LSTM, BiLSTM, BERT, GPT-2 다섯 가지 모델의 감정 분류 성능을 비교한 과제.

---

## 빠른 시작 (Google Colab + Drive)

### 1. Drive에 폴더 만들기

```
MyDrive/sentiment_analysis/
├── utils.py
├── 01_EDA_and_Preprocessing.ipynb
├── 02_RNN_LSTM.ipynb
├── 03_BERT.ipynb
├── 04_GPT.ipynb
├── 05_Comparison.ipynb
└── data/
    ├── train.csv          ← Kaggle에서 받은 원본
    └── test.csv
```

### 2. Kaggle 데이터 받기

[Kaggle 데이터셋 페이지](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)에서 다운로드 후
`train.csv`와 `test.csv`만 `data/` 폴더에 넣는다.

### 3. Colab에서 노트북 열기

### 4. 실행 순서

| 순서 | 노트북 | 환경 | 예상 시간 |
|------|--------|------|----------|
| 1 | `01_EDA_and_Preprocessing.ipynb` | CPU 가능 | 1~2분 |
| 2 | `02_RNN_LSTM.ipynb` | CPU 가능, GPU 권장 | 10~30분 |
| 3 | `03_BERT.ipynb` | **GPU 필수** | 15~25분 |
| 4 | `04_GPT.ipynb` | **GPU 필수** | 20~30분 |
| 5 | `05_Comparison.ipynb` | CPU | 1분 |


---

## 파일별 역할

| 파일 | 내용 |
|------|------|
| `utils.py` | 5개 노트북이 공통으로 사용하는 함수 (전처리, 평가, 시각화) |
| `01_EDA_and_Preprocessing.ipynb` | 결측치 처리, 클래스 분포, 텍스트 길이, 단어 빈도 등 EDA + 전처리 후 클린 CSV 저장 |
| `02_RNN_LSTM.ipynb` | SimpleRNN, LSTM, BiLSTM 학습 및 비교 |
| `03_BERT.ipynb` | DistilBERT 파인튜닝 |
| `04_GPT.ipynb` | GPT-2에 분류 헤드를 부착한 시퀀스 분류 |
| `05_Comparison.ipynb` | 1~4번 결과를 모아 정량·정성 비교, 보고서용 시각화 생성 |

---

## 데이터셋 정보

- 출처: Kaggle "Sentiment Analysis Dataset" (abhi8923shriv)
- 규모: Train 27,481 / Test 4,815 (결측치 제거 후 27,477 / 3,534)
- 클래스: negative (28%), neutral (40%), positive (31%)
- 언어: 영어 트윗

---

## 설계 메모

### 전처리 두 버전을 사용하는 이유

`utils.py`에는 전처리 함수가 두 개 있다.

- `preprocess_classic`: 소문자화, URL/멘션 제거, 구두점 정리 — **RNN/LSTM용**
- `preprocess_transformer`: URL/멘션을 `[URL]`/`[USER]` 토큰으로 치환만 — **BERT/GPT용**

BERT와 GPT는 WordPiece/BPE 토크나이저가 대소문자와 구두점을 활용하므로
원본을 강하게 정제하면 오히려 성능이 떨어진다.

### 평가 지표

- Accuracy
- F1 macro (클래스 불균형 고려)
- Precision / Recall (클래스별)
- Confusion Matrix (오분류 패턴 확인)
