import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# 1. 전처리 함수
URL_RE      = re.compile(r'http\S+|www\.\S+')
MENTION_RE  = re.compile(r'@\w+')
HTML_RE     = re.compile(r'&\w+;')
REPEAT_RE   = re.compile(r'(.)\1{2,}')
MULTI_SPACE = re.compile(r'\s+')


def preprocess_classic(text):
    """RNN/LSTM용 - 강한 정제"""
    t = str(text)
    t = URL_RE.sub(' ', t)
    t = MENTION_RE.sub(' ', t)
    t = HTML_RE.sub(' ', t)
    t = REPEAT_RE.sub(r'\1\1', t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s!?'.,]", ' ', t)
    return MULTI_SPACE.sub(' ', t).strip()


def preprocess_transformer(text):
    """BERT/GPT용 - 최소 정제"""
    t = str(text)
    t = URL_RE.sub('[URL]', t)
    t = MENTION_RE.sub('[USER]', t)
    t = HTML_RE.sub(' ', t)
    t = REPEAT_RE.sub(r'\1\1', t)
    return MULTI_SPACE.sub(' ', t).strip()



# 2. 라벨 매핑
LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL  = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ['negative', 'neutral', 'positive']



# 3. 데이터 로드
def load_data(train_path='train.csv', test_path='test.csv',
              text_col='text_classic'):
    """전처리된 CSV에서 학습/평가용 X, y 반환

    text_col: 'text_classic' (RNN/LSTM) 또는 'text_transformer' (BERT/GPT)
    """
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    X_train = train[text_col].astype(str).values
    y_train = train['label'].values
    X_test  = test[text_col].astype(str).values
    y_test  = test['label'].values
    return X_train, y_train, X_test, y_test



# 4. 평가 함수
def evaluate_model(y_true, y_pred, model_name='Model', verbose=True):
    """모델 평가 결과를 딕셔너리로 반환"""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average='macro', zero_division=0)

    result = {
        'model': model_name,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"📊 {model_name} 평가 결과")
        print(f"{'=' * 50}")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  F1 (macro)      : {f1_macro:.4f}")
        print(f"  F1 (weighted)   : {f1_weighted:.4f}")
        print(f"  Precision macro : {prec_macro:.4f}")
        print(f"  Recall macro    : {rec_macro:.4f}")
        print(f"\n--- Classification Report ---")
        print(classification_report(y_true, y_pred,
                                    target_names=LABEL_NAMES, digits=4))

    return result


def plot_confusion_matrix(y_true, y_pred, model_name='Model', ax=None,
                          cmap='Blues', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    annot = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
                       for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, ax=ax, cbar=False,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                annot_kws={'fontsize': 11})
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return ax



# 5. 오분류 사례 분석
def show_misclassified(X, y_true, y_pred, n=5, seed=42):
    """오분류된 샘플 n개 출력"""
    wrong_idx = np.where(y_pred != y_true)[0]
    if len(wrong_idx) == 0:
        print('완벽 분류!')
        return
    rng = np.random.default_rng(seed)
    sample = rng.choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)
    print(f"\n 오분류 사례 ({n}개, 전체 {len(wrong_idx)}개 중)")
    print('-' * 60)
    for i in sample:
        print(f"[True: {ID2LABEL[y_true[i]]:>8} | Pred: {ID2LABEL[y_pred[i]]:>8}]")
        print(f"  → {X[i][:140]}")
        print()
