import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from dataset import train_loader
from model import CVAE, CVAE_CNN

import lightning as L

# 모델 선택
# USE_CNN = False -> MLP 기반 CVAE 사용
# USE_CNN = True  -> CNN 기반 CVAE 사용
USE_CNN = True

if USE_CNN:
    model = CVAE_CNN(n_dim=2, lr=1e-3)   # CNN 기반 CVAE 모델 생성
else:
    model = CVAE(n_dim=2, lr=1e-3)       # MLP 기반 CVAE 모델 생성

# 트레이너 설정: 20 에폭 학습, GPU/CPU 자동 선택
trainer = L.Trainer(max_epochs=20, accelerator="auto")
trainer.fit(model, train_dataloaders=train_loader)
trainer.save_checkpoint("./cvae.ckpt")

print("학습 완료! 모델이 cvae.ckpt에 저장되었습니다.")
print("사용 디바이스:", model.device)
