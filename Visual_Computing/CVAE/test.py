import torch
import numpy as np
from model import CVAE, CVAE_CNN
from dataset import test_loader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

# 모델 선택 (train.py와 동일하게 설정)
USE_CNN = True

# 학습된 모델 체크포인트 불러오기 
if USE_CNN:
    model = CVAE_CNN.load_from_checkpoint("./cvae.ckpt")
else:
    model = CVAE.load_from_checkpoint("./cvae.ckpt")
model.to(torch.device("cpu"))

print(model.device)

label_mean    = torch.zeros(size=(10,2), dtype=torch.float32)
label_log_var = torch.zeros(size=(10,2), dtype=torch.float32)
label_bool    = np.zeros(shape=(10), dtype=bool)

for step, (x, c) in enumerate(test_loader):
    mean, log_var = model.encoder(x, c)

    for idx, cc in enumerate(c):
        label = int(cc)                        # 텐서를 정수로 변환 (ex: tensor(3) -> 3)
        label_mean[label] = mean[idx]        
        label_log_var[label] = log_var[idx]   
        label_bool[label] = True             

    if (np.all(label_bool)):
        break

label_std = label_log_var.exp().pow(0.5)
print("label_mean", label_mean)
print("cls_std", label)