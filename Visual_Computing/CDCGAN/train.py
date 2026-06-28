from dataset import train_loader
from model import CDCGAN

import lightning as L

num_classes = 10  # MNIST 클래스 수 (0~9)
nz = 5            # 노이즈 벡터 차원 / Gradio 슬라이더 5개와 대응
model = CDCGAN(num_classes=num_classes, nz=nz)

trainer = L.Trainer(max_epochs=3, accelerator="auto")
trainer.fit(model, train_dataloaders=train_loader)
trainer.save_checkpoint("./cdcgan.ckpt")

print(model.device)
