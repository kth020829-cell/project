import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
import einops
from einops.layers.torch import Rearrange


class CGenerator(nn.Module):
    def __init__(self, num_classes=10, nz=100):
        # nz 기본값은 100이지만, train.py에서 nz=5로 설정하여 사용
        super().__init__()

        self.nz = nz
        self.num_classes = num_classes

        # 클래스 레이블을 임베딩 벡터로 변환하는 임베딩 레이어
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # ConvNet Output Size Calculator https://asiltureli.github.io/Convolution-Layer-Calculator/
        # 노이즈 z와 레이블 임베딩을 결합한 벡터를 입력으로 받아 이미지를 생성하는 네트워크
        self.main = nn.Sequential(
            # 입력: nz + num_classes 크기의 벡터
            nn.Linear(self.nz + num_classes, 256 * 7 * 7, bias=False),   
            nn.BatchNorm1d(num_features=256 * 7 * 7),
            nn.LeakyReLU(),
            Rearrange('b (c h w) -> b c h w', c=256, h=7, w=7),                 
            # 전치 합성곱으로 특징 맵 크기를 키워나감
            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=5, stride=1, padding=2, bias=False),  
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1,
                               kernel_size=4, stride=2, padding=1, bias=False),   
            # 출력값을 -1 ~ 1 범위로 정규화
            nn.Tanh(),
        )

    def forward(self, z, c):
        c_emb = self.label_emb(c)
        gen_input = torch.cat([z, c_emb], dim=1)
        return self.main(gen_input)


class CDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes = num_classes

        # 클래스 레이블을 28x28 크기의 채널로 변환하는 임베딩 레이어
        self.label_emb = nn.Sequential(
            nn.Embedding(num_classes, 28 * 28),
            Rearrange('b (h w) -> b 1 h w', h=28, w=28),
        )

        # ConvNet Output Size Calculator https://asiltureli.github.io/Convolution-Layer-Calculator/
        # 이미지(1채널)와 레이블 채널(1채널)을 합쳐 2채널 입력을 받는 CNN
        self.main = nn.Sequential(
            # 입력: 2 x 28 x 28 (이미지 1채널 + 레이블 1채널)
            nn.Conv2d(in_channels=2, out_channels=64,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2, padding=1, bias=False),     
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),                                                 
            nn.Linear(128 * 7 * 7, 1, bias=False),                    
            # 출력을 0~1 확률값으로 변환
            nn.Sigmoid(),
        )

    def forward(self, x, c):
        c_emb = self.label_emb(c)
        disc_input = torch.cat([x, c_emb], dim=1)
        return self.main(disc_input)


# 조건부 생성자와 조건부 판별자를 결합하여 적대적 학습을 수행
class CDCGAN(L.LightningModule):
    def __init__(self, num_classes=10, nz=100, lr=0.0002, beta1=0.5):
        super().__init__()
        self.automatic_optimization = False

        self.num_classes = num_classes
        self.nz = nz
        self.lr = lr
        self.beta1 = beta1
        self.save_hyperparameters()

        self.generator = CGenerator(nz=self.nz, num_classes=self.num_classes)
        self.discriminator = CDiscriminator(num_classes=self.num_classes)

    def forward(self, z, c):
        return self.generator(z, c)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, c = batch
        batch_size = x.size(0)

        opt_g, opt_d = self.optimizers()

        z = torch.randn(batch_size, self.nz, device=self.device)

        # 생성자 학습 단계
        self.toggle_optimizer(opt_g)

        fake = self(z, c)
        pred_fake = self.discriminator(fake, c)

        loss_G = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        self.log("loss_G", loss_G, prog_bar=True)
        self.manual_backward(loss_G)
        opt_g.step()
        opt_g.zero_grad()

        self.untoggle_optimizer(opt_g)


        # 판별자 학습 단계
        self.toggle_optimizer(opt_d)

        real = x
        z = torch.randn(batch_size, self.nz, device=self.device)
        fake = self(z, c).detach()

        pred_real = self.discriminator(real, c)
        pred_fake = self.discriminator(fake, c)

        loss_D_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))
        loss_D_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2
        self.log("loss_D", loss_D, prog_bar=True)
        self.manual_backward(loss_D)
        opt_d.step()
        opt_d.zero_grad()

        self.untoggle_optimizer(opt_d)

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_G, opt_D], []
