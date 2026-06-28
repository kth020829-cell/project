import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td    # torch distribution 임포트

import lightning as L

# MLP 기반 CVAE 모델
class CVAE(L.LightningModule):
    def __init__(self, n_dim=2, lr=1e-3):
        super().__init__()
        self.n_dim = n_dim          
        self.cond_dim = 10          
        self.lr = lr               

        # 인코더 네트워크 (MLP)
        # 입력: 이미지(784) + 조건(10) = 794차원
        self.enc_fc1 = nn.Linear(784 + self.cond_dim, 512)  
        self.enc_fc2 = nn.Linear(512, 256)                   
        self.enc_mean = nn.Linear(256, n_dim)                 # 평균(mean) 출력층
        self.enc_log_var = nn.Linear(256, n_dim)              # 로그 분산(log_var) 출력층

        # 디코더 네트워크 (MLP)
        # 입력: 잠재 벡터(n_dim) + 조건(10)
        self.dec_fc1 = nn.Linear(n_dim + self.cond_dim, 256)
        self.dec_fc2 = nn.Linear(256, 512)          
        self.dec_out = nn.Linear(512, 784)                     # 출력층 (28x28 이미지)


    def encoder(self, x, c):
        x_flattened = x.view(x.shape[0], -1)                  # (batch_size, 784)로 펼치기
        c = F.one_hot(c, num_classes=10).float()               # 원-핫 인코딩 (batch_size, 10)

        # 이미지와 조건을 결합하여 인코더에 입력
        xc = torch.cat([x_flattened, c], dim=1)                # (batch_size, 794)
        h = F.relu(self.enc_fc1(xc))                           
        h = F.relu(self.enc_fc2(h))                      
        mean = self.enc_mean(h)                                # 평균 벡터 출력
        log_var = self.enc_log_var(h)                          # 로그 분산 벡터 출력
        return mean, log_var

    def reparameterize(self, mean, log_var): # 재매개변수화 트릭: 역전파가 가능하도록 샘플링하는 기법
        std = torch.exp(0.5 * log_var)                         # 표준편차 = exp(0.5 * log_var)
        eps = torch.randn_like(std)                            # 표준 정규분포에서 노이즈 샘플링
        z = mean + std * eps                                   # z = mean + std * epsilon
        return z

    def decoder(self, z, c):
        c = F.one_hot(c, num_classes=10).float()               # 원-핫 인코딩 (batch_size, 10)
        zc = torch.cat([z, c], dim=1)                          # 잠재 벡터와 조건 결합

        h = F.relu(self.dec_fc1(zc))                         
        h = F.relu(self.dec_fc2(h))                           
        x_pred = torch.sigmoid(self.dec_out(h))                # 시그모이드로 0~1 범위 출력
        x_pred = x_pred.view(-1, 1, 28, 28)                   # (batch_size, 1, 28, 28)로 복원
        return x_pred


    def training_step(self, batch, batch_idx):
        x, c = batch

        # 인코더로 평균과 로그 분산 계산
        mean, log_var = self.encoder(x, c)
        # 재매개변수화 트릭으로 잠재 벡터 샘플링
        z = self.reparameterize(mean, log_var)
        # 디코더로 이미지 복원
        x_pred = self.decoder(z, c)

        # 복원 손실 (MSE): 원본 이미지와 복원 이미지 사이의 차이
        recon_loss = F.mse_loss(x_pred, x, reduction='sum') / x.shape[0]

        # KL 발산 손실: 잠재 분포가 표준 정규분포에 가까워지도록 유도
        # KL(q(z|x,c) || p(z)) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.shape[0]

        # 전체 손실 = 복원 손실 + KL 발산 손실
        loss = recon_loss + kl_loss

        # 로그에 손실 기록 (학습 진행 모니터링)
        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("kl_loss", kl_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x, c):
        mean, log_var = self.encoder(x, c)                     # 인코딩
        z = self.reparameterize(mean, log_var)                 # 잠재 벡터 샘플링
        x_pred = self.decoder(z, c)                            # 디코딩 (이미지 복원)
        return x_pred


# CNN 기반 CVAE 모델
class CVAE_CNN(L.LightningModule):
    def __init__(self, n_dim=2, lr=1e-3):
        super().__init__()
        self.n_dim = n_dim     
        self.cond_dim = 10        
        self.lr = lr     

        # 인코더 네트워크 (CNN)
        # 입력: 이미지(1채널) + 조건(10채널로 확장) = 11채널
        self.enc_conv1 = nn.Conv2d(1 + 10, 32, kernel_size=3, stride=2, padding=1)   # (11,28,28) -> (32,14,14)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)       # (32,14,14) -> (64,7,7)
        self.enc_fc_mean = nn.Linear(64 * 7 * 7, n_dim)         # 평균 출력층
        self.enc_fc_log_var = nn.Linear(64 * 7 * 7, n_dim)      # 로그 분산 출력층

        # 디코더 네트워크 (CNN)
        # 입력: 잠재 벡터(n_dim) + 조건(10)
        self.dec_fc = nn.Linear(n_dim + 10, 64 * 7 * 7)                              # FC로 특성맵 크기 복원
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # (64,7,7) -> (32,14,14)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)   # (32,14,14) -> (1,28,28)


    def encoder(self, x, c):
        c_onehot = F.one_hot(c, num_classes=10).float()  
        # 조건을 이미지와 같은 공간 크기로 확장: (batch, 10) -> (batch, 10, 28, 28)
        c_map = c_onehot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 28, 28)
        # 이미지(1채널)와 조건(10채널)을 채널 방향으로 결합 -> (batch, 11, 28, 28)
        xc = torch.cat([x, c_map], dim=1)

        h = F.relu(self.enc_conv1(xc))                          
        h = F.relu(self.enc_conv2(h))                   
        h = h.view(h.shape[0], -1)                               # (batch, 64*7*7)로 펼치기
        mean = self.enc_fc_mean(h)                               # 평균 출력
        log_var = self.enc_fc_log_var(h)                         # 로그 분산 출력
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z

    def decoder(self, z, c):
        c_onehot = F.one_hot(c, num_classes=10).float()    
        zc = torch.cat([z, c_onehot], dim=1)             
        
        h = F.relu(self.dec_fc(zc))                              # FC 레이어 + ReLU
        h = h.view(-1, 64, 7, 7)                                 # 특성맵 형태로 재배열
        h = F.relu(self.dec_conv1(h))                            # 첫 번째 전치 합성곱 + ReLU
        x_pred = torch.sigmoid(self.dec_conv2(h))                # 두 번째 전치 합성곱 + 시그모이드
        return x_pred


    def training_step(self, batch, batch_idx):
        x, c = batch

        mean, log_var = self.encoder(x, c)
        z = self.reparameterize(mean, log_var)
        x_pred = self.decoder(z, c)

        # 복원 손실 (MSE)
        recon_loss = F.mse_loss(x_pred, x, reduction='sum') / x.shape[0]
        # KL 발산 손실
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.shape[0]
        loss = recon_loss + kl_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("kl_loss", kl_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x, c):
        mean, log_var = self.encoder(x, c)
        z = self.reparameterize(mean, log_var)
        x_pred = self.decoder(z, c)
        return x_pred
    
