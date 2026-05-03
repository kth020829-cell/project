# CDCGAN - MNIST 조건부 이미지 생성 모델 구현

DCGAN(Deep Convolutional GAN)을 확장하여 클래스 조건부 이미지 생성이 가능한 **CDCGAN(Conditional DCGAN)** 을 PyTorch Lightning 기반으로 구현한 프로젝트입니다.

---

## 1. 모델 설계

- **조건부 생성자 (CGenerator)**: 랜덤 노이즈 벡터 z와 클래스 레이블 c를 Embedding 레이어로 변환 후 concat하여 입력으로 사용. ConvTranspose2d 레이어를 통해 28×28 이미지를 생성하도록 설계.
- **조건부 판별자 (CDiscriminator)**: 클래스 레이블을 이미지와 동일한 28×28 공간으로 임베딩하여 채널 방향으로 결합. 기존 DCGAN 대비 입력 채널을 1→2로 확장하여 조건 인식 판별 구현.
- **DCGAN과의 핵심 차이**: 생성자와 판별자 양쪽 모두에 클래스 조건 c를 적용하여, 원하는 숫자(0~9)를 지정하여 생성할 수 있는 조건부 생성 모델 구현.

---

## 2. 학습 및 결과 분석

- **적대적 학습 (Adversarial Training)**: PyTorch Lightning의 수동 최적화(manual optimization) 모드를 활용하여 생성자와 판별자를 독립적으로 업데이트. BCE 손실 함수 기반으로 학습 진행.
- **학습 손실 분석**: Discriminator 손실(loss_D)이 이론적 수렴값인 ln2(≈0.693) 근방인 0.6~0.7 구간에서 안정적으로 유지되어 두 네트워크가 균형 있게 학습되었음을 확인.
- **클래스별 생성 이미지**: 0~9 각 클래스에 대해 조건이 명확하게 반영된 이미지가 생성됨을 확인. 동일 클래스 내에서도 노이즈 벡터 z에 따라 다양한 스타일이 생성됨.
- **잠재 공간 보간 실험**: nz=5의 작은 잠재 공간에서 z 차원 변화에 따른 이미지 변화를 관찰. 클래스 조건이 모든 z 값에서 일관되게 유지됨을 확인.

---

## 3. 생성 네트워크 활용 (Gradio UI)

Gradio 기반 웹 인터페이스를 구현하여 숫자 클래스(0~9)와 노이즈 벡터(z0~z4)를 슬라이더로 조작하며 실시간 이미지 생성 테스트 가능.

---

## Tech Stack

`Python` `PyTorch` `PyTorch Lightning` `Gradio` `einops` `Google Colab`

---

## 역할

단독 구현 (모델 설계, 학습, 결과 분석, Gradio UI 적용)
