**CVAE - Conditional Variational AutoEncoder (MNIST 손글씨 생성 모델)**

MNIST 손글씨 숫자 데이터셋을 활용하여 숫자 클래스를 조건으로 원하는 숫자 이미지를 생성할 수 있는 Conditional Variational AutoEncoder(CVAE)를 구현한 프로젝트입니다. PyTorch Lightning 기반으로 MLP와 CNN 두 가지 구조를 각각 설계·학습하였으며, Gradio UI를 통해 잠재 벡터(z0, z1)를 조절하며 생성 결과를 인터랙티브하게 확인할 수 있습니다.

**모델 설계**

* **MLP 기반 CVAE**: 이미지를 1차원(784)으로 펼친 뒤 조건 벡터(10차원 원-핫)를 이어붙여 인코더/디코더에 입력. 784+10=794차원 입력으로 512→256 은닉층을 거쳐 2차원 잠재 벡터를 출력.
* **CNN 기반 CVAE**: 조건 벡터를 28x28 공간 크기의 채널로 확장하여 이미지와 채널 방향으로 결합(11채널 입력). Conv2d로 공간을 압축하고 ConvTranspose2d로 복원하는 구조.
* **재매개변수화 트릭(Reparameterization Trick)**: z = mean + std * epsilon 형태로 샘플링하여 역전파가 가능하도록 구현.

**학습**

* 데이터셋: MNIST (훈련 60,000장 / 테스트 10,000장), ToTensor() 전처리로 0~1 정규화
* 손실 함수: Reconstruction Loss(MSE) + KL Divergence Loss (ELBO 최대화)
* 하이퍼파라미터: batch_size=100, lr=1e-3, epochs=20, n_dim=2, optimizer=Adam

**결과**

* MLP/CNN 모두 안정적으로 학습되었으며 MNIST와 같은 단순한 데이터셋에서는 육안으로 유의미한 복원 품질 차이가 관찰되지 않음.
* Gradio UI에서 클래스(0~9) 지정 및 z0, z1 슬라이더 조작으로 같은 숫자의 다양한 스타일 이미지 생성 가능.
* 잠재 공간 차원을 2로 제한했음에도 숫자 클래스별 클러스터가 의미 있게 형성됨.

**사용 기술**: Python, PyTorch, PyTorch Lightning, Gradio, MNIST
