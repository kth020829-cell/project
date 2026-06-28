**DDPM 커스텀 데이터 학습 & U-Net 구조 비교**

포켓몬 스프라이트 이미지(약 800장)를 직접 수집하여 DDPM을 학습하고, U-Net 구조 변형에 따른 성능을 비교한 프로젝트입니다.



**주요 내용**

- **커스텀 데이터셋**: torchvision 빌트인 로더 없이 폴더에서 이미지를 직접 읽는 `FolderImageDataset` 구현. 컬러 64×64 해상도로 학습.
- **DDPM 구현**: Ho et al.(2020) 공개 PyTorch 재구현(Brian Pulfer) 기반. 500 스텝 선형 베타 스케줄 적용.
- **U-Net 세 가지 변형 비교**:
  - `A_baseline`: 기본 구조 (width=1.0, 파라미터 약 91만)
  - `B_wide`: 채널 폭 2배 확장 (width=2.0, 파라미터 약 223만)
  - `C_attention`: 병목 부분에 Self-Attention 추가 (width=1.0, 파라미터 약 92만)
- **결과**: B_wide가 최종 손실 0.0236으로 가장 낮았으나, C_attention도 attention 추가만으로 안정적인 학습 가능성을 확인.



**Tech Stack**

`Python` `PyTorch` `einops` `Google Colab`
