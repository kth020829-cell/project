import torch
import numpy as np
from model import CVAE, CVAE_CNN
from dataset import test_loader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import gradio as gr

# 모델 선택 (train.py와 동일하게 설정)
USE_CNN = True

# 학습된 CVAE 모델 불러오기
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
        label = int(cc)                        # 텐서를 정수로 변환 (ex) tensor(3) -> 3)
        label_mean[label] = mean[idx]        
        label_log_var[label] = log_var[idx]  
        label_bool[label] = True

    if (np.all(label_bool)):
        break

label_std = label_log_var.exp().pow(0.5)


def generate_image(cls, z0, z1):
    # 해당 숫자의 평균에 사용자 입력 잠재 벡터를 더하여 z 생성
    z = label_mean[cls] + torch.tensor([z0, z1])
    c = torch.tensor(cls, dtype=torch.int64)

    # 배치 차원 추가 (디코더 입력 형식에 맞추기)
    z = torch.unsqueeze(z, 0)    # (2,) -> (1, 2)
    c = torch.unsqueeze(c, 0)    # () -> (1,)

    x_pred = model.decoder(z, c)

    gen_img = to_pil_image(x_pred[0])

    return gen_img


# Gradio 웹 UI 구성
with gr.Blocks() as demo:
    gr.Markdown("# Generative Model with Conditional Variational AutoEncoder")
    with gr.Row(equal_height=True):
        cls = gr.Number(value=0, minimum=0, maximum=9)
        z0 = gr.Slider(minimum=-10.0, maximum=10.0, value=0.0, step=0.01, label="z0")
        z1 = gr.Slider(minimum=-10.0, maximum=10.0, value=0.0, step=0.01, label="z1")
        button = gr.Button("Generate", variant="primary")
    with gr.Row(equal_height=True):
        gen_image = gr.Image(height=250, width=250)

    button.click(
        generate_image,
        inputs  = [cls, z0, z1],
        outputs = [gen_image],
    )


# Gradio 서버 실행 / 코랩용
# demo.launch()
demo.launch(share=True)