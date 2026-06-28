import torch
from model import CDCGAN
from dataset import train_dataset

model = CDCGAN.load_from_checkpoint("./cdcgan.ckpt")
model.eval() 
model.to(torch.device("cpu"))
print(model.device)

test_cls = 5

z = torch.randn(1, model.hparams.nz, device=model.device)
c = torch.tensor([test_cls], device=model.device)

print("z:", z)
print("z.shape:", z.shape)
print("c:", c)
print("c.shape:", c.shape)

imgs = model(z, c)

from torchvision.transforms.functional import pil_to_tensor, to_pil_image
PIL_img = to_pil_image(imgs[0])

from PIL import Image
PIL_img.save("output.png")
