import os
import torch
from roboflow import Roboflow

# ----------------------------
# 1. Device Check
# ----------------------------
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Available - Using GPU: {device_name}")
else:
    print("CUDA not available - Using CPU")

# ----------------------------
# 2. Roboflow Dataset Download
# ----------------------------

# API key는 환경변수에서 불러오도록 설정
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if ROBOFLOW_API_KEY is None:
    raise ValueError(
        "Roboflow API key not found. "
        "Please set environment variable: ROBOFLOW_API_KEY"
    )

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

project = rf.workspace("thobjectdetection").project("surface-defect-oyrla-fdryq")
version = project.version(1)

dataset = version.download("yolov8")

# yolo version comparison
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

model.train(
    data="Surface-defect-1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="v8_experiment"
)

model = YOLO("yolo11n.pt")

model.train(
    data="Surface-defect-1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="v11_experiment"
)


model = YOLO("yolo26n.pt")

model.train(
    data="Surface-defect-1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="v26_experiment"
)


print(os.listdir("runs/detect"))

# v8 validation
model_v8 = YOLO("runs/detect/v8_experiment/weights/best.pt")
metrics_v8 = model_v8.val(data="Surface-defect-1/data.yaml")

# v11 validation
model_v11 = YOLO("runs/detect/v11_experiment/weights/best.pt")
metrics_v11 = model_v11.val(data="Surface-defect-1/data.yaml")

# v26 validation
model_v26 = YOLO("runs/detect/v26_experiment/weights/best.pt")
metrics_v26 = model_v26.val(data="Surface-defect-1/data.yaml")


# mAP50, inference trade-off 그래프
import matplotlib.pyplot as plt

models = ["YOLOv8n", "YOLOv11n", "YOLOv26n"]
inference_times = [9.6, 13.1, 10.5]  # ms
map50_scores = [0.597, 0.679, 0.573]

plt.figure()
plt.scatter(inference_times, map50_scores)

for i, model in enumerate(models):
    plt.text(inference_times[i], map50_scores[i], model)

plt.xlabel("Inference Time (ms)")
plt.ylabel("mAP50")
plt.title("Speed vs Accuracy Trade-off")

plt.show()

