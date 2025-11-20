# yolov8.py
from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  # 预训练模型

    model.train(
        data="screen.yaml",   # ⚠️ 这里传 yaml 文件，而不是 dict
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/train",
        name="shared_screen",
        exist_ok=True
    )

    model.save("shared_screen.pt")

if __name__ == "__main__":
    train()
