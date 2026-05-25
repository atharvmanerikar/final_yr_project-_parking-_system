from ultralytics import YOLO

# load base model
model = YOLO("yolov8n.pt")

# train license plate detector
model.train(
    data="config.yaml",
    epochs=80,      # better for overnight training
    imgsz=800,
    batch=8,
    mosaic=1.0,
    mixup=0.1,
    degrees=5,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    patience=30,
    save=True
)