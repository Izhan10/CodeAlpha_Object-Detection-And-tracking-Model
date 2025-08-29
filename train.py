from ultralytics import YOLO

model = YOLO("my_model.pt")

model.train(data = "custom_dataset.yaml", imgsz = 640, batch = 8, epochs = 100, workers = 1, device = "cpu")