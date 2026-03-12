from ultralytics import YOLO

model = YOLO("/your/path/to/best.pt")
model.val(
    data="/your/path/to/data.yaml",
    ch=5,  # multimodal: 5, frame: 3, event: 2
    split='val',
    save=True,
)
