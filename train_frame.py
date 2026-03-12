import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script with custom dataset path")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to frame dataset yaml file (e.g., mydata.yaml)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")
    model.train(
        data=args.data,
        ch=3,
        imgsz=640,
        epochs=500,
        batch=32,
        workers=4,
        device="0",
        optimizer="SGD",
        patience=0,
        amp=False,
        cache=False,
        project="runs/train",
        name="easy-fuse",
        resume=False,
        fraction=1,
    )
