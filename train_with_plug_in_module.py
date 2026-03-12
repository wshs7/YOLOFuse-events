import argparse
import yaml
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = YOLO("ultralytics/cfg/models/fuse/Easy-level-Feature-Fusion_plug.yaml")
    model.train(
        data=args.data,
        ch=5,
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
        name="easy-gate",
        resume=False,
        fraction=1,
    )
