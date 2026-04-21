import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO Multi-modal Validation Script")
    
    parser.add_argument('--model', type=str, default="best.pt", help="Path to model weight (e.g., best.pt)")
    parser.add_argument('--data', type=str, required=True, help="Path to data.yaml")
    parser.add_argument('--ch', type=int, default=5, help="Input channels: multimodal=5, frame=3, event=2")

    args = parser.parse_args()

    model = YOLO(args.model)

    model.val(
        data=args.data,
        ch=args.ch,  # multimodal: 5, frame: 3, event: 2
        split='val',
        save=True,
    )

if __name__ == "__main__":
    main()