from argparse import ArgumentParser
from ultralytics import YOLO


def main():
    parser = ArgumentParser(description="Run YOLO inference on blade images.")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO weight file.")
    parser.add_argument("--source", default="datasets/blade2/test/images", help="Image file or directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(args.source, conf=args.conf, iou=args.iou, save=True)


if __name__ == "__main__":
    main()