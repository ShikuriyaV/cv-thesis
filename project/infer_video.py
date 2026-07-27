import argparse
import csv
from pathlib import Path

import cv2
import torch

from PIL import Image
from torchvision import models

from project.dataset import build_transforms, TARGET_CLASSES


def load_model(checkpoint_path, device):
    """Create ResNet18 and load the trained checkpoint."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(TARGET_CLASSES),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Support two common checkpoint formats.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def infer_video(
    video_path,
    checkpoint_path,
    output_csv,
    frame_step,
):
    """Run frame-level classification on a video."""
    if frame_step <= 0:
        raise ValueError("frame_step must be greater than zero.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = load_model(checkpoint_path, device)

    _, eval_transform = build_transforms()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not read the video FPS.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_id = 0
    prediction_count = 0

    try:
        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "timestamp",
                "frame_id",
                "predicted_class",
                "confidence",
            ])

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_id % frame_step == 0:
                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )
                    image = Image.fromarray(rgb_frame)

                    input_tensor = eval_transform(image)
                    input_tensor = input_tensor.unsqueeze(0)
                    input_tensor = input_tensor.to(device)

                    with torch.no_grad():
                        logits = model(input_tensor)
                        probabilities = torch.softmax(
                            logits,
                            dim=1,
                        )

                        confidence, predicted_index = (
                            probabilities.max(dim=1)
                        )

                    class_name = TARGET_CLASSES[
                        predicted_index.item()
                    ]
                    confidence_value = confidence.item()
                    timestamp = frame_id / fps

                    writer.writerow([
                        f"{timestamp:.2f}",
                        frame_id,
                        class_name,
                        f"{confidence_value:.4f}",
                    ])

                    print(
                        f"Frame {frame_id}: "
                        f"{class_name} "
                        f"({confidence_value:.4f})"
                    )

                    prediction_count += 1

                frame_id += 1

    finally:
        cap.release()

    print(f"Predictions: {prediction_count}")
    print(f"CSV saved to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frame-level video classification."
    )

    parser.add_argument(
        "--video",
        default="demo/demo_video.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--checkpoint",
        default="project/checkpoints/best_model.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="results/video_predictions.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=25,
        help="Run inference every N frames.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    infer_video(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        frame_step=args.frame_step,
    )