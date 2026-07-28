import argparse
from pathlib import Path

import cv2


def get_video_info(video_path: str) -> None:
    """Read and display basic information about a video file."""
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = total_frames / fps if fps > 0 else 0.0

        print("Video opened successfully.")
        print(f"Path: {path}")
        print(f"FPS: {fps:.2f}")
        print(f"Total Frames: {total_frames}")
        print(f"Width: {width}")
        print(f"Height: {height}")
        print(f"Duration: {duration:.2f} seconds")
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display basic information about a video file."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="demo/demo_video.mp4",
        help="Path to the input video.",
    )
    args = parser.parse_args()

    try:
        get_video_info(args.video)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()