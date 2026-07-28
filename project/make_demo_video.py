from pathlib import Path

import cv2
import numpy as np

from project.dataset import TARGET_CLASSES, build_datasets


def main() -> None:
    """Create a short demo video using one image from each target class."""
    output_path = Path("demo/demo_video.mp4")

    fps = 25
    seconds_per_image = 2
    frame_width = 640
    frame_height = 480

    frames_per_image = fps * seconds_per_image

    # Ensure that the output folder exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset loading is placed inside main() to avoid running during import.
    _, _, test_dataset = build_datasets()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    try:
        selected_indices = []
        seen_labels = set()

        # Select one sample from each target class.
        for dataset_index in range(len(test_dataset)):
            _, label = test_dataset[dataset_index]

            if label not in seen_labels:
                selected_indices.append(dataset_index)
                seen_labels.add(label)

            if len(selected_indices) == len(TARGET_CLASSES):
                break

        if len(selected_indices) < len(TARGET_CLASSES):
            raise RuntimeError(
                "Could not find one test sample for every target class."
            )

        for dataset_index in selected_indices:
            original_index = test_dataset.indices[dataset_index]
            image, original_label = test_dataset.base_dataset[original_index]

            label = test_dataset.label_map[original_label]
            class_name = TARGET_CLASSES[label]

            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(
                frame,
                (frame_width, frame_height),
            )

            cv2.putText(
                frame,
                class_name,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )

            for _ in range(frames_per_image):
                writer.write(frame)

    finally:
        writer.release()

    print(f"Demo video saved to: {output_path}")


if __name__ == "__main__":
    main()