import cv2
import numpy as np

from project.dataset import build_datasets, TARGET_CLASSES

# train_dataset, val_dataset, test_dataset
_, _, test_dataset = build_datasets()

output_path = "demo/demo_video.mp4"

fps = 25
seconds_per_image = 2
frame_width = 640
frame_height = 480

frames_per_image = fps * seconds_per_image

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height),
)

if not writer.isOpened():
    raise RuntimeError("Could not create the output video.")

selected_indices = []
seen_labels = set()

for dataset_index in range(len(test_dataset)):
    _, label = test_dataset[dataset_index]

    if label not in seen_labels:
        selected_indices.append(dataset_index)
        seen_labels.add(label)

    if len(selected_indices) == len(TARGET_CLASSES):
        break

for dataset_index in selected_indices:
    original_index = test_dataset.indices[dataset_index]
    image, original_label = test_dataset.base_dataset[original_index]

    label = test_dataset.label_map[original_label]
    class_name = TARGET_CLASSES[label]

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (frame_width, frame_height))

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

writer.release()

print(f"Demo video saved to: {output_path}")