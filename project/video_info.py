import cv2

video_path = "demo/demo_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Duration: {duration:.2f} seconds")

cap.release()