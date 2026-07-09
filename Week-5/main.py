import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model — using your best-performing trained model (mot17_yolov5m2)
model_path = r"C:\Users\rosha\Desktop\SOC\Week-2\YOLO\yolov5\runs\train\mot17_yolov5m2\weights\best.pt"
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
model.conf = 0.25
model.iou = 0.45

from deep_sort_realtime.deepsort_tracker import DeepSort

output_video_folder = "output_videos"
os.makedirs(output_video_folder, exist_ok=True)

input_video_path = "Avenue Dataset/testing_videos/05.avi"
output_video_path = "output_videos/result05.mp4"

# ---- Tune these ----
N_history = 10              # how many recent points to keep per track
smoothing_frames = 15        # how long an anomaly label persists after trigger
velocity_threshold = 250     # px/sec — raised, was way too low at 50
min_track_age = 8            # require at least this many frames of history
                              # before a track is allowed to be flagged anomalous
# ---------------------

# Init DeepSORT tracker
tracker = DeepSort(max_age=30)

# Input video
cap = cv2.VideoCapture(input_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps} FPS, {width}x{height}, {frame_count} frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

read_count = 0
frame_idx = 0
pbar = tqdm(total=frame_count)
track_memory = {}
anomaly_memory = {}
saved_anomaly_ids = set()
max_anomaly_frames = 500

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    detections = detections[detections[:, 5] == 0]  # class 0 = person

    formatted_detections = []
    for d in detections:
        xmin, ymin, xmax, ymax, conf, cls = d
        width_box = xmax - xmin
        height_box = ymax - ymin
        box = [xmin, ymin, width_box, height_box]
        formatted_detections.append([box, conf])

    tracks = tracker.update_tracks(formatted_detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = map(int, ltrb)

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        track_memory.setdefault(track_id, []).append((frame_idx, center_x, center_y))
        track_memory[track_id] = track_memory[track_id][-N_history:]

        history = track_memory[track_id]

        # Only compute/allow anomaly once we have enough history AND the
        # history actually spans min_track_age frames. This stops brand-new
        # or freshly re-spawned tracks (common with DeepSORT ID switches) from
        # producing a bogus huge velocity from very few frames of noisy data.
        if len(history) >= 2 and len(history) >= min(N_history, min_track_age):
            f1, x1, y1 = history[0]
            f2, x2, y2 = history[-1]
            frame_span = f2 - f1
            if frame_span >= min_track_age:
                dt = frame_span / fps
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx**2 + dy**2)
                velocity = distance / (dt + 1e-6)
                anomaly = velocity > velocity_threshold
            else:
                anomaly = False
        else:
            anomaly = False

        if anomaly:
            anomaly_memory[track_id] = frame_idx

        if track_id in anomaly_memory:
            if frame_idx - anomaly_memory[track_id] <= smoothing_frames:
                anomaly = True

        if anomaly:
            label = f"ID {track_id} ANOMALY"
            color = (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    frame_idx += 1
    read_count += 1
    pbar.update(1)

pbar.close()
cap.release()
out.release()
print(f"Frames read: {read_count} / {frame_count}")
print(f"\nDONE — output saved to: {output_video_path}")