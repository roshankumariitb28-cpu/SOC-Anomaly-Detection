# Week 5: Pedestrian Tracking & Rule-Based Anomaly Detection

## Objective

The objective of this week's project was to build a complete pedestrian tracking and anomaly detection pipeline by combining object detection, multi-object tracking, and rule-based anomaly detection on surveillance videos.

The pipeline uses a custom-trained **YOLOv5** model for pedestrian detection, **DeepSORT** for multi-object tracking, and the **CUHK Avenue Dataset** for anomaly detection.

---

## Dataset

### Avenue Dataset (CUHK)

- 16 training videos
- 21 testing videos
- Outdoor surveillance scenes
- Contains both normal and abnormal pedestrian activities

### Examples of Abnormal Events

- Running
- Wrong-direction movement
- Unexpected pedestrian trajectories
- Abnormal pedestrian behavior

---

## Model Selection

The best-performing detector from **Week 3** was selected for this project.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|----------:|-------:|---------:|-------------:|
| YOLOv5s | 0.880 | 0.767 | 0.872 | 0.567 |
| **YOLOv5m** | **0.919** | **0.820** | **0.910** | **0.648** |
| YOLOv5m (Frozen Layers) | 0.659 | 0.683 | 0.670 | 0.477 |

**Selected Model:** **YOLOv5m**

---

## Pipeline

```text
Input Video
      │
      ▼
YOLOv5 Pedestrian Detection
      │
      ▼
DeepSORT Multi-Object Tracking
      │
      ▼
Trajectory History
      │
      ▼
Velocity Estimation
      │
      ▼
Rule-Based Anomaly Detection
      │
      ▼
Annotated Output Video
```

---

## Anomaly Detection Strategy

Each detected pedestrian is assigned a unique tracking ID using **DeepSORT**.

The system maintains a short trajectory history for every tracked pedestrian and computes the pedestrian's velocity over time.

An anomaly is detected when:

- Pedestrian velocity exceeds a predefined threshold.
- The anomaly state is maintained for several frames using temporal smoothing to reduce flickering detections.
- A minimum tracking history is required before evaluating anomalies to reduce false positives caused by newly initialized tracks.

---

## Technologies Used

- Python
- PyTorch
- YOLOv5
- DeepSORT
- OpenCV
- NumPy
- tqdm

---

## Results

- Successfully integrated YOLOv5 with DeepSORT.
- Performed pedestrian detection and tracking on the CUHK Avenue Dataset.
- Maintained unique IDs for tracked pedestrians.
- Detected abnormal pedestrian motion using a velocity-based rule.
- Generated annotated output videos highlighting detected anomalies.

---

## Project Structure

```text
Week-5/
│── main.py
│── output_videos/
│── Avenue Dataset/
│   ├── training_videos/
│   ├── testing_videos/
│   ├── training_vol/
│   └── testing_vol/
└── README.md
```

---

## Future Improvements

- Direction-based anomaly detection
- Trajectory deviation analysis
- Restricted-area intrusion detection
- Integration with Avenue ground-truth evaluation
- Deep learning-based anomaly detection methods

---

## Acknowledgements

- **YOLOv5** by Ultralytics
- **DeepSORT** for multi-object tracking
- **CUHK Avenue Dataset** for abnormal event detection
