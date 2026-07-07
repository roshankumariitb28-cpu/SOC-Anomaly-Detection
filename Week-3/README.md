# SOC Week 3 Report: Pedestrian Detection using YOLOv5 on MOT17

## Objective

The objective of this week was to prepare the MOT17 dataset for YOLOv5 training, convert the annotations into YOLO format, and train pedestrian detection models.

## Work Completed

### 1. Downloaded and Explored MOT17 Dataset

* Downloaded the MOT17 dataset.
* Explored the dataset structure, including the `train` and `test` folders.
* Studied the sequence folders (e.g., `MOT17-02-FRCNN`) and the `gt.txt` annotation files.

### 2. Understood MOT17 Annotations

* Learned the annotation format containing frame number, object ID, bounding box coordinates, confidence, class, and visibility.
* Understood how MOT17 stores tracking annotations.

### 3. Converted MOT17 to YOLO Format

* Developed a Python script (`convert_mot17.py`) to convert MOT17 annotations into YOLO format.
* Converted bounding boxes to normalized YOLO coordinates.
* Generated one label file for each image.

### 4. Prepared the YOLO Dataset

* Created the required YOLO dataset structure:

  * `processed/train/images`
  * `processed/train/labels`
* Generated **5,316 images** and **5,316 corresponding label files**.
* Created the `data.yaml` configuration file for YOLOv5.

### 5. Trained YOLOv5s

* Trained the YOLOv5s model for **25 epochs**.
* Training completed successfully and generated:

  * `best.pt`
  * `last.pt`
* Performance:

  * Precision: **0.884**
  * Recall: **0.763**
  * mAP@50: **0.872**
  * mAP@50-95: **0.567**

### 6. Trained YOLOv5m

* Trained the YOLOv5m model for **25 epochs**.
* Training was performed on the CPU and took approximately **18+ hours**.
* Successfully generated:

  * `best.pt`
  * `last.pt`

## Current Status

* Downloaded MOT17 dataset.
* Understood the dataset structure and annotations.
* Converted MOT17 annotations to YOLO format.
* Created the processed dataset and `data.yaml`.
* Successfully trained YOLOv5s.
* Successfully trained YOLOv5m.
* Successfully trained YOLOv5m with frozen layers.
* 
# YOLOv5 Model Comparison

This project successfully trains and compares three YOLOv5 models:
- YOLOv5s
- YOLOv5m
- YOLOv5m with Frozen Layers


# Objective

The objective of this project was to train different YOLOv5 models on the same dataset and compare their performance. The three models used were:

- YOLOv5s
- YOLOv5m
- YOLOv5m with Frozen Layers

The comparison was based on Precision, Recall, mAP, and training performance.

---

# Dataset

A custom vehicle detection dataset in YOLO format was used.

The dataset contains the following classes:

- Ambulance
- Bus
- Car
- Motorcycle
- Truck

The dataset was divided into training and validation sets.

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | YOLOv5 |
| Language | Python |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Device | CPU |
| Optimizer | SGD |
| Epochs | 25 (YOLOv5s & YOLOv5m), 23 (Frozen Model) |

---

# Results

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|----------:|-------:|---------:|-------------:|
| YOLOv5s | **0.880** | **0.767** | **0.872** | **0.567** |
| YOLOv5m | **0.919** | **0.820** | **0.910** | **0.648** |
| YOLOv5m (Frozen Layers) | **0.620** | **0.693** | **0.681** | **0.488** |

---

# Comparison

### YOLOv5s

- Smallest model.
- Faster training.
- Good detection accuracy.
- Suitable for real-time applications and systems with limited resources.

### YOLOv5m

- Highest accuracy among all models.
- Better Precision, Recall and mAP.
- Requires more computation and training time.

### YOLOv5m with Frozen Layers

- Some layers were frozen during training.
- Reduced the number of trainable parameters.
- Faster fine-tuning than full YOLOv5m.
- Accuracy was lower than the fully trained YOLOv5m.

---

# Challenges Faced

- Training was performed on CPU, which increased training time.
- During frozen-layer training, VS Code stopped because of an Out of Memory (OOM) error.
- Since the checkpoint (`last.pt`) was saved, training was resumed successfully and completed.

---

# Output Files

The following files were generated after training:

- best.pt
- last.pt
- results.csv
- results.png
- train_batch0.jpg
- train_batch1.jpg
- train_batch2.jpg

---

# Model Weights

The trained model generated two weight files:

- **best.pt** – Best-performing model based on validation results.
- **last.pt** – Final checkpoint after training.

Each file is approximately **41 MB**. Since GitHub's web upload has a **25 MB file size limit**, these files are not included in the repository. They can be shared separately using Google Drive or Git LFS if required.

---

# Conclusion

From the comparison, **YOLOv5m** achieved the best overall performance with the highest Precision, Recall, and mAP values. It produced the most accurate detections but required more computation.

**YOLOv5s** provided a good balance between speed and accuracy, making it suitable for lightweight applications.

**YOLOv5m with Frozen Layers** reduced the amount of training by freezing part of the network. Although its accuracy was lower than the fully trained YOLOv5m, it demonstrated how transfer learning can reduce training effort while still achieving reasonable performance.

Overall, this project helped me understand object detection using YOLOv5, transfer learning with frozen layers, checkpoint-based training recovery, and how different model sizes affect detection performance.

---


