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

## Remaining Tasks

* Train YOLOv5m with frozen layers.
* Compare the performance of YOLOv5s, YOLOv5m, and YOLOv5m with frozen layers.

