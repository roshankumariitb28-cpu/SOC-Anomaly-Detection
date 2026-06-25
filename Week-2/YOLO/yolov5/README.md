# YOLOv5 Training and Inference

## Environment Setup

* Installed Git.
* Cloned the YOLOv5 repository.
* Installed all required dependencies using `requirements.txt`.
* Verified the installation of:

  * PyTorch
  * OpenCV
  * NumPy
  * Matplotlib
  * Pandas
  * TensorBoard

---

## Dataset Preparation

* Downloaded and prepared the dataset.
* Verified the project folder structure.

```text
Week-2/
└── YOLO/
    ├── Dataset/
    └── yolov5/
```

---

## Model Training

Run the following command to train the model:

```bash
python train.py --epochs 25
```

During training, the following metrics were monitored:

* Training Loss
* Validation Loss
* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

---

## Training Results

The trained model checkpoints were saved automatically:

```text
runs/train/exp5/weights/
├── best.pt
└── last.pt
```

---

## Visualization

Run the visualization script:

```bash
python visualize_results.py
```

Generated validation prediction images:

```text
val_batch0_pred.jpg
val_batch1_pred.jpg
val_batch2_pred.jpg
```

---

## Image Inference

Run inference on images:

```bash
python detect.py --weights runs/train/exp2/weights/best.pt --source "C:/Users/rosha/Downloads/yolov5_inference_data/inference_images"
```

The model generated object detection results and saved the output images with bounding boxes.

---

## Video Inference

Run inference on videos:

```bash
python detect.py --weights runs/train/exp2/weights/best.pt --source "C:/Users/rosha/Downloads/yolov5_inference_data/inference_videos"
```

The model processed the video frame by frame, detected objects, and saved the output video.

Output location:

```text
runs/detect/exp/
```

Conclusion

In this project, I successfully set up the YOLOv5 environment, prepared the dataset, trained the model for 25 epochs, visualized the training results, and performed inference on both images and videos. This project helped me understand the complete workflow of an object detection pipeline using YOLOv5, from model training to evaluating predictions on new data.
