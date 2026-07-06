# YOLOv5m Frozen Layer Training

## Introduction

In this project, I trained the YOLOv5m model with frozen layers for object detection. The aim was to learn transfer learning and compare the performance of a frozen model while reducing the training time.

## Dataset

I used a custom vehicle detection dataset in YOLO format.

The dataset contains the following classes:

- Ambulance
- Bus
- Car
- Motorcycle
- Truck

The dataset is divided into training and validation sets.

## Model Configuration

- Model: YOLOv5m
- Image Size: 640 × 640
- Batch Size: 16
- Epochs: 24
- Frozen Layers: Yes
- Device: CPU

## Training

The model was trained using YOLOv5. During training, the initial layers of the network were frozen so that only the remaining layers were updated.

Training command:

```bash
python train.py --img 640 --batch 16 --epochs 23 --weights yolov5m.pt --data data.yaml --freeze 10
```

## Results

The final validation results are:

| Metric | Value |
|--------|--------|
| Precision | 0.620 |
| Recall | 0.693 |
| mAP@0.5 | 0.682 |
| mAP@0.5:0.95 | 0.489 |

The model performed well on Ambulance and Bus classes, while Truck detection can still be improved.

## Challenges

I trained the model on CPU, so the training was slower than GPU training. During training, VS Code stopped because of an Out of Memory (OOM) error. Since the checkpoint (`last.pt`) was saved, I resumed the training and completed all 23 epochs.

## Files Generated

After training, the following files were generated:

- `best.pt`
- `last.pt`
- `results.csv`
- `results.png`
- Training batch images

## Conclusion

This project helped me understand how transfer learning works in YOLOv5 using frozen layers. I also learned how to resume interrupted training using checkpoints. The model achieved good performance on the vehicle detection dataset and can be further improved by training for more epochs or using a larger dataset.

## Tools Used

- Python
- PyTorch
- YOLOv5
- OpenCV
- NumPy
- Matplotlib
