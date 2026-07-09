# Zero-Shot Object Tracking using YOLOv5, CLIP, DeepSORT, and Custom YOLOv7

## Objective

The objective of this project was to implement a zero-shot object tracking pipeline using **YOLOv5** for object detection, **OpenAI CLIP** for visual feature extraction, and **DeepSORT** for multi-object tracking. As an extension, a custom **YOLOv7** model was trained on a fish-specific dataset to improve detection accuracy for aquarium videos.

---

## Work Completed

### 1. Environment Setup

- Set up the Zero-Shot Object Tracking repository.
- Installed all required dependencies.
- Configured a Python virtual environment.
- Enabled CUDA acceleration using the NVIDIA RTX 4050 Laptop GPU.

---

### 2. Compatibility Fixes

Resolved compatibility issues with recent versions of PyTorch and NumPy.

#### PyTorch
- Updated `torch.load()` to work with newer PyTorch versions.

#### NumPy
Replaced deprecated aliases throughout the project:

- `np.int` → `int`
- `np.float` → `float`

These changes restored compatibility with the latest software versions.

---

### 3. Zero-Shot Object Tracking

Implemented the complete tracking pipeline consisting of:

- **YOLOv5** for object detection
- **OpenAI CLIP** for appearance feature extraction
- **DeepSORT** for multi-object tracking

The system successfully tracked multiple objects across video frames while maintaining consistent object identities.

---

### 4. Initial Fish Video Testing

The original fish video was tested using the pretrained **YOLOv5 COCO** model.

#### Observation

The pretrained model detected fish with incorrect class labels (such as unrelated COCO object classes). This occurred because the COCO dataset does not contain fish-specific classes suitable for aquarium environments.

---

### 5. Custom Fish Detector using YOLOv7

To improve detection performance, a custom object detector was trained.

#### Dataset

- Downloaded the **Aquarium Dataset** from **Roboflow Universe**
- Exported the dataset in **YOLOv7** format

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv7 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Epochs | 50 |
| GPU | NVIDIA RTX 4050 Laptop GPU |

Training completed successfully and generated the final model weights (`best.pt`).

---

### 6. Retesting on Fish Video

The trained YOLOv7 model was evaluated on the original fish video.

#### Results

- Fish were detected correctly.
- Object labels matched the fish classes learned during training.
- Detection accuracy improved significantly compared to the pretrained YOLOv5 COCO model.
- The custom model produced more reliable and meaningful detections for aquarium scenes.

---

## Challenges Faced

During implementation, several technical challenges were encountered:

- PyTorch checkpoint compatibility issues
- Deprecated NumPy functions
- Dependency installation conflicts
- CLIP integration
- Training a custom YOLOv7 model
- Incorrect fish classification using the generic COCO model

All major compatibility issues were successfully resolved, resulting in a functional object detection and tracking pipeline.

---

## Technologies Used

- Python
- PyTorch
- YOLOv5
- YOLOv7
- OpenAI CLIP
- DeepSORT
- OpenCV
- CUDA
- NumPy
- Roboflow Universe

---

## Learning Outcomes

Through this project, I gained practical experience in:

- Real-time object detection
- Multi-object tracking
- Zero-shot visual feature extraction using CLIP
- Training custom YOLO models
- Preparing datasets for object detection
- Debugging compatibility issues in deep learning frameworks
- Improving model performance using domain-specific datasets

---

## Conclusion

This project successfully implemented a zero-shot object tracking system using **YOLOv5**, **OpenAI CLIP**, and **DeepSORT**. During initial testing, the pretrained YOLOv5 model incorrectly classified fish because it was trained on the generic COCO dataset. To overcome this limitation, a custom **YOLOv7** model was trained using the **Aquarium Dataset** from Roboflow Universe. After retraining, the model correctly detected fish in the test video, demonstrating that training on a domain-specific dataset significantly improves detection accuracy. This project provided valuable hands-on experience in modern computer vision, object detection, multi-object tracking, and custom deep learning model development.
