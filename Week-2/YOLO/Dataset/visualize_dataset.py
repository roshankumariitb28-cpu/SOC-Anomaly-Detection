import glob as glob
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np

np.random.seed(42)

class_names = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Convert YOLO format to xmin, ymin, xmax, ymax
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):
    h, w, _ = image.shape

    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)

        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        class_name = class_names[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2
        )

        cv2.putText(
            image,
            class_name,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return image

def plot(image_paths, label_paths, num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)

    all_training_images.sort()
    all_training_labels.sort()

    num_images = len(all_training_images)

    plt.figure(figsize=(15, 12))

    for i in range(num_samples):
        j = random.randint(0, num_images - 1)

        image = cv2.imread(all_training_images[j])

        bboxes = []
        labels = []

        with open(all_training_labels[j], 'r') as f:
            label_lines = f.readlines()

            for label_line in label_lines:
                values = label_line.strip().split()

                label = int(values[0])

                x_c = float(values[1])
                y_c = float(values[2])
                w = float(values[3])
                h = float(values[4])

                bboxes.append([x_c, y_c, w, h])
                labels.append(label)

        result_image = plot_box(image, bboxes, labels)

        plt.subplot(2, 2, i + 1)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Show 4 random training images
plot(
    image_paths='train/images/*',
    label_paths='train/labels/*',
    num_samples=4
)