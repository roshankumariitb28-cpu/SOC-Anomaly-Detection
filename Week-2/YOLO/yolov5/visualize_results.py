import cv2
import glob
import matplotlib.pyplot as plt

# Show validation prediction images
def show_valid_results(res_dir):
    pred_images = glob.glob(f"runs/train/{res_dir}/*_pred.jpg")

    print("Found images:")   
    for img_path in pred_images:
        print(img_path)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

                
# Show inference output images
def visualize(infer_dir):
    infer_images = glob.glob(f"runs/detect/{infer_dir}/*.jpg")

    print("Found images:")
    for img_path in infer_images:
        print(img_path)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


# Example usage
show_valid_results("exp2")

# Example:
# visualize("exp")
