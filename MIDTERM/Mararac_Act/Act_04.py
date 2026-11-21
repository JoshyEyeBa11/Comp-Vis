# Activity 4: Original, grayscale, grayscale histogram, Sobel X edges
# Needs: pip install opencv-python matplotlib
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "woody.jpg"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not open '{IMAGE_PATH}'. Check the file name or path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

# Everything in one figure
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original (colored)")
axes[0, 0].axis("off")

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale")
axes[0, 1].axis("off")

axes[1, 0].hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
axes[1, 0].set_title("Histogram of grayscale image")
axes[1, 0].set_xlabel("Pixel value")
axes[1, 0].set_ylabel("Count")

axes[1, 1].imshow(sobel_x, cmap="gray")
axes[1, 1].set_title("Edges (Sobel X)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()