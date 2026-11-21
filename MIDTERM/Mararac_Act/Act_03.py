import os
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH_1 = "frog.jpg" 
IMAGE_PATH_2 = "lebron.jpg"  

paths = [IMAGE_PATH_1, IMAGE_PATH_2]

# One figure. Each row is one image:
# original | grayscale histogram | red | green | blue histograms
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for row, path in enumerate(paths):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open '{path}'. Check the file name or path.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Original image
    axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, 0].set_title("Original: " + os.path.basename(path))
    axes[row, 0].axis("off")

    # Histogram of the whole image (grayscale)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    axes[row, 1].plot(hist_gray, color="black")
    axes[row, 1].set_title("Histogram (grayscale)")
    axes[row, 1].set_xlim([0, 256])

    # Separate R, G, B histograms (OpenCV order is B=0, G=1, R=2)
    channels = [(2, "red", "Red"), (1, "green", "Green"), (0, "blue", "Blue")]
    for col, (idx, color, name) in enumerate(channels, start=2):
        hist = cv2.calcHist([img], [idx], None, [256], [0, 256])
        axes[row, col].plot(hist, color=color)
        axes[row, col].set_title(name + " histogram")
        axes[row, col].set_xlim([0, 256])

plt.tight_layout()
plt.show()