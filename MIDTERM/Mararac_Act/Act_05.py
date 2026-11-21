import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = "negus.jpg" 

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not open '{IMAGE_PATH}'. Check the file name or path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)

# 1. Everything in one figure
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original (RGB)")
axes[0, 0].axis("off")

axes[0, 1].imshow(canny, cmap="gray")
axes[0, 1].set_title("Edges (Canny)")
axes[0, 1].axis("off")

axes[1, 0].imshow(gray, cmap="gray")
axes[1, 0].set_title("Grayscale")
axes[1, 0].axis("off")

axes[1, 1].hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
axes[1, 1].set_title("Grayscale histogram")
axes[1, 1].set_xlabel("Pixel value")
axes[1, 1].set_ylabel("Count")

plt.tight_layout()

pil_img = Image.open(IMAGE_PATH)    # PIL tells us the mode (RGB, L, etc.)
width, height = pil_img.size
file_size_kb = os.path.getsize(IMAGE_PATH) / 1024

print("--- Image properties ---")
print("Filename :", os.path.basename(IMAGE_PATH))
print("Format   :", pil_img.mode, f"(file type: {pil_img.format})")
print("Width    :", width, "pixels")
print("Height   :", height, "pixels")
print(f"Size     : {file_size_kb:.2f} KB")

# 3. Get the value of a pixel
y, x = 100, 100
y, x = min(y, height - 1), min(x, width - 1)   # makes sure the pixel exists
b, g, r = img[y, x]
print(f"Pixel at (x={x}, y={y}) -> R={r}, G={g}, B={b}")

plt.show()