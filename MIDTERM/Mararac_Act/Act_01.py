import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "turtle.jpg"  

# Open the image using different flags
color = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)           # normal color
gray_flag = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)   # loads as gray
unchanged = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)   # keeps transparency if any

if color is None:
    raise FileNotFoundError(f"Could not open '{IMAGE_PATH}'. Check the file name or path.")

# Convert image to grayscale
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# Get the value of a pixel (OpenCV gives it in B, G, R order)
y, x = 50, 50
b, g, r = color[y, x]
print(f"Pixel at (x={x}, y={y}) -> R={r}, G={g}, B={b}")
print("Same pixel in grayscale:", gray[y, x])

# Convert the image to black and white
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Show everything in one figure
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.ravel()

axes[0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(gray_flag, cmap="gray")
axes[1].set_title("IMREAD_GRAYSCALE")

if unchanged.ndim == 3 and unchanged.shape[2] == 4:
    axes[2].imshow(cv2.cvtColor(unchanged, cv2.COLOR_BGRA2RGBA))
elif unchanged.ndim == 3:
    axes[2].imshow(cv2.cvtColor(unchanged, cv2.COLOR_BGR2RGB))
else:
    axes[2].imshow(unchanged, cmap="gray")
axes[2].set_title("IMREAD_UNCHANGED")

axes[3].imshow(gray, cmap="gray")
axes[3].set_title("Converted to grayscale")

axes[4].imshow(bw, cmap="gray")
axes[4].set_title("Black and white")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()