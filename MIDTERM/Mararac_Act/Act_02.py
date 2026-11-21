import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "why_me.png" 

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not open '{IMAGE_PATH}'. Check the file name or path.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


h, w = img.shape[:2]
print(f"Size: width = {w}, height = {h}")

# Value of a pixel (R, G, B)
y, x = 100, 100
y, x = min(y, h - 1), min(x, w - 1)   # makes sure the pixel exists
b, g, r = img[y, x]
print(f"Pixel at (x={x}, y={y}) -> R={r}, G={g}, B={b}")

# 4 different edges
sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
sobel_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
laplacian = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
canny = cv2.Canny(gray, 100, 200)   # change 100 and 200 to try different thresholds

# Separate the colors
b_ch, g_ch, r_ch = cv2.split(img)

fig2, ax2 = plt.subplots(1, 4, figsize=(16, 4))
ax2[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax2[0].set_title("Original")
ax2[1].imshow(r_ch, cmap="Reds")
ax2[1].set_title("Red")
ax2[2].imshow(g_ch, cmap="Greens")
ax2[2].set_title("Green")
ax2[3].imshow(b_ch, cmap="Blues")
ax2[3].set_title("Blue")
for ax in ax2:
    ax.axis("off")
plt.tight_layout()
plt.show()