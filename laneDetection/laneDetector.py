#Lane Detection 

from google.colab.patches import cv2_imshow
import cv2
import numpy as np

# =========================
# Image Path
# =========================
image_path = "/content/road.jpg"

frame = cv2.imread(image_path)
height, width = frame.shape[:2]

overlay = frame.copy()

# =========================
# Lane Colors & Speeds
# =========================
lane_colors = [
    ((0,255,255),"30 km/h"),
    ((0,255,0),"40 km/h"),
    ((255,0,0),"50 km/h"),
    ((0,0,255),"60 km/h")
]

# =========================
# Hard-coded Lane Polygons
# =========================

lane4 = np.array([[
    (int(-0.12*width), height), # Bottom-left
    (int(0.19*width), height), # Bottom-right
    (int(0.41*width), int(0.05*height)), # top-right
    (int(0.385*width), int(0.05*height)) # top-left
]], np.int32)

lane3 = np.array([[
    (int(0.19*width), height), # Bottom-left
    (int(0.53*width), height), # Bottom-right
    (int(0.446*width), int(0.05*height)), # top-right
    (int(0.41*width), int(0.05*height)) # top-left
]], np.int32)

lane2 = np.array([[
    (int(0.53*width), height), # bottom-left
    (int(0.81*width), height), # bottom-right
    (int(0.485*width), int(0.05*height)), # top-right
    (int(0.446*width), int(0.05*height)) # top-left
]], np.int32)

lane1 = np.array([[
    (int(0.81*width), height), # bottom-left
    (int(width), height), # bottom-right
    (int(0.53*width), int(0.05*height)), # top-right
    (int(0.485*width), int(0.05*height)) # top-left
]], np.int32)

lanes = [lane1, lane2, lane3, lane4]

# =========================
# Draw Lane Masks
# =========================
for i, lane in enumerate(lanes):

    color, speed = lane_colors[i]

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, lane, color)

    overlay = cv2.addWeighted(overlay, 1.0, mask, 0.25, 0)

    cx = int(np.mean(lane[0][:,0]))
    cy = int(np.mean(lane[0][:,1]))

    cv2.putText(
        overlay,
        speed,
        (cx-40, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA
    )

# =========================
# Show Output
# =========================
cv2_imshow(overlay)

cv2.imwrite("lane_mask_output.jpg", overlay)

print("Output saved as lane_mask_output.jpg")

