#Preprocessing code goes here
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For Colab

# Load video
cap = cv2.VideoCapture("/content/testing_video.mp4")

# Get video properties 
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer for grayscale video
out = cv2.VideoWriter("/content/road_gray_output.mp4", fourcc, fps, (w, h), isColor=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hFrame, wFrame = frame.shape[:2]

   
    #  NEW ROI THAT KEEPS FULL ROAD (trapezoid shape)
    roi_points = np.array([
        [int(0.0 * wFrame), hFrame],              # Bottom-left
        [int(0.75 * wFrame), hFrame],              # Bottom-right
        [int(0.60 * wFrame), int(0.1 * hFrame)],  # Top-right
        [int(0.30 * wFrame), int(0.1 * hFrame)]   # Top-left
    ])

    # Create mask
    mask = np.zeros((hFrame, wFrame), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    # Apply ROI mask to frame
    maskFrame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale 
    grayFrame = cv2.cvtColor(maskFrame, cv2.COLOR_BGR2GRAY)

    # Write output video (now grayscale) 
    out.write(grayFrame)

cap.release()
out.release()

print("Road-only + Grayscale processing complete!")
print("Saved at: /content/road_gray_output.mp4")
