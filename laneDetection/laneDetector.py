import sys
if 'ultralytics' not in sys.modules:
    !pip install ultralytics

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from ultralytics import RTDETR

# =========================
# Load RT-DETR
# =========================
model = RTDETR("rtdetr-l.pt")

vehicle_classes = [2,3,5,7]  # car, motorcycle, bus, truck

# =========================
# Video
# =========================
video_path = "/content/test.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("lane_violation_output.mp4", fourcc, fps, (width, height))

# =========================
# Pixel to Meter Conversion
# =========================
pixels_per_meter = 100

# =========================
# Vehicle Tracking Memory
# =========================
vehicle_positions = {}
next_vehicle_id = 0
max_match_distance = 30

# =========================
# Lane Speed Limits
# =========================
lane_limits = {

    1:(25,35),
    2:(35,45),
    3:(45,55),
    4:(55,65)

}

# =========================
# Lane Colors
# =========================
lane_colors = [

    ((0,0,255),"Lane1 25-35"),
    ((255,0,0),"Lane2 35-45"),
    ((0,255,0),"Lane3 45-55"),
    ((0,255,255),"Lane4 55-65")

]

# =========================
# Frame Loop
# =========================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    h,w = frame.shape[:2]

    # =========================
    # Lane Polygons
    # =========================
    lane4 = np.array([[(int(-0.12*w),h),(int(0.2*w),h),(int(0.41*w),int(0.05*h)),(int(0.385*w),int(0.05*h))]],np.int32)

    lane3 = np.array([[(int(0.2*w),h),(int(0.53*w),h),(int(0.446*w),int(0.05*h)),(int(0.41*w),int(0.05*h))]],np.int32)

    lane2 = np.array([[(int(0.53*w),h),(int(0.81*w),h),(int(0.485*w),int(0.05*h)),(int(0.446*w),int(0.05*h))]],np.int32)

    lane1 = np.array([[(int(0.81*w),h),(int(w),h),(int(0.53*w),int(0.05*h)),(int(0.485*w),int(0.05*h))]],np.int32)

    lanes=[lane1,lane2,lane3,lane4]

    mask = np.zeros_like(frame)

    # =========================
    # Draw Lanes
    # =========================
    for i,lane in enumerate(lanes):

        color,label = lane_colors[i]

        cv2.fillPoly(mask,lane,(255,255,255))
        cv2.polylines(overlay,lane,True,color,4)

        cx=int(np.mean(lane[0][:,0]))
        cy=int(np.mean(lane[0][:,1]))

        cv2.putText(
            overlay,
            label,
            (cx-60,cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3
        )

    masked_frame = cv2.bitwise_and(frame,mask)

    # =========================
    # RT-DETR Detection
    # =========================
    results = model(masked_frame)

    detections=[]

    for r in results:

        boxes=r.boxes.xyxy.cpu().numpy()
        classes=r.boxes.cls.cpu().numpy()

        for box,cls in zip(boxes,classes):

            if int(cls) in vehicle_classes:

                x1,y1,x2,y2 = map(int,box)

                cx=int((x1+x2)/2)
                cy=int((y1+y2)/2)

                detections.append((x1,y1,x2,y2,cx,cy))

    updated_positions = {}

    # =========================
    # Match Vehicles Between Frames
    # =========================
    for det in detections:

        x1,y1,x2,y2,cx,cy = det

        matched_id=None
        min_dist=999999

        for vid,(px,py) in vehicle_positions.items():

            dist=np.sqrt((cx-px)**2+(cy-py)**2)

            if dist < min_dist and dist < max_match_distance:

                min_dist=dist
                matched_id=vid

        if matched_id is None:

            matched_id=next_vehicle_id
            next_vehicle_id+=1

        prev_x,prev_y = vehicle_positions.get(matched_id,(cx,cy))

        # =========================
        # Speed Calculation
        # =========================
        distance_pixels = np.sqrt((cx-prev_x)**2+(cy-prev_y)**2)

        distance_meters = distance_pixels / pixels_per_meter

        speed = distance_meters * fps * 3.6

        updated_positions[matched_id]=(cx,cy)

        # =========================
        # Lane Detection
        # =========================
        lane_id=None

        for i,lane in enumerate(lanes):

            if cv2.pointPolygonTest(lane,(cx,cy),False)>=0:
                lane_id=i+1

        # =========================
        # Violation Detection
        # =========================
        violator=False

        # stopped vehicle
        if speed < 1:
            violator=True

        # lane speed violation
        if lane_id is not None:

            min_s,max_s = lane_limits[lane_id]

            if speed < min_s or speed > max_s:
                violator=True

        # between lane violation
        else:
            violator=True

        # =========================
        # Draw Bounding Box
        # =========================
        color=(0,0,255) if violator else (0,255,0)

        cv2.rectangle(
            overlay,
            (x1,y1),
            (x2,y2),
            color,
            4
        )

        # larger center pointer
        cv2.circle(
            overlay,
            (cx,cy),
            7,
            (255,255,255),
            -1
        )

        text=f"{int(speed)} km/h"

        if violator:
            text+=" VIOLATION"

        cv2.putText(
            overlay,
            text,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    vehicle_positions = updated_positions

    out.write(overlay)

    cv2_imshow(overlay)

cap.release()
out.release()

print("Output saved as lane_violation_output.mp4")
