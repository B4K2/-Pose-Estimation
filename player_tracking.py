import cv2
import pandas as pd
from ultralytics import YOLO

VIDEO_PATH = 'video/input/testing.mp4'

model = YOLO('model/yolo26n-pose.pt') 

cap = cv2.VideoCapture(VIDEO_PATH)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('video/output/output_overlay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

all_keypoints_data = []

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.5)
    annotated_frame = results[0].plot(boxes=False, labels=False, conf=False)

    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        kps = results[0].keypoints.xy[0].cpu().numpy()
        frame_data = {'frame': frame_idx}

        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        for i, (x, y) in enumerate(kps):
            frame_data[f'{keypoint_names[i]}_x'] = x
            frame_data[f'{keypoint_names[i]}_y'] = y
            
        all_keypoints_data.append(frame_data)

        out.write(annotated_frame)
    
    cv2.imshow('YOLO Pose Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

df = pd.DataFrame(all_keypoints_data)
df.to_csv('keypoints.csv', index=False)