import cv2
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep'] 
classifier = models.resnet18(pretrained=False)
num_ftrs = classifier.fc.in_features
classifier.fc = nn.Linear(num_ftrs, len(class_names))

try:
    classifier.load_state_dict(torch.load('model/cricket_shot_classifier.pth', map_location=device))
    print("Classifier loaded successfully.")
except FileNotFoundError:
    print("Error: 'cricket_shot_classifier.pth' not found. Please train the model first.")
    exit()

classifier = classifier.to(device)
classifier.eval() 

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


VIDEO_PATH = 'video/input/testing.mp4'
pose_model = YOLO('model/yolo26n-pose.pt')

cap = cv2.VideoCapture(VIDEO_PATH)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('video/output/output_overlay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

all_keypoints_data = []
frame_idx = 0
current_action = "Stance" 


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 5 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classifier(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
            if confidence.item() > 0.70:
                current_action = f"{class_names[predicted_idx.item()]} ({confidence.item()*100:.0f}%)"
            else:
                current_action = "Stance / Ready"

    results = pose_model(frame, verbose=False, conf=0.5)
    annotated_frame = results[0].plot(boxes=False, labels=False, conf=False)

    cv2.rectangle(annotated_frame, (10, 10), (350, 60), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Action: {current_action}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        kps = results[0].keypoints.xy[0].cpu().numpy()
        frame_data = {'frame': frame_idx, 'action_detected': current_action}

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
    
    cv2.imshow('Analysis', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()