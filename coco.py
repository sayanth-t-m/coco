import torch
import yaml
import cv2
import threading
import os
from datetime import datetime
import pathlib

# Patch for Windows to handle PosixPath issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def save_image(frame):
    # Save the frame as an image with a timestamp.
    save_path = 'CoconutDetection Pictures'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f'coconut_detection_{timestamp}.jpg'
    image_path = os.path.join(save_path, image_name)
    cv2.imwrite(image_path, frame)
    print(f"Image saved: {image_path}")

# Load YOLOv5 model using torch.hub with force_reload=True to refresh cache
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path=r'C:\Users\sayan\Downloads\coco\AIYolov5\content\yolov5\runs\train\yolov5s_results\weights\best.pt', 
                       force_reload=True)

# Attempt to load custom class names from a YAML file.
yaml_path = 'AIYolov5/data.yaml'
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    classes = data['names']
    print(f"Loaded custom class names from {yaml_path}")
else:
    print(f"YAML file not found at {yaml_path}. Using default model class names.")
    # YOLOv5 typically stores class names in model.names.
    classes = model.names if hasattr(model, 'names') else []

# Initialize video capture.
cap = cv2.VideoCapture(0)
cv2.namedWindow('Coconut Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Coconut Detection', 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference on the frame.
    results = model(frame)
    # results.xyxy[0] contains a tensor with detections: [x1, y1, x2, y2, confidence, class]
    detections = results.xyxy[0].cpu().numpy()  # Convert to NumPy array

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.67:
            class_idx = int(cls)
            class_name = classes[class_idx] if class_idx < len(classes) else str(class_idx)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # Save image asynchronously.
            threading.Thread(target=save_image, args=(frame.copy(),), daemon=True).start()

    cv2.imshow('Coconut Detection', frame)
    key = cv2.waitKey(1)
    if key in [ord('q'), ord('Q'), 27]:  # Quit on Q, q, or Esc.
        break
    elif key in [ord('c'), ord('C')]:      # Close window on C or c.
        cv2.destroyAllWindows()
        break
    elif key in [ord('m'), ord('M')]:      # Maximize window on M or m.
        cv2.setWindowProperty('Coconut Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key in [ord('n'), ord('N')]:      # Normalize window on N or n.
        cv2.setWindowProperty('Coconut Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()