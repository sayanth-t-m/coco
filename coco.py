from ultralytics import YOLO
import yaml
import cv2
import threading
import os
from datetime import datetime

def save_image(frame):
    # Save the frame as an image with a timestamp.
    save_path = 'CoconutDetection Pictures'  # Change folder name as desired.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f'coconut_detection_{timestamp}.jpg'
    image_path = os.path.join(save_path, image_name)
    cv2.imwrite(image_path, frame)
    print(f"Image saved: {image_path}")

# Load your custom model using the ultralytics YOLOv8 API.
# Replace the path below with your actual model file.
model = YOLO(r'best1.pt')

# Attempt to load custom class names from a YAML file.
# If not found, fallback to model.names.
yaml_path = 'AIYolov5/data.yaml'
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    classes = data['names']
    print(f"Loaded custom class names from {yaml_path}")
else:
    print(f"YAML file not found at {yaml_path}. Using default model class names.")
    classes = model.names

# Initialize video capture.
cap = cv2.VideoCapture(0)
cv2.namedWindow('Coconut Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Coconut Detection', 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference using the YOLOv8 model.
    # model(frame) returns a list of results (one per image).
    results = model(frame)
    if results:
        # Use the first result.
        result = results[0]
        # YOLOv8 result boxes can be accessed via result.boxes.
        for box in result.boxes:
            # box.xyxy is a tensor with coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy.cpu().numpy()[0]
            confidence = box.conf.cpu().numpy()[0]
            cls = int(box.cls.cpu().numpy()[0])
            if confidence > 0.67:
                # Use custom class names if available.
                class_name = classes[cls] if cls < len(classes) else str(cls)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(xyxy[0]), int(xyxy[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # Launch image saving in a separate thread.
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
