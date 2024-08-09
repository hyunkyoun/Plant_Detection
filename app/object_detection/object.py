import torch
import cv2
from PIL import Image

# Load the YOLO 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Perform inference
    results = model(img)
    
    # Get unique object names
    objects = results.pandas().xyxy[0]['name'].unique().tolist()
    
    return objects

# Object detection
image_path = './object_test/test.jpg'
detected_objects = detect_objects(image_path)

print("Detected objects:")
for obj in detected_objects:
    print(f"- {obj}")