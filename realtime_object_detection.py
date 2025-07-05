import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv5 model (YOLOv8 also works)
model = YOLO('yolov8n.pt')  # 'n' = nano model, fast and small

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Real-Time Object Detection', annotated_frame)

    # Break when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
