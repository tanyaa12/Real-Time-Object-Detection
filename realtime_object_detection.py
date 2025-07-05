import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Start webcam
cap = cv2.VideoCapture(0)

# Your screen resolution (adjust if needed)
screen_width = 1366
screen_height = 768
screen_aspect = screen_width / screen_height

# OpenCV full screen window
cv2.namedWindow('Real-Time Object Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Real-Time Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get camera feed resolution
    h, w = frame.shape[:2]
    cam_aspect = w / h

    # Crop or pad to match screen aspect
    if cam_aspect > screen_aspect:
        # Camera is wider than screen – crop sides
        new_width = int(h * screen_aspect)
        x1 = (w - new_width) // 2
        frame = frame[:, x1:x1+new_width]
    else:
        # Camera is taller – crop top/bottom
        new_height = int(w / screen_aspect)
        y1 = (h - new_height) // 2
        frame = frame[y1:y1+new_height, :]

    # Resize to fit full screen
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Object detection
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    # Show full screen frame
    cv2.imshow('Real-Time Object Detection', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
