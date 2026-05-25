import cv2
from ultralytics import YOLO

# Load model (make sure path is correct)
model = YOLO("runs/detect/train4/weights/best.pt")

# Open video
cap = cv2.VideoCapture("sample1.mp4")

# FPS for smooth playback
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection (force CPU-safe inference)
    results = model.predict(frame, conf=0.3, verbose=False)

    # Draw results
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("License Plate Detection", annotated_frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()