from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv, get_best_plate

results = {}

mot_tracker = Sort()

# Load models
vehicle_detector = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')

# Video
cap = cv2.VideoCapture('sample1.mp4')

vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:
        results[frame_nmr] = {}

        # -------------------------
        # Vehicle Detection
        # -------------------------
        detections = vehicle_detector(frame)[0]
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # -------------------------
        # Vehicle Tracking
        # -------------------------
        track_ids = mot_tracker.update(np.asarray(detections_))

        # -------------------------
        # License Plate Detection
        # -------------------------
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = license_plate

            # Assign plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                license_plate,
                track_ids
            )

            if car_id != -1:

                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # Read plate text
                license_plate_text, license_plate_text_score = read_license_plate(
                    license_plate_crop
                )

                # -------------------------
                # Temporal Voting (NEW)
                # -------------------------
                license_plate_text = get_best_plate(car_id, license_plate_text)

                if license_plate_text is not None:

                    results[frame_nmr][car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2]
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

# Save results
write_csv(results, 'test.csv')

cap.release()