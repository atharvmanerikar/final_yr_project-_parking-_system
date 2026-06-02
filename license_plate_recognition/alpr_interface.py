from ultralytics import YOLO
from util import read_license_plate
import cv2

plate_detector = YOLO("best.pt")


def detect_plate(frame):

    results = plate_detector(frame)[0]

    detections = []

    for result in results.boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = result

        plate_crop = frame[
            int(y1):int(y2),
            int(x1):int(x2)
        ]

        plate_text, confidence = read_license_plate(
            plate_crop
        )

        if plate_text:

            detections.append({
                "plate": plate_text,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    return detections
