import easyocr
import cv2
import numpy as np
import re
import csv
from collections import defaultdict, Counter

# initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

# OCR history for temporal voting
plate_history = defaultdict(list)


# -----------------------------
# Indian License Plate Format
# -----------------------------
def license_complies_format(text):

    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$'

    return re.match(pattern, text) is not None


# -----------------------------
# OCR License Plate
# -----------------------------
def read_license_plate(license_plate_crop):

    if license_plate_crop is None:
        return None, None

    if license_plate_crop.size == 0:
        return None, None

    # enlarge plate
    license_plate_crop = cv2.resize(
        license_plate_crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # grayscale
    gray = cv2.cvtColor(
        license_plate_crop,
        cv2.COLOR_BGR2GRAY
    )

    # denoise
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
    )

    # improve contrast
    gray = cv2.equalizeHist(gray)

    # simple threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR
    detections = reader.readtext(
        thresh,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1,
        paragraph=False
    )

    best_text = None
    best_score = 0

    for detection in detections:

        text = detection[1]
        score = detection[2]

        # uppercase
        text = text.upper()

        # remove spaces
        text = text.replace(" ", "")

        # remove special chars
        text = re.sub(
            r'[^A-Z0-9]',
            '',
            text
        )

        # low confidence skip
        if score < 0.10:
            continue

        # validate format
        if license_complies_format(text):

            if score > best_score:

                best_text = text
                best_score = score

    return best_text, best_score


# -----------------------------
# Temporal Voting
# -----------------------------
def get_best_plate(car_id, text):

    if text is None:
        return None

    plate_history[car_id].append(text)

    # keep only recent predictions
    if len(plate_history[car_id]) > 10:
        plate_history[car_id].pop(0)

    # choose most common prediction
    most_common = Counter(
        plate_history[car_id]
    ).most_common(1)

    return most_common[0][0]


# -----------------------------
# Match plate to vehicle
# -----------------------------
def get_car(license_plate, track_ids):

    x1, y1, x2, y2, score, class_id = license_plate

    for track in track_ids:

        xcar1, ycar1, xcar2, ycar2, car_id = track

        if (
            x1 > xcar1 and
            y1 > ycar1 and
            x2 < xcar2 and
            y2 < ycar2
        ):

            return (
                xcar1,
                ycar1,
                xcar2,
                ycar2,
                car_id
            )

    return -1, -1, -1, -1, -1


# -----------------------------
# Save Results CSV
# -----------------------------
def write_csv(results, output_path):

    with open(output_path, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow([
            'frame_nmr',
            'car_id',
            'car_bbox',
            'license_plate_bbox',
            'license_plate_text',
            'bbox_score',
            'text_score'
        ])

        for frame_nmr in results.keys():

            for car_id in results[frame_nmr].keys():

                car = results[frame_nmr][car_id]

                writer.writerow([
                    frame_nmr,
                    car_id,
                    car['car']['bbox'],
                    car['license_plate']['bbox'],
                    car['license_plate']['text'],
                    car['license_plate']['bbox_score'],
                    car['license_plate']['text_score']
                ])