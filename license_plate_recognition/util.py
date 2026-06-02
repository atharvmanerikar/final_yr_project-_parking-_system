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
# Character correction mappings
# -----------------------------
dict_char_to_int = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}

# -----------------------------
# Indian License Plate Format
# -----------------------------
def license_complies_format(text):

    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$'

    return re.match(pattern, text) is not None

def presentation_fix(text):

    if text is None:
        return None

    if text.startswith("64"):
        text = "GA" + text[2:]

    if text.startswith("6A"):
        text = "GA" + text[2:]

    text = text.replace("OJ", "03")

    return text

def correct_indian_plate(text):

    chars = list(text)

    for i in range(len(chars)):

        # First 2 positions should be letters
        if i < 2:

            if chars[i] == '6':
                chars[i] = 'G'

            elif chars[i] == '0':
                chars[i] = 'O'

            elif chars[i] == '1':
                chars[i] = 'I'

        # State code digits
        elif i < 4:

            if chars[i] in dict_char_to_int:
                chars[i] = dict_char_to_int[chars[i]]

        # Series letters
        elif i < len(chars) - 4:

            if chars[i] in dict_int_to_char:
                chars[i] = dict_int_to_char[chars[i]]

        # Last 4 digits
        else:

            if chars[i] in dict_char_to_int:
                chars[i] = dict_char_to_int[chars[i]]

    return ''.join(chars)
def advanced_plate_fix(text):

    chars = list(text)

    # Position 0-1 = State letters
    for i in range(min(2, len(chars))):

        if chars[i] == '6':
            chars[i] = 'G'

        elif chars[i] == '0':
            chars[i] = 'O'

        elif chars[i] == '1':
            chars[i] = 'I'

        elif chars[i] == '4':
            chars[i] = 'A'

    text = ''.join(chars)

    # Common Goa corrections
    text = text.replace("64", "GA")
    text = text.replace("6A", "GA")
    text = text.replace("G4", "GA")

    # District code mistakes
    text = text.replace("OJ", "03")
    text = text.replace("OI", "01")
    text = text.replace("OZ", "02")

    # Series mistakes
    text = text.replace("JAF", "AF")
    text = text.replace("UH", "4H")

    # Number section
    text = text.replace("S", "5")
    text = text.replace("B", "8")
    text = text.replace("O", "0")

    # Goa specific

    text = text.replace("0AF", "03AF")
    text = text.replace("0J", "03")
    text = text.replace("OJ", "03")

    text = text.replace("KP", "MP")
    text = text.replace("AH", "MH")

    text = text.replace("YC", "Y2")


    return text

def extract_indian_plate(text):

    patterns = [

        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}',
        r'[A-Z]{2}[0-9]{1}[A-Z]{1,2}[0-9]{4}',
        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3}'
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group()

    return text
# -----------------------------
# OCR License Plate
# -----------------------------
def read_license_plate(license_plate_crop):

    if license_plate_crop is None:
        return None, None

    if license_plate_crop.size == 0:
        return None, None

    # upscale
    plate = cv2.resize(
        license_plate_crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    # denoise
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # sharpen
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    gray = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    # morphology
    kernel2 = np.ones((2, 2), np.uint8)

    gray = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel2
    )

    # threshold versions
    _, th1 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, th2 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    # IMPORTANT: include original color image
    versions = [
        plate,
        gray,
        th1,
        th2,
        adaptive
    ]

    best_text = None
    best_score = 0

    for img in versions:

        detections = reader.readtext(
            img,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=1,
            paragraph=False
        )

        for detection in detections:

            text = detection[1]
            score = detection[2]

            text = text.upper()
            text = text.replace(" ", "")
            text = re.sub(
                r'[^A-Z0-9]',
                '',
                text
            )
            text = correct_indian_plate(text)

            text = advanced_plate_fix(text)

            text = extract_indian_plate(text)

            if len(text) > 10:
                continue

            # keep OCR permissive
            if score < 0.35:
                continue

            if len(text) < 6:
                continue

            text = presentation_fix(text)

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
    if len(plate_history[car_id]) > 30:
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