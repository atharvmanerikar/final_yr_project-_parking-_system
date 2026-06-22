"""
backend/utils/ocr.py

Enhanced OCR for Indian vehicle registration plates using the teammate's
exact preprocessing and correction logic from their college project.
"""

import os
import re
import cv2
import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO
from backend.config import settings

# Lazy-loaded models
_easyocr_reader = None
_alpr_yolo_model = None

def get_alpr_model() -> Optional[YOLO]:
    global _alpr_yolo_model
    if _alpr_yolo_model is None:
        model_path = settings.ALPR_MODEL
        if os.path.exists(model_path):
            try:
                _alpr_yolo_model = YOLO(model_path)
                print(f"[ALPR] Loaded custom plate detection model from {model_path}")
            except Exception as e:
                print(f"[ALPR] Error loading custom plate model: {e}")
        else:
            print(f"[ALPR] Custom plate model not found at {model_path}")
    return _alpr_yolo_model

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            print(f"[OCR] EasyOCR Reader initialized (GPU={use_gpu})")
        except ImportError:
            print("[OCR] EasyOCR library not found!")
    return _easyocr_reader


# Character correction mappings from teammate's project
dict_char_to_int = {
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
    'Z': '2', 'S': '5', 'G': '6', 'B': '8'
}

dict_int_to_char = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'
}


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
    if len(text) >= 10:
        prefix = text[:-4]
        suffix = text[-4:]
        corrected = ""
        for c in suffix:
            if c == 'I':
                corrected += '1'
            elif c == 'O':
                corrected += '0'
            else:
                corrected += c
        text = prefix + corrected

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


def read_license_plate(license_plate_crop) -> Tuple[Optional[str], float]:
    """Teammate's exact read_license_plate implementation."""
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, 0.0

    reader = get_easyocr_reader()
    if reader is None:
        return None, 0.0

    # Upscale by a fixed factor of 2 (fx=2, fy=2) using INTER_CUBIC
    plate = cv2.resize(
        license_plate_crop,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Sharpening
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    gray = cv2.filter2D(gray, -1, kernel)

    # Morph close
    kernel2 = np.ones((2, 2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)

    # Thresholding versions
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )

    versions = [plate, gray]
    best_text = None
    best_score = 0.0

    for img in versions:
        try:
            detections = reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,
                paragraph=False
            )
            for detection in detections:
                text = detection[1].upper().replace(" ", "")
                score = float(detection[2])

                text = re.sub(r'[^A-Z0-9]', '', text)
                text = correct_indian_plate(text)
                text = advanced_plate_fix(text)
                text = extract_indian_plate(text)

                if len(text) > 10 or len(text) < 6 or score < 0.35:
                    continue

                text = presentation_fix(text)
                if score > best_score:
                    best_text = text
                    best_score = score
        except Exception as e:
            print(f"[OCR] Error reading version: {e}")

    return best_text, best_score


def read_plate_two_stage(vehicle_crop: np.ndarray, expected_slot: str = None) -> Tuple[Optional[str], float]:
    """Teammate's exact two-stage flow."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, 0.0

    alpr_model = get_alpr_model()
    if alpr_model is not None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = alpr_model(vehicle_crop, verbose=False, device=device)[0]
            best_plate_crop = None
            best_yolo_conf = 0.0

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > best_yolo_conf:
                    h, w = vehicle_crop.shape[:2]
                    pad_x = int((x2 - x1) * 0.08)
                    pad_y = int((y2 - y1) * 0.12)

                    x1p = max(0, int(x1 - pad_x))
                    y1p = max(0, int(y1 - pad_y))
                    x2p = min(w, int(x2 + pad_x))
                    y2p = min(h, int(y2 + pad_y))

                    crop = vehicle_crop[y1p:y2p, x1p:x2p]
                    if crop.size > 0:
                        best_plate_crop = crop
                        best_yolo_conf = score

            if best_plate_crop is not None:
                plate_text, score = read_license_plate(best_plate_crop)
                if plate_text:
                    return plate_text, score
        except Exception as e:
            print(f"[ALPR] Custom plate detection failed: {e}")

    # Fallback to direct OCR on the whole vehicle crop
    return read_license_plate(vehicle_crop)


def crop_from_bbox(frame: np.ndarray, xyxy: list, padding: int = 8) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]
