import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Point = Tuple[int, int]


class ParkingDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parking_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    slot_id INTEGER NOT NULL,
                    event_time TEXT NOT NULL,
                    image_name TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def insert_event(self, plate_number: str, slot_id: int, image_name: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO parking_events (plate_number, slot_id, event_time, image_name) VALUES (?, ?, ?, ?)",
                (plate_number, slot_id, datetime.now().isoformat(timespec="seconds"), image_name),
            )
            conn.commit()
        finally:
            conn.close()


class PlateReader:
    def __init__(self):
        self.reader = None
        self.available = False
        try:
            import easyocr  # type: ignore

            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self.available = True
        except Exception:
            self.reader = None
            self.available = False

    def _extract_plate_roi(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        try:
            # Plates are often in the lower half of the car ROI
            h0, w0 = roi_bgr.shape[:2]
            crop = roi_bgr[int(h0 * 0.45):h0, 0:w0]

            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None

            h_img, w_img = roi_bgr.shape[:2]
            best = None
            best_score = 0.0

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w <= 0 or h <= 0:
                    continue

                area = w * h
                if area < 0.01 * (w_img * h_img):
                    continue

                aspect = w / float(h)
                if aspect < 2.0 or aspect > 6.5:
                    continue

                if w < 60 or h < 18:
                    continue

                score = area * (1.0 - abs(aspect - 3.5) / 3.5)
                if score > best_score:
                    best_score = score
                    best = (x, y, w, h)

            if best is None:
                # Retry in the lower-half crop
                roi_bgr = crop
                if roi_bgr is None or roi_bgr.size == 0:
                    return None

                gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                edges = cv2.Canny(gray, 50, 150)
                edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    return None

                h_img, w_img = roi_bgr.shape[:2]
                best = None
                best_score = 0.0
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    if w <= 0 or h <= 0:
                        continue

                    area = w * h
                    if area < 0.01 * (w_img * h_img):
                        continue

                    aspect = w / float(h)
                    if aspect < 2.0 or aspect > 6.5:
                        continue

                    if w < 60 or h < 18:
                        continue

                    score = area * (1.0 - abs(aspect - 3.5) / 3.5)
                    if score > best_score:
                        best_score = score
                        best = (x, y, w, h)

                if best is None:
                    return None

            x, y, w, h = best
            pad_x = int(round(w * 0.08))
            pad_y = int(round(h * 0.20))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            plate = roi_bgr[y1:y2, x1:x2]
            return plate if plate.size else None
        except Exception:
            return None

    def read_plate(self, roi_bgr: np.ndarray) -> str:
        if not self.available or self.reader is None or roi_bgr.size == 0:
            return "UNKNOWN"

        try:
            plate_roi = self._extract_plate_roi(roi_bgr)
            if plate_roi is None:
                plate_roi = roi_bgr

            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

            # Multi-pass OCR with stronger preprocessing
            candidates: List[np.ndarray] = []
            candidates.append(gray)

            # upscale for better OCR
            h, w = gray.shape[:2]
            scale = 2.0 if max(h, w) < 220 else 1.5
            up = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            candidates.append(up)

            # thresholded versions
            thr1 = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
            thr2 = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
            candidates.append(thr1)
            candidates.append(thr2)

            best_text = "UNKNOWN"
            best_conf = 0.0

            for img in candidates:
                texts = self.reader.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                for t in texts:
                    if len(t) < 3:
                        continue
                    txt = t[1] if len(t) > 1 else ""
                    conf = float(t[2]) if len(t) > 2 else 0.0
                    txt = "".join(ch for ch in txt.upper() if ch.isalnum())
                    if len(txt) < 6:
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        best_text = txt

            return best_text
        except Exception:
            return "UNKNOWN"


def shrink_bbox(b: Tuple[int, int, int, int], shrink: float = 0.15) -> Tuple[int, int, int, int]:
    """Shrink bbox on all sides by `shrink` fraction (0.15 => 15%)."""
    x1, y1, x2, y2 = b
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    dx = int(round(w * shrink / 2.0))
    dy = int(round(h * shrink / 2.0))
    nx1, ny1 = x1 + dx, y1 + dy
    nx2, ny2 = x2 - dx, y2 - dy
    if nx2 <= nx1 or ny2 <= ny1:
        return b
    return (nx1, ny1, nx2, ny2)


def load_marked_slots(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("marked_slots.json must be a list")
    return data


def group_slots_by_image(slots: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for s in slots:
        img = s.get("image_name")
        if not img:
            continue
        out.setdefault(img, []).append(s)
    for img in out:
        out[img].sort(key=lambda x: int(x.get("slot_id", 0)))
    return out


def corners_to_bounds(corners: List[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def bbox_intersection(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def overlap_score(car_bbox: Tuple[int, int, int, int], slot_corners: List[Point]) -> float:
    """Intersection(slot_bbox, car_bbox) / car_area."""
    car_bbox = shrink_bbox(car_bbox, shrink=0.15)
    slot_bbox = corners_to_bounds(slot_corners)
    inter = bbox_intersection(car_bbox, slot_bbox)
    area = bbox_area(car_bbox)
    if area <= 0:
        return 0.0
    return float(inter / area)


def detect_cars_only(model: YOLO, img_bgr: np.ndarray, conf: float = 0.3) -> List[dict]:
    results = model(img_bgr, conf=conf, iou=0.5, verbose=False)
    cars: List[dict] = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls != 2:  # COCO car
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cars.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
            })

    return cars


def draw_results(
    img_bgr: np.ndarray,
    slots: List[dict],
    cars: List[dict],
    occ_threshold: float,
    plate_reader: Optional[PlateReader] = None,
    db: Optional[ParkingDB] = None,
    image_name: str = "",
) -> Tuple[np.ndarray, List[dict]]:
    out = img_bgr.copy()

    used_cars = set()
    slot_results: List[dict] = []

    for slot in slots:
        corners = [tuple(p) for p in slot["corners"]]
        sid = int(slot.get("slot_id", 0))

        best_idx = None
        best_score = 0.0
        for i, c in enumerate(cars):
            if i in used_cars:
                continue
            s = overlap_score(c["bbox"], corners)
            if s > best_score:
                best_score = s
                best_idx = i

        occupied = best_idx is not None and best_score >= occ_threshold
        if occupied:
            used_cars.add(best_idx)

        plate_number = None
        bbox = None
        if occupied and best_idx is not None:
            bbox = cars[best_idx]["bbox"]
            x1, y1, x2, y2 = bbox
            roi = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if plate_reader is not None:
                plate_number = plate_reader.read_plate(roi)
            else:
                plate_number = "UNKNOWN"

            if db is not None:
                db.insert_event(plate_number=plate_number or "UNKNOWN", slot_id=sid, image_name=image_name)

        poly = np.array(corners, dtype=np.int32)
        color = (0, 0, 255) if occupied else (0, 255, 0)

        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.polylines(out, [poly], True, color, 3)

        cx = int(np.mean([p[0] for p in corners]))
        cy = int(np.mean([p[1] for p in corners]))
        label = f"{sid} {'OCCUPIED' if occupied else 'FREE'}"
        cv2.putText(out, label, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(out, label, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        slot_results.append({
            "slot_id": sid,
            "occupied": bool(occupied),
            "overlap": float(best_score),
            "plate_number": plate_number if occupied else None,
            "bbox": bbox if occupied else None,
        })

    # Draw car boxes (optional)
    for c in cars:
        x1, y1, x2, y2 = shrink_bbox(c["bbox"], shrink=0.15)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)

    return out, slot_results


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    marked_path = os.path.join(base, "marked_slots", "marked_slots.json")
    dataset_dir = os.path.join(base, "Dataset")
    results_dir = os.path.join(base, "results")
    db_path = os.path.join(base, "parking.db")

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(marked_path):
        raise FileNotFoundError(f"Missing: {marked_path}")

    slots = load_marked_slots(marked_path)
    by_image = group_slots_by_image(slots)

    model_path = os.path.join(base, "yolov8n.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing: {model_path}")

    model = YOLO(model_path)
    db = ParkingDB(db_path)
    plate_reader = PlateReader()

    occ_threshold = 0.10  # 10%

    summary: Dict[str, dict] = {
        "occupancy_threshold": occ_threshold,
        "images": {},
    }

    for img_name, img_slots in by_image.items():
        img_path = os.path.join(dataset_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            summary["images"][img_name] = {"error": "IMAGE_NOT_FOUND", "path": img_path}
            continue

        cars = detect_cars_only(model, img, conf=0.3)
        vis, slot_results = draw_results(
            img,
            img_slots,
            cars,
            occ_threshold,
            plate_reader=plate_reader,
            db=db,
            image_name=img_name,
        )

        occupied = sum(1 for s in slot_results if s["occupied"])
        free = len(slot_results) - occupied

        out_img = os.path.join(results_dir, f"result_{img_name}")
        cv2.imwrite(out_img, vis)

        summary["images"][img_name] = {
            "total_slots": len(slot_results),
            "occupied": occupied,
            "free": free,
            "cars_detected": len(cars),
            "result_image": out_img,
            "slots": slot_results,
        }

    out_json = os.path.join(results_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved results to:")
    print("-", results_dir)
    print("-", out_json)


if __name__ == "__main__":
    main()
