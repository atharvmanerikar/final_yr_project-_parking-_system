import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Point = Tuple[int, int]


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


def draw_results(img_bgr: np.ndarray, slots: List[dict], cars: List[dict], occ_threshold: float) -> Tuple[np.ndarray, List[dict]]:
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

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(marked_path):
        raise FileNotFoundError(f"Missing: {marked_path}")

    slots = load_marked_slots(marked_path)
    by_image = group_slots_by_image(slots)

    model_path = os.path.join(base, "yolov8n.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing: {model_path}")

    model = YOLO(model_path)

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
        vis, slot_results = draw_results(img, img_slots, cars, occ_threshold)

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
