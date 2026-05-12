import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, send_file, Response, request
from flask_cors import CORS
from ultralytics import YOLO

from parking import (
    PlateReader,
    ParkingDB,
    load_marked_slots,
    group_slots_by_image,
    corners_to_bounds,
    overlap_score,
    detect_cars_only,
    draw_results,
    shrink_bbox,
)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DB_PATH = os.path.join(BASE_DIR, "parking.db")

# Live processing globals
live_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
latest_frame: Optional[np.ndarray] = None
latest_stats: Dict[str, Any] = {}
camera_url: Optional[str] = None
model: Optional[YOLO] = None
plate_reader: Optional[PlateReader] = None
db: Optional[ParkingDB] = None
marked_slots: List[dict] = []
slots_by_image: Dict[str, List[dict]] = {}
occ_threshold = 0.10


def load_summary() -> Dict[str, Any]:
    path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parking_events(limit: int = 50) -> List[Dict[str, Any]]:
    import sqlite3
    events = []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT plate_number, slot_id, event_time, image_name FROM parking_events ORDER BY event_time DESC LIMIT ?",
            (limit,),
        )
        events = [dict(row) for row in cur]
    except Exception:
        pass
    finally:
        if "conn" in locals():
            conn.close()
    return events


def aggregate_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    total_slots = 0
    occupied = 0
    for img_info in summary.get("images", {}).values():
        if isinstance(img_info, dict):
            total_slots += img_info.get("total_slots", 0)
            occupied += img_info.get("occupied", 0)
    free = total_slots - occupied
    occupancy_rate = round((occupied / total_slots * 100) if total_slots else 0, 1)
    return {
        "total_slots": total_slots,
        "occupied": occupied,
        "free": free,
        "occupancy_rate": occupancy_rate,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }


def list_result_images() -> List[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    files = [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")) and f.startswith("result_")]
    files.sort(reverse=True)
    return files


def initialize_live_resources():
    global model, plate_reader, db, marked_slots, slots_by_image
    try:
        model_path = os.path.join(BASE_DIR, "yolov8n.pt")
        model = YOLO(model_path)
        plate_reader = PlateReader()
        db = ParkingDB(DB_PATH)
        marked_path = os.path.join(BASE_DIR, "marked_slots", "marked_slots.json")
        marked_slots = load_marked_slots(marked_path)
        slots_by_image = group_slots_by_image(marked_slots)
        return True
    except Exception as e:
        print(f"[init] Error: {e}")
        return False


def live_processing_loop():
    global latest_frame, latest_stats, camera_url
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print(f"[live] Failed to open camera: {camera_url}")
        return

    print(f"[live] Started processing from {camera_url}")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[live] Frame read failed, retrying...")
            time.sleep(1)
            continue

        # Use first available image's slots
        if not slots_by_image:
            time.sleep(1)
            continue
        img_name, img_slots = next(iter(slots_by_image.items()))

        cars = detect_cars_only(model, frame, conf=0.3)
        vis, slot_results = draw_results(
            frame,
            img_slots,
            cars,
            occ_threshold,
            plate_reader=plate_reader,
            db=db,
            image_name=img_name,
        )
        occupied = sum(1 for s in slot_results if s["occupied"])
        free = len(slot_results) - occupied
        latest_stats = {
            "total_slots": len(slot_results),
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round((occupied / len(slot_results) * 100) if slot_results else 0, 1),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }
        latest_frame = vis.copy()
        time.sleep(1)  # throttle to ~1 FPS
    cap.release()
    print("[live] Stopped")


def video_file_processing_loop(video_path: str):
    global latest_frame, latest_stats
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video] Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(0.5, 1.0 / min(fps, 2))  # cap to 2 FPS max
    print(f"[video] Started processing {video_path} at ~{1/interval:.1f} FPS")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[video] End of video, looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if not slots_by_image:
            time.sleep(0.1)
            continue
        img_name, img_slots = next(iter(slots_by_image.items()))

        cars = detect_cars_only(model, frame, conf=0.3)
        vis, slot_results = draw_results(
            frame,
            img_slots,
            cars,
            occ_threshold,
            plate_reader=plate_reader,
            db=db,
            image_name=img_name,
        )
        occupied = sum(1 for s in slot_results if s["occupied"])
        free = len(slot_results) - occupied
        latest_stats = {
            "total_slots": len(slot_results),
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round((occupied / len(slot_results) * 100) if slot_results else 0, 1),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }
        latest_frame = vis.copy()
        time.sleep(interval)
    cap.release()
    print("[video] Stopped")


@app.route("/status", methods=["GET"])
def status():
    if latest_stats:
        return jsonify(latest_stats)
    summary = load_summary()
    return jsonify(aggregate_status(summary))


@app.route("/results", methods=["GET"])
def results():
    files = list_result_images()
    return jsonify({"results": files})


@app.route("/result/<filename>", methods=["GET"])
def result_image(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        return "Not found", 404
    return send_file(path)


@app.route("/latest_result", methods=["GET"])
def latest_result():
    if latest_frame is not None:
        _, buf = cv2.imencode(".jpg", latest_frame)
        return Response(buf.tobytes(), mimetype="image/jpeg")
    files = list_result_images()
    if not files:
        return "No results", 404
    return send_file(os.path.join(RESULTS_DIR, files[0]))


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global live_thread, stop_event, camera_url
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "Missing 'url'"}), 400
    if live_thread and live_thread.is_alive():
        return jsonify({"error": "Camera already running"}), 409
    if not initialize_live_resources():
        return jsonify({"error": "Failed to initialize resources"}), 500
    camera_url = url
    stop_event.clear()
    live_thread = threading.Thread(target=live_processing_loop, daemon=True)
    live_thread.start()
    return jsonify({"status": "started", "url": url})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global live_thread, stop_event, latest_frame, latest_stats
    if not live_thread or not live_thread.is_alive():
        return jsonify({"status": "already_stopped"})
    stop_event.set()
    live_thread.join(timeout=3)
    latest_frame = None
    latest_stats = {}
    return jsonify({"status": "stopped"})


@app.route("/start_video_file", methods=["POST"])
def start_video_file():
    global live_thread, stop_event
    data = request.get_json(force=True, silent=True) or {}
    print(f"[start_video_file] received data: {data}")
    path = data.get("path")
    print(f"[start_video_file] extracted path: {path}")
    if not path or not os.path.isfile(path):
        print(f"[start_video_file] invalid/missing path: {path}")
        return jsonify({"error": "Invalid or missing 'path'"}), 400
    if live_thread and live_thread.is_alive():
        print("[start_video_file] already running")
        return jsonify({"error": "Already running"}), 409
    if not initialize_live_resources():
        print("[start_video_file] init failed")
        return jsonify({"error": "Failed to initialize resources"}), 500
    stop_event.clear()
    live_thread = threading.Thread(target=video_file_processing_loop, args=(path,), daemon=True)
    live_thread.start()
    print(f"[start_video_file] started thread for path: {path}")
    return jsonify({"status": "started", "path": path})


@app.route("/live_feed", methods=["GET"])
def live_feed():
    def generate():
        while True:
            if latest_frame is not None:
                _, buf = cv2.imencode(".jpg", latest_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            else:
                # placeholder frame
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/parked_vehicles", methods=["GET"])
def parked_vehicles():
    events = load_parking_events(limit=100)
    # Only return vehicles that are currently parked (latest per slot)
    latest_by_slot = {}
    for ev in events:
        slot = ev["slot_id"]
        if slot not in latest_by_slot:
            latest_by_slot[slot] = ev
    return jsonify({"vehicles": list(latest_by_slot.values())})


if __name__ == "__main__":
    print("Starting parking API server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
