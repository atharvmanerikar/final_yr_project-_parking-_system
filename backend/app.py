import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import heapq

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
from vehicle_tracker import VehicleTracker

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
tracker: Optional[VehicleTracker] = None
marked_slots: List[dict] = []
slots_by_image: Dict[str, List[dict]] = {}
occ_threshold = 0.10
GRAPH_PATH = os.path.join(BASE_DIR, "marked_slots", "parking_slots.json")

parking_graph = {}

if os.path.exists(GRAPH_PATH):
    with open(GRAPH_PATH, "r") as f:
        parking_graph = json.load(f)


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

def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def dijkstra(graph, nodes, start, end):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end:
            return path, cost
        neighbors = graph.get(node, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                current_pos = nodes[node]
                neighbor_pos = nodes[neighbor]
                distance = calculate_distance(
                    current_pos,
                    neighbor_pos
                )
                heapq.heappush(
                    queue,
                    (
                        cost + distance,
                        neighbor,
                        path
                    )
                )
    return [], float("inf")

def list_result_images() -> List[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    files = [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")) and f.startswith("result_")]
    files.sort(reverse=True)
    return files

def initialize_live_resources():
    global model, plate_reader, db, tracker, marked_slots, slots_by_image
    try:
        model_path = os.path.join(BASE_DIR, "yolov8n.pt")
        model = YOLO(model_path)
        plate_reader = PlateReader()
        db = ParkingDB(DB_PATH)
        tracker = VehicleTracker(track_thresh=0.3, track_buffer=30, db=db)
        marked_path = os.path.join(BASE_DIR, "marked_slots", "marked_slots.json")
        marked_slots = load_marked_slots(marked_path)
        slots_by_image = group_slots_by_image(marked_slots)
        return True
    except Exception as e:
        print(f"[init] Error: {e}")
        return False

def post_process_tracking(frame_vis: np.ndarray, cars_list: List[dict], slot_results_list: List[dict]) -> np.ndarray:
    global tracker
    if tracker is None:
        return frame_vis
        
    # Update tracker with detections
    active_tracks = tracker.update(cars_list)
    current_time = datetime.now()
    assigned_track_ids = set()
    
    # Map occupied slots to active tracks using bbox overlap
    for res in slot_results_list:
        if res.get("occupied") and res.get("bbox") is not None:
            best_track = None
            best_overlap = 0.0
            slot_bbox = res["bbox"]
            
            for track in active_tracks:
                overlap = tracker._calculate_iou(track.bbox, slot_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_track = track
            
            if best_track is not None and best_overlap > 0.1:
                tracker.assign_parking_slot(best_track.track_id, res["slot_id"], current_time)
                assigned_track_ids.add(best_track.track_id)
                if res.get("plate_number") and res["plate_number"] != "UNKNOWN":
                    tracker.update_plate_number(best_track.track_id, res["plate_number"])
                    
    # Unassign tracks that left their slots
    for track in active_tracks:
        if track.parking_slot is not None and track.track_id not in assigned_track_ids:
            tracker.remove_parking_slot(track.track_id, current_time)
            
    # Overlay annotations on vis
    out = frame_vis.copy()
    for track in active_tracks:
        x1, y1, x2, y2 = track.bbox
        if track.status == 'parked':
            color = (0, 255, 0)
        elif track.status == 'stopped':
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
            
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track.track_id} {track.status.upper()}"
        if track.plate_number and track.plate_number != "UNKNOWN":
            label += f" ({track.plate_number})"
            
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_color = (0, 0, 0) if color != (0, 0, 255) else (255, 255, 255)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        
    return out

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
        vis = post_process_tracking(vis, cars, slot_results)
        occupied = sum(1 for s in slot_results if s["occupied"])
        free = len(slot_results) - occupied
        slot_status = {}
        for s in slot_results:
            slot_name = s.get("slot_id")
            slot_status[slot_name] = (
                "occupied" if s["occupied"] else "free"
            )
        latest_stats = {
            "total_slots": len(slot_results),
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round(
                (occupied / len(slot_results) * 100)
                if slot_results else 0,
                1
            ),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "slot_status": slot_status,
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
        vis = post_process_tracking(vis, cars, slot_results)
        occupied = sum(1 for s in slot_results if s["occupied"])
        free = len(slot_results) - occupied
        slot_status = {}
        for s in slot_results:
            slot_name = s.get("slot_id")
            slot_status[slot_name] = (
                "occupied" if s["occupied"] else "free"
            )
        latest_stats = {
            "total_slots": len(slot_results),
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round(
                (occupied / len(slot_results) * 100)
                if slot_results else 0,
                1
            ),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "slot_status": slot_status,
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


@app.route("/tracking", methods=["GET"])
def get_tracking():
    import sqlite3
    tracks = []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT track_id, plate_number, slot_id, entry_time, exit_time, status FROM vehicle_tracks ORDER BY track_id DESC"
        )
        tracks = [dict(row) for row in cur]
    except Exception as e:
        print(f"[get_tracking] Error: {e}")
    finally:
        if "conn" in locals():
            conn.close()
    return jsonify({"tracks": tracks})


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

@app.route("/path", methods=["GET"])
def get_path():
    if not latest_stats:
        return jsonify({
            "slot": "FULL",
            "coords": []
        })
    slot_status = latest_stats.get("slot_status", {})
    free_slots = [
        slot for slot, status in slot_status.items()
        if status == "free"
    ]
    if not free_slots:
        return jsonify({
            "slot": "FULL",
            "coords": []
        })
    nodes = parking_graph.get("nodes", {})
    graph = parking_graph.get("graph", {})
    entrance = "entry"
    best_slot = None
    best_path = None
    best_distance = float("inf")
    for slot in free_slots:
        path, total_distance = dijkstra(
            graph,
            nodes,
            entrance,
            slot
        )
        if path:
            if best_path is None or total_distance < best_distance:
                best_path = path
                best_slot = slot
                best_distance = total_distance
    if not best_path:
        return jsonify({
            "slot": "FULL",
            "coords": []
        })
    coords = []
    for node in best_path:
        if node in nodes:
            coords.append(nodes[node])
    return jsonify({
        "slot": best_slot,
        "coords": coords
    })

if __name__ == "__main__":
    print("Starting parking API server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
