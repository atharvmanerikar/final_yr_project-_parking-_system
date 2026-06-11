"""
backend/detector.py

Core computer vision pipeline:
- VideoCapture (camera or local video file)
- YOLOv8 vehicle detection
- ByteTrack object tracking
- Slot assignment (Shapely polygon container tests)
- Plate recognition (custom ALPR detector + EasyOCR)
- Event logs & state updates saved directly to SQLite database.

Runs in a separate thread, providing an MJPEG feed and snapshotted stats.
"""

import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Optional

import supervision as sv
from ultralytics import YOLO

from backend.config import settings
from backend.utils.ocr import read_plate_two_stage, crop_from_bbox
from backend.utils.slot_manager import SlotManager
from backend.database.sync_db import log_event, update_slot_state, init_sync_db

VEHICLE_CLASSES = {2: "car"}  # Class 2 in COCO is Car


class ParkingState:
    """Thread-safe shared state communicating detector output to FastAPI."""
    def __init__(self):
        self._lock = threading.RLock()
        self.latest_frame: Optional[bytes] = None  # Annotated JPEG bytes
        self.slot_snapshot: list = []             # List of serializable slot dicts
        self.events: list = []                    # Recent 20 events cache
        self.alerts: list = []                    # Active parking violation alerts
        self.stats = {
            "total": 0,
            "free": 0,
            "occupied": 0,
            "avg_dwell_mins": 0.0,
            "fps": 0.0,
            "current_source": ""
        }

    def update_frame(self, jpeg: bytes):
        with self._lock:
            self.latest_frame = jpeg

    def update_slots(self, snapshot: list, free: int, occ: int, total: int, fps: float):
        with self._lock:
            self.slot_snapshot = snapshot
            self.stats["free"] = free
            self.stats["occupied"] = occ
            self.stats["total"] = total
            self.stats["fps"] = round(fps, 1)

    def push_event(self, event: dict):
        with self._lock:
            self.events.insert(0, event)
            if len(self.events) > 50:
                self.events.pop()

    def update_avg_dwell(self, secs: Optional[int]):
        if secs is None:
            return
        with self._lock:
            exits = [e for e in self.events if e["event_type"] == "exiting" and e.get("dwell_secs")]
            if exits:
                avg = sum(e["dwell_secs"] for e in exits) / len(exits)
                self.stats["avg_dwell_mins"] = round(avg / 60, 1)

    def set_source(self, source_name: str):
        with self._lock:
            self.stats["current_source"] = source_name

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_frame

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "slots": self.slot_snapshot,
                "stats": self.stats,
                "events": self.events[:20],
                "alerts": self.alerts[:50],
            }

    def add_alert(self, alert: dict):
        with self._lock:
            # Check if similar alert already exists (avoid duplicates)
            for existing in self.alerts:
                if (existing.get("track_id") == alert.get("track_id") and
                    existing.get("type") == alert.get("type") and
                    existing.get("slot_id") == alert.get("slot_id")):
                    # Update timestamp instead of adding duplicate
                    existing["timestamp"] = alert["timestamp"]
                    return
            self.alerts.insert(0, alert)
            if len(self.alerts) > 100:
                self.alerts.pop()

    def remove_alert(self, track_id: int, alert_type: str = None):
        with self._lock:
            if alert_type:
                self.alerts = [a for a in self.alerts if not (a.get("track_id") == track_id and a.get("type") == alert_type)]
            else:
                self.alerts = [a for a in self.alerts if a.get("track_id") != track_id]


class ParkingDetector:
    def __init__(
        self,
        camera_source,
        slots_config: str,
        yolo_model: str,
        confidence: float,
        process_width: int,
        ocr_cooldown: int,
        state: ParkingState
    ):
        self.source = camera_source
        self.slots_config = slots_config
        self.confidence = confidence
        self.process_width = process_width
        self.ocr_cooldown = ocr_cooldown
        self.state = state
        self._running = False
        self._thread: Optional[threading.Thread] = None

        print(f"[Detector] Loading YOLO detection model: {yolo_model}")
        self.model = YOLO(yolo_model)
        print("[Detector] YOLO loaded successfully.")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=25
        )

        self.slots = SlotManager(slots_config)
        self.state.slots_snapshot = self.slots.snapshot()
        self.state.stats["total"] = len(self.slots.slots)
        
        # Track-specific metadata
        self._entry_times: dict[int, datetime] = {}
        self._ocr_in_progress: dict[int, bool] = defaultdict(bool)
        self._track_plates: dict[int, str] = {}
        self._track_slot_assignments: dict[int, str] = {}
        self._track_positions = defaultdict(list)
        self._track_statuses: dict[int, str] = {}
        self._plate_histories = defaultdict(list)
        self._alpr_attempts: dict[int, int] = defaultdict(int)
        self._wrong_parking_start_times: dict[int, datetime] = {}  # Track when wrong parking started
        self._alert_cooldowns: dict[int, datetime] = {}  # Prevent alert spam

        # Annotators
        self._box_ann = sv.BoxAnnotator(thickness=2)
        self._label_ann = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_position=sv.Position.TOP_LEFT
        )

        # Initialize SQLite database
        init_sync_db()

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[Detector Error] Cannot open camera source: {self.source}")
            self._running = False
            return

        # Attempt high resolution capture settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        source_label = "Webcam" if self.source == 0 else os.path.basename(str(self.source))
        self.state.set_source(source_label)

        fps_timer = time.time()
        fps_frames = 0
        fps = 0.0

        print(f"[Detector] Pipeline started on source: {source_label}")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                # Loop video files if they end
                if isinstance(self.source, str) and os.path.exists(self.source):
                    print(f"[Detector] Video ended. Re-opening source: {self.source}")
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    time.sleep(0.1)  # Brief pause to prevent CPU pegging if read failures persist
                    continue
                else:
                    print("[Detector Warning] Video source read failed, sleeping...")
                    time.sleep(0.5)
                    continue

            # Resize frame to optimize processing
            h, w = frame.shape[:2]
            scale = self.process_width / w
            proc = cv2.resize(frame, (self.process_width, int(h * scale)))

            # 1. YOLO inference
            # Detect multiple vehicle types (bicycle, car, motorcycle, bus, truck) to let YOLO
            # differentiate them correctly, then filter out everything except cars (class 2).
            results = self.model(
                proc,
                classes=[1, 2, 3, 5, 7],
                conf=self.confidence,
                verbose=False
            )[0]
            
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 2]

            # 2. ByteTrack tracking
            detections = self.tracker.update_with_detections(detections)

            # Scale detections back to the high-resolution frame space
            if len(detections) > 0:
                scale_back = w / self.process_width
                detections.xyxy = detections.xyxy * scale_back

            # 3. Slot assignment & OCR
            active_track_ids = set()

            if detections.tracker_id is not None:
                for i, track_id in enumerate(detections.tracker_id):
                    if track_id is None:
                        continue
                    
                    track_id = int(track_id)
                    active_track_ids.add(track_id)

                    xyxy = detections.xyxy[i]
                    
                    # Track position history for status classification
                    cx_val = float((xyxy[0] + xyxy[2]) / 2)
                    cy_val = float((xyxy[1] + xyxy[3]) / 2)
                    pos_hist = self._track_positions[track_id]
                    if len(pos_hist) >= 5:
                        pos_hist.pop(0)
                    pos_hist.append((cx_val, cy_val))

                    cx = float((xyxy[0] + xyxy[2]) / 2)
                    cy = float(xyxy[3])  # Use bottom center (ground contact point) instead of centroid to match ground slot geometry
                    slot_id = self.slots.get_slot_for_centroid(cx, cy)

                    if slot_id is None:
                        # Vehicle is in search space, make sure it is unassigned
                        old_slot = self._track_slot_assignments.pop(track_id, None)
                        if old_slot:
                            self._handle_vehicle_exit(track_id, old_slot)

                        # Compute status for non-parked vehicles (stopped vs searching)
                        if len(pos_hist) >= 3:
                            total_movement = 0
                            for idx in range(1, len(pos_hist)):
                                dx = pos_hist[idx][0] - pos_hist[idx-1][0]
                                dy = pos_hist[idx][1] - pos_hist[idx-1][1]
                                total_movement += np.sqrt(dx**2 + dy**2)
                            avg_movement = total_movement / (len(pos_hist) - 1)
                            is_stopped = avg_movement < 8.0
                            self._track_statuses[track_id] = "jst stopped" if is_stopped else "searching"

                            # Wrong parking detection: vehicle stopped but not in a slot
                            if is_stopped:
                                if track_id not in self._wrong_parking_start_times:
                                    self._wrong_parking_start_times[track_id] = datetime.utcnow()

                                # Alert if stopped for more than 5 seconds and not in cooldown
                                stopped_duration = (datetime.utcnow() - self._wrong_parking_start_times[track_id]).total_seconds()
                                if stopped_duration > 5.0:
                                    cooldown_end = self._alert_cooldowns.get(track_id, datetime.min)
                                    if datetime.utcnow() > cooldown_end:
                                        plate = self._track_plates.get(track_id, "Unknown")
                                        alert = {
                                            "track_id": track_id,
                                            "type": "wrong_parking",
                                            "message": f"Vehicle stopped in non-designated area",
                                            "plate": plate,
                                            "slot_id": None,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "duration_seconds": int(stopped_duration)
                                        }
                                        self.state.add_alert(alert)
                                        print(f"[Alert] Wrong parking detected: Track {track_id} (Plate: {plate}) stopped for {int(stopped_duration)}s")
                                        # Set cooldown for 30 seconds to prevent spam
                                        self._alert_cooldowns[track_id] = datetime.utcnow() + timedelta(seconds=30)
                            else:
                                # Vehicle is moving again, clear wrong parking alert
                                if track_id in self._wrong_parking_start_times:
                                    del self._wrong_parking_start_times[track_id]
                                self.state.remove_alert(track_id, "wrong_parking")
                        else:
                            self._track_statuses[track_id] = "searching"
                        continue

                    # Vehicle is in a slot, so it is parked
                    self._track_statuses[track_id] = "parked"

                    # Clear wrong parking alert if vehicle was previously wrongly parked
                    if track_id in self._wrong_parking_start_times:
                        del self._wrong_parking_start_times[track_id]
                    self.state.remove_alert(track_id, "wrong_parking")

                    old_slot = self._track_slot_assignments.get(track_id)
                    
                    # Check if this is a newly assigned slot (entry event)
                    if old_slot != slot_id:
                        # If it was assigned to a different slot previously, free that one first
                        if old_slot:
                            self._handle_vehicle_exit(track_id, old_slot)
                            
                        self._track_slot_assignments[track_id] = slot_id
                        self._entry_times[track_id] = datetime.utcnow()
                        self.slots.occupy(slot_id, track_id)
                        
                        # Log Entry event to DB
                        log_event(track_id, slot_id, None, None, "entry")
                        update_slot_state(slot_id, "occupied", track_id, None, self._entry_times[track_id])
                        
                        self.state.push_event({
                            "track_id": track_id,
                            "slot_id": slot_id,
                            "plate": None,
                            "ocr_conf": None,
                            "event_type": "entry",
                            "timestamp": datetime.utcnow().isoformat(),
                            "dwell_secs": None
                        })
                        print(f"[Detector] Entry: Track {track_id} -> Slot {slot_id}")

                    # 4. Trigger Two-Stage ALPR (with temporal voting) asynchronously
                    # Only trigger plate detection once the vehicle has been parked in the slot for at least 1.2 seconds
                    time_in_slot = (datetime.utcnow() - self._entry_times[track_id]).total_seconds() if track_id in self._entry_times else 0
                    if time_in_slot >= 1.2:
                        if not self._ocr_in_progress[track_id] and self._alpr_attempts[track_id] < 10:
                            # Start OCR in background thread
                            self._ocr_in_progress[track_id] = True
                            self._alpr_attempts[track_id] += 1
                            
                            # Crop from high-resolution frame
                            crop = crop_from_bbox(frame, xyxy)
                            threading.Thread(
                                target=self._async_ocr_worker,
                                args=(crop, track_id, slot_id),
                                daemon=True
                            ).start()

            # 5. Handle disappeared tracks (exits)
            all_assigned_track_ids = list(self._track_slot_assignments.keys())
            for tid in all_assigned_track_ids:
                if tid not in active_track_ids:
                    # Vehicle has exited
                    slot_id = self._track_slot_assignments.pop(tid, None)
                    if slot_id:
                        self._handle_vehicle_exit(tid, slot_id)

            # Clean up metadata for inactive tracks
            all_tracked_ids = list(self._track_positions.keys())
            for tid in all_tracked_ids:
                if tid not in active_track_ids:
                    self._track_positions.pop(tid, None)
                    self._track_statuses.pop(tid, None)
                    self._ocr_in_progress.pop(tid, None)
                    self._wrong_parking_start_times.pop(tid, None)
                    self._alert_cooldowns.pop(tid, None)
                    self.state.remove_alert(tid)

            # Update live stats
            self.state.update_slots(
                self.slots.snapshot(),
                self.slots.free_count,
                self.slots.occupied_count,
                len(self.slots.slots),
                fps
            )

            # Annotate stream frame (use high-resolution frame)
            annotated = self._annotate(frame, detections)
            
            # Encode frame as JPEG
            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self.state.update_frame(jpg.tobytes())

            # FPS counter
            fps_frames += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                print(f"[Detector] FPS updated: {fps:.2f} (frames: {fps_frames}, elapsed: {elapsed:.2f}s)")
                fps_frames = 0
                fps_timer = time.time()

        cap.release()
        print("[Detector] Detector thread stopped.")

    def _handle_vehicle_exit(self, track_id: int, slot_id: str):
        """Processes vehicle leaving a parking slot."""
        entry_time = self._entry_times.pop(track_id, None)
        dwell_secs = int((datetime.utcnow() - entry_time).total_seconds()) if entry_time else None
        plate = self._track_plates.pop(track_id, None)
        self.slots.free(slot_id)
        
        # Log exit to database
        log_event(track_id, slot_id, plate, None, "exiting", dwell_secs)
        update_slot_state(slot_id, "free", None, None, None)
        
        exit_event = {
            "track_id": track_id,
            "slot_id": slot_id,
            "plate": plate,
            "ocr_conf": None,
            "event_type": "exiting",
            "timestamp": datetime.utcnow().isoformat(),
            "dwell_secs": dwell_secs
        }
        self.state.push_event(exit_event)
        self.state.update_avg_dwell(dwell_secs)
        
        dwell_str = f"{dwell_secs // 60}m {dwell_secs % 60}s" if dwell_secs else "N/A"
        print(f"[Detector] Exit: Track {track_id} left Slot {slot_id} (Dwell: {dwell_str})")

    def _async_ocr_worker(self, crop: np.ndarray, track_id: int, slot_id: str):
        """Worker method to execute EasyOCR in background thread and update states."""
        try:
            plate_text, score = read_plate_two_stage(crop, slot_id)
            if plate_text:
                self._plate_histories[track_id].append(plate_text)
                best_plate = Counter(self._plate_histories[track_id]).most_common(1)[0][0]
                
                # Retrieve entry time safely
                entry_time = self._entry_times.get(track_id)
                
                old_plate = self._track_plates.get(track_id)
                if best_plate != old_plate:
                    self._track_plates[track_id] = best_plate
                    self.slots.update_plate(slot_id, best_plate)
                    
                    # Log Event and update state in DB
                    log_event(track_id, slot_id, best_plate, score, "ocr_update")
                    update_slot_state(slot_id, "occupied", track_id, best_plate, entry_time)
                    
                    # Update current cache list
                    self.state.push_event({
                        "track_id": track_id,
                        "slot_id": slot_id,
                        "plate": best_plate,
                        "ocr_conf": score,
                        "event_type": "ocr_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "dwell_secs": None
                    })
                    print(f"[Detector Async OCR] Plate read (Voting): Track {track_id} -> '{best_plate}' ({score:.2f})")
        except Exception as e:
            print(f"[Detector Async OCR Error] {e}")
        finally:
            self._ocr_in_progress[track_id] = False

    def _annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        out = frame.copy()
        
        # Render slot overlays
        for slot in self.slots.slots.values():
            pts = np.array(list(slot.polygon.exterior.coords[:-1]), dtype=np.int32)
            color = (46, 216, 130) if slot.status == "free" else (92, 92, 255)  # BGR green / red
            
            # Semi-transparent overlay
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
            cv2.polylines(out, [pts], True, color, 2)
            
            # Center label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = f"Slot {slot.id}"
            cv2.putText(out, label, (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(out, label, (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Render vehicle tracks
        if detections.tracker_id is not None:
            for idx, track_id in enumerate(detections.tracker_id):
                if track_id is None:
                    continue
                track_id = int(track_id)
                xyxy = detections.xyxy[idx]
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Determine status
                status = self._track_statuses.get(track_id, "searching")
                
                # Colors mapping (BGR)
                if status == "parked":
                    color = (46, 216, 130)  # Green
                    text_color = (0, 0, 0)
                elif status == "jst stopped":
                    color = (92, 92, 255)  # Red
                    text_color = (255, 255, 255)
                else:
                    color = (46, 211, 251)  # Yellow/Orange
                    text_color = (0, 0, 0)
                    
                # Draw bounding box
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"ID {track_id} ({status})"
                plate = self._track_plates.get(track_id)
                if plate:
                    label += f" [{plate}]"
                    
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
            
        return out

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
