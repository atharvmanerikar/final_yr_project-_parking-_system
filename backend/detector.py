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
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional
from shapely.geometry import Polygon

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
        self.violations: list = []                # Active wrong parking alerts
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
                "violations": self.violations
            }


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
        
        # Violations metadata
        self._outside_parked_frames: dict[int, int] = defaultdict(int)
        self._improper_parked_frames: dict[int, int] = defaultdict(int)
        self._proper_parked_frames: dict[int, int] = defaultdict(int)
        self._active_violations: dict[int, dict] = {}
        self._track_slot_overlaps: dict[int, float] = {}
        self._track_missing_count: dict[int, int] = defaultdict(int)
        self.last_valid_H = None

        # Annotators
        self._box_ann = sv.BoxAnnotator(thickness=2)
        self._label_ann = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_position=sv.Position.TOP_LEFT
        )

        # Load reference calibration image for camera alignment/stabilization
        self.ref_img_path = str(settings.PROJECT_ROOT / "backend/marked_slots/temp_calibration_frame.jpg")
        self.ref_img = cv2.imread(self.ref_img_path)
        if self.ref_img is not None:
            h_ref, w_ref = self.ref_img.shape[:2]
            scale_ref = self.process_width / w_ref
            self.ref_img_proc = cv2.resize(self.ref_img, (self.process_width, int(h_ref * scale_ref)))
            self.orb = cv2.ORB_create(nfeatures=1000)
            self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_img_proc, None)
            print(f"[Detector] Loaded reference image for alignment: {self.ref_img_path} ({len(self.ref_kp)} keypoints)")
        else:
            self.ref_des = None
            print("[Detector Warning] Reference image for alignment not found.")

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

        # Get video properties for real-time skipping
        import numpy as np
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0
            
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        is_video_file = total_frames > 0
        video_start_time = time.time()
        frame_counter = 0

        print(f"[Detector] Pipeline started on source: {source_label}")

        while self._running:
            if is_video_file:
                elapsed_wall = time.time() - video_start_time
                target_frame = int(elapsed_wall * video_fps)
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                skip_count = target_frame - current_frame
                
                if skip_count > 0:
                    if skip_count > 100: # large jump, seek directly
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    else: # small jump, fast grab
                        for _ in range(int(skip_count - 1)):
                            cap.grab()

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
                    video_start_time = time.time() # Reset clock!
                    frame_counter = 0
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

            frame_counter += 1

            # Align current frame to reference frame using homography (run once every 10 frames)
            H = None
            if self.ref_img is not None and self.ref_des is not None and (frame_counter % 10 == 0 or self.last_valid_H is None):
                try:
                    kp, des = self.orb.detectAndCompute(proc, None)
                    if des is not None and len(kp) > 10:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(self.ref_des, des)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = matches[:50]
                        if len(good_matches) >= 10:
                            # Scale keypoint coordinates back to high-resolution
                            scale_back = w / self.process_width
                            src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_back
                            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_back
                            H_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if H_est is not None:
                                # Validate homography to prevent radical distortions
                                det = H_est[0, 0] * H_est[1, 1] - H_est[0, 1] * H_est[1, 0]
                                if 0.8 < det < 1.25 and abs(H_est[2, 0]) < 0.0015 and abs(H_est[2, 1]) < 0.0015:
                                    H = H_est
                                    self.last_valid_H = H_est
                except Exception as e:
                    print(f"[Detector Alignment Error] {e}")

            # Fallback to last known valid homography if current one failed
            if H is None and self.last_valid_H is not None:
                H = self.last_valid_H

            # Warp slot polygons to match current camera perspective
            active_slots = {}
            if H is not None:
                for slot_id, slot in self.slots.slots.items():
                    try:
                        coords = list(slot.polygon.exterior.coords[:-1])
                        warped_coords = []
                        for sx, sy in coords:
                            pt = np.array([sx, sy, 1.0])
                            warped_pt = H @ pt
                            w_z = warped_pt[2] if warped_pt[2] != 0 else 1.0
                            warped_coords.append((warped_pt[0] / w_z, warped_pt[1] / w_z))
                        active_slots[slot_id] = Polygon(warped_coords)
                    except Exception:
                        active_slots[slot_id] = slot.polygon
            else:
                for slot_id, slot in self.slots.slots.items():
                    active_slots[slot_id] = slot.polygon

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

            # Extract center points in processed space (640px) for movement tracking
            proc_centers = {}
            if detections.tracker_id is not None:
                for idx, track_id in enumerate(detections.tracker_id):
                    if track_id is not None:
                        t_id = int(track_id)
                        p_xyxy = detections.xyxy[idx]
                        cx_proc = float((p_xyxy[0] + p_xyxy[2]) / 2)
                        cy_proc = float((p_xyxy[1] + p_xyxy[3]) / 2)
                        proc_centers[t_id] = (cx_proc, cy_proc)

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
                    
                    # Track position history in processed space for status classification
                    proc_pt = proc_centers.get(track_id)
                    if proc_pt:
                        pos_hist = self._track_positions[track_id]
                        if len(pos_hist) >= 5:
                            pos_hist.pop(0)
                        pos_hist.append(proc_pt)

                    # Compute movement speed in processed space to see if vehicle is stopped
                    is_stopped = False
                    if len(pos_hist) >= 3:
                        total_movement = 0
                        for idx in range(1, len(pos_hist)):
                            dx = pos_hist[idx][0] - pos_hist[idx-1][0]
                            dy = pos_hist[idx][1] - pos_hist[idx-1][1]
                            total_movement += np.sqrt(dx**2 + dy**2)
                        avg_movement = total_movement / (len(pos_hist) - 1)
                        if avg_movement < 12.0:  # Increased threshold to 12.0 pixels to prevent camera jitter glitches
                            is_stopped = True
                    else:
                        avg_movement = 999.0  # Assumed moving until history is loaded

                    # Calculate vehicle footprint (bottom 25% of its bounding box)
                    x1_f, y1_f, x2_f, y2_f = xyxy
                    y_footprint = y2_f - 0.25 * (y2_f - y1_f)
                    footprint_poly = Polygon([(x1_f, y_footprint), (x2_f, y_footprint), (x2_f, y2_f), (x1_f, y2_f)])
                    
                    # Find slot with highest footprint overlap
                    slot_id = None
                    max_overlap_area = 0.0
                    best_overlap_ratio = 0.0
                    
                    for s_id, slot_poly in active_slots.items():
                        try:
                            intersection = footprint_poly.intersection(slot_poly)
                            area = intersection.area
                            if area > max_overlap_area:
                                max_overlap_area = area
                                slot_id = s_id
                                best_overlap_ratio = area / footprint_poly.area
                        except Exception:
                            pass
                            
                    # If highest overlap ratio is too low, treat as not in slot
                    if slot_id and best_overlap_ratio > 0.15:
                        self._track_slot_overlaps[track_id] = best_overlap_ratio
                    else:
                        slot_id = None
                        self._track_slot_overlaps[track_id] = 0.0

                    old_status = self._track_statuses.get(track_id)
                    new_status = "searching"

                    # If the car is not in a slot, or if the car is still moving (not stopped),
                    # it does NOT occupy any slot.
                    if slot_id is None or not is_stopped:
                        old_assigned_slot = self._track_slot_assignments.pop(track_id, None)
                        if old_assigned_slot:
                            self._handle_vehicle_exit(track_id, old_assigned_slot)
                            
                        if slot_id is None:
                            # Outside all slots
                            if is_stopped:
                                self._outside_parked_frames[track_id] += 1
                                if self._outside_parked_frames[track_id] >= 15:
                                    new_status = "outside_parking"
                                    self._active_violations[track_id] = {
                                        "type": "outside_parking",
                                        "track_id": track_id,
                                        "description": f"Vehicle ID {track_id} is parked outside any marked slots (blocking lane)."
                                    }
                                else:
                                    new_status = "jst stopped"
                                    self._active_violations.pop(track_id, None)
                            else:
                                self._outside_parked_frames[track_id] = 0
                                new_status = "searching"
                                self._active_violations.pop(track_id, None)
                        else:
                            # Moving inside a slot
                            self._outside_parked_frames[track_id] = 0
                            new_status = "searching"
                            self._active_violations.pop(track_id, None)
                    else:
                        # Vehicle is stationary in a slot, so it is parked
                        self._outside_parked_frames[track_id] = 0
                        
                        overlap_ratio = self._track_slot_overlaps.get(track_id, 1.0)
                        is_previously_parked = old_status in ("parked", "improperly_parked")
                        
                        if not is_previously_parked:
                            # Fresh parking event: set state instantly based on overlap to avoid latency
                            self._improper_parked_frames[track_id] = 0
                            self._proper_parked_frames[track_id] = 0
                            if overlap_ratio < 0.90:  # Increased threshold to 90%
                                new_status = "improperly_parked"
                                self._active_violations[track_id] = {
                                    "type": "improper_parking",
                                    "track_id": track_id,
                                    "slot_id": slot_id,
                                    "overlap_ratio": round(overlap_ratio * 100, 1),
                                    "description": f"Vehicle ID {track_id} is spilling out of Slot {slot_id} ({round(overlap_ratio*100)}% inside)."
                                }
                            else:
                                new_status = "parked"
                                self._active_violations.pop(track_id, None)
                        else:
                            # Hysteresis for established parked state to prevent glitching/flicker
                            if overlap_ratio < 0.90:  # Increased threshold to 90%
                                self._improper_parked_frames[track_id] += 1
                                self._proper_parked_frames[track_id] = 0
                                if self._improper_parked_frames[track_id] >= 15:
                                    new_status = "improperly_parked"
                                    self._active_violations[track_id] = {
                                        "type": "improper_parking",
                                        "track_id": track_id,
                                        "slot_id": slot_id,
                                        "overlap_ratio": round(overlap_ratio * 100, 1),
                                        "description": f"Vehicle ID {track_id} is spilling out of Slot {slot_id} ({round(overlap_ratio*100)}% inside)."
                                    }
                                else:
                                    new_status = old_status
                            else:
                                self._proper_parked_frames[track_id] += 1
                                self._improper_parked_frames[track_id] = 0
                                if self._proper_parked_frames[track_id] >= 15:
                                    new_status = "parked"
                                    self._active_violations.pop(track_id, None)
                                else:
                                    new_status = old_status

                        old_assigned_slot = self._track_slot_assignments.get(track_id)
                        
                        # Check if this is a newly assigned slot (entry event)
                        if old_assigned_slot != slot_id:
                            # If it was assigned to a different slot previously, free that one first
                            if old_assigned_slot:
                                self._handle_vehicle_exit(track_id, old_assigned_slot)
                                
                            self._track_slot_assignments[track_id] = slot_id
                            self._entry_times[track_id] = datetime.utcnow()
                            
                            # Log Entry event to DB
                            log_event(track_id, slot_id, None, None, "entry")
                            
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

                        # Update slot occupancy states in local memory and DB
                        db_status = "improperly_parked" if new_status == "improperly_parked" else "occupied"
                        self.slots.occupy(slot_id, track_id, status=db_status)
                        update_slot_state(slot_id, db_status, track_id, self._track_plates.get(track_id), self._entry_times[track_id])

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

                    # Save and log status transitions
                    if new_status != old_status:
                        print(f"[Detector Status Change] Track {track_id}: {old_status} -> {new_status} (avg_movement={avg_movement:.2f}px)")
                    self._track_statuses[track_id] = new_status

            # 5. Handle disappeared tracks (exits) and metadata cleanup with a grace period
            all_tracked_ids = set(self._track_positions.keys()) | set(self._track_slot_assignments.keys())
            for tid in all_tracked_ids:
                if tid not in active_track_ids:
                    is_parked = tid in self._track_slot_assignments
                    is_outside = self._track_statuses.get(tid) == "outside_parking"
                    
                    if is_parked or is_outside:
                        self._track_missing_count[tid] += 1
                        if self._track_missing_count[tid] < 30:  # 30 frames grace period
                            continue
                    
                    # If grace period expires (or it was just a searching car), clean up
                    slot_id = self._track_slot_assignments.pop(tid, None)
                    if slot_id:
                        self._handle_vehicle_exit(tid, slot_id)
                    
                    self._track_positions.pop(tid, None)
                    self._track_statuses.pop(tid, None)
                    self._ocr_in_progress.pop(tid, None)
                    self._outside_parked_frames.pop(tid, None)
                    self._improper_parked_frames.pop(tid, None)
                    self._proper_parked_frames.pop(tid, None)
                    self._active_violations.pop(tid, None)
                    self._track_slot_overlaps.pop(tid, None)
                    self._track_missing_count.pop(tid, None)
                else:
                    self._track_missing_count[tid] = 0

            # Update live stats
            self.state.update_slots(
                self.slots.snapshot(),
                self.slots.free_count,
                self.slots.occupied_count,
                len(self.slots.slots),
                fps
            )
            # Expose active violations to state
            with self.state._lock:
                self.state.violations = list(self._active_violations.values())

            # Annotate stream frame (use high-resolution frame)
            annotated = self._annotate(frame, detections, active_slots)
            
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
                db_status = "improperly_parked" if self._track_statuses.get(track_id) == "improperly_parked" else "occupied"
                
                old_plate = self._track_plates.get(track_id)
                if best_plate != old_plate:
                    self._track_plates[track_id] = best_plate
                    self.slots.update_plate(slot_id, best_plate)
                    
                    # Log Event and update state in DB
                    log_event(track_id, slot_id, best_plate, score, "ocr_update")
                    update_slot_state(slot_id, db_status, track_id, best_plate, entry_time)
                    
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

    def _annotate(self, frame: np.ndarray, detections: sv.Detections, active_slots: dict) -> np.ndarray:
        out = frame.copy()
        
        # Render slot overlays
        for slot in self.slots.slots.values():
            # Get warped coordinates
            poly = active_slots.get(slot.id, slot.polygon)
            pts = np.array(list(poly.exterior.coords[:-1]), dtype=np.int32)
            
            # Draw color depending on status (free: green, improperly_parked: orange, occupied: red)
            if slot.status == "free":
                color = (46, 216, 130)            # Green (BGR)
            elif slot.status == "improperly_parked":
                color = (0, 140, 255)             # Orange (BGR)
            else:
                color = (92, 92, 255)             # Red (BGR)
            
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
                    color = (46, 216, 130)        # Green
                    text_color = (0, 0, 0)
                elif status == "improperly_parked":
                    color = (0, 140, 255)         # Orange
                    text_color = (255, 255, 255)
                elif status == "outside_parking":
                    color = (255, 0, 255)         # Purple/Magenta
                    text_color = (255, 255, 255)
                elif status == "jst stopped":
                    color = (92, 92, 255)         # Red
                    text_color = (255, 255, 255)
                else:
                    color = (46, 211, 251)        # Yellow/Orange
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
