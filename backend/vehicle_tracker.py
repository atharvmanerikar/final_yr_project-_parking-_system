"""
Vehicle Tracking System for Parking Detection
Implements ByteTrack-like functionality for vehicle tracking and logs events to database.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Track:
    """Represents a tracked vehicle"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    status: str  # 'parked', 'stopped', or 'searching'
    parking_slot: Optional[int] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    last_seen: datetime = None
    position_history: List[Tuple[int, int]] = None
    plate_number: Optional[str] = "UNKNOWN"
    
    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.now()
        if self.position_history is None:
            self.position_history = []


class VehicleTracker:
    """
    Vehicle tracking system with ByteTrack-like functionality and database storage support.
    """
    
    def __init__(self, track_thresh: float = 0.3, track_buffer: int = 30, db=None):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.db = db
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.max_disappeared = 50  # frames before removing track
        
        # Movement detection parameters
        self.movement_threshold = 10  # pixels
        self.position_history_length = 5
        
        # Parking events log
        self.parking_events = []
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracker with new detections
        detections: List of {'bbox': (x1, y1, x2, y2), 'confidence'/'conf': float}
        """
        self.frame_count += 1
        current_time = datetime.now()
        
        # Clean and standardize detections to handle keys 'confidence' or 'conf'
        clean_detections = []
        for det in detections:
            bbox = det.get('bbox')
            confidence = det.get('confidence', det.get('conf', 0.0))
            clean_detections.append({'bbox': bbox, 'confidence': confidence})

        # Match detections to existing tracks
        matched_detections, unmatched_detections = self._match_detections_to_tracks(clean_detections)
        
        # Update matched tracks
        for detection, track_id in matched_detections:
            self._update_track(track_id, detection, current_time)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            # Only create tracks for detections with confidence > threshold
            if detection['confidence'] >= self.track_thresh:
                self._create_new_track(detection, current_time)
        
        # Remove old tracks
        self._remove_old_tracks(current_time)
        
        # Update vehicle status (moving/stopped)
        self._update_vehicle_status()
        
        return list(self.tracks.values())
    
    def _match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[Dict, int]], List[Dict]]:
        """Match detections to existing tracks using IoU"""
        matched = []
        unmatched = detections.copy()
        used_track_ids = set()
        
        for detection in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                
                iou = self._calculate_iou(detection['bbox'], track.bbox)
                if iou > best_iou and iou > 0.3:  # IoU threshold
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id:
                matched.append((detection, best_track_id))
                used_track_ids.add(best_track_id)
                unmatched.remove(detection)
        
        return matched, unmatched
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _update_track(self, track_id: int, detection: Dict, current_time: datetime):
        """Update existing track with new detection"""
        track = self.tracks[track_id]
        
        # Update position history
        center_x = (detection['bbox'][0] + detection['bbox'][2]) // 2
        center_y = (detection['bbox'][1] + detection['bbox'][3]) // 2
        
        if len(track.position_history) >= self.position_history_length:
            track.position_history.pop(0)
        track.position_history.append((center_x, center_y))
        
        # Update track info
        track.bbox = detection['bbox']
        track.confidence = detection['confidence']
        track.last_seen = current_time
        
        # Upsert status/position changes in DB
        self._db_upsert(track)
    
    def _create_new_track(self, detection: Dict, current_time: datetime):
        """Create new track from detection"""
        center_x = (detection['bbox'][0] + detection['bbox'][2]) // 2
        center_y = (detection['bbox'][1] + detection['bbox'][3]) // 2
        
        track = Track(
            track_id=self.next_track_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            status='searching',  # Default status
            last_seen=current_time,
            position_history=[(center_x, center_y)]
        )
        
        self.tracks[self.next_track_id] = track
        self._db_upsert(track)
        self.next_track_id += 1
    
    def _remove_old_tracks(self, current_time: datetime):
        """Remove tracks that haven't been seen for a while"""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            time_diff = (current_time - track.last_seen).total_seconds()
            if time_diff > self.max_disappeared * 0.1:  # Assuming 10 FPS
                if track.parking_slot is not None and track.exit_time is None:
                    track.exit_time = current_time
                    self._log_parking_event(track, 'exit')
                
                track.status = 'exited'
                self._db_upsert(track)
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _update_vehicle_status(self):
        """Update vehicle status: parked, stopped, searching"""
        for track in self.tracks.values():
            if track.parking_slot is not None:
                if track.status != 'parked':
                    track.status = 'parked'
                    self._db_upsert(track)
                continue
            
            if len(track.position_history) >= 3:
                positions = track.position_history[-3:]
                total_movement = 0
                
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    total_movement += np.sqrt(dx**2 + dy**2)
                
                avg_movement = total_movement / (len(positions) - 1)
                
                new_status = 'stopped' if avg_movement < self.movement_threshold else 'searching'
            else:
                new_status = 'stopped'
                
            if track.status != new_status:
                track.status = new_status
                self._db_upsert(track)
    
    def assign_parking_slot(self, track_id: int, slot_id: int, current_time: datetime):
        """Assign a parking slot to a tracked vehicle"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            if track.parking_slot != slot_id:
                track.parking_slot = slot_id
                track.entry_time = current_time
                track.exit_time = None
                track.status = 'parked'
                self._log_parking_event(track, 'entry')
                self._db_upsert(track)
    
    def remove_parking_slot(self, track_id: int, current_time: datetime):
        """Remove parking slot assignment from a tracked vehicle"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            if track.parking_slot is not None:
                track.exit_time = current_time
                self._log_parking_event(track, 'exit')
                track.parking_slot = None
                track.status = 'stopped'
                self._db_upsert(track)
                
    def update_plate_number(self, track_id: int, plate_number: str):
        """Associate number plate with a track ID"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            if track.plate_number != plate_number:
                track.plate_number = plate_number
                self._db_upsert(track)
    
    def _log_parking_event(self, track: Track, event_type: str):
        """Log parking entry/exit events to local list for backup"""
        event = {
            'track_id': track.track_id,
            'event_type': event_type,
            'parking_slot': track.parking_slot,
            'timestamp': datetime.now().isoformat(),
            'entry_time': track.entry_time.isoformat() if track.entry_time else None,
            'exit_time': track.exit_time.isoformat() if track.exit_time else None
        }
        self.parking_events.append(event)
        
    def _db_upsert(self, track: Track):
        """Helper to log track to SQLite database if available"""
        if self.db is not None:
            try:
                self.db.upsert_vehicle_track(
                    track_id=track.track_id,
                    plate_number=track.plate_number,
                    slot_id=track.parking_slot,
                    entry_time=track.entry_time.isoformat() if track.entry_time else None,
                    exit_time=track.exit_time.isoformat() if track.exit_time else None,
                    status=track.status
                )
            except Exception as e:
                print(f"[Tracker DB Error] Failed to upsert track ID {track.track_id}: {e}")
    
    def get_parking_events(self) -> List[Dict]:
        """Get all parking events"""
        return self.parking_events
    
    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks"""
        return list(self.tracks.values())
