"""
backend/utils/slot_manager.py

Loads slot polygon configurations.
Provides centroid-to-slot assignment using Shapely polygon intersection.
Supports both user and teammate calibration file structures.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from shapely.geometry import Point, Polygon


@dataclass
class Slot:
    id:        str
    polygon:   Polygon
    status:    str = "free"          # free | occupied
    track_id:  Optional[int] = None
    plate:     Optional[str] = None


class SlotManager:
    """
    Manages all parking slots. Call `get_slot_for_centroid` each frame
    to update which track owns which slot.
    """

    def __init__(self, config_path: str):
        self.slots: dict[str, Slot] = {}
        self._load(config_path)

    def _load(self, path: str):
        if not Path(path).exists():
            print(f"[SlotManager Warning] Slot configuration not found at: {path}")
            return
            
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        
        # Handle both list and dict formats
        slots_list = []
        if isinstance(data, list):
            slots_list = data
        elif isinstance(data, dict) and "slots" in data:
            slots_list = data["slots"]

        for s in slots_list:
            slot_id = str(s.get("slot_id") or s.get("id") or "")
            corners = s.get("corners") or s.get("polygon")
            
            if not slot_id or not corners:
                continue
                
            poly = Polygon(corners)
            self.slots[slot_id] = Slot(id=slot_id, polygon=poly)
            
        print(f"[SlotManager] Loaded {len(self.slots)} parking slots from {path}")

    # ── public API ────────────────────────────────────────────────────────────

    def get_slot_for_centroid(self, cx: float, cy: float) -> Optional[str]:
        """Return the slot ID whose polygon contains (cx, cy), or None."""
        pt = Point(cx, cy)
        for slot in self.slots.values():
            if slot.polygon.contains(pt):
                return slot.id
        return None

    def occupy(self, slot_id: str, track_id: int, plate: Optional[str] = None, status: str = "occupied"):
        if slot_id in self.slots:
            s = self.slots[slot_id]
            s.status   = status
            s.track_id = track_id
            if plate:
                s.plate = plate

    def free(self, slot_id: str):
        if slot_id in self.slots:
            s = self.slots[slot_id]
            s.status   = "free"
            s.track_id = None
            s.plate    = None

    def update_plate(self, slot_id: str, plate: str):
        if slot_id in self.slots:
            self.slots[slot_id].plate = plate

    def get_track_slot(self, track_id: int) -> Optional[str]:
        """Return slot currently assigned to this track_id."""
        for slot in self.slots.values():
            if slot.track_id == track_id:
                return slot.id
        return None

    def snapshot(self) -> list[dict]:
        """Return serialisable state of all slots for the API."""
        return [
            {
                "slot_id":   s.id,
                "status":    s.status,
                "track_id":  s.track_id,
                "plate":     s.plate,
                "polygon":   [[int(pt[0]), int(pt[1])] for pt in s.polygon.exterior.coords[:-1]],
            }
            for s in self.slots.values()
        ]

    @property
    def free_count(self) -> int:
        return sum(1 for s in self.slots.values() if s.status == "free")

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self.slots.values() if s.status != "free")
