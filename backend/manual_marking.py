"""Manual parking-slot marker (4-point professional)

- Left click: add point
- Auto-completes a slot after exactly 4 points (no line connects to the 5th)
- Draws a closed polygon box + semi-transparent fill
- Per-image slots, with next/prev navigation
- Saves JSON for later occupancy detection / training

Controls:
  n = next image
  p = previous image
  s = save
  u = undo last slot (current image)
  c = clear current points (in-progress slot)
  r = remove all slots (current image)
  q = quit (will also save)

Output:
  marked_slots/marked_slots.json
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass
class Slot:
    slot_id: int
    corners: List[Point]  # 4 points


class ManualMarking4Pro:
    def __init__(self, dataset_dir: str = "Dataset", out_dir: str = "marked_slots", image_path: Optional[str] = None):
        self.dataset_dir = dataset_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        if image_path:
            self.images: List[Optional[str]] = [image_path]
        else:
            found = self._list_images(self.dataset_dir)
            self.images = found if found else [None]

        self.idx = 0
        self.window = "Manual Slot Marker"

        self.img_bgr: Optional[np.ndarray] = None
        self.img_path: Optional[str] = None
        self.img_name: Optional[str] = None
        self.h: int = 0
        self.w: int = 0

        self.scale: float = 1.0
        self.disp_w: int = 0
        self.disp_h: int = 0

        self.current_points: List[Point] = []
        self.slots_by_image: Dict[str, List[Slot]] = {}

        self._load_current_image()

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _list_images(self, folder: str) -> List[str]:
        if not os.path.isdir(folder):
            return []
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        files.sort()
        return [os.path.join(folder, f) for f in files]

    def _make_demo_image(self) -> np.ndarray:
        img = np.full((720, 1280, 3), 245, dtype=np.uint8)

        # simple grid
        for x in range(0, img.shape[1], 80):
            cv2.line(img, (x, 0), (x, img.shape[0] - 1), (220, 220, 220), 1)
        for y in range(0, img.shape[0], 80):
            cv2.line(img, (0, y), (img.shape[1] - 1, y), (220, 220, 220), 1)

        title = "DEMO IMAGE - Manual Slot Marking"
        line1 = "Left click 4 points to create one slot (auto completes)"
        line2 = "Keys: n/p images | s save | u undo | c clear points | r reset image slots | q quit"
        cv2.putText(img, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(img, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        cv2.putText(img, line1, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(img, line2, (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        return img

    def _order_4_points(self, pts: List[Point]) -> List[Point]:
        """Order 4 arbitrary points into a stable polygon order (non-twisted).

        Uses angle sort around centroid; then rotates so the first point is top-left-ish.
        """
        if len(pts) != 4:
            return pts

        c_x = float(np.mean([p[0] for p in pts]))
        c_y = float(np.mean([p[1] for p in pts]))

        def angle(p: Point) -> float:
            return float(np.arctan2(p[1] - c_y, p[0] - c_x))

        ordered = sorted(pts, key=angle)

        # rotate so first point is top-left-ish (min x+y)
        start_i = min(range(4), key=lambda i: ordered[i][0] + ordered[i][1])
        ordered = ordered[start_i:] + ordered[:start_i]

        # Ensure clockwise order (positive area means CCW in image coords with y down; we enforce CW)
        area2 = 0
        for i in range(4):
            x1, y1 = ordered[i]
            x2, y2 = ordered[(i + 1) % 4]
            area2 += (x1 * y2 - x2 * y1)
        if area2 > 0:
            ordered = [ordered[0], ordered[3], ordered[2], ordered[1]]

        return ordered

    def _load_current_image(self) -> None:
        self.img_path = self.images[self.idx]
        if self.img_path is None:
            self.img_name = "DEMO"
            self.img_bgr = self._make_demo_image()
        else:
            self.img_name = os.path.basename(self.img_path)
            self.img_bgr = cv2.imread(self.img_path)
            if self.img_bgr is None:
                raise RuntimeError(f"Failed to read image: {self.img_path}")

        self.h, self.w = self.img_bgr.shape[:2]

        # Fit image to a reasonable screen size while keeping aspect ratio.
        max_w, max_h = 1280, 720
        self.scale = min(max_w / self.w, max_h / self.h, 1.0)
        self.disp_w = int(self.w * self.scale)
        self.disp_h = int(self.h * self.scale)

        self.current_points = []
        self.slots_by_image.setdefault(self.img_name, [])

    def _img_to_disp(self, p: Point) -> Point:
        return (int(p[0] * self.scale), int(p[1] * self.scale))

    def _disp_to_img(self, p: Point) -> Point:
        x = int(round(p[0] / self.scale))
        y = int(round(p[1] / self.scale))
        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))
        return (x, y)

    def _next_slot_id(self) -> int:
        # global unique incremental id across all images (stable)
        mx = 0
        for slots in self.slots_by_image.values():
            for s in slots:
                mx = max(mx, s.slot_id)
        return mx + 1

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.img_bgr is None or self.img_name is None:
            return

        img_point = self._disp_to_img((x, y))
        self.current_points.append(img_point)

        # Auto-complete after 4 points
        if len(self.current_points) == 4:
            corners = self._order_4_points(self.current_points)
            slot = Slot(slot_id=self._next_slot_id(), corners=corners)
            self.slots_by_image[self.img_name].append(slot)
            self.current_points = []

    def _draw(self) -> np.ndarray:
        assert self.img_bgr is not None
        base = self.img_bgr.copy()

        # Draw saved slots for this image
        slots = self.slots_by_image.get(self.img_name or "", [])
        for slot in slots:
            pts = np.array(slot.corners, dtype=np.int32)
            overlay = base.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            base = cv2.addWeighted(overlay, 0.25, base, 0.75, 0)
            cv2.polylines(base, [pts], True, (0, 255, 0), 3)

            cx = int(np.mean([p[0] for p in slot.corners]))
            cy = int(np.mean([p[1] for p in slot.corners]))
            cv2.putText(base, str(slot.slot_id), (cx - 12, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(base, str(slot.slot_id), (cx - 12, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # Draw in-progress points (current slot)
        for i, p in enumerate(self.current_points):
            cv2.circle(base, p, 7, (0, 0, 255), -1)
            cv2.putText(base, str(i + 1), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw lines for in-progress polygon (only within current 4)
        if len(self.current_points) >= 2:
            for i in range(len(self.current_points) - 1):
                cv2.line(base, self.current_points[i], self.current_points[i + 1], (255, 255, 0), 2)

        # HUD
        slots_count = len(self.slots_by_image.get(self.img_name or "", []))
        hud1 = f"Image {self.idx + 1}/{len(self.images)}: {self.img_name}"
        hud2 = f"Slots on this image: {slots_count} | Current points: {len(self.current_points)}/4"
        hud3 = "Keys: n next | p prev | s save | u undo | c clear points | r remove image slots | q quit"

        y = 28
        for line in (hud1, hud2, hud3):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(base, (8, y - th - 10), (8 + tw + 16, y + 8), (255, 255, 255), -1)
            cv2.putText(base, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y += 26

        # Resize for display
        disp = cv2.resize(base, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        return disp

    def _serialize(self) -> List[dict]:
        out: List[dict] = []
        for image_name, slots in self.slots_by_image.items():
            for s in slots:
                corners = [(int(x), int(y)) for (x, y) in s.corners]
                cx = int(np.mean([p[0] for p in s.corners]))
                cy = int(np.mean([p[1] for p in s.corners]))
                out.append(
                    {
                        "slot_id": int(s.slot_id),
                        "image_name": image_name,
                        "corners": corners,
                        "center": (cx, cy),
                    }
                )
        # Keep stable order
        out.sort(key=lambda d: (d["image_name"], d["slot_id"]))
        return out

    def save(self) -> str:
        path = os.path.join(self.out_dir, "marked_slots.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._serialize(), f, indent=2)
        print(f"Saved: {path}")
        return path

    def run(self) -> None:
        print("ManualMarking4Pro running")
        print("Left click 4 points per slot. Slot auto-completes after 4 points.")

        while True:
            frame = self._draw()
            cv2.imshow(self.window, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                self.save()
                break
            if key == ord('s'):
                self.save()
            if key == ord('n'):
                if self.idx < len(self.images) - 1:
                    self.idx += 1
                    self._load_current_image()
            if key == ord('p'):
                if self.idx > 0:
                    self.idx -= 1
                    self._load_current_image()
            if key == ord('c'):
                self.current_points = []
            if key == ord('u'):
                slots = self.slots_by_image.get(self.img_name or "", [])
                if slots:
                    slots.pop()
            if key == ord('r'):
                self.slots_by_image[self.img_name or ""] = []
                self.current_points = []

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--image", type=str, default=None, help="Single image path to mark (optional)")
    parser.add_argument("--dataset", type=str, default="Dataset", help="Dataset folder with images")
    args = parser.parse_args()

    tool = ManualMarking4Pro(dataset_dir=args.dataset, image_path=args.image)
    tool.run()


if __name__ == "__main__":
    main()
