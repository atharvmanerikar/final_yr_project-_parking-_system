"""
backend/calibrate_slots.py

Interactive tool to define parking slot polygons on a still frame.
Can be triggered from the React dashboard.

Usage:
    python backend/calibrate_slots.py --image path/to/frame.jpg --output backend/marked_slots/marked_slots.json

Controls:
    Left click      : add a point to current polygon
    Right click     : undo last point
    ENTER           : finish current slot polygon
    D               : delete last completed slot
    S               : save and exit
    ESC             : exit without saving
    R               : reset all slots
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path

# Color palette for slots
COLOURS = [
    (52, 211, 153), (251, 191, 36), (96, 165, 250), (248, 113, 113),
    (167, 139, 250), (251, 146, 60), (34, 211, 238), (244, 114, 182),
    (163, 230, 53), (232, 121, 0),
]


class SlotCalibrator:
    def __init__(self, image: np.ndarray):
        self.orig = image.copy()
        self.frame = image.copy()
        self.slots: list[dict] = []          # Completed slots
        self.current_pts: list[tuple] = []   # Points for slot being drawn
        self.window = "Slot Calibrator"

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)
        cv2.setMouseCallback(self.window, self._mouse_cb)

    # ── Mouse Handler ──
    def _mouse_cb(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_pts:
                self.current_pts.pop()
                self._redraw()
        elif event == cv2.EVENT_MOUSEMOVE:
            self._redraw(cursor=(x, y))

    # ── Drawing ──
    def _redraw(self, cursor=None):
        self.frame = self.orig.copy()

        # Draw completed slots
        for i, slot in enumerate(self.slots):
            pts = np.array(slot["polygon"], dtype=np.int32)
            color = COLOURS[i % len(COLOURS)]
            overlay = self.frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, self.frame, 0.75, 0, self.frame)
            cv2.polylines(self.frame, [pts], True, color, 2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(self.frame, slot["id"], (cx - 14, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Draw in-progress polygon
        if self.current_pts:
            color = COLOURS[len(self.slots) % len(COLOURS)]
            for pt in self.current_pts:
                cv2.circle(self.frame, pt, 5, color, -1)
            if len(self.current_pts) > 1:
                pts = np.array(self.current_pts, dtype=np.int32)
                cv2.polylines(self.frame, [pts], False, color, 2)
            # Live edge to cursor
            if cursor:
                cv2.line(self.frame, self.current_pts[-1], cursor, color, 1)

        # HUD Overlay text info
        self._draw_hud()
        cv2.imshow(self.window, self.frame)

    def _draw_hud(self):
        lines = [
            f"Slots defined: {len(self.slots)}",
            f"Drawing slot: P{len(self.slots)+1}  ({len(self.current_pts)} pts)",
            "LClick=add pt  RClick=undo  ENTER=finish slot",
            "D=delete last  S=save  ESC=quit  R=reset",
        ]
        for i, line in enumerate(lines):
            y = 24 + i * 22
            cv2.rectangle(self.frame, (8, y - 16), (8 + len(line) * 9, y + 6),
                          (0, 0, 0), -1)
            cv2.putText(self.frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # ── Actions ──
    def _finish_slot(self):
        if len(self.current_pts) < 3:
            print("[Calibrator Warning] Need at least 3 points to define a slot.")
            return
        slot_id = f"P{len(self.slots)+1}"
        self.slots.append({"id": slot_id, "polygon": self.current_pts.copy()})
        print(f"[Calibrator] Slot {slot_id} saved with {len(self.current_pts)} points.")
        self.current_pts = []
        self._redraw()

    def _delete_last(self):
        if self.slots:
            removed = self.slots.pop()
            print(f"[Calibrator] Deleted last slot: {removed['id']}")
            self._redraw()

    # ── Main Event Loop ──
    def run(self) -> list[dict] | None:
        print("[Calibrator] Slot Calibrator Window Opened.")
        print("Instructions: Draw polygons around each parking slot.")
        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13:                  # ENTER
                self._finish_slot()
            elif key == ord('d'):
                self._delete_last()
            elif key == ord('r'):
                self.slots = []
                self.current_pts = []
                print("[Calibrator] All slots cleared.")
                self._redraw()
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return self.slots
            elif key == 27:                # ESC
                cv2.destroyAllWindows()
                return None


def main():
    parser = argparse.ArgumentParser(description="Parking slot calibration tool")
    parser.add_argument("--image", type=str, required=True, help="Path to still image of parking lot")
    parser.add_argument("--output", type=str, default="backend/marked_slots/marked_slots.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # Load calibration target image
    img = cv2.imread(args.image)
    if img is None:
        print(f"[Calibrator Error] Cannot read target image: {args.image}")
        return

    calibrator = SlotCalibrator(img)
    slots = calibrator.run()

    if slots is None:
        print("[Calibrator] Calibration cancelled. No changes saved.")
        return

    if not slots:
        print("[Calibrator] No slots defined. Nothing saved.")
        return

    # Convert to teammate list format compatibility
    output_slots = []
    for s in slots:
        corners = s["polygon"]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        center = [int(sum(xs)/len(xs)), int(sum(ys)/len(ys))]
        slot_num = int(s["id"].replace("P", ""))
        
        output_slots.append({
            "slot_id": slot_num,
            "image_name": "temp_calibration_frame.jpg",
            "corners": corners,
            "center": center
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_slots, indent=2), encoding="utf-8")
    print(f"[Calibrator] Successfully saved {len(output_slots)} slots -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
