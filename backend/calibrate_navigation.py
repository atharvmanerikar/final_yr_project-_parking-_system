"""
backend/calibrate_navigation.py

Interactive tool to define routing nodes (roads) and slot center points.
- Phase 0: Mark road nodes (entry, R2, R3...). Press ENTER when done.
- Phase 1: Mark parking slot center points. Type slot ID in terminal.
- Press S to save and exit.
"""

import cv2
import json
import os
import sys
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
backend_json_path = PROJECT_ROOT / "backend/marked_slots/parking_slots.json"
frontend_json_path = PROJECT_ROOT / "frontend/src/assets/parking_slots.json"
map_image_path = PROJECT_ROOT / "frontend/public/aitd_parking_lot_main.png"

if not map_image_path.exists():
    print(f"[Calibrator Error] Map image not found at: {map_image_path}")
    sys.exit(1)

img = cv2.imread(str(map_image_path))
H_orig, W_orig = img.shape[:2]

# Create a clean display frame with extra bottom margin for instructions (leaves map fully visible)
margin_bottom = 200
display_h = H_orig + margin_bottom
display_w = W_orig

# Calibration State
road_nodes = []     # List of [x, y] coordinates for the road
slot_spots = []     # List of {"name": str, "center": [x, y], "closest_road": str}
phase = 0           # 0: Road Nodes, 1: Slot Centers
window_name = "Navigation Map Calibrator"

def mouse_callback(event, x, y, flags, param):
    global phase, road_nodes, slot_spots
    
    # Restrict clicks to image area
    if y >= H_orig:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        if phase == 0:
            # Add road node
            road_nodes.append([x, y])
            node_name = "entry" if len(road_nodes) == 1 else f"R{len(road_nodes)}"
            print(f"[Calibrator] Road node placed: {node_name} at [{x}, {y}]")
        elif phase == 1:
            # Place slot center
            print("\n[Calibrator Action Required] Go to terminal and enter Slot ID/Name:")
            
            # Temporary UI feedback
            alert_frame = make_base_display()
            draw_overlays(alert_frame)
            # Alert banner on top of image
            cv2.rectangle(alert_frame, (100, H_orig // 2 - 60), (display_w - 100, H_orig // 2 + 40), (15, 23, 42), -1)
            cv2.rectangle(alert_frame, (100, H_orig // 2 - 60), (display_w - 100, H_orig // 2 + 40), (59, 130, 246), 3)
            cv2.putText(alert_frame, "ACTION REQUIRED: Enter Slot ID/Name in your terminal", 
                        (200, H_orig // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.imshow(window_name, alert_frame)
            cv2.waitKey(100)
            
            slot_name = input("Enter Slot Name/ID (e.g. 1, 2, 3): ").strip()
            if not slot_name:
                slot_name = str(len(slot_spots) + 1)
                
            # Find closest road node
            closest_node_name = None
            min_dist = float("inf")
            for idx, pt in enumerate(road_nodes):
                name = "entry" if idx == 0 else f"R{idx+1}"
                dist = np.hypot(x - pt[0], y - pt[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_node_name = name
                    
            slot_spots.append({
                "name": slot_name,
                "center": [x, y],
                "closest_road": closest_node_name
            })
            print(f"[Calibrator] Saved Slot {slot_name} (routed to {closest_node_name}).")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if phase == 0:
            if road_nodes:
                removed = road_nodes.pop()
                print(f"[Calibrator] Removed road node: {removed}")
        elif phase == 1:
            if slot_spots:
                removed = slot_spots.pop()
                print(f"[Calibrator] Removed Slot: {removed['name']}")

def make_base_display():
    # Create canvas with bottom margin
    canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    # Slate background for margin
    canvas[H_orig:, :] = (40, 29, 15) # BGR slate-900 (#0f172a)
    # Copy map image to top
    canvas[:H_orig, :] = img
    return canvas

def draw_overlays(frame):
    scale = W_orig / 1000.0
    
    # 1. Draw Road path
    for idx, pt in enumerate(road_nodes):
        name = "entry" if idx == 0 else f"R{idx+1}"
        cv2.circle(frame, (pt[0], pt[1]), int(8 * scale), (249, 115, 22), -1) # Orange dot
        cv2.circle(frame, (pt[0], pt[1]), int(8 * scale), (255, 255, 255), int(1.5 * scale) if int(1.5 * scale) >= 1 else 1)
        cv2.putText(frame, name, (pt[0] + int(12 * scale), pt[1] + int(5 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), int(2 * scale) if int(2 * scale) >= 1 else 1)
                    
        # Connect to previous node
        if idx > 0:
            prev_pt = road_nodes[idx - 1]
            cv2.line(frame, (prev_pt[0], prev_pt[1]), (pt[0], pt[1]), (249, 115, 22), int(3 * scale))

    # 2. Draw Slot center points & routes to closest road nodes
    for slot in slot_spots:
        cx, cy = slot["center"]
        # Draw slot box centered at point
        w = int(70 * scale)
        h = int(45 * scale)
        cv2.rectangle(frame, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (16, 185, 129), int(2 * scale)) # Green box
        cv2.circle(frame, (cx, cy), int(5 * scale), (16, 185, 129), -1)
        cv2.putText(frame, f"Slot {slot['name']}", (cx - int(24 * scale), cy - int(12 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), int(1.5 * scale) if int(1.5 * scale) >= 1 else 1)
        
        # Draw dotted/dashed line connecting slot to its closest road node
        if slot["closest_road"] and road_nodes:
            # Find coordinates of closest road node
            road_idx = 0 if slot["closest_road"] == "entry" else int(slot["closest_road"][1:]) - 1
            if road_idx < len(road_nodes):
                rx, ry = road_nodes[road_idx]
                cv2.line(frame, (cx, cy), (rx, ry), (14, 165, 233), int(1.5 * scale), cv2.LINE_AA) # Light blue connection

    # 3. Draw instructions in bottom margin (never covers the map image!)
    y_start = H_orig + int(40 * scale)
    y_step = int(30 * scale)
    font_scale = 0.62 * scale
    thickness = int(2 * scale) if int(2 * scale) >= 1 else 1

    if phase == 0:
        lines = [
            "PHASE 1: ROAD CALIBRATION",
            "Click on the map in sequence to draw the road path (first click is entry).",
            f"Nodes placed: {len(road_nodes)}  (LClick=Add node | RClick=Undo last)",
            "Press ENTER when you are done drawing the road path."
        ]
    else:
        lines = [
            "PHASE 2: PARKING SPOTS CALIBRATION",
            "Click on map to mark Slot center points. Enter Slot ID in your terminal window.",
            f"Spots marked: {len(slot_spots)}  (LClick=Add spot | RClick=Undo last)",
            "Press S to save configurations & Exit | Press ESC to quit without saving"
        ]

    for i, line in enumerate(lines):
        color = (59, 130, 246) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (int(30 * scale), y_start + i * y_step), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def redraw():
    frame = make_base_display()
    draw_overlays(frame)
    cv2.imshow(window_name, frame)

# Main Loop
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720 + int(margin_bottom * (1280 / W_orig)))
cv2.setMouseCallback(window_name, mouse_callback)

print("=== Phase 1: Calibrating Road Nodes ===")
print("Instructions: Click to draw the road path sequence. Press ENTER when finished.")

while True:
    redraw()
    key = cv2.waitKey(30) & 0xFF
    
    if key == 27: # ESC
        print("[Calibrator] Exited without saving.")
        cv2.destroyAllWindows()
        break
        
    elif key == 13: # ENTER (Confirm road path and switch phase)
        if phase == 0:
            if len(road_nodes) < 2:
                print("[Calibrator Warning] Please place at least 2 road nodes before confirming.")
            else:
                phase = 1
                print("\n=== Phase 2: Calibrating Slot Center Points ===")
                print("Instructions: Click on map to place slot centers. Type names in console.")
                
    elif key == ord('s') or key == ord('S'): # Save and Exit
        if phase == 1:
            if len(slot_spots) == 0:
                print("[Calibrator Warning] Mark at least one parking spot before saving.")
                continue
                
            cv2.destroyAllWindows()
            
            # 1. Build backend json nodes & graph connections
            final_nodes = {}
            final_graph = {}
            
            # Map road nodes
            for idx, pt in enumerate(road_nodes):
                name = "entry" if idx == 0 else f"R{idx+1}"
                final_nodes[name] = pt
                
                # Connect sequentially
                connections = []
                if idx > 0:
                    prev_name = "entry" if idx == 1 else f"R{idx}"
                    connections.append(prev_name)
                if idx < len(road_nodes) - 1:
                    next_name = f"R{idx+2}"
                    connections.append(next_name)
                final_graph[name] = connections
                
            # Map slots center nodes
            for slot in slot_spots:
                name = slot["name"]
                final_nodes[name] = slot["center"]
                
                # Connect slot to its closest road node
                closest_road = slot["closest_road"]
                final_graph[name] = [closest_road]
                
                # Bidirectional connection
                if closest_road in final_graph:
                    if name not in final_graph[closest_road]:
                        final_graph[closest_road].append(name)
                        
            backend_data = {
                "nodes": final_nodes,
                "graph": final_graph
            }
            
            # 2. Build frontend slot center coordinates json
            frontend_slots = []
            for slot in slot_spots:
                frontend_slots.append({
                    "name": slot["name"],
                    "center": slot["center"]
                })
            frontend_data = {
                "slots": frontend_slots
            }
            
            # Write files
            backend_json_path.write_text(json.dumps(backend_data, indent=2), encoding="utf-8")
            frontend_json_path.write_text(json.dumps(frontend_data, indent=2), encoding="utf-8")
            print("\n[Calibrator] Calibration completed and configs saved successfully!")
            print(f"  - Backend graph: {backend_json_path}")
            print(f"  - Frontend slots: {frontend_json_path}")
            
            # 3. Rebuild frontend React app
            import subprocess
            print("\n[Calibrator] Rebuilding frontend assets for production...")
            cmd = ["cmd.exe", "/c", "npm run build"]
            try:
                subprocess.run(cmd, cwd=str(PROJECT_ROOT / "frontend"), check=True)
                print("[Calibrator] Rebuild complete! Production dashboard updated.")
            except Exception as e:
                print(f"[Calibrator Error] Failed to rebuild frontend: {e}")
                
            break
