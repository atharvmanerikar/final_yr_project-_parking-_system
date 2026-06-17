"""
backend/calibrate_navigation.py

Interactive tool to define routing nodes and slot coordinates on the navigation map.
Saves backend graph nodes to backend/marked_slots/parking_slots.json and 
frontend slot polygons to frontend/src/assets/parking_slots.json.
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

# Load existing configurations
nodes = {}
graph = {}
if backend_json_path.exists():
    try:
        data = json.loads(backend_json_path.read_text(encoding="utf-8"))
        nodes = data.get("nodes", {})
        graph = data.get("graph", {})
    except Exception as e:
        print(f"Error loading existing backend config: {e}")

# Standard corridor nodes
corridor_nodes = ["entry", "turn1", "turn2", "center1", "center2", "center3"]

# Calibration State
calibrated_corridor = {}
calibrated_slots = [] # list of {"name": str, "points": list}
current_points = []
phase = 0 # 0: corridor nodes, 1: slot polygons
corridor_idx = 0
window_name = "Navigation Map Calibrator"

def get_node_under_cursor(x, y, threshold=30):
    for name, pt in calibrated_corridor.items():
        if np.hypot(x - pt[0], y - pt[1]) < threshold:
            return name
    return None

def mouse_callback(event, x, y, flags, param):
    global corridor_idx, phase, current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if phase == 0:
            # Place corridor node
            node_name = corridor_nodes[corridor_idx]
            calibrated_corridor[node_name] = [x, y]
            print(f"[Calibrator] Set {node_name} at [{x}, {y}]")
            corridor_idx += 1
            if corridor_idx >= len(corridor_nodes):
                phase = 1
                print("\n=== Phase 2: Calibrating Slot Polygons ===")
                print("Instructions: Click 4 corner points to draw a slot. Press ENTER to complete the slot.")
        elif phase == 1:
            # Draw slot polygon point
            if len(current_points) < 4:
                current_points.append([x, y])
                print(f"[Calibrator] Added slot vertex {len(current_points)}: [{x}, {y}]")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if phase == 0:
            if corridor_idx > 0:
                corridor_idx -= 1
                node_name = corridor_nodes[corridor_idx]
                if node_name in calibrated_corridor:
                    del calibrated_corridor[node_name]
                print(f"[Calibrator] Undo last corridor node. Redefining: {node_name}")
        elif phase == 1:
            if current_points:
                removed = current_points.pop()
                print(f"[Calibrator] Removed slot vertex: {removed}")

def draw_hud(frame):
    # Header Info box
    y_offset = 30
    cv2.rectangle(frame, (10, 10), (500, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (500, 110), (255, 255, 255), 1)

    if phase == 0:
        curr_node = corridor_nodes[corridor_idx]
        lines = [
            "PHASE 1: Corridor Nodes Placement",
            f"Click on the map to define: {curr_node}",
            f"Progress: {corridor_idx}/{len(corridor_nodes)} nodes set",
            "LClick=Place  RClick=Undo last node  ESC=Quit"
        ]
    else:
        lines = [
            "PHASE 2: Slot Polygons Drawing",
            "Click 4 corners to draw a slot, then press ENTER",
            f"Slots calibrated: {len(calibrated_slots)}",
            "ENTER=Save slot  S=Save config & Exit  ESC=Quit"
        ]
        
    for i, line in enumerate(lines):
        color = (96, 165, 250) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (20, y_offset + i * 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

def redraw():
    frame = img.copy()
    
    # 1. Draw corridor nodes
    for name, pt in calibrated_corridor.items():
        cv2.circle(frame, (pt[0], pt[1]), 8, (37, 99, 235), -1) # Blue dot
        cv2.circle(frame, (pt[0], pt[1]), 8, (255, 255, 255), 1.5)
        cv2.putText(frame, name, (pt[0] + 12, pt[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, name, (pt[0] + 12, pt[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (37, 99, 235), 1)
                    
    # 2. Draw calibrated slots
    for i, slot in enumerate(calibrated_slots):
        pts = np.array(slot["points"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (34, 197, 94), 2) # Green outline
        
        # Draw slot label inside centroid
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.circle(frame, (cx, cy), 5, (16, 185, 129), -1)
        cv2.putText(frame, f"Slot {slot['name']}", (cx - 20, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Slot {slot['name']}", (cx - 20, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 185, 129), 1)

    # 3. Draw in-progress slot points
    if current_points:
        color = (244, 63, 94) # Rose dot
        for pt in current_points:
            cv2.circle(frame, (pt[0], pt[1]), 6, color, -1)
            cv2.circle(frame, (pt[0], pt[1]), 6, (255, 255, 255), 1)
        if len(current_points) > 1:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, 2)

    draw_hud(frame)
    cv2.imshow(window_name, frame)

# Main Execution Flow
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.setMouseCallback(window_name, mouse_callback)

print("=== Phase 1: Calibrating Corridor Nodes ===")
print("Instructions: Click on the map to define entry point and corridor turn nodes.")

while True:
    redraw()
    key = cv2.waitKey(30) & 0xFF
    
    if key == 27: # ESC
        print("[Calibrator] Exited without saving.")
        cv2.destroyAllWindows()
        break
        
    elif key == 13: # ENTER (save current slot polygon)
        if phase == 1:
            if len(current_points) < 4:
                print("[Calibrator Warning] Click 4 points to define a slot.")
            else:
                # Prompt user for slot name in terminal
                # Bring terminal focus or just input
                print("\n[Calibrator Action Required] Go to terminal and enter Slot ID/Name:")
                # Temporarily draw active text on CV window to alert user
                alert_frame = img.copy()
                cv2.rectangle(alert_frame, (100, 300), (1100, 420), (0,0,0), -1)
                cv2.rectangle(alert_frame, (100, 300), (1100, 420), (0,0,255), 2)
                cv2.putText(alert_frame, "ACTION REQUIRED: Enter Slot ID in your command window", 
                            (120, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, alert_frame)
                cv2.waitKey(100)
                
                slot_name = input("Enter Slot Name/ID (e.g. 1, 2, 3): ").strip()
                if not slot_name:
                    slot_name = str(len(calibrated_slots) + 1)
                    
                calibrated_slots.append({
                    "name": slot_name,
                    "points": current_points.copy()
                })
                print(f"[Calibrator] Saved Slot {slot_name}.")
                current_points = []
                
    elif key == ord('s') or key == ord('S'): # Save and Exit
        if len(calibrated_slots) == 0 and phase == 1:
            print("[Calibrator Warning] No slots defined yet. Define at least one slot or press ESC to quit.")
            continue
            
        cv2.destroyAllWindows()
        
        # 1. Update backend json
        # Build nodes map
        final_nodes = {}
        for k, v in calibrated_corridor.items():
            final_nodes[k] = v
            
        for slot in calibrated_slots:
            name = slot["name"]
            pts = np.array(slot["points"], dtype=np.int32)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            final_nodes[name] = [cx, cy]
            
        # Build connections graph
        # Try to preserve edges from loaded config, otherwise map automatically
        final_graph = graph.copy()
        
        # Ensure new slot nodes are in graph connected to center1 as default corridor junction
        for slot in calibrated_slots:
            name = slot["name"]
            if name not in final_graph:
                final_graph[name] = ["center1"]
                # Add slot to center1 connections
                if "center1" in final_graph:
                    if name not in final_graph["center1"]:
                        final_graph["center1"].append(name)
                else:
                    final_graph["center1"] = [name]
                    
        # Remove any deleted slots from connections
        active_slot_ids = {slot["name"] for slot in calibrated_slots}
        for k in list(final_graph.keys()):
            if k not in corridor_nodes and k not in active_slot_ids:
                del final_graph[k]
            else:
                final_graph[k] = [node for node in final_graph[k] if node in corridor_nodes or node in active_slot_ids]
                
        backend_data = {
            "nodes": final_nodes,
            "graph": final_graph
        }
        
        # 2. Update frontend json
        frontend_data = {
            "slots": calibrated_slots
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
