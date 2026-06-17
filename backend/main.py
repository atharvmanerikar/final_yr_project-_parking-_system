"""
backend/main.py

FastAPI Application server.
Exposes endpoints for:
- Snapshot slot states, stats, and events.
- Historic logs (events & tracking database).
- Navigation routing (Dijkstra).
- Analytics aggregations.
- Video feed MJPEG stream.
- Live system restart controls (switching between webcam and demo video).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Query, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database.models import init_db, get_db, ParkingEvent, SlotState
from backend.detector import ParkingDetector, ParkingState
from backend.utils.pathfinder import ParkingPathfinder

# Shared state
parking_state = ParkingState()
detector: Optional[ParkingDetector] = None

# Initialize pathfinder with portable path
pathfinder = ParkingPathfinder(settings.make_absolute("backend/marked_slots/parking_slots.json"))



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes Database and Detector on Startup."""
    global detector
    
    # Init DB
    await init_db()
    
    # Load first section details from sections_config if it exists
    import json
    from pathlib import Path
    slots_file = settings.SLOTS_CONFIG
    camera_source = settings.CAMERA_SOURCE
    
    if Path(settings.SECTIONS_CONFIG).exists():
        try:
            sec_data = json.loads(Path(settings.SECTIONS_CONFIG).read_text(encoding="utf-8"))
            sections = sec_data.get("sections", [])
            if sections:
                first_sec = sections[0]
                slots_file = settings.make_absolute(first_sec.get("slots_file", slots_file))
                camera_source = settings.make_absolute(first_sec.get("source", camera_source))
        except Exception as e:
            print(f"[Lifespan Error] Loading default section config failed: {e}")
            
    # Start Vision Pipeline
    detector = ParkingDetector(
        camera_source  = camera_source,
        slots_config   = slots_file,
        yolo_model     = settings.YOLO_MODEL,
        confidence     = settings.DETECTION_CONFIDENCE,
        process_width  = settings.PROCESS_WIDTH,
        ocr_cooldown   = settings.OCR_COOLDOWN_FRAMES,
        state          = parking_state
    )
    detector.start()
    
    yield
    
    # Stop Vision Pipeline on shutdown
    if detector:
        detector.stop()


app = FastAPI(
    title="NextGen Smart Parking API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files (React build frontend/dist)
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


# ── REST API Routing ──

@app.get("/api/snapshot")
async def get_snapshot():
    """Return latest snapshotted slot configurations, cache logs, and general occupancy stats."""
    return parking_state.get_snapshot()


@app.get("/api/path")
@app.get("/api/navigation/path")
async def get_navigation_path():
    """Calculates shortest route from entry to the closest free parking slot using Dijkstra."""
    snapshot = parking_state.get_snapshot()
    slots_list = snapshot.get("slots", [])
    free_slots = [str(s["slot_id"]) for s in slots_list if s["status"] == "free"]
    
    path_info = pathfinder.find_shortest_path_to_available_slot(free_slots)
    return path_info



@app.get("/api/control/sections")
async def get_sections():
    """Retrieve the parking sections & floor configuration."""
    import json
    from pathlib import Path
    if Path(settings.SECTIONS_CONFIG).exists():
        try:
            return json.loads(Path(settings.SECTIONS_CONFIG).read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": str(e), "sections": []}
    return {"sections": []}


@app.post("/api/control/sections")
async def save_sections(config: dict = Body(...)):
    """Save the updated parking sections & floor configuration."""
    import json
    from pathlib import Path
    try:
        Path(settings.SECTIONS_CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")
        return {"status": "success", "message": "Sections configuration saved."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/events")
async def get_events(
    limit:      int = Query(50, le=200),
    event_type: Optional[str] = Query(None, description="entry | exiting | ocr_update"),
    slot_id:    Optional[str] = Query(None),
    plate:      Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve historical logs from SQLite database with dynamic filters."""
    query = select(ParkingEvent).order_by(desc(ParkingEvent.timestamp)).limit(limit)
    
    if event_type:
        query = query.where(ParkingEvent.event_type == event_type)
    if slot_id:
        query = query.where(ParkingEvent.slot_id == slot_id)
    if plate:
        query = query.where(ParkingEvent.plate.ilike(f"%{plate}%"))
        
    result = await db.execute(query)
    rows = result.scalars().all()
    
    if rows:
        return {
            "events": [
                {
                    "id": r.id,
                    "track_id": r.track_id,
                    "slot_id": r.slot_id,
                    "plate": r.plate,
                    "ocr_conf": r.ocr_conf,
                    "event_type": r.event_type,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "dwell_secs": r.dwell_secs
                }
                for r in rows
            ]
        }
        
    # In-memory fallback if DB query yields nothing
    events = parking_state.get_snapshot()["events"]
    if event_type:
        events = [e for e in events if e["event_type"] == event_type]
    if slot_id:
        events = [e for e in events if e["slot_id"] == slot_id]
    if plate:
        events = [e for e in events if plate.lower() in (e["plate"] or "").lower()]
        
    return {"events": events[:limit]}


@app.post("/api/events/clear")
async def clear_events(db: AsyncSession = Depends(get_db)):
    """Deletes all historic events from the database and clears visual history."""
    try:
        from sqlalchemy import delete
        
        # 1. Clear database events table
        await db.execute(delete(ParkingEvent))
        await db.commit()
        
        # 2. Clear in-memory event cache
        parking_state.events.clear()
        parking_state.stats["avg_dwell_mins"] = 0.0
        
        return {"status": "success", "message": "All historic logs cleared successfully."}
    except Exception as e:
        print(f"[Clear Events Error] {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/analytics")
async def get_analytics(db: AsyncSession = Depends(get_db)):
    """Computes aggregate analytics metrics (occupancy rates, peak times, slot usages)."""
    snapshot = parking_state.get_snapshot()
    
    # Query database records
    slot_utilization = {}
    peak_hours = {str(h).zfill(2): 0 for h in range(24)}
    avg_dwell_secs = 0
    dwell_exits_count = 0
    
    try:
        query = select(ParkingEvent)
        result = await db.execute(query)
        events = result.scalars().all()
        
        for ev in events:
            # Slot utilization
            if ev.slot_id:
                slot_data = slot_utilization.setdefault(ev.slot_id, {"entries": 0, "exits": 0, "total": 0})
                slot_data["total"] += 1
                if ev.event_type == "entry":
                    slot_data["entries"] += 1
                elif ev.event_type == "exiting":
                    slot_data["exits"] += 1
                    
            # Peak hours (based on entries)
            if ev.event_type == "entry" and ev.timestamp:
                hour_str = str(ev.timestamp.hour).zfill(2)
                peak_hours[hour_str] = peak_hours.get(hour_str, 0) + 1
                
            # Dwell duration aggregates
            if ev.event_type == "exiting" and ev.dwell_secs:
                avg_dwell_secs += ev.dwell_secs
                dwell_exits_count += 1
                
        if dwell_exits_count > 0:
            avg_dwell_secs = round(avg_dwell_secs / dwell_exits_count, 1)
            
    except Exception as e:
        print(f"[Analytics API Error] Database aggregation failed: {e}")
        
    # Formatting slot list
    slots_util = []
    for s_id, data in slot_utilization.items():
        slots_util.append({
            "slot_id": s_id,
            "entries": data["entries"],
            "exits": data["exits"],
            "total_usage": data["total"]
        })
        
    # Peak hours chart formatting
    peak_hours_list = [
        {"hour": f"{h}:00", "entries": count} for h, count in peak_hours.items()
    ]
    
    # Overall summary stats
    total_slots = snapshot["stats"]["total"]
    occupied_slots = snapshot["stats"]["occupied"]
    free_slots = snapshot["stats"]["free"]
    occupancy_rate = round((occupied_slots / total_slots * 100) if total_slots > 0 else 0.0, 1)

    return {
        "status": {
            "total": total_slots,
            "occupied": occupied_slots,
            "free": free_slots,
            "occupancy_rate": occupancy_rate
        },
        "slot_utilization": slots_util,
        "peak_hours": peak_hours_list,
        "avg_dwell_mins": round(avg_dwell_secs / 60.0, 1) if avg_dwell_secs > 0 else snapshot["stats"]["avg_dwell_mins"]
    }


# ── Dynamic Camera Control Feeds ──

@app.post("/api/control/start")
async def start_pipeline(payload: dict = Body(...)):
    """Dynamically start or restart pipeline with a selected section ID."""
    global detector
    
    section_id = payload.get("section_id")
    if not section_id:
        return {"status": "error", "message": "Missing section_id in request body."}
        
    import json
    from pathlib import Path
    
    slots_file = settings.SLOTS_CONFIG
    camera_source = settings.CAMERA_SOURCE
    
    if Path(settings.SECTIONS_CONFIG).exists():
        try:
            sec_data = json.loads(Path(settings.SECTIONS_CONFIG).read_text(encoding="utf-8"))
            sections = sec_data.get("sections", [])
            target_sec = next((s for s in sections if s.get("id") == section_id), None)
            if target_sec:
                slots_file = settings.make_absolute(target_sec.get("slots_file", slots_file))
                camera_source = settings.make_absolute(target_sec.get("source", camera_source))
            else:
                return {"status": "error", "message": f"Section {section_id} not found."}
        except Exception as e:
            print(f"[Control API Error] Loading section config failed: {e}")
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": "Sections config file not found."}
        
    if detector:
        detector.stop()
        
    print(f"[Control API] Restarting vision pipeline for section: {section_id} (source: {camera_source}, slots: {slots_file})")
    detector = ParkingDetector(
        camera_source  = camera_source,
        slots_config   = slots_file,
        yolo_model     = settings.YOLO_MODEL,
        confidence     = settings.DETECTION_CONFIDENCE,
        process_width  = settings.PROCESS_WIDTH,
        ocr_cooldown   = settings.OCR_COOLDOWN_FRAMES,
        state          = parking_state
    )
    detector.start()
    return {"status": "started", "section_id": section_id, "source": str(camera_source)}


@app.post("/api/control/stop")
async def stop_pipeline():
    """Stops the active detector pipeline."""
    global detector
    if detector:
        detector.stop()
        return {"status": "stopped"}
    return {"status": "already_stopped"}


@app.post("/api/control/calibrate")
async def start_calibration():
    """Extract a frame from source, stop detector, and open the interactive desktop calibration window."""
    import subprocess
    import cv2
    global detector
    
    if not detector:
        return {"error": "Detector pipeline is not active."}
        
    # Grab a frame from the current active source
    source = detector.source
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return {"error": f"Failed to capture frame from video source: {source}"}
        
    temp_img_path = str(settings.PROJECT_ROOT / "backend/marked_slots/temp_calibration_frame.jpg")
    cv2.imwrite(temp_img_path, frame)
    
    slots_file = detector.slots_config
    
    # Stop the detector thread to release the webcam/file locks
    detector.stop()
        
    # Open the desktop calibration OpenCV window using the python script
    python_exe = r"E:\smart_parking_v2\venv\Scripts\python.exe"
    script_path = str(settings.PROJECT_ROOT / "backend/calibrate_slots.py")
    
    cmd = [python_exe, script_path, "--image", temp_img_path, "--output", slots_file]
    print(f"[Control API] Opening calibration subprocess: {' '.join(cmd)}")
    subprocess.Popen(cmd)
    
    return {"status": "calibration_window_opened"}


@app.post("/api/control/calibrate_navigation")
async def start_navigation_calibration():
    """Stops the detector and launches the interactive navigation map calibration window."""
    import subprocess
    global detector
    
    # Stop the detector thread to free CPU/devices
    if detector:
        detector.stop()
        
    # Open the map calibration window
    python_exe = r"E:\smart_parking_v2\venv\Scripts\python.exe"
    script_path = str(settings.PROJECT_ROOT / "backend/calibrate_navigation.py")
    
    cmd = [python_exe, script_path]
    print(f"[Control API] Opening navigation calibration: {' '.join(cmd)}")
    subprocess.Popen(cmd)
    
    return {"status": "navigation_calibration_opened"}



# ── Video Streaming MJPEG Feed ──

async def _frame_generator():
    """Yield MJPEG frames at ~15 FPS."""
    while True:
        frame = parking_state.get_frame()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        await asyncio.sleep(0.067)  # Throttle ~15 FPS


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Serve React Frontend static SPA ──

@app.get("/", response_class=HTMLResponse)
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str = ""):
    # Skip routing backend paths
    if full_path.startswith("api/") or full_path == "video_feed":
        return HTMLResponse("Not Found", status_code=404)
        
    # Check if requested static asset exists in build output
    file_path = FRONTEND_DIST / full_path
    if full_path and file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))

    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
        
    return HTMLResponse(
        "<h2>NextGen Smart Parking Dashboard</h2>"
        "<p>React frontend build not found. Run <code>npm run build</code> inside the frontend directory.</p>"
    )
