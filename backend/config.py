"""
backend/config.py - loads settings from environment
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def make_absolute(path_str: str | int) -> str | int:
    if isinstance(path_str, int):
        return path_str
    if not path_str:
        return path_str
    if isinstance(path_str, str) and path_str.isdigit():
        return int(path_str)
    # If it's a URL or rtsp/http path, don't modify
    if "://" in path_str or path_str.startswith("rtsp://") or path_str.startswith("http://") or path_str.startswith("https://"):
        return path_str
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


class Settings(BaseSettings):
    CAMERA_SOURCE:         str | int = "videos/aitd_parking_lot.mp4"
    SLOTS_CONFIG:          str       = "backend/marked_slots/slots_ground_floor.json"
    SECTIONS_CONFIG:       str       = "backend/marked_slots/sections_config.json"
    YOLO_MODEL:            str       = "backend/yolov8s.pt"
    ALPR_MODEL:            str       = "license_plate_recognition/best.pt"
    DETECTION_CONFIDENCE:  float     = 0.35
    OCR_COOLDOWN_FRAMES:   int       = 45
    API_HOST:              str       = "0.0.0.0"
    API_PORT:              int       = 8000
    DATABASE_URL:          str       = "sqlite+aiosqlite:///backend/parking.db"
    PROCESS_WIDTH:         int       = 640
    TESSERACT_PATH:        str       = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    PROJECT_ROOT:          Path      = PROJECT_ROOT

    class Config:
        env_file = ".env"
        extra = "ignore"

    def make_absolute(self, path_str: str | int) -> str | int:
        return make_absolute(path_str)

    def model_post_init(self, __context):
        # Resolve CAMERA_SOURCE
        object.__setattr__(self, "CAMERA_SOURCE", make_absolute(self.CAMERA_SOURCE))
        
        # Resolve other configuration files
        object.__setattr__(self, "SLOTS_CONFIG", make_absolute(self.SLOTS_CONFIG))
        object.__setattr__(self, "SECTIONS_CONFIG", make_absolute(self.SECTIONS_CONFIG))
        object.__setattr__(self, "YOLO_MODEL", make_absolute(self.YOLO_MODEL))
        object.__setattr__(self, "ALPR_MODEL", make_absolute(self.ALPR_MODEL))
        
        # Resolve DATABASE_URL
        db_url = self.DATABASE_URL
        if db_url.startswith("sqlite+aiosqlite:///"):
            db_path_part = db_url[20:]
            if not Path(db_path_part).is_absolute() and not db_path_part.startswith("/"):
                abs_db_path = (PROJECT_ROOT / db_path_part).resolve()
                abs_db_path_str = str(abs_db_path).replace("\\", "/")
                object.__setattr__(self, "DATABASE_URL", f"sqlite+aiosqlite:///{abs_db_path_str}")
