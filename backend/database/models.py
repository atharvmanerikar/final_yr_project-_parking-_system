"""
backend/database/models.py
SQLAlchemy models + async DB setup for Smart Parking
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ParkingEvent(Base):
    """Every entry and exit is logged here."""
    __tablename__ = "parking_events"

    id          = Column(Integer, primary_key=True, index=True)
    track_id    = Column(Integer, index=True)           # ByteTrack ID
    slot_id     = Column(String, index=True)            # e.g. "1", "2", "3"
    plate       = Column(String, nullable=True)         # OCR result, may be None
    ocr_conf    = Column(Float, nullable=True)          # OCR confidence 0-1
    event_type  = Column(String)                        # "entry" | "exit" | "parked"
    timestamp   = Column(DateTime, default=datetime.utcnow)
    dwell_secs  = Column(Integer, nullable=True)        # filled on exit


class SlotState(Base):
    """Current state snapshot of every parking slot."""
    __tablename__ = "slot_states"

    slot_id     = Column(String, primary_key=True)
    status      = Column(String, default="free")        # free | occupied
    track_id    = Column(Integer, nullable=True)
    plate       = Column(String, nullable=True)
    entry_time  = Column(DateTime, nullable=True)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency injector for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        yield session
