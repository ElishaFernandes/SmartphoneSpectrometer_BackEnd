"""
Persistent storage of analysis sessions.

DATABASE_URL controls the backend:
    sqlite:///./spectrometer.db          (default; zero-config development)
    postgresql+psycopg2://user:pw@host/db (production)

The schema uses portable column types (JSON, Float, String, DateTime) so
no code changes are required when migrating from SQLite to Postgres. JSON
columns serialise to TEXT on SQLite and to JSONB-equivalent on Postgres.
"""
from __future__ import annotations

import os
import uuid
import datetime as dt
from typing import Optional, List

from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./spectrometer.db")

_engine_kwargs: dict = {}
if DATABASE_URL.startswith("sqlite"):
    # required so FastAPI can use the same connection across threads
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(String, primary_key=True)
    preset = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=dt.datetime.utcnow)
    concentration = Column(Float, nullable=True)

    wavelength_data = Column(JSON, nullable=False)
    reflectance_data = Column(JSON, nullable=False)
    derivative_data = Column(JSON, nullable=True)
    second_derivative_data = Column(JSON, nullable=True)

    ratios = Column(JSON, nullable=True)
    peak_values = Column(JSON, nullable=True)
    integrated_reflectance = Column(Float, nullable=True)
    graph_paths = Column(JSON, nullable=True)

    # preset-specific extras (chlorophyll proxy, bilirubin band features, etc.)
    extras = Column(JSON, nullable=True)


# ---------- lifecycle ----------

def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a SQLAlchemy session and closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- CRUD ----------

def _record_to_dict(rec: AnalysisSession) -> dict:
    return {
        "id": rec.id,
        "preset": rec.preset,
        "timestamp": rec.timestamp.isoformat() if rec.timestamp else None,
        "concentration": rec.concentration,
        "wavelength_data": rec.wavelength_data,
        "reflectance_data": rec.reflectance_data,
        "derivative_data": rec.derivative_data,
        "second_derivative_data": rec.second_derivative_data,
        "ratios": rec.ratios,
        "peak_values": rec.peak_values,
        "integrated_reflectance": rec.integrated_reflectance,
        "graph_paths": rec.graph_paths,
        "extras": rec.extras,
    }


def create_session_record(db: Session,
                          preset: str,
                          concentration: Optional[float],
                          result: dict) -> str:
    """Persist an analysis result and return the new session id."""
    sid = str(uuid.uuid4())
    extras_keys = ("fractional_derivative_alpha_0_3", "chlorophyll_proxy", "bilirubin_features")
    extras = {k: result[k] for k in extras_keys if k in result}

    rec = AnalysisSession(
        id=sid,
        preset=preset,
        concentration=concentration,
        wavelength_data=result["wavelengths_nm"],
        reflectance_data=result["reflectance"],
        derivative_data=result.get("first_derivative"),
        second_derivative_data=result.get("second_derivative"),
        ratios=result.get("ratios"),
        peak_values=result.get("peaks"),
        integrated_reflectance=result.get("integrated_reflectance"),
        extras=extras or None,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return sid


def get_session_record(db: Session, session_id: str) -> Optional[dict]:
    rec = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    return _record_to_dict(rec) if rec else None


def list_session_records(db: Session) -> List[dict]:
    recs = db.query(AnalysisSession).order_by(AnalysisSession.timestamp.desc()).all()
    return [_record_to_dict(r) for r in recs]


def delete_session_record(db: Session, session_id: str) -> bool:
    rec = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if rec is None:
        return False
    db.delete(rec)
    db.commit()
    return True