"""
FastAPI entry point for the smartphone spectrometer backend.

Endpoints:
    POST   /analyse              run preset pipeline + save session
    GET    /sessions             list saved sessions
    GET    /sessions/{id}        get one session
    DELETE /sessions/{id}        delete one session
    POST   /compare              overlay two or more sessions
    POST   /calibration          regression on concentration-labelled sessions
    POST   /export/csv           export a session as CSV
    POST   /export/pdf           export a session as PDF report
    GET    /health               liveness probe
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import (
    init_db,
    get_db,
    create_session_record,
    get_session_record,
    list_session_records,
    delete_session_record,
)
from input import handle_upload
from analysis import analyse_spectrum
from storage_export import export_csv, export_pdf
from comparison import compare_sessions, run_calibration


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Spectrometer Backend", version="0.1.0", lifespan=lifespan)

# Permissive CORS for local Flutter development. Tighten before deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request body models ----------

class CompareRequest(BaseModel):
    session_ids: List[str]


class CalibrationRequest(BaseModel):
    session_ids: List[str]
    x_metric: str = "integrated_reflectance"  # or "peak_reflectance"


class ExportRequest(BaseModel):
    session_id: str


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse")
async def analyse_endpoint(
    sample: UploadFile = File(...),
    white: UploadFile = File(...),
    dark: UploadFile = File(...),
    preset: str = Form(...),
    concentration: Optional[float] = Form(None),
    crop_x1: int = Form(...),
    crop_y1: int = Form(...),
    crop_x2: int = Form(...),
    crop_y2: int = Form(...),
    db: Session = Depends(get_db),
):
    parsed = await handle_upload(
        sample, white, dark, preset, concentration,
        (crop_x1, crop_y1, crop_x2, crop_y2),
    )
    result = analyse_spectrum(
        preset=parsed["preset"],
        wavelengths=parsed["wavelengths"],
        sample_lum=parsed["sample_lum"],
        white_lum=parsed["white_lum"],
        dark_lum=parsed["dark_lum"],
        sample_rgb=parsed["sample_rgb"],
        white_rgb=parsed["white_rgb"],
        dark_rgb=parsed["dark_rgb"],
    )
    sid = create_session_record(
        db,
        preset=parsed["preset"],
        concentration=parsed["concentration"],
        result=result,
    )
    return {
        "session_id": sid,
        "preset": parsed["preset"],
        "concentration": parsed["concentration"],
        "result": result,
    }


@app.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    return list_session_records(db)


@app.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    rec = get_session_record(db, session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return rec


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    ok = delete_session_record(db, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.post("/compare")
def compare(req: CompareRequest, db: Session = Depends(get_db)):
    if len(req.session_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sessions to compare.")
    try:
        return compare_sessions(db, req.session_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/calibration")
def calibrate(req: CalibrationRequest, db: Session = Depends(get_db)):
    if len(req.session_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sessions.")
    try:
        return run_calibration(db, req.session_ids, x_metric=req.x_metric)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/export/csv")
def export_csv_endpoint(req: ExportRequest, db: Session = Depends(get_db)):
    rec = get_session_record(db, req.session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Session not found")
    path = export_csv(rec)
    return FileResponse(path, media_type="text/csv", filename=f"{req.session_id}.csv")


@app.post("/export/pdf")
def export_pdf_endpoint(req: ExportRequest, db: Session = Depends(get_db)):
    rec = get_session_record(db, req.session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Session not found")
    path = export_pdf(rec)
    return FileResponse(path, media_type="application/pdf", filename=f"{req.session_id}.pdf")