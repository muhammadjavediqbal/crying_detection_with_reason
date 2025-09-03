from __future__ import annotations
import uuid
import json
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from schemas import VisionAnalysisResult, RuleBasedFusion, HeuristicAudioDetector, HeuristicVisionAnalyzer
from core import db_conn, AUDIO_DIR, IMAGE_DIR
from utils import now_iso, save_upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baby-mvp")

app = FastAPI(title="Baby Cry Reasoning – MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_detector = HeuristicAudioDetector()
vision_analyzer = HeuristicVisionAnalyzer()
fuser = RuleBasedFusion()


@app.get("/health")
def health():
    return {"status": "ok", "time": now_iso()}


@app.post("/detect_cry")
async def detect_cry(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    content = await file.read()
    res = audio_detector.detect(content)

    # Save
    audio_path = save_upload(file, AUDIO_DIR)
    sid = str(uuid.uuid4())
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO samples (id, created_at, audio_path, is_crying, cry_score, cry_type) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, now_iso(), audio_path, int(res.is_crying), float(res.score), res.cry_type),
        )
        conn.commit()

    return JSONResponse({
        "sample_id": sid,
        "is_crying": res.is_crying,
        "score": res.score,
        "cry_type": res.cry_type,
        "details": res.details,
    })


@app.post("/analyze_face")
async def analyze_face(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    content = await file.read()
    res = vision_analyzer.analyze(content)

    # Save
    image_path = save_upload(file, IMAGE_DIR)
    sid = str(uuid.uuid4())
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO samples (id, created_at, image_path, emotion, posture) VALUES (?, ?, ?, ?, ?)",
            (sid, now_iso(), image_path, res.emotion, res.posture),
        )
        conn.commit()

    return JSONResponse({
        "sample_id": sid,
        "emotion": res.emotion,
        "posture": res.posture,
        "details": res.details,
    })


@app.post("/predict_reason")
async def predict_reason(audio: UploadFile = File(...), image: Optional[UploadFile] = File(None)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")

    audio_bytes = await audio.read()
    ares = audio_detector.detect(audio_bytes)
    audio_path = save_upload(audio, AUDIO_DIR)

    vres = VisionAnalysisResult()
    image_path = None
    if image and image.filename:
        img_bytes = await image.read()
        vres = vision_analyzer.analyze(img_bytes)
        image_path = save_upload(image, IMAGE_DIR)

    fusion = fuser.predict(ares, vres)

    sid = str(uuid.uuid4())
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO samples (id, created_at, audio_path, image_path, is_crying, cry_score, cry_type, emotion, posture, predicted_reason, reason_scores)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid, now_iso(), audio_path, image_path, int(ares.is_crying), float(ares.score), ares.cry_type,
                vres.emotion, vres.posture, fusion.predicted_reason, json.dumps(fusion.scores),
            ),
        )
        conn.commit()

    return JSONResponse({
        "sample_id": sid,
        "audio": {
            "is_crying": ares.is_crying,
            "score": ares.score,
            "cry_type": ares.cry_type,
            "details": ares.details,
        },
        "vision": {
            "emotion": vres.emotion,
            "posture": vres.posture,
            "details": vres.details,
        },
        "fusion": {
            "predicted_reason": fusion.predicted_reason,
            "scores": fusion.scores,
            "rationale": fusion.rationale,
        }
    })

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
