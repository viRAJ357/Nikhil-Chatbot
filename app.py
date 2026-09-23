"""
app.py — FastAPI backend for Nikhil AI Chatbot
Model is loaded ONCE at startup and reused for every request.
"""

import json, logging
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

from config import API_TITLE, API_VERSION, API_HOST, API_PORT, METRICS_PATH, MODEL_INFO_PATH
from chatbot_engine import ChatbotEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title=API_TITLE, version=API_VERSION,
              description="Production-grade AI Chatbot — CLINC150 dataset")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Global engine (loaded once) ──────────────────────────────────────────────
engine: Optional[ChatbotEngine] = None

@app.on_event("startup")
async def startup():
    global engine
    logger.info("🚀 Loading Nikhil AI chatbot engine …")
    try:
        engine = ChatbotEngine()
        logger.info("✅ Chatbot engine ready!")
    except FileNotFoundError:
        logger.error("❌ Model not found! Run:  python train.py  first.")
        raise

# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

    @field_validator("message")
    @classmethod
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > 1000:
            raise ValueError("Message too long (max 1000 chars)")
        return v

class ChatResponse(BaseModel):
    response: str
    intent: Optional[str]
    confidence: float
    is_confident: bool
    top_k: List[dict]
    timestamp: str

class PredictRequest(BaseModel):
    text: str

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": engine is not None, "version": API_VERSION}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if engine is None:
        raise HTTPException(503, "Chatbot not initialised — run train.py first")
    try:
        result = engine.chat(req.message)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/predict")
async def predict(req: PredictRequest):
    if engine is None:
        raise HTTPException(503, "Chatbot not initialised")
    return engine.predictor.predict(req.text)

@app.get("/api/model-info")
async def model_info():
    if not MODEL_INFO_PATH.exists():
        raise HTTPException(404, "Run train.py first")
    return json.loads(MODEL_INFO_PATH.read_text())

@app.get("/api/metrics")
async def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Run train.py first")
    return json.loads(METRICS_PATH.read_text())

@app.get("/api/history")
async def history():
    if engine is None:
        raise HTTPException(503, "Chatbot not initialised")
    return {"history": engine.get_history(), "stats": engine.get_stats()}

@app.delete("/api/history")
async def clear_history():
    if engine is None:
        raise HTTPException(503, "Chatbot not initialised")
    engine.clear()
    return {"message": "Conversation cleared ✅"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", API_PORT))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
