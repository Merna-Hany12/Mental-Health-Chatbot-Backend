"""
FastAPI Application — Mental Health Chatbot
============================================
Endpoints:
    POST /api/chat      → Main chat endpoint
    POST /api/feedback  → User feedback
    GET  /api/health    → Health check
    GET  /api/modules   → Module status
    POST /api/index     → Trigger dataset indexing (admin)

Run:
    .venv\\Scripts\\Activate.ps1
    uvicorn main:app
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from orchestrator import Orchestrator

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Chatbot API",
    description="RAG-powered empathetic mental health support chatbot.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:80",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "https://merna-hany12.github.io/chatbot-frontend",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    show_sources: bool = Field(True, description="Include RAG sources in response")


class ChatResponse(BaseModel):
    answer: str
    language: dict
    intent: dict
    emotion: Optional[dict]
    sources: list
    used_rag: bool
    latency_ms: float


class FeedbackRequest(BaseModel):
    vote: str  # "up" or "down"
    user_message: str
    bot_response: str
    session_id: Optional[str] = None


# ── Orchestrator (singleton) ───────────────────────────────────────────────────

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(
            groq_api_key=os.getenv("groq_api_key", ""),
            qdrant_url=os.getenv("qdrant_url", ""),
            qdrant_api_key=os.getenv("qdrant_api_key", ""),
        )
    return _orchestrator


@app.on_event("startup")
def warm_up_orchestrator() -> None:
    """Load models and connect external services before the first chat request."""
    t0 = time.perf_counter()
    logger.info("Starting application warmup...")
    try:
        get_orchestrator().warm_up()
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"Application warmup complete in {elapsed} ms.")
    except Exception:
        logger.exception(
            "Application warmup failed. The first chat request will retry initialization."
        )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Runs the full pipeline:
    Language Detection → Intent Classification →
    (Emotion Classification) → (RAG Answer)
    """
    t0 = time.perf_counter()
    try:
        logger.info(f"/api/chat received message ({len(req.message)} chars).")

        bot = get_orchestrator()
        logger.info("Calling orchestrator chat pipeline...")
        result = bot.chat(req.message)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"/api/chat pipeline finished in {latency} ms.")

        if not req.show_sources:
            result["sources"] = []

        return ChatResponse(
            answer=result["answer"],
            language=result["language"],
            intent=result["intent"],
            emotion=result.get("emotion"),
            sources=result.get("sources", []),
            used_rag=result["used_rag"],
            latency_ms=latency,
        )
    except Exception as e:
        import traceback

        logger.error(f"Chat endpoint error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> JSONResponse:
    """Accept thumbs up/down feedback from the frontend."""
    if req.vote not in ("up", "down"):
        logger.warning(f"Unexpected vote value received: {req.vote!r}")
    logger.info(f"Feedback received: vote={req.vote}, msg_len={len(req.user_message)}")
    return JSONResponse({"status": "ok"})


@app.get("/api/health")
async def health() -> JSONResponse:
    """Health check — returns module status."""
    try:
        bot = get_orchestrator()
        status = bot.health_check()
        ok = status["language_detector"] and status["intent_classifier"]
        if not ok:
            logger.warning(f"Health check degraded: {status}")
        return JSONResponse(
            content={"status": "ok" if ok else "degraded", "modules": status},
            status_code=200 if ok else 207,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/modules")
async def modules_info() -> JSONResponse:
    """Detailed module information."""
    return JSONResponse(
        {
            "modules": [
                {
                    "id": 1,
                    "name": "Language Detector",
                    "tech": "TF-IDF (char n-grams) + Logistic Regression",
                    "languages": [
                        "English",
                        "Arabic",
                        "French",
                        "Spanish",
                        "German",
                        "Italian",
                        "Portuguese",
                        "Russian",
                        "Turkish",
                        "Hindi",
                    ],
                },
                {
                    "id": 2,
                    "name": "Emotion Classifier",
                    "tech": "DistilBERT fine-tuned on dair-ai/emotion",
                    "emotions": ["sadness", "joy", "love", "anger", "fear", "surprise"],
                },
                {
                    "id": 3,
                    "name": "Intent Classifier",
                    "tech": "Few-shot prompting via Groq LLM",
                    "intents": [
                        "greeting",
                        "goodbye",
                        "gratitude",
                        "asking_mental_health_question",
                        "out_of_scope",
                    ],
                },
                {
                    "id": 4,
                    "name": "RAG Pipeline",
                    "tech": "LangChain + Qdrant Cloud + SentenceTransformer + Groq",
                    "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
                    "llm": "openai/gpt-oss-120b",
                    "vector_db": "Qdrant Cloud",
                },
            ]
        }
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("APP_DEBUG", "false").lower() == "true",
        workers=1,
    )
