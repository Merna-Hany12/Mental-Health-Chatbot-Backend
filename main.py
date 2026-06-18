"""
FastAPI Application — Mental Health Chatbot
============================================
Endpoints:
    POST /api/chat      → Main chat endpoint
    POST /api/feedback  → User feedback
    GET  /api/health    → Health check
    GET  /api/modules   → Module status

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
from telemetry import setup_telemetry, timed_histogram  # 💡 Imported context manager tool

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Telemetry Initialization ──────────────────────────────────────────────────
# 💡 Captures the single metrics dictionary cleanly without unpacking errors!
METRICS = setup_telemetry()

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
        "https://merna-hany12.github.io",
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
    logger.info("Starting application warmup...")
    try:
        get_orchestrator().warm_up()
        logger.info("Application warmup complete.")
    except Exception:
        logger.exception(
            "Application warmup failed. The first chat request will retry initialization."
        )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    Runs the full pipeline with live end-to-end telemetry profiling.
    """
    t0 = time.perf_counter()
    logger.info(f"/api/chat received message ({len(req.message)} chars).")

    # 🧠 Use your custom context manager to measure absolute pipeline duration in seconds
    with timed_histogram(METRICS["http_request_duration"], {"endpoint": "/api/chat"}):
        try:
            bot = get_orchestrator()
            result = bot.chat(req.message)

            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.info(f"/api/chat pipeline finished in {latency_ms} ms.")

            # Extract granular fields
            detected_intent = result.get("intent", {}).get("intent", "unknown")
            detected_language = result.get("language", {}).get("language", "unknown")
            detected_emotion = (
                result.get("emotion", {}).get("emotion", "none")
                if result.get("emotion")
                else "none"
            )

            # ── 📊 Core OTel Metric Registrations ──────────────────────────────────
            METRICS["http_requests"].add(1, {"endpoint": "/api/chat", "status": "200"})
            METRICS["language_detected"].add(1, {"language": detected_language})
            METRICS["intent_classified"].add(1, {"intent": detected_intent})
            METRICS["emotion_detected"].add(1, {"emotion": detected_emotion})
            METRICS["input_length"].record(len(req.message))

            if result.get("used_rag"):
                METRICS["llm_calls"].add(1, {"route": "rag_triggered"})
            else:
                METRICS["llm_calls"].add(1, {"route": "general_response"})

            if not req.show_sources:
                result["sources"] = []

            return ChatResponse(
                answer=result["answer"],
                language=result["language"],
                intent=result["intent"],
                emotion=result.get("emotion"),
                sources=result.get("sources", []),
                used_rag=result["used_rag"],
                latency_ms=latency_ms,
            )

        except Exception as e:
            # 🛑 Register the exact platform breakdown metric
            METRICS["http_requests"].add(1, {"endpoint": "/api/chat", "status": "500"})
            METRICS["http_errors"].add(1, {"endpoint": "/api/chat"})

            # Check for critical keywords to flag system safety issues
            if detected_intent == "crisis":
                METRICS["crisis_escalation"].add(1, {"crisis_type": "self_harm"})

            logger.error(f"Chat endpoint error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> JSONResponse:
    """Accept thumbs up/down feedback from the frontend."""
    METRICS["http_requests"].add(1, {"endpoint": "/api/feedback", "status": "200"})

    # 📊 Record the feedback choice straight to your OTel Counter metric
    if req.vote in ("up", "down"):
        METRICS["feedback_votes"].add(1, {"vote": req.vote})
        logger.info(f"Feedback metric recorded: vote={req.vote}")
    else:
        logger.warning(f"Unexpected vote value received: {req.vote!r}")

    return JSONResponse({"status": "ok"})


@app.get("/api/health")
async def health() -> JSONResponse:
    """Health check — returns module status."""
    try:
        bot = get_orchestrator()
        status = bot.health_check()
        ok = status["language_detector"] and status["intent_classifier"]

        status_code = "200" if ok else "207"
        METRICS["http_requests"].add(1, {"endpoint": "/api/health", "status": status_code})

        if not ok:
            logger.warning(f"Health check degraded: {status}")

        return JSONResponse(
            content={"status": "ok" if ok else "degraded", "modules": status},
            status_code=200 if ok else 207,
        )
    except Exception as e:
        METRICS["http_requests"].add(1, {"endpoint": "/api/health", "status": "500"})
        METRICS["http_errors"].add(1, {"endpoint": "/api/health"})
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/modules")
async def modules_info() -> JSONResponse:
    """Detailed module information."""
    METRICS["http_requests"].add(1, {"endpoint": "/api/modules", "status": "200"})
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
