import logging
import os
from contextlib import contextmanager
from time import perf_counter

import requests
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

_log = logging.getLogger(__name__)
AXIOM_TOKEN = os.getenv("AXIOM_TOKEN", "")
AXIOM_DATASET = os.getenv("AXIOM_DATASET", "mental_health")


_METER_NAME = "serenity.metrics"


def setup_telemetry():
    # Identify application name for Axiom
    resource = Resource.create(
        attributes={
            "service.name": "serenity-backend",
            "service.version": os.getenv("APP_VERSION", "unknown"),
        }
    )
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    headers = dict(pair.split("=", 1) for pair in headers_env.split(",") if "=" in pair) or None
    exporter = OTLPMetricExporter(
        endpoint=endpoint,
        headers=headers,
        insecure=endpoint.startswith("http://"),
    )
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=int(os.getenv("OTEL_EXPORT_INTERVAL_MS", "15000")),
    )
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter(_METER_NAME)
    instruments = {}
    # ── 1. System / Infrastructure ──────────────────────────────────────────
    instruments["http_requests"] = meter.create_counter(
        "serenity_http_requests_total",
        description="Total API requests, labelled by endpoint and status code",
    )
    instruments["http_request_duration"] = meter.create_histogram(
        "serenity_http_request_duration_seconds",
        description="End-to-end request latency — drives P95/P99 panels",
        unit="s",
    )
    instruments["http_errors"] = meter.create_counter(
        "serenity_http_errors_total",
        description="Failed requests (5xx / unhandled exceptions), labelled by endpoint",
    )
    # ── 2. NLP Pipeline — Language & Intent ─────────────────────────────────
    instruments["language_detected"] = meter.create_counter(
        "serenity_language_detected_total",
        description="Detected input language, labelled by language code",
    )
    instruments["intent_classified"] = meter.create_counter(
        "serenity_intent_classified_total",
        description="Classified intent, labelled by intent label "
        "(mental_health, general_chat, out_of_scope, crisis)",
    )
    # ── 3. Emotion Monitoring ────────────────────────────────────────────────
    instruments["emotion_detected"] = meter.create_counter(
        "serenity_emotion_detected_total",
        description="Detected emotion, labelled by emotion class "
        "(sadness, joy, love, anger, fear, surprise)",
    )
    # ── 4. RAG Retrieval Quality ─────────────────────────────────────────────
    instruments["retrieval_top1_score"] = meter.create_histogram(
        "serenity_retrieval_top1_score",
        description="Top-1 cosine similarity score per query — "
        "alert if rolling average drops below 0.5",
    )
    instruments["retrieved_chunk_count"] = meter.create_histogram(
        "serenity_retrieved_chunk_count",
        description="Number of chunks retrieved per RAG query",
    )
    # ── 5. Cost / Routing Efficiency ─────────────────────────────────────────
    instruments["llm_calls"] = meter.create_counter(
        "serenity_llm_calls_total",
        description="LLM invocations, labelled by route "
        "(rag_triggered vs general_response) — used to compute RAG activation rate",
    )
    # ── 6. Safety / Crisis Monitoring ────────────────────────────────────────
    instruments["crisis_escalation"] = meter.create_counter(
        "serenity_crisis_escalation_total",
        description="Crisis escalations triggered, labelled by crisis_type "
        "(self_harm, emergency_phrase, other)",
    )
    # ── 7. User Input Shape ───────────────────────────────────────────────────
    # Anomalously long inputs can indicate prompt injection / misuse.
    instruments["input_length"] = meter.create_histogram(
        "serenity_input_length_chars",
        description="User message length in characters",
    )
    # ── 8. Feedback ───────────────────────────────────────────────────────────
    instruments["feedback_votes"] = meter.create_counter(
        "serenity_feedback_votes_total",
        description="Thumbs up/down votes, labelled by vote (up, down)",
    )
    return instruments


@contextmanager
def timed_histogram(histogram, attributes: dict | None = None):
    """
    Convenience context manager for timing a block and recording it
    to a histogram in seconds.
        with timed_histogram(METRICS["http_request_duration"], {"endpoint": "/chat"}):
            do_work()
    """
    start = perf_counter()
    try:
        yield
    finally:
        histogram.record(perf_counter() - start, attributes or {})


def log_chat_event(
    latency_ms: float,
    language: str,
    emotion: str,
    intent: str,
    used_rag: bool,
    message_length: int,
    session_id: str = "",
) -> None:
    """
    Send a structured chat_completed event directly to Axiom's ingest API.
    This is used instead of OTel histograms because Axiom does not store
    OTel histogram metrics as queryable fields.
    Dashboard queries can filter with:  body == "chat_completed"
    """
    if not AXIOM_TOKEN:
        _log.warning("AXIOM_TOKEN not set — skipping log_chat_event")
        return
    try:
        requests.post(
            f"https://api.axiom.co/v1/datasets/{AXIOM_DATASET}/ingest",
            headers={
                "Authorization": f"Bearer {AXIOM_TOKEN}",
                "Content-Type": "application/json",
            },
            json=[
                {
                    "body": "chat_completed",
                    "attributes.latency_ms": latency_ms,
                    "attributes.language": language,
                    "attributes.emotion": emotion,
                    "attributes.intent": intent,
                    "attributes.used_rag": used_rag,
                    "attributes.message_length": message_length,
                    "attributes.session_id": session_id,
                }
            ],
            timeout=3,
        )
    except Exception as exc:
        # Never let telemetry block or crash the chat response
        _log.warning("log_chat_event failed: %s", exc)
