"""
telemetry.py — OpenTelemetry Logs instrumentation (raw events only)
======================================================================
Sends raw, per-request key-value events to a local OTel Collector via
gRPC (port 4317). The collector forwards everything to Axiom over HTTPS.

No pre-aggregation happens here — every event is sent as-is with full
attributes, so all counting/averaging/grouping happens in Axiom queries.

Events emitted:
    http_request     — server: one row per API call (endpoint, status)
    feedback_vote     — data:   one row per thumbs up/down
    chat_completed     — model:  one row per chat pipeline run
                                  (language, emotion, intent, latency_ms, used_rag)
    app_lifecycle       — system: startup/shutdown/error events
"""

import logging
import os

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource


def setup_telemetry():
    """
    Initialize OTel LoggerProvider only — no metrics, no aggregation.
    Returns the loggers used to emit raw events, plus the provider for shutdown.
    Call once at app startup.
    """
    resource = Resource.create({"service.name": "serenity-backend"})
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    log_exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)

    # Bridge Python's standard logging → OTel so every logger.info() is exported
    otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.getLogger().addHandler(otel_handler)

    # Loggers — each name becomes a filterable "scope" in Axiom
    request_logger = logging.getLogger("serenity.request")
    feedback_logger = logging.getLogger("serenity.feedback")
    chat_logger = logging.getLogger("serenity.chat")
    system_logger = logging.getLogger("serenity.system")

    # Explicitly attach handler + level to each logger too — belt and suspenders,
    # avoids relying solely on propagation to root in case something resets it
    for lg in (request_logger, feedback_logger, chat_logger, system_logger):
        lg.setLevel(logging.INFO)
        lg.addHandler(otel_handler)

    return (
        request_logger,
        feedback_logger,
        chat_logger,
        system_logger,
        logger_provider,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Surface gRPC export errors instead of failing silently
    logging.getLogger("opentelemetry.exporter.otlp.proto.grpc.exporter").setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.WARNING)

    print("Running telemetry smoke test...")
    print("Make sure the OTel Collector is running: docker compose up otel-collector")
    print()

    (
        request_logger,
        feedback_logger,
        chat_logger,
        system_logger,
        logger_provider,
    ) = setup_telemetry()

    # ── Raw event examples — each is a single row in Axiom ────────────────────
    system_logger.info("app_startup", extra={"event": "warmup_complete", "warmup_ms": 1200})

    request_logger.info("http_request", extra={"endpoint": "/api/chat", "status": 200})
    request_logger.info("http_request", extra={"endpoint": "/api/feedback", "status": 200})

    feedback_logger.info("feedback_vote", extra={"vote": "up"})
    feedback_logger.info("feedback_vote", extra={"vote": "down"})

    chat_logger.info(
        "chat_completed",
        extra={
            "language": "English",
            "emotion": "sadness",
            "intent": "asking_mental_health_question",
            "latency_ms": 842.3,
            "used_rag": True,
            "message_length": 87,
            "session_id": "test-session-001",
        },
    )
    chat_logger.info(
        "chat_completed",
        extra={
            "language": "Arabic",
            "emotion": "fear",
            "intent": "asking_mental_health_question",
            "latency_ms": 910.1,
            "used_rag": True,
            "message_length": 64,
            "session_id": "test-session-002",
        },
    )
    chat_logger.info(
        "chat_completed",
        extra={
            "language": "English",
            "emotion": "none",
            "intent": "greeting",
            "latency_ms": 48.5,
            "used_rag": False,
            "message_length": 5,
            "session_id": "test-session-003",
        },
    )

    print("Raw events emitted. Forcing flush...")
    flush_result = logger_provider.force_flush(timeout_millis=10000)
    print(f"force_flush returned: {flush_result}")

    logger_provider.shutdown()
    print("Done. Check Axiom — each call above should appear as one raw row.")
