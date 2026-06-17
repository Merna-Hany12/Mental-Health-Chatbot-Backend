#!/bin/bash
# start.sh — runs both the OTel Collector and the FastAPI app in one container
set -e

echo "[start.sh] Starting OTel Collector..."
/usr/local/bin/otelcol-contrib --config=/etc/otelcol/config.yaml &
OTEL_PID=$!

# Give the collector a moment to bind its gRPC port before the app starts
sleep 2

echo "[start.sh] Starting FastAPI app..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
APP_PID=$!

# If either process dies, stop the container — restart policy handles recovery
wait -n "$OTEL_PID" "$APP_PID"
exit $?