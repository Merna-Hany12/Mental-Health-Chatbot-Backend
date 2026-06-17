FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ── OTel Collector binary ───────────────────────────────────────────────────
# Downloaded directly into the final image — no separate container needed.
# Pin the version to match what you tested locally (0.154.0).
ARG OTEL_COLLECTOR_VERSION=0.154.0
RUN apt-get update && apt-get install -y --no-install-recommends curl bash \
    && curl -L -o /tmp/otelcol.tar.gz \
       "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTEL_COLLECTOR_VERSION}/otelcol-contrib_${OTEL_COLLECTOR_VERSION}_linux_amd64.tar.gz" \
    && tar -xzf /tmp/otelcol.tar.gz -C /usr/local/bin otelcol-contrib \
    && chmod +x /usr/local/bin/otelcol-contrib \
    && rm /tmp/otelcol.tar.gz \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ── Collector config + startup script ───────────────────────────────────────
COPY otel-collector-config.yaml /etc/otelcol/config.yaml
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ── App ───────────────────────────────────────────────────────────────────
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

# App listens to the collector on localhost now — no Docker networking needed
ENV OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

EXPOSE 8000
CMD ["/start.sh"]