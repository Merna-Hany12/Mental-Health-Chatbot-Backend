"""
Minimal isolated test — bypasses telemetry.py entirely.
Run this directly: python minimal_test.py
"""

import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s: %(message)s")

import time
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry._logs import set_logger_provider

print("=" * 60)
print("STEP 1: Setting up LoggerProvider with OTLP gRPC exporter")
print("=" * 60)

resource = Resource.create({"service.name": "minimal-test"})
provider = LoggerProvider(resource=resource)

otlp_exporter = OTLPLogExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_log_record_processor(BatchLogRecordProcessor(otlp_exporter))

# ALSO add a console exporter so we can see locally what's being attempted
console_exporter = ConsoleLogExporter()
provider.add_log_record_processor(BatchLogRecordProcessor(console_exporter))

set_logger_provider(provider)

print("=" * 60)
print("STEP 2: Emitting one log record")
print("=" * 60)

test_logger = logging.getLogger("minimal.test")
otel_handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
test_logger.addHandler(otel_handler)
test_logger.setLevel(logging.INFO)

test_logger.info("minimal_test_event", extra={"foo": "bar", "count": 1})

print("=" * 60)
print("STEP 3: Forcing flush (this is where export happens)")
print("=" * 60)

result = provider.force_flush(timeout_millis=10000)
print(f"force_flush returned: {result}")

print("=" * 60)
print("STEP 4: Shutdown")
print("=" * 60)
provider.shutdown()

print("DONE.")
