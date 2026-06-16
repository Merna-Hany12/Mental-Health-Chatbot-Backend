import os
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

def setup_telemetry():
    # Identify your application name for Axiom
    resource = Resource.create(attributes={"service.name": "serenity-backend"})
    
    # Check where our background OTel Collector proxy is running
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    # Set up local streaming exporter
    exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    reader = PeriodicExportingMetricReader(exporter)
    
    # Register the system
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    
    meter = metrics.get_meter("serenity.metrics")
    
    # Define the 3 explicit rubric metrics
    server_requests = meter.create_counter(
        "serenity_http_requests_total",
        description="Total incoming API traffic calls"
    )
    data_feedback = meter.create_counter(
        "serenity_feedback_votes_total",
        description="Frontend thumbs up/down vote counter"
    )
    model_latency = meter.create_histogram(
        "serenity_pipeline_duration_seconds",
        description="Pipeline inference execution duration"
    )
    
    return server_requests, data_feedback, model_latency