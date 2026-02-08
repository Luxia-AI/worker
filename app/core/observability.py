import os
import time
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

worker_jobs_consumed_total = Counter("worker_jobs_consumed_total", "Worker jobs consumed")
worker_jobs_completed_total = Counter("worker_jobs_completed_total", "Worker jobs completed")
worker_jobs_failed_total = Counter("worker_jobs_failed_total", "Worker jobs failed")
worker_kafka_send_failures_total = Counter("worker_kafka_send_failures_total", "Kafka send failures from worker")
worker_kafka_consume_failures_total = Counter("worker_kafka_consume_failures_total", "Kafka consume/process failures")
worker_fallback_total = Counter("worker_fallback_total", "Fallback responses due to unavailable pipeline")
worker_external_calls_total = Counter(
    "worker_external_calls_total",
    "External dependency calls by provider and status",
    ["provider", "status"],
)
worker_stage_duration_seconds = Histogram(
    "worker_stage_duration_seconds",
    "Worker stage duration seconds",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
worker_job_duration_seconds = Histogram(
    "worker_job_duration_seconds",
    "Worker total job processing time",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
worker_jobs_in_flight = Gauge("worker_jobs_in_flight", "Worker jobs currently in flight")
worker_queue_estimate = Gauge("worker_queue_depth_estimate", "Best-effort worker queue depth estimate")

_TRACING_INITIALIZED = False


def setup_tracing(app) -> None:
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    service_name = os.getenv("OTEL_SERVICE_NAME", "worker")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    _TRACING_INITIALIZED = True


def get_trace_context() -> dict[str, Optional[str]]:
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if not span_context or not span_context.is_valid:
        return {"trace_id": None, "span_id": None, "parent_span_id": None}
    parent = span.parent
    return {
        "trace_id": f"{span_context.trace_id:032x}",
        "span_id": f"{span_context.span_id:016x}",
        "parent_span_id": f"{parent.span_id:016x}" if parent else None,
    }


@contextmanager
def stage_timer(stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        worker_stage_duration_seconds.labels(stage=stage).observe(time.perf_counter() - start)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
