"""
OpenTelemetry 全链路追踪

把每一步（检索、重排、生成、缓存）都打上 Trace
当用户反馈"回答不对"时，能定位到具体是哪一步出了问题

部署方案：
- 应用侧用 opentelemetry-sdk 采集
- 通过 OTLP exporter 发送到 Jaeger / Tempo
- 在 Grafana 里看全链路看板
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)


def setup_tracing(endpoint: str, service_name: str):
    """初始化 OpenTelemetry 追踪"""
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def get_tracer():
    """获取全局 tracer"""
    return trace.get_tracer("ai-agent-qa")
