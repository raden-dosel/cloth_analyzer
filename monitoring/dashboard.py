from prometheus_client import start_http_server, CollectorRegistry
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

class PerformanceMetrics:
    def __init__(self):
        self.latency = GaugeMetricFamily(
            'inference_latency_seconds',
            'Inference processing time',
            labels=['model_version']
        )
        
        self.throughput = CounterMetricFamily(
            'requests_processed_total',
            'Total number of requests',
            labels=['status']
        )

    def collect(self):
        yield self.latency
        yield self.throughput

# Start exporter
start_http_server(8002)
registry = CollectorRegistry()
registry.register(PerformanceMetrics())