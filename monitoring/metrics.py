from prometheus_client import Counter, Histogram, start_http_server
import threading
import time

# Counters and histograms for basic observability
inference_requests = Counter(
    "capstone_inference_requests_total", "Total number of poster inference requests"
)

gaze_analysis_requests = Counter(
    "capstone_gaze_analysis_requests_total", "Total number of gaze analysis requests"
)

processing_seconds = Histogram(
    "capstone_processing_seconds", "Processing time for inference and gaze analysis"
)


def _start_server(port: int = 8000) -> None:
    # This call blocks, so run it in a background thread when needed.
    start_http_server(port)


def start_metrics_server(port: int = 8000) -> None:
    thread = threading.Thread(target=_start_server, args=(port,), daemon=True)
    thread.start()


# Convenience context manager for timing
class timeit:
    def __init__(self, histogram: Histogram):
        self.histogram = histogram

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self._start
        try:
            self.histogram.observe(elapsed)
        except Exception:
            pass
