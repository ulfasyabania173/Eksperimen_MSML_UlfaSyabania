from prometheus_client import start_http_server, Summary, Counter
import time
import random

REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests')
REQUEST_LATENCY = Summary('inference_latency_seconds', 'Inference latency in seconds')
ERROR_COUNT = Counter('inference_errors_total', 'Total inference errors')

@REQUEST_LATENCY.time()
def inference():
    REQUEST_COUNT.inc()
    if random.random() < 0.1:
        ERROR_COUNT.inc()
        raise Exception("Random error")
    time.sleep(random.uniform(0.1, 0.5))
    return "ok"

if __name__ == '__main__':
    start_http_server(8001)
    while True:
        try:
            inference()
        except Exception:
            pass
        time.sleep(1)
