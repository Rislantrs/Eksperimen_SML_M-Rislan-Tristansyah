from prometheus_client import start_http_server, Counter, Gauge, Summary
import psutil
import time
import threading

# 1-3. Request Counters
REQUEST_COUNT = Counter('model_requests_total', 'Total number of requests made to the model')
SUCCESS_COUNT = Counter('model_success_responses_total', 'Total number of successful responses')
ERROR_COUNT = Counter('model_error_responses_total', 'Total number of failed responses')

# 4. Latency
LATENCY = Summary('model_inference_latency_seconds', 'Time spent processing inference request')

# 5-6. Prediction Classes
PRED_CLASS_0 = Counter('model_predictions_class_0_total', 'Total predictions for class 0 (e.g., No)')
PRED_CLASS_1 = Counter('model_predictions_class_1_total', 'Total predictions for class 1 (e.g., Yes)')

# 7-8. System Metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Current CPU usage percent')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Current memory usage in bytes')

# 9-10. ML Specific Metrics
DRIFT_SCORE = Gauge('model_drift_score', 'Simulated data drift score')
CONFIDENCE_SCORE = Gauge('model_confidence_score_avg', 'Average confidence score of the model')

def update_system_metrics():
    """Background thread to update system metrics every 5 seconds"""
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        time.sleep(5)

def start_exporter(port=8000):
    print(f"Starting Prometheus Exporter on port {port}...")
    start_http_server(port)
    
    # Start background thread for system metrics
    sys_thread = threading.Thread(target=update_system_metrics, daemon=True)
    sys_thread.start()

if __name__ == '__main__':
    start_exporter()
    while True:
        time.sleep(1)
