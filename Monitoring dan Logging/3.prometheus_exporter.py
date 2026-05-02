from prometheus_client import start_http_server, Counter, Gauge, Histogram
import psutil
import time
import threading

# =============================================================================
# METRIK 1: Total Request ke Model (Counter)
# Menghitung jumlah total permintaan yang diterima oleh model inferensi.
# =============================================================================
REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total number of inference requests received by the model'
)

# =============================================================================
# METRIK 2: Latensi Inferensi (Histogram)
# Mengukur distribusi waktu yang dibutuhkan model untuk memproses satu request.
# Histogram lebih baik dari Summary karena mendukung agregasi antar-instance.
# =============================================================================
INFERENCE_LATENCY = Histogram(
    'model_inference_latency_seconds',
    'Histogram of inference latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# =============================================================================
# METRIK 3: Penggunaan CPU Sistem (Gauge)
# Memantau persentase penggunaan CPU secara real-time selama model berjalan.
# =============================================================================
CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Current system CPU usage in percent'
)

# =============================================================================
# Metrik Tambahan (untuk level Advance / kelengkapan)
# =============================================================================
# Jumlah respons sukses
SUCCESS_COUNT = Counter(
    'model_success_responses_total',
    'Total number of successful model responses (HTTP 200)'
)

# Jumlah respons gagal / error
ERROR_COUNT = Counter(
    'model_error_responses_total',
    'Total number of failed model responses (non-200 or exception)'
)

# Distribusi prediksi per kelas
PRED_CLASS_0 = Counter(
    'model_predictions_class_0_total',
    'Total predictions for class 0 (No Depression)'
)
PRED_CLASS_1 = Counter(
    'model_predictions_class_1_total',
    'Total predictions for class 1 (Depression)'
)

# Penggunaan memori sistem
MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'Current system memory usage in bytes'
)

# Skor drift data (simulasi)
DRIFT_SCORE = Gauge(
    'model_drift_score',
    'Simulated data drift score (0 = no drift, 1 = full drift)'
)

# Rata-rata confidence score prediksi
CONFIDENCE_SCORE = Gauge(
    'model_confidence_score_avg',
    'Average prediction confidence score of the model'
)


def update_system_metrics():
    """
    Background thread yang memperbarui metrik sistem (CPU, RAM) setiap 5 detik.
    Berjalan sebagai daemon sehingga otomatis berhenti saat program utama selesai.
    """
    while True:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        time.sleep(5)


def start_exporter(port: int = 8000):
    """
    Memulai HTTP server Prometheus pada port yang ditentukan dan menjalankan
    thread pembaruan metrik sistem di latar belakang.

    Args:
        port (int): Port yang digunakan untuk mengekspos metrik Prometheus.
                    Default: 8000.
    """
    print(f"[Prometheus Exporter] Starting on port {port}...")
    start_http_server(port)
    print(f"[Prometheus Exporter] Metrics available at http://localhost:{port}/metrics")

    # Jalankan thread pembaruan metrik sistem di latar belakang
    sys_thread = threading.Thread(target=update_system_metrics, daemon=True)
    sys_thread.start()
    print("[Prometheus Exporter] System metrics updater thread started.")


if __name__ == '__main__':
    start_exporter(port=8000)
    print("[Prometheus Exporter] Running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
