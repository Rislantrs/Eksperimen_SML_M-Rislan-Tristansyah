import requests
import time
import random
import sys
import os
import importlib.util

# =============================================================================
# Import modul prometheus_exporter menggunakan importlib karena nama file
# dimulai dengan angka (tidak bisa di-import langsung dengan 'import').
# =============================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
_exporter_path = os.path.join(_script_dir, "3.prometheus_exporter.py")

spec = importlib.util.spec_from_file_location("prometheus_exporter", _exporter_path)
exporter = importlib.util.module_from_spec(spec)
sys.modules["prometheus_exporter"] = exporter
spec.loader.exec_module(exporter)


def simulate_inference():
    """
    Mensimulasikan pengiriman request inferensi ke model MLflow yang sedang
    berjalan, sambil mengekspos metrik ke Prometheus.

    Metrik yang diperbarui setiap iterasi:
        - model_requests_total          : Jumlah total request
        - model_success_responses_total : Jumlah request berhasil (HTTP 200)
        - model_error_responses_total   : Jumlah request gagal
        - model_inference_latency_seconds: Histogram latensi inferensi
        - model_predictions_class_0_total: Prediksi kelas 0 (No Depression)
        - model_predictions_class_1_total: Prediksi kelas 1 (Depression)
        - model_drift_score              : Simulasi skor data drift
        - model_confidence_score_avg     : Simulasi rata-rata confidence score
    """
    # Mulai server Prometheus Exporter pada port 8000
    exporter.start_exporter(8000)
    print("Inference simulation started. Sending requests to MLflow model...")
    print("Press Ctrl+C to stop.\n")

    # Endpoint model MLflow yang sedang di-serve
    URL = "http://localhost:5001/invocations"

    while True:
        # Catat satu request
        exporter.REQUEST_COUNT.inc()
        start_time = time.time()

        # Buat data dummy sesuai skema fitur model mental health
        dummy_data = {
            "dataframe_split": {
                "columns": [
                    "age", "gender", "daily_social_media_hours", "sleep_hours",
                    "screen_time_before_sleep", "academic_performance", "physical_activity",
                    "social_interaction_level", "stress_level", "anxiety_level",
                    "addiction_level", "Screen_Sleep_Ratio", "platform_usage_Instagram",
                    "platform_usage_TikTok", "Age_Group_Late Teen"
                ],
                "data": [[
                    random.randint(15, 25),
                    random.randint(0, 1),
                    random.uniform(1, 10),
                    random.uniform(4, 9),
                    random.uniform(1, 5),
                    random.uniform(1, 5),
                    random.uniform(0, 5),
                    random.uniform(1, 5),
                    random.uniform(1, 5),
                    random.uniform(1, 5),
                    random.uniform(1, 5),
                    random.uniform(0.1, 2),
                    random.randint(0, 1),
                    random.randint(0, 1),
                    random.randint(0, 1)
                ]]
            }
        }

        try:
            # Kirim request ke model; timeout 2 detik agar tidak menggantung lama
            response = requests.post(URL, json=dummy_data, timeout=2)

            if response.status_code == 200:
                exporter.SUCCESS_COUNT.inc()

                # Parse prediksi dan tentukan kelasnya
                try:
                    pred = response.json()[0]
                    if pred == 0 or pred == "No":
                        exporter.PRED_CLASS_0.inc()
                    else:
                        exporter.PRED_CLASS_1.inc()
                except (KeyError, IndexError, ValueError):
                    # Jika format respons tidak sesuai, simulasikan prediksi acak
                    if random.random() > 0.5:
                        exporter.PRED_CLASS_1.inc()
                    else:
                        exporter.PRED_CLASS_0.inc()
            else:
                exporter.ERROR_COUNT.inc()
                print(f"[WARN] Model returned HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            exporter.ERROR_COUNT.inc()
            print("[ERROR] Cannot connect to MLflow model on port 5001. Is 'mlflow models serve' running?")
        except requests.exceptions.Timeout:
            exporter.ERROR_COUNT.inc()
            print("[ERROR] Request timed out after 2 seconds.")
        except requests.exceptions.RequestException as e:
            exporter.ERROR_COUNT.inc()
            print(f"[ERROR] Unexpected request error: {e}")

        # Hitung latensi dan catat ke Histogram Prometheus
        latency = time.time() - start_time
        exporter.INFERENCE_LATENCY.observe(latency)

        # Simulasikan data drift dan confidence score
        exporter.DRIFT_SCORE.set(random.uniform(0.01, 0.15))
        exporter.CONFIDENCE_SCORE.set(random.uniform(0.75, 0.99))

        # Jeda acak 1-3 detik sebelum request berikutnya
        time.sleep(random.uniform(1, 3))


if __name__ == '__main__':
    simulate_inference()
