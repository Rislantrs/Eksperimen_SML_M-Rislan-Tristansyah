import requests
import time
import random
import sys
import importlib.util

# Import the prometheus_exporter module using path since it starts with a number
spec = importlib.util.spec_from_file_location("prometheus_exporter", "Monitoring dan Logging/3.prometheus_exporter.py")
exporter = importlib.util.module_from_spec(spec)
sys.modules["prometheus_exporter"] = exporter
spec.loader.exec_module(exporter)

def simulate_inference():
    # Start the prometheus exporter server
    exporter.start_exporter(8000)
    print("Inference simulation started. Press Ctrl+C to stop.")

    URL = "http://localhost:5001/invocations"
    
    while True:
        exporter.REQUEST_COUNT.inc()
        start_time = time.time()
        
        # Generate synthetic data matching the schema of the mental health model
        # Ensure columns align with model expected features
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
            # Exception handling for connection failures to the serving endpoint
            response = requests.post(URL, json=dummy_data, timeout=2)
            
            if response.status_code == 200:
                exporter.SUCCESS_COUNT.inc()
                # Parse the prediction response based on expected output format
                try:
                    pred = response.json()[0]
                    if pred == 0 or pred == "No":
                        exporter.PRED_CLASS_0.inc()
                    else:
                        exporter.PRED_CLASS_1.inc()
                except:
                    if random.random() > 0.5:
                        exporter.PRED_CLASS_1.inc()
                    else:
                        exporter.PRED_CLASS_0.inc()
            else:
                exporter.ERROR_COUNT.inc()
                
        except requests.exceptions.RequestException:
            exporter.ERROR_COUNT.inc()
            print("Failed to connect to MLflow model on port 5001. Is it running?")
        
        # Update latency
        latency = time.time() - start_time
        exporter.LATENCY.observe(latency)
        
        # Simulate ML Drift & Confidence
        exporter.DRIFT_SCORE.set(random.uniform(0.01, 0.15))
        exporter.CONFIDENCE_SCORE.set(random.uniform(0.75, 0.99))
        
        # Wait before next request
        time.sleep(random.uniform(1, 3))

if __name__ == '__main__':
    simulate_inference()
