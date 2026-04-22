from locust import HttpUser, task, between
import random
import uuid

class FraudDetectionUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task
    def predict_transaction(self):
        headers = {"X-API-Key": "dev_key_123"}
        payload = {
            "transaction_id": str(uuid.uuid4()),
            "Time": random.uniform(0, 100000),
            "Amount": random.uniform(10.0, 1000.0)
        }
        for i in range(1, 29):
            payload[f"V{i}"] = random.uniform(-2.0, 2.0)
            
        self.client.post("/predict", json=payload, headers=headers)
