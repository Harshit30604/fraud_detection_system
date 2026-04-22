import requests
import time
import random
import uuid
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"
HEADERS = {"X-API-Key": "dev_key_123"}

def generate_transaction():
    # Simulate 99.8% legit, 0.2% fraud distribution
    is_fraud = random.random() < 0.002
    
    tx = {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:8]}",
        "Time": random.uniform(0, 172800),
        "Amount": random.uniform(1.0, 5000.0) if not is_fraud else random.uniform(200.0, 10000.0)
    }
    
    for i in range(1, 29):
        mean = 0.0 if not is_fraud else random.choice([-3.0, 3.0])
        tx[f"V{i}"] = np.random.normal(mean, 1.0)
        
    return tx

if __name__ == "__main__":
    print("Starting live transaction simulation (1 tx/sec)...")
    while True:
        tx_data = generate_transaction()
        try:
            response = requests.post(API_URL, json=tx_data, headers=HEADERS)
            print(f"Sent {tx_data['transaction_id']} -> Status: {response.status_code}")
        except Exception as e:
            print(f"Connection error: {e}")
        time.sleep(1)
