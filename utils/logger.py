import json
import os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.json"

def append_log(record: dict):
    """Appends a prediction record to the JSON log file."""
    # Ensure timestamp is present
    if "timestamp" not in record:
        record["timestamp"] = datetime.utcnow().isoformat()
        
    # Read existing logs if file exists
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            pass # File might be empty
            
    logs.append(record)
    
    # Write back to file
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)
