from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import hashlib
import joblib
import pandas as pd
import time
import json
from datetime import datetime
from typing import List
from pathlib import Path
import asyncio

from api.schemas import TransactionInput, PredictionOutput
from utils.logger import append_log
from utils.preprocess import engineer_features

# Global state
ml_models = {}
START_TIME = time.time()
LOG_FILE = Path("../logs/predictions.json")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models on startup
    try:
        ml_models["model"] = joblib.load("../models/best_model.pkl")
        ml_models["scaler"] = joblib.load("../models/scaler.pkl")
    except FileNotFoundError:
        print("Warning: Models not found. Using mock inference.")
    ml_models["threshold"] = 0.42
    ml_models["predictions_count"] = 0
    yield
    # Cleanup on shutdown
    ml_models.clear()

# Rate Limiter Setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# API Key Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

VALID_API_KEYS = {
    hashlib.sha256("dev_key_123".encode()).hexdigest(),
    hashlib.sha256("prod_key_999".encode()).hexdigest()
}

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not api_key_header:
        raise HTTPException(status_code=401, detail="API Key missing")
    
    key_hash = hashlib.sha256(api_key_header.encode()).hexdigest()
    if key_hash not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key_header

def get_risk_level(prob: float) -> str:
    if prob < 0.3: return "LOW"
    if prob <= 0.7: return "MEDIUM"
    return "HIGH"

@app.post("/predict", response_model=PredictionOutput)
@limiter.limit("100/minute")
async def predict(request: Request, transaction: TransactionInput, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    # 1. Input Validation
    if transaction.Amount < 0:
        raise HTTPException(status_code=422, detail="Transaction amount cannot be negative")
        
    pca_features = [getattr(transaction, f"V{i}") for i in range(1, 29)]
    if all(v == 0 for v in pca_features):
        raise HTTPException(status_code=422, detail="Invalid transaction: All PCA features are zero")

    # 2. Inference
    if "model" in ml_models and "scaler" in ml_models:
        df = pd.DataFrame([transaction.model_dump(exclude={"transaction_id"})])
        df_engineered = engineer_features(df)
        df_engineered[['Amount', 'Time']] = ml_models["scaler"].transform(df_engineered[['Amount', 'Time']])
        prob = float(ml_models["model"].predict_proba(df_engineered)[0, 1])
    else:
        # Mock inference if model not loaded
        prob = 0.85 if sum(pca_features) > 10 else 0.15

    pred = 1 if prob >= ml_models.get("threshold", 0.42) else 0
    
    result = PredictionOutput(
        transaction_id=transaction.transaction_id,
        fraud_probability=prob,
        prediction=pred,
        risk_level=get_risk_level(prob),
        timestamp=datetime.utcnow().isoformat()
    )
    
    # 3. Background Task for Logging
    background_tasks.add_task(append_log, result.model_dump())
    ml_models["predictions_count"] = ml_models.get("predictions_count", 0) + 1
    
    return result

@app.get("/health")
async def health():
    uptime = time.time() - START_TIME
    return {
        "status": "ok", 
        "model_version": "1.0.0", 
        "uptime_seconds": round(uptime, 2),
        "total_predictions": ml_models.get("predictions_count", 0)
    }

@app.get("/stats")
async def stats():
    if not LOG_FILE.exists():
        return {"total": 0, "fraud_count": 0, "fraud_rate": 0.0, "avg_fraud_prob": 0.0}
    
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
    
    total = len(logs)
    fraud_count = sum(1 for log in logs if log["prediction"] == 1)
    avg_prob = sum(log["fraud_probability"] for log in logs) / total if total > 0 else 0
    
    return {
        "total": total,
        "fraud_count": fraud_count,
        "fraud_rate": round(fraud_count / total, 4) if total > 0 else 0.0,
        "avg_fraud_prob": round(avg_prob, 4)
    }

@app.post("/batch_predict", response_model=list[PredictionOutput])
async def batch_predict(request: Request, transactions: list[TransactionInput], background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions per batch")
    
    coroutines = [predict(request, tx, background_tasks, api_key) for tx in transactions]
    results = await asyncio.gather(*coroutines)
    return results
