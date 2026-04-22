# Fraud Detection System

A 15-week production-grade machine learning pipeline for detecting financial fraud.

## Architecture

[Raw Data] --> [utils/preprocess.py] --> [SMOTE Resampling]
                                                |
                                                v
[Streamlit Dashboard] <---(JSON Logs)--- [XGBoost Model (best_model.pkl)]
       (Port 8501)                              ^
                                                |
[simulate_transactions.py] ---(REST API)---> [FastAPI Backend]
                                                (Port 8000)

## Tech Stack
- **Data Processing:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn
- **Machine Learning:** XGBoost, LightGBM, TensorFlow (Keras)
- **Backend API:** FastAPI, Uvicorn, Pydantic, SlowAPI
- **Frontend Dashboard:** Streamlit, Plotly
- **MLOps & Deployment:** Docker, Docker Compose, GitHub Actions, Evidently AI

## API Reference
- `POST /predict`: Scores a single transaction and returns risk level
- `GET /health`: Returns API uptime, model version, and status
- `GET /stats`: Returns real-time aggregate fraud statistics
- `POST /batch_predict`: Scores an array of up to 1000 transactions

## Run Instructions
```bash
# 1. Clone the repository
git clone https://github.com/Harshit30604/fraud_detection_system.git
cd fraud_detection_system

# 2. Set up environment variables
cp .env.example .env

# 3. Run with Docker Compose
docker-compose up --build -d

# 4. Access Services
# FastAPI Docs: http://localhost:8000/docs
# Streamlit Dashboard: http://localhost:8501

# 5. Run Live Simulator
python simulate_transactions.py
```
