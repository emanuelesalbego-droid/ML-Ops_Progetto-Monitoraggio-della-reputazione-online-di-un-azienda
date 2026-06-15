import pytest
import os
os.environ["DATABASE_URL"] = "sqlite:///./data/sentiment_logs.db"
from fastapi.testclient import TestClient
from app.main import app# import main  

client = TestClient(app)

def test_read_root():
    """Verifica che l'endpoint root risponda correttamente"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    """Verifica l'endpoint di predizione sentiment"""
    payload = {"text": "Questo progetto è fantastico!"}
    response = client.post("/predict", json=payload)
    if response.status_code != 200:
        print(f"Errore del server: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data

def test_metrics_endpoint():
    """Verifica che l'endpoint metrics sia raggiungibile"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_predictions_logged" in response.json()

def test_models_list():
    """Verifica l'endpoint della lista modelli"""
    response = client.get("/models_list")
    assert response.status_code == 200
    assert "models" in response.json()