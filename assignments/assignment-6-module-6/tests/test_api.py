from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

GOOD_PAYLOAD = {
    "Time": 406.0,
    "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99, "V5": -0.52,
    "V6": -1.43, "V7": -2.54, "V8": 1.39, "V9": -2.77, "V10": -2.77,
    "V11": 3.20, "V12": -2.90, "V13": -0.60, "V14": -4.29, "V15": 0.39,
    "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42, "V20": 0.13,
    "V21": 0.52, "V22": -0.04, "V23": -0.47, "V24": 0.32, "V25": 0.04,
    "V26": 0.18, "V27": 0.26, "V28": -0.14,
    "Amount": 239.93
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_valid_prediction():
    response = client.post("/predict", json=GOOD_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "decision" in data
    assert 0 <= data["fraud_probability"] <= 1

def test_missing_field():
    # Missing 'Amount'
    bad_payload = GOOD_PAYLOAD.copy()
    del bad_payload["Amount"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_invalid_type():
    # Amount is a string instead of number
    bad_payload = GOOD_PAYLOAD.copy()
    bad_payload["Amount"] = "two hundred"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422