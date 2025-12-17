from api.app import app

def test_health():
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 200

def test_metadata():
    c = app.test_client()
    r = c.get("/metadata")
    assert r.status_code in (200, 404)

def test_predict_ok():
    c = app.test_client()
    r = c.post("/predict", json={"text": "Win a free iPhone now!", "threshold": 0.5})
    assert r.status_code == 200
    data = r.get_json()
    assert "label" in data
    assert "prediction" in data
