from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_and_chat_endpoints():
    r1 = client.get("/")
    assert r1.status_code == 200
    assert "ShopWise AI" in r1.text

    r2 = client.get("/chat")
    assert r2.status_code == 200
    assert "ShopWise AI" in r2.text


def test_health_endpoints():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8002

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    r_data = ready.json()
    assert r_data["ready"] is True


def test_catalog_endpoints():
    res = client.get("/api/v1/catalog/products")
    assert res.status_code == 200
    data = res.json()
    assert "count" in data
    assert "products" in data


def test_policies_endpoint():
    res = client.get("/api/v1/policies")
    assert res.status_code == 200
    data = res.json()
    assert "count" in data
    assert "policies" in data


def test_chat_query_endpoint():
    payload = {
        "query": "What is the return policy for opened items?",
        "top_k": 3
    }
    res = client.post("/api/v1/chat/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "intent" in data
    assert "sources" in data
