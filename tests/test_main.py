from fastapi.testclient import TestClient

from app.main import app  # make sure this matches your actual path

client = TestClient(app)


def test_homepage_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers["content-type"]
