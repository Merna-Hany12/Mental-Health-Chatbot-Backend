import pytest
from httpx import ASGITransport, AsyncClient

from main import app  # your FastAPI app

transport = ASGITransport(app=app)  # your FastAPI app


@pytest.mark.asyncio
async def test_chat_happy_path():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "message": "Hello, I feel anxious",
            "session_id": "test123",
            "show_sources": True,
        }
        response = await ac.post("/api/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "language" in data
        assert "intent" in data
        assert "emotion" in data
        assert isinstance(data["sources"], list)
        assert "used_rag" in data
        assert "latency_ms" in data


@pytest.mark.asyncio
async def test_chat_missing_message():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"session_id": "test123"}
        response = await ac.post("/api/chat", json=payload)
        assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_feedback_happy_path():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "vote": "up",
            "user_message": "Hello",
            "bot_response": "Hi there",
            "session_id": "test123",
        }
        response = await ac.post("/api/feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_feedback_missing_vote():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"user_message": "Hello", "bot_response": "Hi there", "session_id": "test123"}
        response = await ac.post("/api/feedback", json=payload)
        assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_health_happy_path():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/health")
        assert response.status_code in (200, 207)
        data = response.json()
        assert "status" in data
        assert "modules" in data


@pytest.mark.asyncio
async def test_modules_happy_path():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/modules")
        assert response.status_code == 200
        data = response.json()

        # Response should be a dict with "modules" key
        assert "modules" in data
        assert isinstance(data["modules"], list)
        assert len(data["modules"]) >= 4

        # Check first module structure
        first = data["modules"][0]
        assert "id" in first
        assert "name" in first
        assert "tech" in first

        # Example: Language Detector should have languages list
        if first["name"] == "Language Detector":
            assert "languages" in first
            assert isinstance(first["languages"], list)
