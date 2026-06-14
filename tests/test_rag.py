import os
import pytest
from modules.rag_pipeline import RAGPipeline

@pytest.fixture(scope="session")
def rag():
    return RAGPipeline(
        os.getenv("qdrant_url", ""),
        os.getenv("qdrant_api_key", ""),
        os.getenv("groq_api_key", ""),
    )

def test_rag_happy_path(rag):
    result = rag.ask("I feel very anxious and cannot sleep", show_sources=False)
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert "sources" in result
    assert isinstance(result["sources"], list)

def test_rag_with_sources(rag):
    result = rag.ask("How can I manage stress?", show_sources=True)
    assert "sources" in result
    assert isinstance(result["sources"], list)

def test_rag_empty_question(rag):
    result = rag.ask("", show_sources=False)
    assert "answer" in result
    assert isinstance(result["answer"], str)
    # Should degrade gracefully, not crash

def test_rag_emotion_tone(rag):
    result = rag.ask(
        "I feel sad about my future",
        emotion_tone="Be empathetic.",
        emotion_label="sadness",
        language="English",
        language_code="en",
    )
    assert "answer" in result
    assert isinstance(result["answer"], str)

def test_rag_non_english(rag):
    result = rag.ask("أشعر بالقلق ولا أستطيع النوم", language="Arabic", language_code="ar")
    assert "answer" in result
    assert isinstance(result["answer"], str)
