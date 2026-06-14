import os
from dotenv import load_dotenv
import pytest
from modules.language_detector import LanguageDetector
from modules.emotion_classifier import EmotionClassifier
from modules.intent_classifier import IntentClassifier

load_dotenv()

# ── Fixtures ───────────────────────────────────────────────
@pytest.fixture(scope="session")
def language_detector():
    return LanguageDetector()

@pytest.fixture(scope="session")
def emotion_clf():
    return EmotionClassifier()

@pytest.fixture(scope="session")
def intent_clf():
    # Will use Groq if GROQ_API_KEY is set, otherwise rule-based fallback
    return IntentClassifier(groq_api_key=os.environ.get("GROQ_API_KEY", ""))

# ── Tests ─────────────────────────────────────────────────
def test_language_detector_empty(language_detector):
    result = language_detector.predict("")
    assert result["language"] in ["Unknown", "English", "Arabic"]  # depending on fallback
    assert result["confidence"] <= 0.5

def test_language_detector_mixed(language_detector):
    result = language_detector.predict("حاسس إني burnt out جدا الفترة الي فاتت")
    assert result["language"] == "Arabic"

def test_language_detector_gibberish(language_detector):
    result = language_detector.predict("asdfghjkl")
    assert result["low_confidence"] is True

def test_emotion_classifier_neutral(emotion_clf):
    result = emotion_clf.predict("The sky is blue.")
    assert result["emotion"] in ["neutral", "joy", "surprise"]  # depending on training
    assert result["confidence"] < 0.8

def test_emotion_classifier_short_input(emotion_clf):
    result = emotion_clf.predict("Hi")
    assert "emotion" in result
    assert result["confidence"] <= 0.9

def test_emotion_classifier_batch(emotion_clf):
    texts = ["I’m thrilled!", "I’m angry!", "I’m scared!"]
    results = [emotion_clf.predict(t) for t in texts]
    emotions = [r["emotion"] for r in results]
    assert "joy" in emotions
    assert "anger" in emotions
    assert "fear" in emotions

def test_intent_classifier_empty(intent_clf):
    result = intent_clf.predict("")
    assert result["intent"] == "greeting"
    assert result["confidence"] == "low"

def test_intent_classifier_goodbye(intent_clf):
    result = intent_clf.predict("See you later")
    assert result["intent"] in ["goodbye", "out_of_scope"]

def test_intent_classifier_batch(intent_clf):
    texts = ["Hello", "Thanks", "Bye", "Can you help me with anxiety?"]
    results = intent_clf.batch_predict(texts)
    intents = [r["intent"] for r in results]
    assert "greeting" in intents
    assert "gratitude" in intents
    assert any(i in intents for i in ["goodbye", "asking_mental_health_question"])

