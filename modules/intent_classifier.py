"""
Module 3 — Intent Classifier
================================
Zero-shot LLM prompting via Groq to classify user intent into:
    greeting | goodbye | gratitude | asking_mental_health_question | out_of_scope

Falls back to rule-based classification if the API is unavailable.

Usage:
    from modules.intent_classifier import IntentClassifier
    clf = IntentClassifier(groq_api_key="...")
    result = clf.predict("I feel very depressed and cannot cope")
    # → {"intent": "asking_mental_health_question", "confidence": "medium", "reasoning": "..."}
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
VALID_INTENTS = [
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope",
]

# ── Rule-based Fallback ────────────────────────────────────────────────────────
_GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"}
_GOODBYE_WORDS = {"bye", "goodbye", "see you", "talk later", "farewell", "take care", "good night"}
_GRATITUDE_WORDS = {"thank", "thanks", "grateful", "appreciate", "cheers", "thx", "ty"}
_MH_KEYWORDS = {
    "anxious", "anxiety", "depressed", "depression", "sad", "hopeless", "stress",
    "stressed", "worried", "panic", "trauma", "grief", "lonely", "loneliness",
    "suicidal", "self-harm", "therapy", "therapist", "counseling", "mental health",
    "cope", "coping", "emotion", "feeling", "mood", "sleep", "insomnia",
    "worthless", "empty", "numb", "overwhelmed", "burnout", "phobia", "ocd", "ptsd",
}


def _rule_based_predict(text: str) -> Dict[str, Any]:
    """Simple keyword-based fallback when the LLM API is unavailable."""
    lower = text.lower().strip()
    words = set(re.findall(r"\w+", lower))

    if any(g in lower for g in _GOODBYE_WORDS):
        return {"intent": "goodbye", "confidence": "medium", "reasoning": "Farewell keywords detected."}
    if any(g in lower for g in _GRATITUDE_WORDS):
        return {"intent": "gratitude", "confidence": "medium", "reasoning": "Thank-you keywords detected."}
    if any(g in lower for g in _GREETING_WORDS):
        return {"intent": "greeting", "confidence": "medium", "reasoning": "Greeting keywords detected."}
    if words & _MH_KEYWORDS:
        return {"intent": "asking_mental_health_question", "confidence": "medium", "reasoning": "Mental health keywords detected."}
    return {"intent": "out_of_scope", "confidence": "low", "reasoning": "No known pattern matched."}


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def build_intent_prompt(user_query: str) -> str:
    return f"""Classify the intent of the following user query into exactly one of these intents:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

Rules:
- Use "asking_mental_health_question" for anything related to emotions, mental health, therapy, coping, or psychological wellbeing.
- Use "out_of_scope" for anything unrelated to mental health (e.g. coding, geography, recipes).
- Respond with only the intent label, nothing else.

Query: "{user_query}"
Intent:"""


# ── Classifier Class ───────────────────────────────────────────────────────────
class IntentClassifier:
    """
    Zero-shot intent classifier backed by Groq LLM.
    Automatically falls back to rule-based classification on API errors.
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        model: str = "openai/gpt-oss-120b",
    ):
        self._api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._model = model
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self._api_key:
            logger.warning("No GROQ_API_KEY — intent classifier will use rule-based fallback.")
            return
        try:
            from groq import Groq

            self._client = Groq(api_key=self._api_key)
            logger.info("Groq client initialized for intent classification.")
        except ImportError:
            logger.warning("groq package not installed — using rule-based fallback.")

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify the intent of *text*.

        Returns
        -------
        {
            "intent": "asking_mental_health_question",
            "confidence": "medium",
            "reasoning": "Matched intent keyword in model output."
        }
        """
        if not text or not text.strip():
            return {"intent": "greeting", "confidence": "low", "reasoning": "Empty input."}

        if self._client is None:
            return _rule_based_predict(text)

        try:
            prompt = build_intent_prompt(text.strip())
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are an intent classifier. Respond with only the intent label."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip().lower()

            for intent in VALID_INTENTS:
                if intent in raw:
                    return {
                        "intent": intent,
                        "confidence": "medium",
                        "reasoning": f"Model output contained '{intent}'.",
                    }
            return {"intent": "out_of_scope", "confidence": "low", "reasoning": "No matching intent in model output."}

        except Exception as e:
            logger.error(f"Groq API error in intent classification: {e}")
            return _rule_based_predict(text)

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]

    def needs_rag(self, text: str) -> bool:
        """Quick helper: True only if intent is 'asking_mental_health_question'."""
        return self.predict(text)["intent"] == "asking_mental_health_question"


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    clf = IntentClassifier(groq_api_key=os.getenv("GROQ_API_KEY"))
    samples = [
        "Hello there!",
        "Thank you so much for your support.",
        "Goodbye, see you tomorrow.",
        "I feel very depressed and cannot cope with life.",
        "What is the capital of France?",
        "I keep having panic attacks at night, what should I do?",
    ]
    print("\n=== Intent Classification Test ===")
    for s in samples:
        r = clf.predict(s)
        print(f"  [{r['intent']:35}] ({r['confidence']:6}) → {s}")