"""
Module 2 — Emotion Classifier
================================
Multi-class classifier using a fine-tuned Transformer (DistilBERT).
Trained on the GoEmotions / dair-ai/emotion dataset.

Emotion labels: sadness, joy, love, anger, fear, surprise

Integration with RAG pipeline:
    from modules.emotion_classifier import EmotionClassifier

    clf = EmotionClassifier()
    result = clf.predict("I feel so hopeless and lost")
    # → {"emotion": "sadness", "confidence": 0.94, "tone_hint": "empathetic_supportive", ...}

    rag.ask(
        question      = user_text,
        emotion_label = result["emotion"],
        emotion_tone  = result["tone_hint"],
    )

Usage:
    python emotion_classifier.py --train     # fine-tune + save
    python emotion_classifier.py             # run demo inference table
"""

from pathlib import Path

from transformers import AutoTokenizer, pipeline

MODEL_DIR = Path(__file__).parent.parent / "models" / "emotion_model"

EMOTION_TONE = {
    "sadness": "empathetic_supportive",
    "joy": "celebratory_positive",
    "love": "warm_affirming",
    "anger": "calm_de-escalating",
    "fear": "reassuring_grounding",
    "surprise": "curious_engaging",
}


class EmotionClassifier:
    def __init__(self, model_dir: str = str(MODEL_DIR)):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_input_names = [
            n for n in tokenizer.model_input_names if n != "token_type_ids"
        ]

        self.pipe = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            device=-1,
        )

    def predict(self, text: str):
        if not text.strip():
            return {
                "emotion": "sadness",
                "confidence": 0.0,
                "tone_hint": EMOTION_TONE["sadness"],
                "all_scores": {},
                "crisis_flag": False,
            }

        outputs = self.pipe(text)

        scores = outputs[0] if isinstance(outputs[0], list) else outputs
        top = max(scores, key=lambda x: x["score"])

        return {
            "emotion": top["label"],
            "confidence": round(top["score"], 4),
            "tone_hint": EMOTION_TONE.get(top["label"], "neutral"),
            "all_scores": {s["label"]: round(s["score"], 4) for s in scores},
            "crisis_flag": top["label"] in {"sadness", "fear"} and top["score"] >= 0.90,
        }
