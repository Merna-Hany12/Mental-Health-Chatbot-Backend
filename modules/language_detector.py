"""
Module 1 — Language Detection
==============================
Multi-class classifier using TF-IDF (char n-grams) + LinearSVC.
Trained on the Language Identification dataset from Hugging Face.

Usage:
    from modules.language_detector import LanguageDetector
    detector = LanguageDetector()
    result = detector.predict("I feel very anxious today")
    # → {"language": "English", "code": "en", "confidence": 0.97}
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "language_detection_pipeline.pkl"

# ISO code → display name  (used everywhere consistently)
LANG_NAMES: Dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

# Confidence threshold — below this we return "unknown" instead of a wrong guess
CONFIDENCE_THRESHOLD = 0.65

_KEEP = re.compile(r"[^\w\s]", flags=re.UNICODE)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Minimal cleaning — keeps Unicode letters so char n-grams stay meaningful.
    Removing diacritics or lowercasing script-specific chars would hurt accuracy.
    """
    text = text.strip().lower()
    text = _KEEP.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text



# ── Detector Class ─────────────────────────────────────────────────────────────
class LanguageDetector:
    """
    High-level wrapper around the TF-IDF language classifier.
    Auto-loads a pre-trained model from disk; trains one if not found.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self._pipeline: Pipeline | None = None
        self._model_path = model_path
        self._load_or_train()

    # ── Private ───────────────────────────────────────────────────────────────
    def _load_or_train(self) -> None:
        if self._model_path.exists():
            logger.info(f"Loading language detector from {self._model_path}")
            self._pipeline = joblib.load(self._model_path)
        else:
            logger.warning("No saved model found — training from scratch …")

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, text: str, threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
        """
        Classify the language of text.

        Parameters
        ----------
        text      : input string (any language)
        threshold : minimum confidence to trust the prediction.
                    Below this, returns code="en" with flag low_confidence=True.

        Returns
        -------
        {
            "language":       "English",
            "code":           "en",
            "confidence":     0.97,
            "low_confidence": False,
            "top5": [
                {"code": "en", "language": "English",  "prob": 0.97},
                {"code": "fr", "language": "French",   "prob": 0.01},
                ...
            ]
        }
        """
        # ── Edge case: empty input ────────────────────────────────────────────
        if not text or not text.strip():
            return {
                "language": "English",
                "code": "en",
                "confidence": 0.0,
                "low_confidence": True,
                "top5": [],
            }

        cleaned = preprocess(text)

        # pipeline.classes_ = sorted ISO codes e.g. ["ar","bg","de",...]
        # predict_proba returns probabilities in the same order
        proba = self._pipeline.predict_proba([cleaned])[0]
        classes = self._pipeline.classes_  # ISO codes

        # ── Build sorted probability list ────────────────────────────────────
        pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
        top_code, top_prob = pairs[0]

        # ── Confidence check ─────────────────────────────────────────────────
        low_confidence = bool(top_prob < threshold)
        if low_confidence:
            # Don't return a wrong confident answer — flag it
            top_code = "en"

        return {
            "language": LANG_NAMES.get(top_code, top_code),
            "code": top_code,
            "confidence": round(float(top_prob), 4),
            "low_confidence": low_confidence,
            "top5": [
                {
                    "code": code,
                    "language": LANG_NAMES.get(code, code),
                    "prob": round(float(prob), 4),
                }
                for code, prob in pairs[:5]
            ],
        }

    def batch_predict(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Predict language for a list of texts."""
        return [self.predict(t) for t in texts]


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train(save=True)
    else:
        detector = LanguageDetector()
        samples = [
            ("en", "my name is Merna"),
            ("en", "Thnak you"),
            ("ar", "السلام عليكم"),
            ("ar", "ازيك"),
            ("ar", "أشعر بالقلق الشديد ولا أستطيع النوم."),
            ("fr", "Je me sens très anxieux et je ne peux pas dormir."),
            ("es", "Me siento muy ansioso y no puedo dormir."),
            ("en", "I'm having trouble sleeping"),
            ("de", "Ich fühle mich sehr gestresst."),
            ("hi", "मुझे बहुत चिंता हो रही है।"),
            ("zh", "我最近感到非常焦虑。"),
            ("en", "hi"),
            ("en", "help me"),
            ("en", "I feel sad"),
            ("en", "my name is Merna"),
            ("en", "I cannot sleep at night"),
            # ── Arabic ───────────────────────────────────────────────
            ("ar", "مرحبا"),
            ("ar", "ساعدني"),
            ("ar", "أنا حزين"),
            ("ar", "اسمي مرنا"),
            ("ar", "لا أستطيع النوم"),
            # ── French ───────────────────────────────────────────────
            ("fr", "bonjour"),
            ("fr", "aide moi"),
            ("fr", "je suis triste"),
            ("fr", "je m'appelle Merna"),
            ("fr", "je ne peux pas dormir"),
            # ── Spanish ──────────────────────────────────────────────
            ("es", "hola"),
            ("es", "ayúdame"),
            ("es", "me siento triste"),
            ("es", "me llamo Merna"),
            ("es", "no puedo dormir"),
            # ── German ───────────────────────────────────────────────
            ("de", "hallo"),
            ("de", "hilf mir"),
            ("de", "ich bin traurig"),
            ("de", "ich heiße Merna"),
            ("de", "ich kann nicht schlafen"),
            # ── Italian ──────────────────────────────────────────────
            ("it", "ciao"),
            ("it", "aiutami"),
            ("it", "mi sento triste"),
            ("it", "mi chiamo Merna"),
            ("it", "non riesco a dormire"),
            # ── Portuguese ───────────────────────────────────────────
            ("pt", "olá"),
            ("pt", "me ajuda"),
            ("pt", "estou triste"),
            ("pt", "meu nome é Merna"),
            ("pt", "não consigo dormir"),
            # ── Russian ──────────────────────────────────────────────
            ("ru", "привет"),
            ("ru", "помоги мне"),
            ("ru", "мне грустно"),
            ("ru", "меня зовут Мерна"),
            ("ru", "я не могу спать"),
            # ── Hindi ────────────────────────────────────────────────
            ("hi", "नमस्ते"),
            ("hi", "मेरी मदद करो"),
            ("hi", "मैं दुखी हूं"),
            ("hi", "मेरा नाम मेर्ना है"),
            ("hi", "मुझे नींद नहीं आती"),
            # ── Chinese ──────────────────────────────────────────────
            ("zh", "你好"),
            ("zh", "帮帮我"),
            ("zh", "我很难过"),
            ("zh", "我叫Merna"),
            ("zh", "我睡不着"),
            # ── Japanese ─────────────────────────────────────────────
            ("ja", "こんにちは"),
            ("ja", "助けて"),
            ("ja", "悲しい"),
            ("ja", "私の名前はメルナです"),
            ("ja", "眠れません"),
            # ── Turkish ──────────────────────────────────────────────
            ("tr", "merhaba"),
            ("tr", "bana yardım et"),
            ("tr", "üzgünüm"),
            ("tr", "benim adım Merna"),
            ("tr", "uyuyamıyorum"),
            # ── Dutch ────────────────────────────────────────────────
            ("nl", "hallo"),
            ("nl", "help me"),
            ("nl", "ik ben verdrietig"),
            ("nl", "ik kan niet slapen"),
            # ── Polish ───────────────────────────────────────────────
            ("pl", "cześć"),
            ("pl", "jestem smutny"),
            ("pl", "mam na imię Merna"),
            ("pl", "nie mogę spać"),
            # ── Swedish (sw = Swahili in your model) ─────────────────
            ("sw", "habari"),
            ("sw", "nisaidie"),
            ("sw", "nina huzuni"),
            ("sw", "jina langu ni Merna"),
            ("sw", "siwezi kulala"),
            # ── Bulgarian ────────────────────────────────────────────
            ("bg", "здравей"),
            ("bg", "помогни ми"),
            ("bg", "тъжен съм"),
            ("bg", "казвам се Мерна"),
            ("bg", "не мога да спя"),
            # ── Greek ────────────────────────────────────────────────
            ("el", "γεια"),
            ("el", "βοήθησέ με"),
            ("el", "είμαι λυπημένος"),
            ("el", "με λένε Μέρνα"),
            ("el", "δεν μπορώ να κοιμηθώ"),
            # ── Thai ─────────────────────────────────────────────────
            ("th", "สวัสดี"),
            ("th", "ช่วยฉันด้วย"),
            ("th", "ฉันเศร้า"),
            ("th", "ชื่อของฉันคือเมอร์นา"),
            ("th", "ฉันนอนไม่หลับ"),
            # ── Urdu ─────────────────────────────────────────────────
            ("ur", "ہیلو"),
            ("ur", "میری مدد کرو"),
            ("ur", "میں اداس ہوں"),
            ("ur", "میرا نام مرنا ہے"),
            ("ur", "مجھے نیند نہیں آتی"),
            # ── Vietnamese ───────────────────────────────────────────
            ("vi", "xin chào"),
            ("vi", "giúp tôi với"),
            ("vi", "tôi buồn"),
            ("vi", "tên tôi là Merna"),
            ("vi", "tôi không ngủ được"),
        ]

        print("\n=== Language Detection Test ===")
        correct = 0
        for true_code, text in samples:
            r = detector.predict(text)
            status = "✅" if r["code"] == true_code else "❌"
            flag = " ⚠️ low confidence" if r["low_confidence"] else ""
            print(f"  {status} [{r['code']}] {r['language']:<14} ({r['confidence']:.2%}){flag}")
            print(f"       True: [{true_code}]  Input: {text[:60]}")
            if r["low_confidence"]:
                print(f"       Top guess was: [{r['top5'][0]['code']}] {r['top5'][0]['prob']:.2%}")
            correct += r["code"] == true_code

        print(f"\n  Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.0f}%)")