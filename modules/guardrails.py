"""
Module — Safety Guardrails
================================
Centralized safety checks applied before and after the NLP pipeline.

Responsibilities:
    - Detect crisis/self-harm signals
    - Provide localized crisis resources
    - Detect off-topic / jailbreak-style task requests
      (while allowing legitimate therapeutic writing requests)
    - Screen LLM output for unsafe content (diagnosis, dosages)

Known limitations (documented in README):
    - Regex-based detection is a first line of defense, not exhaustive.
    - No multi-turn/conversation-level context — each message evaluated alone.
    - Coverage is English-focused; relies on upstream translation quality
      for non-English crisis/off-topic phrasing.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Crisis keyword patterns (multi-language + indirect phrasing) ──────────────
_CRISIS_PATTERNS = [
    # Direct
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bend my life\b",
    r"\bend it all\b",
    r"\bwant to die\b",
    r"\bwish i (was|were) dead\b",
    r"\bself[\s-]?harm\b",
    r"\bhurt myself\b",
    r"\bcut myself\b",
    # Indirect / euphemistic
    r"\bdon'?t want to (be here|exist) anymore\b",
    r"\bno point in (living|going on)\b",
    r"\btired of (living|being alive|everything)\b",
    r"\bbetter off without me\b",
    r"\bcan'?t (do this|go on) anymore\b",
    # Arabic
    r"\bأريد أن أموت\b",
    r"\bانتحار\b",
    r"\bأريد الموت\b",
    # Spanish
    r"\bquiero morir\b",
    r"\bsuicidio\b",
    r"\bno quiero vivir\b",
    # French
    r"\bje veux mourir\b",
    r"\bsuicide\b",
]
_CRISIS_REGEX = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)

# Patterns indicating PAST crisis the person says they've recovered from —
# reduces false positives like "I used to think about suicide but I'm better now".
_PAST_TENSE_RECOVERY_PATTERNS = [
    r"\bused to (think about|feel|want)\b.{0,40}\b(suicide|dying|kill myself|hurt myself)\b",
    r"\b(but|and) (i'?m|i am) (better|okay|fine|doing well|in a better place) now\b",
    r"\bnot anymore\b",
]
_PAST_TENSE_RECOVERY_REGEX = re.compile("|".join(_PAST_TENSE_RECOVERY_PATTERNS), re.IGNORECASE)

# ── Localized crisis resources ─────────────────────────────────────────────────
CRISIS_RESOURCES: Dict[str, str] = {
    "en": (
        "I'm really concerned about how you're feeling right now. "
        "You deserve immediate support from someone who can help.\n\n"
        "If you're in the US: call or text 988 (Suicide & Crisis Lifeline).\n"
        "If you're outside the US: please contact your local emergency number "
        "or find a helpline at findahelpline.com.\n\n"
        "Please reach out to one of these — you don't have to go through this alone."
    ),
}

# ── Off-topic / jailbreak task patterns ─────────────────────────────────────────
_OFF_TOPIC_TASK_PATTERNS = [
    r"\bwrite (me )?(a |some |an )?[\w\s]{0,20}\b(code|script|program|function|app|game)\b",
    r"\bhelp me (build|code|develop|program)\b",
    r"\bdebug (this|my) code\b",
    r"\bgenerate (a |an )?(sql|python|javascript|html|css)\b",
    r"\bsolve (this|my) (math|homework|assignment)\b",
    r"\bwrite (an? )?(essay|article)\b",
    r"\btranslate (this|the following)\b",
    r"\bwhat('?s| is) the (capital|population|gdp) of\b",
]
_OFF_TOPIC_REGEX = re.compile("|".join(_OFF_TOPIC_TASK_PATTERNS), re.IGNORECASE)

# Legitimate therapeutic writing requests — checked FIRST, exempts from off-topic block.
# e.g. "can you help me write a poem about how I feel" should be allowed.
_THERAPEUTIC_WRITING_EXCEPTION = re.compile(
    r"\bwrite\b.{0,30}\b(poem|journal|diary|letter)\b.{0,40}"
    r"\b(feel|feeling|feelings|emotions?|myself|my life|going through)\b",
    re.IGNORECASE,
)

OFF_TOPIC_RESPONSE = (
    "I'm a mental health support assistant, so I'm not able to help with "
    "coding, writing unrelated content, or other tasks outside emotional "
    "wellbeing — even if they were suggested as a hobby or activity. "
    "If you'd like, I'm happy to talk about how you're feeling, or we can "
    "explore activities and coping strategies together."
)

# ── Unsafe output patterns (diagnosis / dosage) ─────────────────────────────────
_UNSAFE_OUTPUT_PATTERNS = [
    # Numeric dosages
    r"\btake \d+\s*(mg|milligrams|pills|tablets|drops)\b",
    r"\b\d+\s*(mg|milligrams)\s*(of|per day|daily)\b",
    # Spelled-out dosages
    r"\b(a |the |your )?(typical |usual |starting )?dose (is|of|would be)\b",
    r"\btake (one|two|three|a|several) (tablets?|pills?|capsules?)\b",
    # Direct diagnosis of the user
    r"\byou (have|definitely have|are showing signs of|are suffering from)\s+"
    r"(depression|bpd|ptsd|bipolar|anxiety disorder|adhd|ocd)\b",
    r"\bthis (sounds|looks) like (classic |textbook )?"
    r"(depression|bpd|ptsd|bipolar|anxiety disorder|adhd|ocd)\b",
]
_UNSAFE_OUTPUT_REGEX = re.compile("|".join(_UNSAFE_OUTPUT_PATTERNS), re.IGNORECASE)

FALLBACK_SAFE_RESPONSE = (
    "I want to make sure I give you safe and responsible information. "
    "I'm not able to diagnose conditions or recommend medication — "
    "please consider speaking with a licensed mental health professional "
    "who can properly evaluate your situation."
)


# ── Public API ───────────────────────────────────────────────────────────────────
def check_crisis_input(text: str, emotion_result: Optional[dict] = None) -> bool:
    """
    Returns True if the message should be treated as a crisis.

    Combines:
      1. Emotion classifier's crisis_flag (high-confidence sadness/fear)
      2. Keyword/phrase pattern match (direct + indirect + multilingual)

    Reduces false positives for messages that describe a PAST crisis
    the person says they've recovered from.
    """
    crisis_phrase_found = bool(_CRISIS_REGEX.search(text))
    emotion_flag = bool(emotion_result and emotion_result.get("crisis_flag"))
    HIGH_RISK_PATTERNS = re.compile(
        r"\b(suicide|kill myself|want to die|end my life|hurt myself|self harm)\b",
        re.IGNORECASE,
    )
    if crisis_phrase_found and _PAST_TENSE_RECOVERY_REGEX.search(text):
        logger.info("Crisis phrase found but framed as past/recovered — not flagging.")
        crisis_phrase_found = False

    if emotion_flag and HIGH_RISK_PATTERNS.search(text):
        return True

    if crisis_phrase_found:
        logger.warning("guardrail_triggered: crisis (source=keyword_match)")
        return True

    return False


def get_crisis_response(language_code: str) -> str:
    return CRISIS_RESOURCES.get(language_code, CRISIS_RESOURCES["en"])


def check_off_topic_task(text: str) -> bool:
    """
    Returns True if the message asks for an unrelated task (coding, essays, etc.).

    Exempts legitimate therapeutic writing requests, e.g.
    "can you help me write a poem about how I feel right now".
    """
    if _THERAPEUTIC_WRITING_EXCEPTION.search(text):
        logger.info("Writing request matched therapeutic-expression exception — allowing.")
        return False

    if _OFF_TOPIC_REGEX.search(text):
        logger.warning("guardrail_triggered: off_topic (source=keyword_match)")
        return True

    return False


def get_off_topic_response() -> str:
    return OFF_TOPIC_RESPONSE


def check_unsafe_output(answer: str) -> bool:
    """Returns True if the LLM's answer contains unsafe content (diagnosis, dosages)."""
    if _UNSAFE_OUTPUT_REGEX.search(answer):
        logger.warning("guardrail_triggered: unsafe_output (source=output_filter)")
        return True
    return False


# ── Prompt injection / jailbreak attempt patterns ───────────────────────────────
_PROMPT_INJECTION_PATTERNS = [
    r"\bignore (previous|all|the above|prior) instructions\b",
    r"\byou are now\b",
    r"\bpretend (you are|to be)\b",
    r"\breveal your (system prompt|instructions|prompt)\b",
    r"\bwhat (is|are) your (system prompt|instructions)\b",
    r"\bact as\b.{0,30}\b(developer|admin|root|dan)\b",
    r"\bDAN mode\b",
    r"\bdisregard (your|all) (rules|guidelines|programming)\b",
    r"\bjailbreak\b",
]
_PROMPT_INJECTION_REGEX = re.compile("|".join(_PROMPT_INJECTION_PATTERNS), re.IGNORECASE)

PROMPT_INJECTION_RESPONSE = (
    "I'm not able to change how I operate or share my internal instructions. "
    "I'm here to support you with how you're feeling — is there something on "
    "your mind you'd like to talk about?"
)


def check_prompt_injection(text: str) -> bool:
    """True if the message attempts to override instructions or extract the system prompt."""
    if _PROMPT_INJECTION_REGEX.search(text):
        logger.warning("guardrail_triggered: prompt_injection (source=keyword_match)")
        return True
    return False


def get_prompt_injection_response() -> str:
    return PROMPT_INJECTION_RESPONSE


def sanitize_response(answer: str) -> str:
    """Replace unsafe LLM output with a safe fallback message."""
    if check_unsafe_output(answer):
        return FALLBACK_SAFE_RESPONSE
    return answer
