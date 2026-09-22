"""
Single source of truth for the classism classifier's target classes.

Per CLASSISM-SPEC-REC REQ-2.2.1 / REQ-4.1, no other module may define its
own class list; everything imports `CLASSES` and `CLASS_DEFINITION_VERSION`
from here.
"""

CLASS_DEFINITION_VERSION = "classism-v1"

# Choseong (initial) consonant jamo, including the five tense doubles that
# are typed as a single dubeolsik keystroke (Shift + base key).
_CONSONANTS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# Jungseong (medial) vowel jamo producible by a single dubeolsik keystroke
# (compound vowels such as ㅘ/ㅝ/ㅢ are produced by two sequential
# keystrokes downstream and are therefore excluded from this per-keystroke
# class set).
_VOWELS = [
    "ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ",
    "ㅐ", "ㅒ", "ㅔ", "ㅖ", "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅢ",
]

CLASSES = tuple(_CONSONANTS + _VOWELS)

assert len(CLASSES) == len(set(CLASSES)), "duplicate class label in labels.py"
