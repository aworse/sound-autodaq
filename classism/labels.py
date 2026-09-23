"""
Single source of truth for the classism classifier's target classes.

Per CLASSISM-SPEC-REC REQ-2.2.1 / REQ-4.1, no other module may define its
own class list; everything imports `CLASSES` and `CLASS_DEFINITION_VERSION`
from here.

PLACEHOLDER: the real classism project's class definition (38 classes per
the spec) was not available when this repository was created. This file
lists only the jamo a single dubeolsik keystroke can produce, so every
trial asks for something physically typeable. Replace this file with the
project's actual labels.py before collecting training data; the recorder
derives the class count from len(CLASSES) and needs no other change.
"""

CLASS_DEFINITION_VERSION = "classism-placeholder-v2"

# Choseong consonants: 14 base keys plus the 5 tense doubles typed as
# Shift + base key.
_CONSONANTS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# Vowels with their own key (12) plus the 2 produced by Shift (ㅒ, ㅖ).
# Compound vowels (ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ) need two keystrokes and are
# therefore not per-keystroke classes.
_VOWELS = [
    "ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ",
    "ㅐ", "ㅒ", "ㅔ", "ㅖ",
]

CLASSES = tuple(_CONSONANTS + _VOWELS)

assert len(CLASSES) == len(set(CLASSES)), "duplicate class label in labels.py"
