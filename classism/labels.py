"""
Single source of truth for the classism classifier's target classes.

Per CLASSISM-SPEC-REC REQ-2.2.1 / REQ-4.1, no other module may define its
own class list; everything imports `CLASSES` and `CLASS_DEFINITION_VERSION`
from here.

Mirrors the classifier's label space in aworse/classism (spec.md §4,
src/labels.py and src/hangul.py at commit c6606a0): 33 base jamo followed
by 5 special-key tokens, in the same order, so a class index means the
same thing on both sides.
"""

CLASS_DEFINITION_VERSION = "classism-afe-38@c6606a0"

_CONSONANTS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]  # 14
_TENSE = ["ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"]  # 5, typed as Shift + base key
_VOWELS = ["ㅏ", "ㅐ", "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"]  # 12
_SHIFT_VOWELS = ["ㅒ", "ㅖ"]  # 2, typed as Shift + ㅐ / ㅔ

JAMO = tuple(_CONSONANTS + _TENSE + _VOWELS + _SHIFT_VOWELS)  # 33
SPECIAL = ("<sp>", "<bs>", "<shift>", "<caps>", "<other>")  # space, backspace, shift, caps lock, any other key

CLASSES = JAMO + SPECIAL

assert len(CLASSES) == 38 and len(set(CLASSES)) == 38, "label space must be 38 unique symbols"
