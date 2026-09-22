"""
Thin re-export of the project's single class definition (REQ-56.2).

This module MUST NOT define classes itself; it only imports from
`classism.labels`. REQ-4.4: if the class definition cannot be imported,
the caller must abort before opening any audio device.
"""

from .errors import ClassDefinitionError

try:
    from classism.labels import CLASSES, CLASS_DEFINITION_VERSION
except Exception as exc:  # pragma: no cover - exercised via load_classes()
    _IMPORT_ERROR = exc
    CLASSES = None
    CLASS_DEFINITION_VERSION = None
else:
    _IMPORT_ERROR = None


def load_classes():
    """Return (classes, version), raising ClassDefinitionError on failure."""
    if _IMPORT_ERROR is not None or not CLASSES:
        raise ClassDefinitionError(
            f"could not load class definition from classism.labels: {_IMPORT_ERROR}"
        )
    return tuple(CLASSES), CLASS_DEFINITION_VERSION
