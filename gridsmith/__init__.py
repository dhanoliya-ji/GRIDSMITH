"""gridsmith -- deterministic table and context extraction from large, messy PDFs.

No language model touches the document.  Usage: python -m gridsmith run --input doc.pdf --out ./out --offline
"""
from .core import DEFAULT, Config  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Config", "DEFAULT", "__version__"]
