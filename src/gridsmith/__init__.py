"""gridsmith -- deterministic table and context extraction from large, messy PDFs.

No language model touches the document.  See docs/ARCHITECTURE.md for the stage map and
docs/DECISIONS.md for why each non-obvious choice was made.
"""
from .config import DEFAULT, Config  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Config", "DEFAULT", "__version__"]
