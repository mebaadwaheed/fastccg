from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ModelResponse:
    """A structured response from the model."""
    content: str
    tokens_used: Optional[int] = None
    provider: Optional[str] = None
    raw: Optional[Any] = None