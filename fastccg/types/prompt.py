from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelPrompt:
    """A structured prompt for the model."""
    role: Literal["user", "assistant", "system"]
    content: str

    def to_dict(self) -> dict:
        """Convert the prompt to a dictionary."""
        return {"role": self.role, "content": self.content}