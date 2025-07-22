import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

from .types.prompt import ModelPrompt


class MemoryManager:
    """
    Manages short-term and long-term memory for conversational models.

    - Short-term memory: A simple list of prompts in the current session.
    - Long-term memory: Persists conversations to a JSONL file, allowing
      models to retrieve context from past sessions.
    """

    def __init__(self, long_term_path: str = ".fcvs"):
        self.history: List[ModelPrompt] = []
        self._is_long_term_enabled = False
        self.long_term_path = long_term_path
        self.memory_file = os.path.join(long_term_path, "memory.jsonl")

    def enable_long_term(self, enable: bool):
        """Enable or disable long-term memory persistence."""
        self._is_long_term_enabled = enable
        if enable and not os.path.exists(self.long_term_path):
            os.makedirs(self.long_term_path)

    def append(self, prompt: ModelPrompt):
        """Add a prompt to the short-term history."""
        self.history.append(prompt)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history in a serializable format."""
        return [p.to_dict() for p in self.history]

    def clear(self):
        """Clear the short-term memory."""
        self.history = []

    def save_turn(self, user_prompt: ModelPrompt, assistant_prompt: ModelPrompt):
        """Save a user-assistant turn to the long-term memory file if enabled."""
        if not self._is_long_term_enabled:
            return

        turn_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_prompt.to_dict(),
            "assistant": assistant_prompt.to_dict(),
        }
        with open(self.memory_file, "a") as f:
            f.write(json.dumps(turn_record) + "\n")

    def load_recent_history(self, num_turns: int = 5) -> List[ModelPrompt]:
        """
        Load the most recent turns from long-term memory.

        Args:
            num_turns: The number of recent conversational turns to load.

        Returns:
            A list of ModelPrompt objects representing the loaded history.
        """
        if not self._is_long_term_enabled or not os.path.exists(self.memory_file):
            return []

        try:
            with open(self.memory_file, "r") as f:
                lines = f.readlines()

            recent_turns = []
            for line in reversed(lines[-num_turns:]):
                record = json.loads(line.strip())
                # Insert in reverse order to maintain chronology
                recent_turns.insert(0, ModelPrompt.from_dict(record["assistant"]))
                recent_turns.insert(0, ModelPrompt.from_dict(record["user"]))
            return recent_turns
        except (IOError, json.JSONDecodeError):
            return []
