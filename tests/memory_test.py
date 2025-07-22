import os
import unittest
import json
from fastccg.models.mock import MockModel


class TestMemoryFeatures(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.long_term_path = ".fcvs"
        self.memory_file = os.path.join(self.long_term_path, "memory.jsonl")

    def tearDown(self):
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        if os.path.exists(self.long_term_path) and not os.listdir(self.long_term_path):
            os.rmdir(self.long_term_path)

    def test_short_term_memory(self):
        """Test that the model correctly tracks conversation history in a session."""
        self.model.ask("Hello, who are you?")
        self.model.ask("What was my first question?")

        history = self.model.get_history()
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "Hello, who are you?")
        self.assertEqual(history[1].role, "assistant")
        self.assertTrue(history[1].content.startswith("This is a mock response"))
        self.assertEqual(history[2].role, "user")
        self.assertEqual(history[2].content, "What was my first question?")

    def test_long_term_memory_saving(self):
        """Test that enabling long-term memory saves the conversation to a file."""
        self.model.enable_memory(long_term=True)
        self.model.ask("This is a test for long-term memory.")

        self.assertTrue(os.path.exists(self.memory_file))

        with open(self.memory_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["user"]["content"], "This is a test for long-term memory.")
            self.assertTrue(record["assistant"]["content"].startswith("This is a mock response"))

    def test_long_term_memory_loading(self):
        """Test that the model loads recent history when long-term memory is enabled."""
        # 1. Prepare a dummy memory file
        if not os.path.exists(self.long_term_path):
            os.makedirs(self.long_term_path)
        
        dummy_turn = {
            "timestamp": "2023-01-01T12:00:00Z",
            "user": {"role": "user", "content": "Hello from the past!"},
            "assistant": {"role": "assistant", "content": "Hi, past user!"}
        }
        with open(self.memory_file, "w") as f:
            f.write(json.dumps(dummy_turn) + "\n")

        # 2. Create a new model and enable memory to trigger loading
        new_model = MockModel()
        new_model.enable_memory(long_term=True, recent_history_turns=1)

        # 3. Check if the history was loaded
        history = new_model.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "Hello from the past!")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "Hi, past user!")

if __name__ == "__main__":
    unittest.main()
