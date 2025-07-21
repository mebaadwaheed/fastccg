from . import gpt
from . import claude
from . import gemini
from . import mistral
from . import mock

_mock_key_set = False

def add_mock_key():
    """Adds a mock API key for testing purposes."""
    global _mock_key_set
    if not _mock_key_set:
        from fastccg.keys import add_key
        add_key("mock", "mock_key")
        _mock_key_set = True
