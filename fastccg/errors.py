# fastccg/errors.py

class FastCCGError(Exception): pass

class QuotaExceeded(FastCCGError):
    def __init__(self):
        super().__init__("❌ You've exceeded your AI usage quota. Check your billing dashboard.")

class ModelUnavailable(FastCCGError):
    def __init__(self, msg: str):
        super().__init__(msg)

class APIRequestFailed(FastCCGError):
    def __init__(self, msg: str):
        super().__init__(f"AI API request failed: {msg}")
