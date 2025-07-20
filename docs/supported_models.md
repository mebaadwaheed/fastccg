# Supported Models

FastCCG provides a unified API for a range of models from leading providers. This document lists all the currently supported and tested models.

To use any of these models, import the model class from its submodule (e.g., `fastccg.models.gpt`) and pass it to the `fastccg.init_model()` function.

---

## OpenAI

-   **Provider Name**: `openai`
-   **API Key Function**: `fastccg.add_openai_key()`

| Class Name      | Model ID          |
| --------------- | ----------------- |
| `gpt_4o`        | `gpt-4o`          |
| `gpt_3_5_turbo` | `gpt-3.5-turbo`   |

**Example:**
```python
import fastccg
from fastccg.models.gpt import gpt_4o

api_key = fastccg.add_openai_key("sk-...")
model = fastccg.init_model(gpt_4o, api_key=api_key)
```

---

## Gemini

-   **Provider Name**: `gemini`
-   **API Key Function**: `fastccg.add_gemini_key()`

| Class Name          | Model ID                   |
| ------------------- | -------------------------- |
| `gemini_pro_1_5`    | `gemini-1.5-pro-latest`    |
| `gemini_flash_1_5`  | `gemini-1.5-flash-latest`  |

**Example:**
```python
import fastccg
from fastccg.models.gemini import gemini_flash_1_5

api_key = fastccg.add_gemini_key("AIzaSy...")
model = fastccg.init_model(gemini_flash_1_5, api_key=api_key)
```

---

## Mistral

-   **Provider Name**: `mistral`
-   **API Key Function**: `fastccg.add_mistral_key()`

| Class Name         | Model ID           |
| ------------------ | ------------------ |
| `mistral_tiny`     | `mistral-tiny`     |
| `mistral_small`    | `mistral-small`    |
| `mistral_medium`   | `mistral-medium`   |

**Example:**
```python
import fastccg
from fastccg.models.mistral import mistral_tiny

api_key = fastccg.add_mistral_key("your-mistral-key")
model = fastccg.init_model(mistral_tiny, api_key=api_key)
```

---

## Anthropic (Claude)

Support for Anthropic's Claude models is planned but not yet fully integrated or tested. The `claude.py` module exists but is not currently used in the test suite.
