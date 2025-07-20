# API Reference

This document provides a detailed reference for the core public functions and classes in the FastCCG library.

---

## Top-Level Functions

These functions are available directly from the `fastccg` package.

#### `init_model(model_class, api_key: str) -> ModelBase`

Initializes a model instance.

-   **Parameters**:
    -   `model_class`: The class of the model to initialize (e.g., `gpt_4o`).
    -   `api_key` (str): The API key for the provider.
-   **Returns**: An instance of the specified model, ready to use.

#### `load_model(path: str, api_key: str) -> ModelBase`

Loads a model's state from a saved session file.

-   **Parameters**:
    -   `path` (str): The file path to the saved session (`.json`).
    -   `api_key` (str): The API key for the provider.
-   **Returns**: An instance of the model with its history and configuration restored.

#### `run_terminal(model: ModelBase) -> None`

Starts an interactive chat session in the terminal with the given model instance.

#### `add_<provider>_key(key: str) -> str`

Adds an API key for a specific provider (e.g., `add_openai_key`, `add_google_key`).

-   **Parameters**:
    -   `key` (str): Your API key.
-   **Returns**: The API key that was passed in.

---

## `ModelBase` Class

All model classes inherit from `ModelBase` and share this common interface.

### Core Methods

#### `.ask(prompt: str) -> ModelResponse`

Sends a prompt and waits for the complete response.

#### `.ask_async(prompt: str) -> Coroutine[ModelResponse]`

Sends a prompt and returns a coroutine for an asynchronous response.

#### `.ask_stream(prompt: str) -> AsyncGenerator[ModelResponse, None]`

Streams the response, yielding chunks as they arrive.

### Configuration Methods (Chainable)

#### `.sys_prompt(msg: str) -> ModelBase`

Sets a system-level prompt to guide the model.

#### `.temperature(val: float) -> ModelBase`

Sets the creativity of the response (e.g., `0.0` to `1.0`).

#### `.max_tokens(n: int) -> ModelBase`

Sets the maximum length of the response.

### State Management

#### `.save(path: str) -> None`

Saves the current session (history and configuration) to a file.

#### `.reset() -> ModelBase`

Clears the conversation history and resets all configurations.

#### `.get_history() -> List[ModelPrompt]`

Returns the full conversation history.

---

Next, see a complete list of all the models you can use in **[Supported Models](./supported_models.md)**.
