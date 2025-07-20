# CLI Usage

FastCCG comes with a powerful command-line interface (CLI) that allows you to interact with models directly from your terminal. It's built with `typer` and `rich` to provide a great user experience.

---

## 1. Listing Available Models

To see a list of all the model presets available in FastCCG, you can use the `models` command.

```bash
fastccg models
```

This will print a formatted table of all supported models, along with their providers, making it easy to see what you can use.

## 2. Sending a Single Prompt (`ask`)

The `ask` command allows you to send a single, one-shot prompt to any model. This is useful for quick questions or testing a prompt.

```bash
fastccg ask "What is the capital of France?" --model gpt-4o --api-key "sk-...
```

### Options

The `ask` command comes with several options to customize the request:

-   `--model` / `-m`: (Required) The ID of the model you want to use (e.g., `gpt-4o`, `gemini-1.5-pro-latest`).
-   `--api-key`: Your API key for the provider. If not provided, the CLI will look for an environment variable (e.g., `OPENAI_API_KEY`).
-   `--temperature`: Set the creativity of the response (e.g., `0.8`).
-   `--max-tokens`: Limit the length of the response.

**Example with options:**

```bash
fastccg ask "Tell me a short story." -m mistral-small --temperature 0.9 --max-tokens 100
```

## 3. Interactive Chat (`chat`)

For a full conversational experience, use the `chat` command. This will launch an interactive session in your terminal where the model remembers the context of the conversation.

```bash
fastccg chat --model gpt-4o
```

### Chat Features

-   **Streaming by Default**: Responses are streamed back to you in real-time.
-   **Conversation History**: The model remembers previous turns in the conversation.
-   **System Prompts**: You can set a system prompt to guide the model's behavior for the entire session.
-   **Saving Sessions**: You can save the entire conversation to a JSON file and resume it later.

### Chat Options

-   `--model` / `-m`: (Required) The model to chat with.
-   `--api-key`: Your API key.
-   `--sys-prompt`: Set a system prompt for the session (e.g., `"You are a helpful assistant."`).
-   `--save-path`: A file path to save the conversation history (e.g., `my_chat.json`).

### In-Chat Commands

While in a chat session, you can use these special commands:

-   `exit`: Quits the chat session.
-   `reset`: Clears the conversation history and starts fresh.

---

Next, let's dive into the more powerful programmatic features in the **[Advanced Usage](./advanced_usage.md)** guide.
