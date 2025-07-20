# Quick Start

This guide provides the fastest way to get up and running with FastCCG. We'll show you how to install the library, get your first response from a model, and then run an interactive chat session in your terminal.

---

## 1. Installation

First, install the FastCCG library using pip:

```bash
pip install fastccg
```

## 2. Get Your First Response

The core workflow of FastCCG is simple: add your API key, initialize a model, and start asking questions.

Here's a complete example using OpenAI's GPT-4o:

```python
# main.py
import fastccg
from fastccg.models.gpt import gpt_4o

# 1. Add your API key
# Replace "sk-..." with your actual OpenAI API key
api_key = fastccg.add_openai_key("sk-...")

# 2. Initialize the model
# The init_model function sets up the model for you.
model = fastccg.init_model(gpt_4o, api_key=api_key)

# 3. Ask a question
# The .ask() method sends the prompt and returns a response object.
response = model.ask("What is the best thing about Large Language Models?")

# 4. Print the content
print(response.content)
```

Save this as `main.py`, and run it from your terminal:

```bash
python main.py
```

## 3. Interactive Terminal Chat

FastCCG also comes with a powerful utility to launch an interactive chat session directly in your terminal. This is the easiest way to have a conversation with any supported model.

Simply pass your initialized model to the `run_terminal()` function:

```python
# chat.py
import fastccg
from fastccg.models.gpt import gpt_4o

api_key = fastccg.add_openai_key("sk-...")
model = fastccg.init_model(gpt_4o, api_key=api_key)

# This will start an interactive session
fastccg.run_terminal(model)
```

Run this file, and you'll be able to chat with the model, with full conversation history managed for you.

---

Next, learn how to use the full power of the command-line interface in our **[CLI Usage](./cli_usage.md)** guide.
