# FastCCG (Fast Conversational & Completion Gateway)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/fastccg.svg)](https://pypi.org/project/fastccg/)
[![GitHub Stars](https://img.shields.io/github/stars/mebaadwaheed/fastccg.svg)](https://github.com/mebaadwaheed/fastccg/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/mebaadwaheed/fastccg.svg)](https://github.com/mebaadwaheed/fastccg/issues)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://github.com/mebaadwaheed/fastccg/tree/main/docs)

**FastCCG** is a simple, powerful, and developer-friendly Python library for interacting with Large Language Models (LLMs). It provides a clean, unified API to work with models from leading providers like OpenAI, Google, Anthropic, and Mistral, making it easy to build, test, and deploy AI-powered applications.

## 🚀 Key Features

- **🔄 Unified API**: Switch between different LLM providers with minimal code changes
- **⚡ Async Support**: Built-in asynchronous operations for high-performance applications
- **🧠 Retrieval-Augmented Generation (RAG)**: Build powerful Q&A systems over your own documents
- **✨ Text Embedding**: Convert text into vector representations for semantic search
- **🌊 Streaming**: Real-time response streaming for interactive experiences
- **💾 Session Management**: Save and restore conversation history
- **🖥️ CLI Interface**: Powerful command-line tools for quick testing and interaction
- **🔧 Easy Configuration**: Chainable methods for clean, readable code
- **🛡️ Error Handling**: Robust error handling with custom exceptions

## 🏗️ Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-4o, GPT-3.5 Turbo | ✅ Fully Supported |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash | ✅ Fully Supported |
| **Mistral** | Mistral Tiny, Small, Medium | ✅ Fully Supported |
| **Anthropic** | Claude 3 Sonnet | ✅ Fully Supported |

## 📦 Installation

```bash
pip install fastccg
```

## ⚡ Quick Start

```python
import fastccg
from fastccg.models.gpt import gpt_4o

# Add your API key
api_key = fastccg.add_openai_key("sk-...")

# Initialize the model
model = fastccg.init_model(gpt_4o, api_key=api_key)

# Ask a question
response = model.ask("What is the best thing about Large Language Models?")
print(response.content)
```

## 🖥️ CLI Usage

FastCCG comes with a powerful CLI for quick interactions:

```bash
# List available models
fastccg models

# Ask a single question
fastccg ask "What is the capital of France?" --model gpt_4o

# Start an interactive chat session
fastccg chat --model gpt_4o
```

## 🧠 Retrieval-Augmented Generation (RAG)

Build a powerful question-answering system over your own documents with just a few lines of code. FastCCG handles the complexity of embedding, indexing, and context retrieval for you.

```python
import asyncio
import fastccg
from fastccg.models.gpt import gpt_4o
from fastccg.embedding import OpenAIEmbedding
from fastccg.rag import RAGModel

# 1. Setup API keys and models
api_key = fastccg.add_openai_key("sk-...")
llm = fastccg.init_model(gpt_4o, api_key=api_key)
embedder = OpenAIEmbedding(api_key=api_key)

# 2. Create and configure the RAG model
rag = RAGModel(llm=llm, embedder=embedder)

# 3. Index your documents
documents = {
    "doc1": "The sky is blue during a clear day.",
    "doc2": "The grass in the park is typically green."
}

# 4. Ask a question related to your documents
async def main():
    response = await rag.ask_async("What color is the sky?")
    print(response.content)
    # Expected output will be based on the indexed context

asyncio.run(main())

# 5. Save your knowledge base for later use
rag.save("my_knowledge.fcvs", pretty_print=True)
```

## 🔄 Advanced Features

### Asynchronous Operations
```python
import asyncio

async def main():
    # Run multiple prompts concurrently
    task1 = model.ask_async("What is the speed of light?")
    task2 = model.ask_async("What is the capital of Australia?")
    
    responses = await asyncio.gather(task1, task2)
    for response in responses:
        print(response.content)

asyncio.run(main())
```

### Streaming Responses
```python
async def stream_example():
    async for chunk in model.ask_stream("Tell me a story"):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_example())
```

### Session Management
```python
# Save conversation
model.save("my_session.json")

# Load conversation later
loaded_model = fastccg.load_model("my_session.json", api_key=api_key)
```

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) directory:

- **[Quick Start Guide](./docs/quick_start.md)** - Get up and running in minutes
- **[CLI Usage](./docs/cli_usage.md)** - Command-line interface guide
- **[FCVS CLI Tool](./docs/fcvs_cli.md)** - Manage `.fcvs` vector store files
- **[Embedding and RAG](./docs/embedding_and_rag.md)** - Guides for embedding and RAG
- **[API Reference](./docs/api_reference.md)** - Complete API documentation
- **[Supported Models](./docs/supported_models.md)** - All available models and providers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Why FastCCG?

- **Developer Experience**: Clean, intuitive API that just works
- **Performance**: Built with async-first architecture for scalable applications
- **Flexibility**: Easy to switch between providers and models
- **Reliability**: Comprehensive error handling and testing
- **Community**: Open source with active development and support

---

**[📖 Read the Full Documentation](./docs/index.md)** | **[🚀 Get Started Now](./docs/quick_start.md)** | **[💬 Join the Discussion](https://github.com/mebaadwaheed/fastccg/discussions)**