import sys
from fastccg.core.model_base import ModelBase

def run_terminal(model: ModelBase):
    """
    Starts an interactive terminal chat session with the given model.

    Supports special commands:
    - 'reset': Clears the conversation history.
    - 'exit': Exits the terminal chat.
    """
    model_name = model.model_info().get("model", "unknown model")
    print(f"🧠 fastccg Terminal Started — model: {model_name}")
    print("   Type 'reset' to clear history, 'exit' to quit.")

    while True:
        try:
            prompt = input(">>> ")

            if prompt.lower().strip() == "exit":
                print("[Goodbye 👋]")
                break

            if prompt.lower().strip() == "reset":
                model.reset()
                print("[Context cleared]")
                continue

            response = model.ask(prompt)
            print(response.content)

        except KeyboardInterrupt:
            print("\n[Goodbye 👋]")
            break
        except Exception as e:
            print(f"[Error: {e}]", file=sys.stderr)