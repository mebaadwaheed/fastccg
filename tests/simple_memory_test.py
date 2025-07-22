import os
import shutil
import fastccg
from fastccg.models.mock import MockModel


def print_history(model, title):
    print(f"--- {title} ---")
    history = model.get_history()
    if not history:
        print("History is empty.")
        return
    for prompt in history:
        print(f"{prompt.role.capitalize()}: {prompt.content}")
    print("-" * (len(title) + 6) + "\n")


def run_simple_test():
    # Clean up previous runs
    if os.path.exists(".fcvs"):
        shutil.rmtree(".fcvs")

    # --- Part 1: First conversation --- #
    print("--- Starting First Conversation (Long-Term Memory Enabled) ---")
    api = fastccg.add_mock_key("akdosad")
    model1 = fastccg.init_model(MockModel, api_key=api)
    model1.enable_memory(long_term=True)

    model1.ask("What is the capital of France?")
    model1.ask("And what is its population?")

    print_history(model1, "History of Model 1")

    # --- Part 2: Second conversation, loading from memory --- #
    print("--- Starting Second Conversation (Loading from Long-Term Memory) ---")
    api = fastccg.add_mock_key("akdosad")
    model2 = fastccg.init_model(MockModel, api_key=api)
    print_history(model2, "History of Model 2 (Before Loading)")

    # Enable memory to trigger loading from the .fcvs file
    model2.enable_memory(long_term=True, recent_history_turns=5)
    print_history(model2, "History of Model 2 (After Loading)")

    model2.ask("Based on that, what language do they speak?")
    print_history(model2, "History of Model 2 (After a New Question)")

if __name__ == "__main__":
    run_simple_test()
