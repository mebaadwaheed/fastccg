import asyncio
import os
import fastccg

from fastccg.models.gemini import gemini_flash_1_5

# --- Configuration ---
# IMPORTANT: Please replace "YOUR_MISTRAL_API_KEY" with your actual key
API_KEY = "AIzaSyA-2hkj7Nv6GMIlGOjGe3P7MDcyvaBnGF0"
MODEL_CLASS = gemini_flash_1_5
SAVE_PATH = "test_session.json"

async def main():
    """Runs advanced tests for async, streaming, and save/load features."""
    if API_KEY == "YOUR_MISTRAL_API_KEY":
        print("Please set your Mistral API key in tests/advanced_test.py")
        return

    print("--- Initializing Model ---")
    fastccg.add_gemini_key(API_KEY)
    model = fastccg.init_model(MODEL_CLASS)
    model.sys_prompt("You are a helpful assistant that provides concise answers.")
    print(f"Model: {model.model_name}")
    print("-" * 20)

    # 1. Test ask_async
    print("\n--- Testing ask_async ---")
    prompt1 = "What is the capital of France?"
    print(f"> User: {prompt1}")
    response_async = await model.ask_async(prompt1)
    print(f"< AI: {response_async.content}")
    print("-" * 20)

    # 2. Test ask_stream
    print("\n--- Testing ask_stream ---")
    prompt2 = "List three benefits of exercise."
    print(f"> User: {prompt2}")
    print("< AI (streaming): ", end="", flush=True)
    full_stream_response = ""
    async for chunk in model.ask_stream(prompt2):
        print(chunk.content, end="", flush=True)
        full_stream_response += chunk.content
    print("\n" + "-" * 20)

    # 3. Test save and load
    print("\n--- Testing Save/Load ---")
    print(f"Saving model state to {SAVE_PATH}...")
    model.save(SAVE_PATH)

    print("Loading model from file...")
    # We need to provide the API key again for the new instance
    loaded_model = fastccg.load_model(SAVE_PATH, api_key=API_KEY)

    print("Verifying loaded state...")
    print(f"Loaded model name: {loaded_model.model_name}")
    print(f"Loaded system prompt: {loaded_model._sys_prompt.content}")
    print(f"History length: {len(loaded_model.get_history())}")

    # Verify history is correct (should contain the last two interactions)
    history = loaded_model.get_history()
    if (
        len(history) == 4 and
        history[0].content == prompt1 and
        history[1].content == response_async.content and
        history[2].content == prompt2 and
        history[3].content.strip() == full_stream_response.strip()
    ):
        print("History verification successful!")
    else:
        print("History verification FAILED!")
        print("Expected history length 4.")
        print(f"Actual history: {history}")

    print("-" * 20)

    # 4. Test asking a question with the loaded model
    print("\n--- Testing loaded model ---")
    prompt3 = "Based on our conversation, what was the first question I asked?"
    print(f"> User: {prompt3}")
    response_loaded = await loaded_model.ask_async(prompt3)
    print(f"< AI: {response_loaded.content}")
    print("-" * 20)

    # Clean up the saved file
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print(f"Cleaned up {SAVE_PATH}.")

if __name__ == "__main__":
    asyncio.run(main())
