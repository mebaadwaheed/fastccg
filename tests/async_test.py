import asyncio
import fastccg
from fastccg.models.gemini import gemini_pro_1_5

api_key = fastccg.add_gemini_key("AIzaSyAtCXR7jLHvZWqamCp22UcMpHY1XgRFGBc")
model = fastccg.init_model(gemini_pro_1_5, api_key=api_key)

async def main():
    # You can run multiple prompts concurrently
    task1 = model.ask_async("What is the speed of light?")
    task2 = model.ask_async("What is the capital of Australia?")

    # Wait for both to complete
    responses = await asyncio.gather(task1, task2)

    for response in responses:
        print(response.content)

asyncio.run(main())