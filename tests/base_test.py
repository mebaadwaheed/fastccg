import fastccg
from fastccg.models.mistral import mistral_tiny

# Set API key
# IMPORTANT: Please replace "YOUR_MISTRAL_API_KEY" with your actual key
api = fastccg.add_mistral_key("YOUR_MISTRAL_API_KEY")

# Init model
model = fastccg.init_model(mistral_tiny, api_key=api)

# Configure
model.sys_prompt("You're a wise teacher.")
model.reply_filter(lambda x: x.strip().upper())

# Ask something
response = model.ask("Explain photosynthesis.")
print(response.content)

# Open a terminal chat interface
fastccg.run_terminal(model)

