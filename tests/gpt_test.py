import fastccg
from fastccg.models.gpt import gpt_4o

api = fastccg.add_openai_key("edahdsuhdsa")
model = fastccg.init_model(gpt_4o, api_key=api)

fastccg.run_terminal(model)