import fastccg
from fastccg.models.mock import MockModel

api = fastccg.add_mock_key("akdosad")
model = fastccg.init_model(MockModel, api_key=api)

fastccg.run_terminal(model)