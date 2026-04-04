import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipelines.full_pipeline import run_full_pipeline
from src.utils.model_utils import load_model
from src.utils.ollama_chat import chat

if __name__ == "__main__":
    user_input = "My laptop is overheating and shutting down"
    # load price model
    price_model = load_model(path=os.path.join(os.path.dirname(__file__), '..', 'models', 'PricePrediction', 'best_model.pkl'))
    # load recommend model
    recommend_model = load_model(path=os.path.join(os.path.dirname(__file__), '..', 'models', 'Recommendation', 'best_model.pkl'))
    result = run_full_pipeline(user_input, price_model=price_model, recommend_model=recommend_model)
    print("\nFINAL OUTPUT:\n")
    print(result)
    print("\n-----------------------------------------------")
    system_instruction = f"""You are a 'Service Provider Recommendation System' who is responsible to recommend 
    service provider to the user from the given recommendation data. Your output should be 'Provider name' and 'reason'.
    Explain the user the reason behind the recommended provider by refering from given data. Do not ask any follow up question to the user.
    Recommendation data: {result}
    User query: {user_input}"""
    chat(instruction_prompt=system_instruction, input_query=user_input)
