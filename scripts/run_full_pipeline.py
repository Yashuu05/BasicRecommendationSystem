import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipelines.full_pipeline import run_full_pipeline
from src.utils.model_utils import load_model

if __name__ == "__main__":
    user_input = "My laptop is overheating and shutting down urgently"
    # load price model
    price_model = load_model(path=os.path.join(os.path.dirname(__file__), '..', 'models', 'PricePrediction', 'best_model.pkl'))
    # load recommend model
    recommend_model = load_model(path=os.path.join(os.path.dirname(__file__), '..', 'models', 'Recommendation', 'best_model.pkl'))
    result = run_full_pipeline(user_input, price_model=price_model, recommend_model=recommend_model)
    print("\nFINAL OUTPUT:\n")
    print(result)
