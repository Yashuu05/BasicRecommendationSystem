import pandas as pd
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nlp.nlp_pipeline import run_nlp_pipeline
from src.utils.model_utils import predict_price, predict_recommendation
#from src.models.predict_recommendation import predict_recommendation


# Mock Provider Generator (TEMP)

def generate_providers(n=5):
    providers = []

    for i in range(n):
        provider = {
            "rating": round(random.uniform(3.0, 5.0), 1),
            "num_reviews": random.randint(50, 2000),
            "success_rate": round(random.uniform(0.6, 0.95), 2),
            "experience_years": random.randint(1, 10),
            "distance_km": round(random.uniform(0.5, 10), 2),
            "response_time_min": random.randint(5, 60),
            "base_price": random.randint(300, 2000),
            "service_type": random.choice(["home_service", "shop_visit"]),
            "availability": random.choice(["available", "busy"]),
            "provider_name": f"Provider_{i+1}"
        }
        providers.append(provider)
    return providers

provider = generate_providers(n=5)
print("Generated providers:", provider)

def run_full_pipeline(user_text, price_model, recommend_model):
    
    # Step 1
    print("step 1: NLP processing...")
    nlp_result = run_nlp_pipeline(user_text)
    print("\nNLP output: ", nlp_result)

    # Step 2
    print("\nstep 2: Price Prediction...")

    # prepare data
    price_input = {
        "issue" : nlp_result['issue'],
        "device" : nlp_result['device'],
        "severity" : nlp_result['severity'],
        "urgent" : nlp_result['urgent'],
        "brand" : "HP",
        "device_age_years" : 4,
        "service_type" : "home_service",
        "warranty_status" : "out_of_warranty",
        "city_tier" : "tier1",
        "technician_experience" : 10
    }

    # prdict the price
    price = predict_price(input_data=price_input,model=price_model)
    print(f"predicted price is INR {price}")

    # step 3: generate providers
    print("step 3")
    print("generating providers...")
    providers = generate_providers(n=8)

    # step 4: prepare data for recommendation
    print("\nstep 4: recommendation ranking...")

    rec_input = []
    for provider in providers:
        row = {
            "issue":nlp_result["issue"],
            "device":nlp_result["device"],
            "severity":nlp_result["severity"],
            "urgent":nlp_result["urgent"],
            **provider
        }
        rec_input.append(row)

    df = pd.DataFrame(rec_input)

    ranked = predict_recommendation(input_data=df, model=recommend_model, top_k=3)
    print("\nFinal Recommendation:\n") 
    result = [] 
    for _, row in ranked.iterrows(): 
        result.append({ "provider_name": row["provider_name"], 
                       "rating": row["rating"], 
                       "distance_km": row["distance_km"], 
                       "score": round(row["score"], 2), 
                       "estimated_price": float(price) 
                    }) 
        
    return { "input": user_text, "parsed": nlp_result, "estimated_price": price, "top_providers": result } 