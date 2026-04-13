from flask import Flask, render_template, request, Response, stream_with_context
from src.utils.model_utils import load_model
from src.pipelines.full_pipeline import run_full_pipeline_structured
from src.nlp.nlp_extracter import extract_entities
from src.utils.ollama_chat import stream_chat
import os
import json

app = Flask(__name__)

# Preload models to improve performance
models_loaded = False
price_model = None
recommend_model = None

def get_models():
    global models_loaded, price_model, recommend_model
    if not models_loaded:
        try:
            price_model = load_model(path="models/PricePrediction/best_model.pkl")
            recommend_model = load_model(path="models/Recommendation/best_model.pkl")
            models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
    return price_model, recommend_model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        try:
            # get user input
            user_text = request.form.get("issue")
            # extract the keywords from iuser text
            extracted_input = extract_entities(text=user_text)
            # Capture input from form
            form_data = {
                "issue": extracted_input["issue"],
                "device": extracted_input['device'],
                "severity": extracted_input['severity'],
                "urgent": extracted_input['urgent'],
                "brand": request.form.get("brand"),
                "device_age_years": float(request.form.get("device_age_years", 0)),
                "service_type": request.form.get("service_type"),
                "warranty_status": request.form.get("warranty_status"),
                "city_tier": request.form.get("city_tier"),
                "technician_experience": int(request.form.get("technician_experience", 0))
            }
            
            p_model, r_model = get_models()
            
            if p_model is None or r_model is None:
                return render_template("index.html", error="Models not loaded properly.")

            # Run pipeline
            results = run_full_pipeline_structured(form_data, p_model, r_model)
            
            return render_template("index.html", 
                                 results=results["top_providers"], 
                                 estimated_price=results["estimated_price"],
                                 user_query=user_text,
                                 scroll_to_results=True)
            
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")

    return render_template("index.html")

@app.route("/stream_explanation")
def stream_explanation():
    user_query = request.args.get("query")
    results_json = request.args.get("results")
    
    if not user_query or not results_json:
        return "Missing data", 400

    system_instruction = f"""You are a 'Service Provider Recommendation System' who is responsible to recommend 
    service provider to the user from the given recommendation data. Your output should be 'Provider name' and 'reason'.
    Explain the user the reason behind the recommended provider by refering from given data. Do not ask any follow up question to the user.
    Recommendation data: {results_json}
    User query: {user_query}"""

    def generate():
        for chunk in stream_chat(instruction_prompt=system_instruction, input_query=user_query):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True)
