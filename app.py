from flask import Flask, render_template, request
from src.utils.model_utils import load_model
from src.pipelines.full_pipeline import run_full_pipeline_structured
from src.nlp.nlp_extracter import extract_entities

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
                                 scroll_to_results=True)
            
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

