# recommendation_script.py

import random
import numpy as np
import pandas as pd

# Config

NUM_SAMPLES = 2300

issues = ["overheating", "not_working", "broken_screen", "slow_performance", "wifi_issue", "power_issue", "strange_noise"]
devices = ["laptop", "desktop", "printer", "router", "scanner", "cctv"]
severity_levels = ["low", "medium", "high"]
service_types = ["home_service", "shop_visit"]
availability_status = ["available", "busy", "offline"]

# Helper Functions

def generate_provider_features():
    rating = round(np.random.uniform(1.5, 5.0), 1)
    reviews = random.randint(10, 3000)
    success_rate = round(np.random.uniform(0.3, 0.95), 2)
    experience = random.randint(0, 15)
    distance = round(np.random.uniform(0.5, 15), 2)
    response_time = random.randint(5, 120)
    base_price = random.randint(200, 3000)
    service_type = random.choice(service_types)
    availability = random.choices(
    availability_status, weights=[0.6, 0.3, 0.1]
    )[0]

    return rating, reviews, success_rate, experience, distance, response_time, base_price, service_type, availability

def compute_selection_probability(rating, success_rate, distance, response_time, availability):
    score = (
    (rating / 5) * 0.4 +
    success_rate * 0.3 +
    (1 / (1 + distance)) * 0.15 +
    (1 / (1 + response_time)) * 0.1 
    )

    if availability == "available":
        score += 0.05
    elif availability == "offline":
        score -= 0.2
    return min(max(score, 0), 1)

# Data Generation

data = []

for _ in range(NUM_SAMPLES):

    issue = random.choice(issues)
    device = random.choice(devices)
    severity = random.choice(severity_levels)
    urgent = random.choice(["yes", "no"])

    rating, reviews, success_rate, experience, distance, response_time, base_price, service_type, availability = generate_provider_features()

    # Adjust probability based on urgency
    prob = compute_selection_probability(rating, success_rate, distance, response_time, availability)

    if urgent == "yes":
        prob += 0.05  # urgency favors fast/available providers

    prob = min(prob, 1)
    selected = np.random.choice([1, 0], p=[prob, 1 - prob])
    data.append([
        issue, device, severity, urgent,
        rating, reviews, success_rate, experience,
        distance, response_time, base_price,
        service_type, availability,
        selected
    ])

# Create DataFrame

columns = [
"issue", "device", "severity", "urgent",
"rating", "num_reviews", "success_rate", "experience_years",
"distance_km", "response_time_min", "base_price",
"service_type", "availability",
"selected"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset

df.to_csv("recommendation_dataset.csv", index=False)

print("✅ Recommendation dataset generated successfully!")