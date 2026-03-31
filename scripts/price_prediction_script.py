# price_prediction_script.py

import random
import numpy as np
import pandas as pd


# Config
NUM_SAMPLES = 2000

issues = ["overheating", "broken_screen", "motherboard_damage", "software_issue", "network_issue", "strange_noise"]
devices = ["laptop", "desktop", "printer", "router", "cctv"]
severity_levels = ["low", "medium", "high"]
brands = ["HP", "Dell", "Lenovo", "Apple", "Asus", "Acer", "Canon"]
service_types = ["home_service", "shop_visit"]
warranty_status = ["in_warranty", "out_of_warranty"]
city_tiers = ["tier1", "tier2", "tier3"]

# Helper Functions

def base_price_by_issue(issue):
    mapping = {
    "overheating": 800,
    "broken_screen": 2500,
    "motherboard_damage": 4000,
    "software_issue": 500,
    "network_issue": 700,
    "strange_noise": 300
    }
    return mapping[issue]

def severity_multiplier(severity):
    return {"low": 0.8, "medium": 1.0, "high": 1.5}[severity]

def brand_multiplier(brand):
    premium = ["Apple"]
    return 1.5 if brand in premium else 1.0

def service_type_multiplier(service):
    return 1.2 if service == "home_service" else 1.0

def city_multiplier(city):
    return {"tier1": 1.3, "tier2": 1.1, "tier3": 0.9}[city]

# Data Generation

data = []

for _ in range(NUM_SAMPLES):
    issue = random.choice(issues)
    device = random.choice(devices)
    severity = random.choice(severity_levels)
    urgent = random.choice(["yes", "no"])
    brand = random.choice(brands)
    device_age = round(np.random.uniform(0, 8), 1)
    service_type = random.choice(service_types)
    warranty = random.choice(warranty_status)
    city = random.choice(city_tiers)
    technician_exp = random.randint(0, 15)

# Price calculation logic
    price = base_price_by_issue(issue)
    price *= severity_multiplier(severity)
    price *= brand_multiplier(brand)
    price *= service_type_multiplier(service_type)
    price *= city_multiplier(city)

    if urgent == "yes":
        price *= 1.2

    if warranty == "in_warranty":
        price *= 0.5

    if device_age > 5:
        price *= 1.1

# Add noise
    noise = np.random.normal(0, 100)
    price = max(100, int(price + noise))

    data.append([
        issue, device, severity, urgent, brand,
        device_age, service_type, warranty,
        city, technician_exp,
        price   
    ])

# Create DataFrame

columns = [
"issue", "device", "severity", "urgent", "brand",
"device_age_years", "service_type", "warranty_status",
"city_tier", "technician_experience",
"price"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset

df.to_csv("price_prediction_dataset.csv", index=False)

print("Price prediction dataset generated successfully!")
