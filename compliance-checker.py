import json

# Mock compliance checker (e.g., scan for PII)
with open("data/sample_meta.json", "r") as f:
    data = json.load(f)

if "ssn" in data:
    raise Exception("Sensitive data found in training set metadata")
