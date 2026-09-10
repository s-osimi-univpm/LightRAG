import json, re
from collections import defaultdict

#INPUT_FILE = "reproduce/results/agriculture_evaluation.json"
#OUTPUT_FILE = "reproduce/results/agriculture_summary.json"
INPUT_FILE = "reproduce/results/mix_evaluation.json"
OUTPUT_FILE = "reproduce/results/mix_summary.json"

# Load your input JSON file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Categories to track
categories = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Overall Winner"
]

# Initialize counters
results = {
    category: {
        "Answer 1": 0,
        "Answer 2": 0
    }
    for category in categories
}

# Process evaluations

for item in data:

    evaluation_text = item["evaluation"]

    # Remove markdown code fences safely
    evaluation_text = re.sub(r"^```json\s*", "", evaluation_text)
    evaluation_text = re.sub(r"\s*```$", "", evaluation_text)

    try:
        evaluation = json.loads(evaluation_text)

    except json.JSONDecodeError as e:
        print("\nFAILED TO PARSE:")
        print(e)
        print(evaluation_text[:1000])
        continue

    for category in categories:

        winner = evaluation.get(category, {}).get("Winner")

        if winner in ["Answer 1", "Answer 2"]:
            results[category][winner] += 1


# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

# Print result
print(json.dumps(results, indent=4))