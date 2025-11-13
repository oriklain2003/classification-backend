import json
import os
from openai import OpenAI

# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY environment variable is not set")

# Initialize the client safely
client = OpenAI(api_key=api_key)

# Load classification rules from file
with open("rules.json", "r", encoding="utf-8") as f:
    rules = json.load(f)
# track 141 alt 32125  445knts
# Example: a single flight track (could be multiple points)
flight_track = [
    { "lat":  33.177790, "lon":35.386963, "alt": 5000, "speed": 700, "heading": 180},
    { "lat":  33.122026, "lon":35.390396, "alt": 5000, "speed": 700, "heading": 180},
    { "lat":  33.063349, "lon":35.397949, "alt": 5000, "speed": 700, "heading": 180},

]
"""
flight_track = [
    { "lat":  33.177790, "lon":35.386963, "alt": 5000, "speed": 100, "heading": 180},
    { "lat":  33.122026, "lon":35.390396, "alt": 5000, "speed": 100, "heading": 180},
    { "lat":  33.063349, "lon":35.397949, "alt": 5000, "speed": 100, "heading": 180},

]
drone

"""
"""
flight_track = [
    { "lat":  33.195029, "lon":36.051636, "alt": 2952, "speed": 41, "heading": 230},
    { "lat":  33.078885, "lon":35.822296, "alt": 2952, "speed": 41, "heading": 230},
    { "lat":  32.894579, "lon":35.662994, "alt": 2952, "speed": 41, "heading": 230},

]

birds day
"""
"""
flight_track = [
    { "lat": 33.970698, "lon": 31.360474, "alt": 32125, "speed": 445, "heading": 130},
    { "lat": 32.444885, "lon": 32.860107, "alt": 32125, "speed": 445, "heading": 130},
    { "lat": 32.236036, "lon": 34.414673, "alt": 32125, "speed": 445, "heading": 130},

]
Airbus A321-271NX

"""
# Prompt: instruct GPT to return structured JSON output

from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data:dict):
    data = data["data"]
    system_prompt = """
    You are an aerospace classification assistant.
    Given flight tracks and classification rules, estimate the probability for each category
    (birds, uav, aircraft), choose the most likely subtype, and explain why.
    Only use given classes and subtypes. Return exact subtype name

    Output strictly as JSON with the following structure:
    {
      "probabilities": {"birds": float, "uav": float, "aircraft": float, "missile":float},
      "best_guess": "category/subtype",
      "reasoning": "string"
    }
    in case there are 2 options based on the rules you can say that based on that data you cant deside what is the right answer and give high score to both.
    Probabilities must sum to 1.0
    """

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-5",  # or "gpt-4o-mini" for cheaper/faster version
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classification rules:\n{json.dumps(rules, ensure_ascii=False)}"},
            {"role": "user", "content": f"Flight track:\n{json.dumps(data, ensure_ascii=False)}"}
        ],
        temperature=1  # lower = more consistent
    )
    result_text = response.choices[0].message.content
    try:
        result = json.loads(result_text)
        return result
    except json.JSONDecodeError:
        print("Model output not valid JSON:")
        print(result_text)
        exit()





if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=80)