from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VALID_PERSONAS = ["Technical Expert", "Frustrated User", "Business Executive"]

def detect_persona(user_message: str) -> str:
    prompt = f"""You are a classifier for a customer support system.
Classify the following customer message into EXACTLY ONE persona category.

Categories:
- "Technical Expert": Uses technical terminology, asks about APIs, logs, error codes, configurations, wants detailed/precise explanations.
- "Frustrated User": Uses emotional language, expresses urgency or anger, mentions repeated failed attempts, wants quick resolution.
- "Business Executive": Outcome-focused, asks about business impact, timelines, cost, prefers concise answers, less interested in technical detail.

Customer Message: "{user_message}"

Respond with ONLY the persona name, exactly as written above. No explanation, no punctuation.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception:
        return "Frustrated User"  # safe fallback if API call fails

    for persona in VALID_PERSONAS:
        if persona.lower() in raw_output.lower():
            return persona

    return "Frustrated User"


if __name__ == "__main__":
    test_messages = [
        "Can you explain the API authentication failure and provide error details?",
        "I've tried everything and nothing works! This is so frustrating!",
        "How does this issue impact our operations and when will it be resolved?"
    ]

    for msg in test_messages:
        persona = detect_persona(msg)
        print(f"Message: {msg}")
        print(f"Detected Persona: {persona}")
        print("-" * 50)