from groq import Groq
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

TONE_INSTRUCTIONS = {
    "Technical Expert": "Be detailed and technical. Include root cause analysis where relevant, error codes, and step-by-step troubleshooting. Use precise technical terminology.",
    "Frustrated User": "Be empathetic and reassuring. Acknowledge their frustration briefly at the start. Use simple, clear language. Be action-oriented and get straight to the solution.",
    "Business Executive": "Be concise and outcome-focused. Avoid technical jargon. Mention business impact and estimated resolution time where applicable. Keep it brief."
}

def generate_response(user_message: str, persona: str, retrieved_results: list):
    """
    retrieved_results: list of (Document, score) tuples from rag_pipeline.retrieve()
    Returns: (response_text, list_of_sources)
    """
    context = "\n\n".join([doc.page_content for doc, _ in retrieved_results])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc, _ in retrieved_results]))
    tone = TONE_INSTRUCTIONS.get(persona, "")

    prompt = f"""You are a customer support agent. Answer the customer's question using ONLY the information in the context below. Do NOT make up or assume any information that isn't in the context.

Tone Instructions: {tone}

Context from Knowledge Base:
{context}

Customer Message: {user_message}

If the context does not contain enough information to answer, clearly say so instead of guessing.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"⚠️ Temporary issue reaching the AI service. Please wait a few seconds and try again. ({str(e)[:100]})"

    return answer, sources


if __name__ == "__main__":
    from src.rag_pipeline import retrieve
    from src.persona_detector import detect_persona

    test_query = "Can you explain the API authentication failure and provide error details?"

    persona = detect_persona(test_query)
    results = retrieve(test_query)
    answer, sources = generate_response(test_query, persona, results)

    print(f"Query: {test_query}")
    print(f"Detected Persona: {persona}\n")
    print(f"Response:\n{answer}\n")
    print(f"Sources: {sources}")