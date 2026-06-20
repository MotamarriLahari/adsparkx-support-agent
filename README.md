# Persona-Adaptive Customer Support Agent

A persona-aware AI customer support agent built using Retrieval-Augmented Generation (RAG), LLMs, and human-in-the-loop escalation. The system detects the type of customer interacting with it, retrieves relevant information from a knowledge base, generates a response in a tone suited to that customer, and escalates to a human agent when necessary.

Built as part of the Adsparkx AI Assignment (NxtWave hiring process).

---

## 1. Project Overview

This support agent automatically identifies whether a customer is a **Technical Expert**, a **Frustrated User**, or a **Business Executive**, based on their message. It then retrieves relevant content from a knowledge base of support documents and generates a grounded, persona-adapted response. If the query touches a sensitive area (billing, legal, security), if the knowledge base has no good answer, or if the user remains unresolved after multiple turns, the system escalates the conversation and produces a structured handoff summary for a human agent.

---

## 2. Tech Stack

| Component | Tool / Library | Version |
|---|---|---|
| Language | Python | 3.13 |
| LLM | Groq (Llama 3.1 8B Instant) | `llama-3.1-8b-instant` |
| Agent Framework | LangChain (community + text-splitters) | 0.3.x |
| Embedding Model | Sentence Transformers (`all-MiniLM-L6-v2`) | via `langchain_community.embeddings` |
| Vector Database | ChromaDB | latest (persisted locally) |
| UI | Streamlit | latest |
| PDF Handling | pypdf | latest |

**Note on LLM choice:** The project was originally built against Google Gemini, but the free tier of the Gemini API returned a persistent `RESOURCE_EXHAUSTED` (quota = 0) error that could not be resolved even after regenerating the API key. The project was migrated to Groq's free tier (Llama 3.1 8B Instant), which is fast and reliable. This is documented here for transparency, since the assignment explicitly permits any LLM provider.

---

## 3. Architecture

```
User Query
    │
    ▼
Persona Detection (Groq LLM classifier)
    │
    ▼
Knowledge Base Retrieval (ChromaDB similarity search, top-k=4)
    │
    ▼
Escalation Check (keywords / low confidence / repeated turns)
    │
    ├── If escalate = True ──► Human Handoff Summary (structured JSON)
    │
    └── If escalate = False ─► Adaptive Response Generation (Groq LLM, persona-specific tone)
                                       │
                                       ▼
                                Response shown in Streamlit UI with sources
```

---

## 4. Persona Detection Strategy

**Method:** Zero-shot LLM classification (not a separate ML model — the same LLM used for generation is prompted to classify).

**Prompt design:** The model is given the three persona categories with their defining characteristics (technical terminology vs. emotional language vs. outcome-focus) and the raw user message, and asked to return exactly one label with no extra text. The output is validated against the three allowed labels — if the model returns something unexpected, the system defaults to "Frustrated User" as a safe fallback (since this persona triggers the most cautious, empathetic response style).

**Rules used:**
- Technical Expert → terms like API, error code, logs, configuration
- Frustrated User → emotional/urgent language, repeated complaints
- Business Executive → impact, timeline, ROI-oriented phrasing

---

## 5. RAG Pipeline Design

**Chunking strategy:** Documents are split using `RecursiveCharacterTextSplitter` with a chunk size of 500 characters and 50 characters of overlap. This balances enough context per chunk against retrieval precision.

**Embedding model:** `all-MiniLM-L6-v2` (Sentence Transformers) — a small, fast, locally-run embedding model that avoids any API cost or rate limits for the embedding step.

**Vector database:** ChromaDB, persisted locally to `./chroma_db`. Chosen for its simplicity and zero external dependency (no server or account needed).

**Retrieval strategy:** Top-k similarity search (k=4) using cosine/L2 distance. Each result includes the source document filename as metadata, which is surfaced in the UI and included in the handoff summary.

---

## 6. Escalation Logic

The system escalates a conversation when **any** of the following is true:

| Trigger | Condition |
|---|---|
| Sensitive topic | Message contains a keyword related to billing disputes, legal threats, fraud, account compromise, or explicit request for a manager |
| Low retrieval confidence | Best similarity score from ChromaDB exceeds a configurable distance threshold (`1.2`), meaning no chunk in the knowledge base is a good match |
| No results at all | Knowledge base returns zero relevant chunks |
| Repeated dissatisfaction | The user has been in the conversation for 3+ turns and remains unresolved |

**Confidence threshold:** `LOW_CONFIDENCE_THRESHOLD = 1.2` (ChromaDB L2 distance — lower is better; this can be tuned in `src/escalation.py`).

When escalation triggers, `src/handoff.py` builds a structured JSON summary containing the detected persona, the issue, full conversation history, documents used, attempted steps, escalation reason, and a context-aware recommendation for the human agent.

---

## 7. Setup Instructions

### Prerequisites
- Python 3.11+
- A free Groq API key from https://console.groq.com/keys

### Installation

```bash
# Clone the repository
git clone https://github.com/MotamarriLahari/adsparkx-support-agent.git
cd adsparkx-support-agent

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Build the knowledge base (run once)

```bash
python ingest.py
```

This loads all documents from `/data`, chunks them, generates embeddings, and stores them in a local ChromaDB at `./chroma_db`.

### Run the app

```bash
streamlit run app.py
```

This opens the chat interface at `http://localhost:8501`.

---

## 8. Example Queries

| # | Query | Expected Behavior |
|---|---|---|
| 1 | "Can you explain the API authentication failure and provide error details?" | Detected as **Technical Expert** → detailed, technical response with error codes |
| 2 | "I've tried everything and nothing works! How do I export my data?" | Detected as **Frustrated User** → empathetic, step-by-step response |
| 3 | "How does this issue impact our operations and when will it be resolved?" | Detected as **Business Executive** → concise, impact-focused response |
| 4 | "I want to file a legal complaint, my account was hacked" | **Escalated** — sensitive keyword detected, handoff summary generated |
| 5 | "asdkjaslkdj random gibberish query xyz123" | **Escalated** — low retrieval confidence, no relevant knowledge found |

---

## 9. Knowledge Base

Located in `/data`, containing 11 documents covering a SaaS product support domain:
- Password reset & login troubleshooting
- Billing policy & pricing plans
- API authentication & rate limits
- Webhook setup
- Account suspension policy
- SLA policy
- Data export
- General FAQ (PDF)

---

## 10. Known Limitations

- **Single-session memory only:** Conversation history is held in Streamlit's session state and is lost on page refresh; there is no persistent database for long-term conversation storage.
- **Escalation thresholds are static:** The confidence threshold and keyword list are hardcoded (though configurable in `src/escalation.py`); a production system would tune these against real traffic.
- **Persona detection can occasionally misclassify** short, neutral queries (e.g., a plain question can sometimes be labeled "Frustrated User" by the LLM in the absence of stronger signals).
- **No authentication/user accounts:** This is a prototype interface, not a production-ready multi-user system.
- **Embedding model runs locally on CPU:** For larger knowledge bases, a hosted embedding API would scale better.

### Future Improvements
- Add multi-turn conversation memory backed by SQLite
- Add confidence scoring displayed in the UI
- Add a feedback ("was this helpful?") loop to improve retrieval over time
- Implement LangGraph for explicit multi-step agentic orchestration
