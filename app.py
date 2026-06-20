import streamlit as st
from src.persona_detector import detect_persona
from src.rag_pipeline import retrieve
from src.response_generator import generate_response
from src.escalation import should_escalate
from src.handoff import generate_handoff_summary

st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="centered")
st.title("🤖 Persona-Adaptive Customer Support Agent")
st.caption("Built with RAG + LLM + Human Escalation")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.turn_count = 0
    st.session_state.docs_used = []
    st.session_state.attempted_steps = []
    st.session_state.escalated = False
    st.session_state.last_issue = ""

# Display past chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Sidebar info
with st.sidebar:
    st.header("Session Info")
    st.write(f"**Turns:** {st.session_state.turn_count}")
    st.write(f"**Escalated:** {st.session_state.escalated}")
    if st.button("🔄 Start New Conversation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Chat input
user_input = st.chat_input("Describe your issue...", disabled=st.session_state.escalated)

if user_input and not st.session_state.escalated:
    st.session_state.turn_count += 1
    st.session_state.last_issue = user_input

    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Step 1: Detect persona
    persona = detect_persona(user_input)

    # Step 2: Retrieve from knowledge base
    results = retrieve(user_input)
    sources = list(set([doc.metadata.get("source", "Unknown") for doc, _ in results]))
    st.session_state.docs_used.extend(sources)

    # Step 3: Check escalation
    unresolved = st.session_state.turn_count >= 3
    escalate, reason = should_escalate(user_input, results, st.session_state.turn_count, unresolved)

    if escalate:
        st.session_state.escalated = True
        summary = generate_handoff_summary(
            persona=persona,
            issue=st.session_state.last_issue,
            conversation_history=[m["content"] for m in st.session_state.history],
            documents_used=list(set(st.session_state.docs_used)),
            attempted_steps=st.session_state.attempted_steps,
            escalation_reason=reason
        )
        with st.chat_message("assistant"):
            st.error(f"🚨 **Escalating to Human Support**\n\nReason: {reason}")
            st.subheader("📋 Handoff Summary")
            st.code(summary, language="json")
        st.session_state.history.append({"role": "assistant", "content": f"Escalated: {reason}"})

    else:
        # Step 4: Generate persona-adapted response
        answer, used_sources = generate_response(user_input, persona, results)
        st.session_state.attempted_steps.append(user_input[:60])

        with st.chat_message("assistant"):
            st.info(f"🎭 **Detected Persona:** {persona}")
            st.write(answer)
            st.caption(f"📚 **Sources:** {', '.join(used_sources)}")

        st.session_state.history.append({"role": "assistant", "content": answer})