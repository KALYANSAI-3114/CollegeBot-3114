import streamlit as st
import logging

# silence langchain logs
logging.getLogger("langchain").setLevel(logging.ERROR)

from rag_pipeline import answer_question

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="CollegeBot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 CollegeBot")
st.caption("Ask anything about VVIT – courses, faculty, placements, campus")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- CHAT INPUT ----------------
question = st.chat_input("Ask your question here...")

if question:
    # user message
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.spinner("Thinking..."):
        try:
            response = answer_question(question)
            answer_text = response.get("answer", "No answer found.")
            sources = response.get("source_details", [])
        except Exception as e:
            answer_text = "⚠️ Something went wrong. Please try again."

    # assistant message
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_text}
    )

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
