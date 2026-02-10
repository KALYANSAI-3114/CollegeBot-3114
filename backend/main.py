import streamlit as st
import logging

logging.getLogger("langchain").setLevel(logging.ERROR)

from backend.rag_pipeline import answer_question 

st.set_page_config(
    page_title="CollegeBot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 CollegeBot")
st.caption("Ask anything about VVIT – courses, faculty, placements, campus")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Ask your question here...")

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.spinner("Thinking..."):
        try:
            response = answer_question(question)
            answer_text = response.get("answer", "No answer found.")
        except Exception as e:
            answer_text = f"⚠️ Error: {str(e)}"

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_text}
    )

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
