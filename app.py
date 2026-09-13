import os
import streamlit as st

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from rag import answer

st.set_page_config(page_title="Thesis RAG Assistant", page_icon="📄")
st.title("Thesis RAG Assistant")
st.caption("Ask questions about the research on AR-assisted indoor navigation and emergency evacuation.")

SUGGESTED_QUESTIONS = [
    "How can AR guide people during an emergency evacuation?",
    "Why is cognitive load a key factor in AR wayfinding?",
    "How does QR-code-based indoor positioning work?",
    "How much does phone AR tracking drift while walking?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.write("Not sure where to start? Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"suggested_{i}", use_container_width=True):
            st.session_state.pending_question = q

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed = st.chat_input("Ask a question about the research")
question = typed or st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the papers..."):
            text, sources = answer(question)
        cited = text + "\n\n**Sources:**\n" + "\n".join(
            f"- {s['source']} (page {s['page']})" for s in sources
        )
        st.markdown(cited)
    st.session_state.messages.append({"role": "assistant", "content": cited})
    st.rerun()