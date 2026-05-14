"""
app.py
Streamlit UI — wiring only.  No business logic lives here.

Delegates to:
  config.py      → env validation & shared clients
  transcriber.py → YouTube caption / Whisper extraction
  retriever.py   → chunking, FAISS indexing, context retrieval
  chatbot.py     → Groq LLM prompt building and response
"""

import logging
import streamlit as st

from config import load_config
from transcriber import extract_video_id, get_transcript
from retriever import chunk_transcript, build_vector_index, retrieve_context
from chatbot import get_answer

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="VideoMind", page_icon="🎥", layout="centered")

# ── Load CSS ───────────────────────────────────────────────────────────────
def _load_css() -> None:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

# ── Config (cached so it only runs once per session) ───────────────────────
@st.cache_resource
def _get_config():
    return load_config()

config = _get_config()

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:3rem 0 2rem 0; display:flex; align-items:center; gap:1.0rem;'>
  <svg width="90" height="90" viewBox="0 0 80 90" style="flex-shrink:0; margin-top:25px;">
    <path d="M16 10 L40 58 L64 10" fill="none" stroke="rgba(255,255,255,0.88)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M26 10 L40 42 L54 10" fill="none" stroke="rgba(255,255,255,0.20)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="40" cy="58" r="4" fill="rgba(255,255,255,0.88)"/>
  </svg>
  <div>
    <h1 style='font-family:Lora,Georgia,serif; font-size:2.4rem; font-weight:400;
    color:rgba(255,255,255,0.92); margin:0 0 0.3rem 0; line-height:1.1;'>VideoMind</h1>
    <p style='color:rgba(255,255,255,0.38); font-size:0.88rem; margin:0; font-weight:300;
    letter-spacing:0.03em;'>Ask anything about any YouTube video</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0;'>
        <h3 style='color:white; font-weight:600;'>⚙️ Settings</h3>
        <hr style='border-color:rgba(56,189,248,0.2); margin:12px 0;'>
        <p style='color:rgba(255,255,255,0.35); font-size:0.7rem;
        text-transform:uppercase; letter-spacing:0.1em; font-weight:600;
        margin-bottom:12px;'>HOW IT WORKS</p>
        <div style='display:flex; flex-direction:column; gap:12px;'>
            <div style='display:flex; align-items:center; gap:10px;'>
                <div style='width:30px; height:30px; border-radius:50%;
                background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3);
                display:flex; align-items:center; justify-content:center;
                font-size:0.8rem; flex-shrink:0;'>📺</div>
                <div>
                    <div style='color:rgba(255,255,255,0.35); font-size:0.68rem;'>Step 1</div>
                    <div style='color:white; font-size:0.82rem; font-weight:500;'>Paste a YouTube URL</div>
                </div>
            </div>
            <div style='display:flex; align-items:center; gap:10px;'>
                <div style='width:30px; height:30px; border-radius:50%;
                background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3);
                display:flex; align-items:center; justify-content:center;
                font-size:0.8rem; flex-shrink:0;'>⚡</div>
                <div>
                    <div style='color:rgba(255,255,255,0.35); font-size:0.68rem;'>Step 2</div>
                    <div style='color:white; font-size:0.82rem; font-weight:500;'>We grab subtitles</div>
                </div>
            </div>
            <div style='display:flex; align-items:center; gap:10px;'>
                <div style='width:30px; height:30px; border-radius:50%;
                background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3);
                display:flex; align-items:center; justify-content:center;
                font-size:0.8rem; flex-shrink:0;'>💬</div>
                <div>
                    <div style='color:rgba(255,255,255,0.35); font-size:0.68rem;'>Step 3</div>
                    <div style='color:white; font-size:0.82rem; font-weight:500;'>Ask anything!</div>
                </div>
            </div>
        </div>
        <hr style='border-color:rgba(56,189,248,0.2); margin:16px 0 8px 0;'>
        <p style='color:rgba(255,255,255,0.2); font-size:0.72rem; text-align:center;'>
        Powered by FAISS · Groq · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────
for key in ("messages", "video_url", "transcript", "vector_index"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# ── URL input ──────────────────────────────────────────────────────────────
url = st.text_input("🔗 Paste YouTube URL here:")

if url:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("❌ Invalid YouTube URL. Please check it and try again.")
        st.stop()

    # Show thumbnail immediately
    st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

    # If it's a new URL, clear state and re-process
    if st.session_state.get("video_url") != url:
        st.session_state.messages = []
        st.session_state.video_url = url
        st.session_state.transcript = None
        st.session_state.vector_index = None

        with st.spinner("⚡ Extracting transcript..."):
            try:
                transcript, method = get_transcript(url, config.groq_client)
            except ValueError as exc:
                st.error(f"⚠️ {exc}")
                st.stop()

        if not transcript:
            st.error("❌ Could not extract captions for this video. Try another!")
            st.stop()

        method_label = "📝 Captions" if method == "captions" else "🎙️ Whisper"
        st.session_state.transcript = transcript

        with st.spinner("🧠 Building knowledge base..."):
            chunks = chunk_transcript(transcript)
            if not chunks:
                st.error("❌ Transcript was too short or empty.")
                st.stop()
            st.session_state.vector_index = build_vector_index(chunks, config.embedder)

        st.success(f"✅ Ready ({method_label}, {len(chunks)} chunks). Ask anything!")

    # Transcript expander
    if st.session_state.transcript:
        with st.expander("📄 See full transcript"):
            st.write(st.session_state.transcript)

# ── Suggested starter questions ────────────────────────────────────────────
if st.session_state.vector_index and not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    col1, col2, col3 = st.columns(3)
    starters = {
        col1: ("📌 What is this video about?", "What is this video about?"),
        col2: ("🔑 What are the key points?", "What are the key points?"),
        col3: ("📝 Give me a summary", "Give me a summary"),
    }
    for col, (label, question) in starters.items():
        with col:
            if st.button(label):
                st.session_state["_starter"] = question

# ── Chat history display ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input + answer ────────────────────────────────────────────────────
question = st.chat_input("Ask anything about the video...")

# Allow starter buttons to inject a question
if "_starter" in st.session_state:
    question = st.session_state.pop("_starter")

if question:
    if not st.session_state.vector_index:
        st.warning("⚠️ Please paste a YouTube URL first!")
        st.stop()

    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    context = retrieve_context(question, st.session_state.vector_index)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Pass history excluding the message we just appended
                reply = get_answer(
                    question=question,
                    context=context,
                    history=st.session_state.messages[:-1],
                    groq_client=config.groq_client,
                )
            except Exception as exc:
                reply = f"Sorry, I hit an error: {exc}"
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})