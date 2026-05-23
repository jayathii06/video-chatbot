"""
app.py — Streamlit UI wiring only.
Delegates to config.py, transcriber.py, retriever.py, chatbot.py
"""

import logging
import streamlit as st

from config import load_config
from transcriber import extract_video_id, get_transcript
from retriever import chunk_transcript, build_vector_index, retrieve_context
from chatbot import get_answer

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="VideoMind", page_icon="🎥", layout="wide")

def _load_css() -> None:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

@st.cache_resource
def _get_config():
    return load_config()

config = _get_config()

# ── Session state ──────────────────────────────────────────────────────────
for key in ("messages", "video_url", "transcript", "vector_index", "video_id"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# ── Two-column split layout ────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="small")

# ══════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════
with left:
    # Logo
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;padding:0.5rem 0 2rem 0;'>
        <div style='width:38px;height:38px;background:rgba(255,255,255,0.05);
        border:1px solid rgba(255,255,255,0.10);border-radius:10px;
        display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <polygon points="5,3 19,12 5,21" fill="rgba(255,255,255,0.80)"/>
            </svg>
        </div>
        <div>
            <div style='font-size:1rem;font-weight:500;color:rgba(255,255,255,0.88);
            letter-spacing:-0.01em;font-family:Lora,Georgia,serif;'>VideoMind</div>
            <div style='font-size:0.68rem;color:rgba(255,255,255,0.30);'>
            ask anything about videos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # URL input
    st.markdown("<div class='input-label'>YouTube URL</div>", unsafe_allow_html=True)
    url = st.text_input("", placeholder="https://youtube.com/watch?v=...",
                        key="url_input", label_visibility="collapsed")
    
    if not url:
        st.markdown("""
    <div class='hint-block'>
        <div class='hint-block-title'>How it works</div>

        <div class='hint-item'>
            <div class='hint-dot'></div>
            <div class='hint-text'>Paste any YouTube URL above</div>
        </div>

        <div class='hint-item'>
            <div class='hint-dot'></div>
            <div class='hint-text'>We extract the transcript automatically</div>
        </div>

        <div class='hint-item'>
            <div class='hint-dot'></div>
            <div class='hint-text'>Ask anything — summaries, key points, quotes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    

    if url:
        video_id = extract_video_id(url)

        if not video_id:
            st.error("Invalid URL.")
        else:
            # FIX 1: Video card — removed raw ID line, fixed border-radius
            thumbnail = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            st.markdown(f"""
            <div class='video-card'>
                <img src='{thumbnail}' style='width:100%;height:136px;
                object-fit:cover;border-radius:10px 10px 0 0;display:block;'/>
                <div style='padding:12px 14px;'>
                    <div style='font-size:0.80rem;font-weight:500;
                    color:rgba(255,255,255,0.82);margin-bottom:6px;'>
                    YouTube Video</div>
                    <span class='status-pill'>🎬 Ready to analyse</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Process new URL
            if st.session_state.get("video_url") != url:
                st.session_state.messages = []
                st.session_state.video_url = url
                st.session_state.video_id = video_id
                st.session_state.transcript = None
                st.session_state.vector_index = None

                with st.spinner("Extracting transcript..."):
                    try:
                        transcript, method = get_transcript(url, config.groq_client)
                    except ValueError as exc:
                        st.error(f"{exc}")
                        st.stop()

                if not transcript:
                    st.error("Could not extract captions. Try another video.")
                    st.stop()

                st.session_state.transcript = transcript

                with st.spinner("Building knowledge base..."):
                    chunks = chunk_transcript(transcript)
                    if not chunks:
                        st.error("Transcript too short.")
                        st.stop()
                    st.session_state.vector_index = build_vector_index(
                        chunks, config.embedder)

                method_label = "Captions" if method == "captions" else "Whisper"
                st.success(f"Ready · {method_label} · {len(chunks)} chunks")

            # Transcript expander
            if st.session_state.transcript:
                with st.expander("📄 Full transcript"):
                    st.write(st.session_state.transcript)

    # FIX 2: Quick question pills — left-aligned text, more spacing
    if st.session_state.vector_index:
        st.markdown("<div class='pills-label'>Quick questions</div>",
                    unsafe_allow_html=True)
        pills = [
            ("✦  What is this video about?", "What is this video about?"),
            ("✦  Give me the key points",    "What are the key points?"),
            ("✦  Summarise for me",           "Give me a summary"),
        ]
        for label, question in pills:
            if st.button(label, use_container_width=True, key=f"pill_{question}"):
                st.session_state["_starter"] = question

    # FIX 3: Footer lowercase
    st.markdown("""
    <div style='position:fixed;bottom:1.4rem;left:1.5rem;
    font-size:0.62rem;color:rgba(255,255,255,0.10);letter-spacing:0.05em;'>
    faiss · groq · streamlit
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Chat
# ══════════════════════════════════════════════════════════════════════════
with right:
    st.markdown("""
    <div class='chat-header'>
        <div class='chat-dot'></div>
        <span class='chat-header-txt'>Chat with this video</span>
    </div>
    """, unsafe_allow_html=True)

    # Empty state
    if not st.session_state.vector_index:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>🎬</div>
            <div class='empty-title'>Paste a YouTube URL to start</div>
            <div class='empty-sub'>I'll analyse the video and answer anything you ask</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # FIX 4: Chat body — scrollable, padded, max-width on bubbles
        st.markdown("<div class='chat-body'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='bubble-row-user'>
                    <div class='bubble-user'>{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='bubble-row-ai'>
                    <div class='ai-avatar'>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                            <circle cx="12" cy="12" r="4" fill="rgba(255,255,255,0.50)"/>
                            <circle cx="12" cy="12" r="9"
                            stroke="rgba(255,255,255,0.20)" stroke-width="1.5"/>
                        </svg>
                    </div>
                    <div class='bubble-ai'>{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("Ask anything about the video...")

    if "_starter" in st.session_state:
        question = st.session_state.pop("_starter")

    if question:
        if not st.session_state.vector_index:
            st.warning("Paste a YouTube URL first!")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        context = retrieve_context(question, st.session_state.vector_index)

        with st.spinner("Thinking..."):
            try:
                reply = get_answer(
                    question=question,
                    context=context,
                    history=st.session_state.messages[:-1],
                    groq_client=config.groq_client,
                )
            except Exception as exc:
                reply = f"Sorry, I hit an error: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()