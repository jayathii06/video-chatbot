import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import imageio_ffmpeg
import re
import yt_dlp
import requests
import tempfile

os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

st.set_page_config(page_title="Video RAG App", page_icon="🎥")
# Load CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.5rem 0;'>
    <div style='display:inline-block; background:rgba(56,189,248,0.1); 
    border:1px solid rgba(56,189,248,0.25); border-radius:999px; 
    padding:3px 14px; margin-bottom:10px;'>
        <span style='color:#38bdf8; font-size:11px; font-weight:500;'>
        ● AI Video Intelligence</span>
    </div>
    <h1 style='font-family:Inter,sans-serif; font-size:2.8rem; font-weight:700;
    color:white; margin:0; line-height:1.1;'>🎬 VideoMind</h1>
    <p style='color:rgba(255,255,255,0.35); font-size:1rem; margin-top:0.5rem;'>
    Ask questions about any YouTube video instantly</p>
</div>
<div style='display:flex; gap:10px; margin:1.2rem 0;'>
    <div style='flex:1; background:rgba(56,189,248,0.06); 
    border:1px solid rgba(56,189,248,0.15); border-radius:12px; 
    padding:1rem; text-align:center;'>
        <div style='font-size:1.3rem;'>⚡</div>
        <div style='color:white; font-size:0.85rem; font-weight:500; margin-top:4px;'>
        Lightning fast</div>
        <div style='color:rgba(255,255,255,0.35); font-size:0.75rem; margin-top:2px;'>
        Subtitles in seconds</div>
    </div>
    <div style='flex:1; background:rgba(56,189,248,0.06); 
    border:1px solid rgba(56,189,248,0.15); border-radius:12px; 
    padding:1rem; text-align:center;'>
        <div style='font-size:1.3rem;'>🎯</div>
        <div style='color:white; font-size:0.85rem; font-weight:500; margin-top:4px;'>
        Accurate answers</div>
        <div style='color:rgba(255,255,255,0.35); font-size:0.75rem; margin-top:2px;'>
        Vector search</div>
    </div>
    <div style='flex:1; background:rgba(56,189,248,0.06); 
    border:1px solid rgba(56,189,248,0.15); border-radius:12px; 
    padding:1rem; text-align:center;'>
        <div style='font-size:1.3rem;'>💬</div>
        <div style='color:white; font-size:0.85rem; font-weight:500; margin-top:4px;'>
        Natural chat</div>
        <div style='color:rgba(255,255,255,0.35); font-size:0.75rem; margin-top:2px;'>
        Conversational follow-ups</div>
    </div>
</div>
""", unsafe_allow_html=True)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def parse_vtt(vtt_text):
    lines = vtt_text.splitlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if (not line or
            line.startswith("WEBVTT") or
            line.startswith("NOTE") or
            "-->" in line or
            line.isdigit()):
            continue
        line = re.sub(r'<[^>]+>', '', line)
        if line.strip():
            text_lines.append(line.strip())
    # Remove duplicate consecutive lines
    deduped = []
    for line in text_lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return " ".join(deduped)

def get_subtitles(video_id):
    # Step 1 — download subtitle file directly
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "te", "hi", "ta", "kn", "ml"],
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(tmpdir, "sub"),
                "ffmpeg_location": ffmpeg_path,
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find downloaded subtitle file
            for file in os.listdir(tmpdir):
                if file.endswith(".vtt"):
                    with open(os.path.join(tmpdir, file), "r", encoding="utf-8") as f:
                        text = parse_vtt(f.read())
                    if text.strip():
                        return text.strip()

    except Exception as e:
        st.warning(f"⚠️ Subtitles not found: {e}")

    # Step 2 — Groq Whisper fallback
    try:
        st.info("🎙️ Transcribing with Groq Whisper...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_path,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                }],
                "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            mp3_path = audio_path + ".mp3"
            file_size = os.path.getsize(mp3_path) / (1024 * 1024)

            if file_size > 25:
                st.warning(f"⚠️ Audio is {file_size:.1f}MB — too large. Try a shorter video!")
                return None

            api_key = st.secrets.get("GROQ_API_KEY", "")
            client = Groq(api_key=api_key)
            with open(mp3_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    response_format="text"
                )
            return transcription

    except Exception as e:
        st.error(f"❌ Failed: {e}")
        return None

def chunk_text(text, chunk_size=5, overlap=1):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

def retrieve(query, index, chunks, k=8):
    q_embed = embedder.encode([query])
    _, I = index.search(np.array(q_embed), k=k)
    return "\n---\n".join(chunks[i] for i in I[0])

# ── Sidebar ────────────────────────────────────────────
api_key = st.secrets.get("GROQ_API_KEY", "")
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Paste a YouTube URL")
st.sidebar.markdown("2. We grab subtitles ⚡")
st.sidebar.markdown("3. Ask anything about the video!")
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by FAISS + Groq")

# ── URL input ──────────────────────────────────────────
url = st.text_input("🔗 Paste YouTube URL here:")

if url and api_key:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("❌ Invalid YouTube URL! Please check and try again.")
    else:
        thumbnail = f"https://img.youtube.com/vi/{video_id}/0.jpg"
        st.image(thumbnail, use_container_width=True)

        if "video_url" not in st.session_state or st.session_state.video_url != url:
            st.session_state.messages = []
            st.session_state.video_url = url

            with st.spinner("⚡ Fetching subtitles..."):
                transcript = get_subtitles(video_id)

            if not transcript:
                st.error("❌ Could not get subtitles for this video. Try another!")
                st.stop()

            with st.spinner("🧠 Building knowledge base..."):
                chunks = chunk_text(transcript)
                st.session_state.chunks = chunks
                st.session_state.index = build_index(chunks)
                st.session_state.transcript = transcript

            st.success("✅ Video ready! Ask me anything about it.")

        with st.expander("📄 See full transcript"):
            st.write(st.session_state.transcript)

elif url and not api_key:
    st.warning("⚠️ API key not configured.")

# ── Chat history ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggested questions ────────────────────────────────
if "index" in st.session_state and len(st.session_state.messages) == 0:
    st.markdown("### 💡 Try asking:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 What is this video about?"):
            st.session_state.starter = "What is this video about?"
    with col2:
        if st.button("🔑 What are the key points?"):
            st.session_state.starter = "What are the key points?"
    with col3:
        if st.button("📝 Give me a summary"):
            st.session_state.starter = "Give me a summary"

# ── Display messages ───────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ─────────────────────────────────────────
question = st.chat_input("Ask anything about the video...")

if "starter" in st.session_state:
    question = st.session_state.starter
    del st.session_state.starter

# ── Answer ─────────────────────────────────────────────
if question and "index" in st.session_state and api_key:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    context = retrieve(question, st.session_state.index, st.session_state.chunks)

    system_prompt = f"""You are an expert assistant that answers questions about a YouTube video transcript.
You have access to the most relevant sections of the transcript below.
Answer based on the transcript. If you can infer the answer from context, do so.
Only say "That wasn't covered" if the topic is completely absent from the context.
Be detailed and specific in your answers.

Relevant Transcript Sections:
{context}
"""
    messages = (
        [{"role": "system", "content": system_prompt}]
        + st.session_state.messages[:-1]
        + [{"role": "user", "content": question}]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1024,
            )
            reply = response.choices[0].message.content
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

elif question and "index" not in st.session_state:
    st.warning("⚠️ Please paste a YouTube URL first!")