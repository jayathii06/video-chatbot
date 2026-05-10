import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
import faiss
import numpy as np
import os
import imageio_ffmpeg
import re
import yt_dlp
import tempfile

os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

st.set_page_config(page_title="VideoMind", page_icon="🎥", layout="wide")

def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
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
    <p style='color:rgba(255,255,255,0.45); font-size:1rem; margin-top:0.5rem;'>
    Ask questions about any YouTube video instantly</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

api_key = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=api_key) if api_key else None

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

def transcribe_audio_with_groq(video_url):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "quiet": True,
            "noplaylist": True,
            "retries": 3,
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        mp3_path = audio_path + ".mp3"
        if not os.path.exists(mp3_path):
            return None

        size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
        if size_mb > 25:
            st.warning(f"Audio is {size_mb:.1f}MB. Try a shorter video.")
            return None

        with open(mp3_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )
        return transcription

def get_transcript(video_url, video_id):
    cache_key = f"transcript_{video_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Method 1 — youtube-transcript-api
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)
        transcript = transcript_list.find_transcript(
            ['en', 'en-US', 'en-GB', 'hi', 'te', 'ta', 'kn', 'ml']
        ).fetch()
        text = " ".join(entry.text for entry in transcript)
        if text.strip():
            st.session_state[cache_key] = text
            return text
    except Exception as e:
        st.info(f"⚡ Caption error: {e}")

    # Method 2 — Groq Whisper fallback
    if client:
        try:
            st.info("🎙️ Transcribing with Groq Whisper...")
            transcript = transcribe_audio_with_groq(video_url)
            if transcript and transcript.strip():
                st.session_state[cache_key] = transcript
                return transcript
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")

    return None

def chunk_text(text, chunk_size=120, overlap=25):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(sentences):
        chunk = " ".join(sentences[start:start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def build_index(chunks):
    if not chunks:
        return None
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, k=5):
    q_embed = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_embed, min(k, len(chunks)))
    retrieved = []
    for i in ids[0]:
        if i != -1 and i < len(chunks):
            retrieved.append(chunks[i])
    return "\n---\n".join(retrieved)

# ── Sidebar ────────────────────────────────────────────
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

# ── URL input ──────────────────────────────────────────
url = st.text_input("🔗 Paste YouTube URL here:")

if "messages" not in st.session_state:
    st.session_state.messages = []

if url and api_key:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("❌ Invalid YouTube URL. Please check it and try again.")
    else:
        thumbnail = f"https://img.youtube.com/vi/{video_id}/0.jpg"
        st.image(thumbnail, use_container_width=True)

        if st.session_state.get("video_url") != url:
            st.session_state.messages = []
            st.session_state.video_url = url
            st.session_state.video_id = video_id
            st.session_state.pop("transcript", None)
            st.session_state.pop("chunks", None)
            st.session_state.pop("index", None)

            with st.spinner("⚡ Loading video text..."):
                transcript = get_transcript(url, video_id)

            if not transcript:
                st.error("❌ Could not extract captions for this video. Try another!")
                st.stop()

            with st.spinner("🧠 Building knowledge base..."):
                chunks = chunk_text(transcript)
                index = build_index(chunks)
                if not chunks or index is None:
                    st.error("❌ Transcript was too short or empty.")
                    st.stop()
                st.session_state.transcript = transcript
                st.session_state.chunks = chunks
                st.session_state.index = index

            st.success("✅ Video ready. Ask anything about it!")

        with st.expander("📄 See full transcript"):
            st.write(st.session_state.transcript)

elif url and not api_key:
    st.warning("⚠️ API key not configured.")

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
if question and "index" in st.session_state and client:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    context = retrieve(question, st.session_state.index, st.session_state.chunks)

    system_prompt = f"""You answer questions about a YouTube video transcript.
Use only the provided transcript context.
If the answer is not in the transcript, say it is not covered.
Be specific, clear, and concise.

Transcript context:
{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[:-1]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    max_tokens=900,
                    temperature=0.2,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"Sorry, I hit an error: {e}"
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

elif question and "index" not in st.session_state:
    st.warning("⚠️ Please paste a YouTube URL first!")