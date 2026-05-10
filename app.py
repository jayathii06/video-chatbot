import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import imageio_ffmpeg
import re
import yt_dlp
import tempfile
import time

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

api_key = os.secrets.get("GROQ_API_KEY", "")
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

def parse_vtt(vtt_text):
    lines = vtt_text.splitlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE") or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            text_lines.append(line)
    deduped = []
    for line in text_lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return " ".join(deduped)

def run_yt_dlp(url, ydl_opts):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False) if ydl_opts.get("skip_download") else ydl.download([url])

def pick_caption_url(info):
    captions = info.get("subtitles", {}) or {}
    auto_caps = info.get("automatic_captions", {}) or {}

    for lang_group in [captions, auto_caps]:
        for lang in ["en", "en-US", "en-GB", "hi", "te", "ta", "kn", "ml"]:
            tracks = lang_group.get(lang)
            if tracks:
                for track in tracks:
                    if track.get("ext") == "vtt" and track.get("url"):
                        return track["url"]
                for track in tracks:
                    if track.get("url"):
                        return track["url"]

    for lang_group in [captions, auto_caps]:
        for tracks in lang_group.values():
            for track in tracks:
                if track.get("url"):
                    return track["url"]
    return None

def download_caption_text(caption_url):
    import requests
    r = requests.get(caption_url, timeout=20)
    r.raise_for_status()
    return parse_vtt(r.text)

def transcribe_audio_with_groq(video_url):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "quiet": True,
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
            "extractor_retries": 3,
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

    info = None
    try:
        info = run_yt_dlp(video_url, {
            "skip_download": True,
            "quiet": True,
            "noplaylist": True,
            "retries": 2,
            "extractor_retries": 2,
            "socket_timeout": 20,
        })
    except Exception:
        info = None

    if info:
        caption_url = pick_caption_url(info)
        if caption_url:
            try:
                transcript = download_caption_text(caption_url)
                if transcript.strip():
                    st.session_state[cache_key] = transcript
                    return transcript
            except Exception as e:
                st.info(f"Caption fetch failed, trying fallback transcription. {e}")

    if client is None:
        return None

    try:
        st.info("🎙️ No usable captions found. Trying audio transcription...")
        transcript = transcribe_audio_with_groq(video_url)
        if transcript and transcript.strip():
            st.session_state[cache_key] = transcript
            return transcript
    except Exception as e:
        st.error(f"Transcription failed: {e}")

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

st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Paste a YouTube URL")
st.sidebar.markdown("2. Fetch captions or transcribe audio")
st.sidebar.markdown("3. Ask questions about the video")
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by FAISS + Groq")

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
                st.error("❌ Could not extract captions or transcribe this video.")
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

            st.success("✅ Video ready. Ask anything about it.")

        with st.expander("📄 See full transcript"):
            st.write(st.session_state.transcript)

elif url and not api_key:
    st.warning("⚠️ API key not configured.")

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask anything about the video...")

if "starter" in st.session_state:
    question = st.session_state.starter
    del st.session_state.starter

if question and "index" in st.session_state and client:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    context = retrieve(question, st.session_state.index, st.session_state.chunks)

    system_prompt = f"""
You answer questions about a YouTube video transcript.
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
                reply = f"Sorry, I hit an error while generating the answer: {e}"
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

elif question and "index" not in st.session_state:
    st.warning("⚠️ Please paste a YouTube URL first!")