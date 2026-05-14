# 🎬 VideoMind — AI Video Q&A

Ask questions about any YouTube video in natural language. VideoMind extracts the transcript, builds a semantic search index, and uses an LLM to answer questions grounded in the video content.

---

## How It Works

```
YouTube URL
    │
    ▼
┌─────────────────────────────┐
│  transcriber.py             │
│  1. Try captions via yt-dlp │
│  2. Fallback: Groq Whisper  │
└────────────┬────────────────┘
             │ plain text transcript
             ▼
┌─────────────────────────────┐
│  retriever.py               │
│  1. Character-bounded chunks│
│  2. Embed with MiniLM-L6-v2 │
│  3. FAISS inner-product idx │
└────────────┬────────────────┘
             │ top-k relevant chunks
             ▼
┌─────────────────────────────┐
│  chatbot.py                 │
│  Groq LLaMA-3.1-8b-instant  │
│  Capped conversation history│
└─────────────────────────────┘
```

---

## Project Structure

```
videomind/
├── app.py           # Streamlit UI (wiring only — no business logic)
├── config.py        # Env validation & shared client initialisation
├── transcriber.py   # YouTube caption fetching + Whisper fallback
├── retriever.py     # Text chunking, FAISS indexing, context retrieval
├── chatbot.py       # LLM prompt building and Groq API call
├── style.css        # Dark-theme UI styles
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## Quick Start

### Local

```bash
# 1. Clone
git clone <your-repo-url>
cd videomind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY=gsk_...

# 4. Run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### Docker

```bash
docker build -t videomind .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... videomind
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Transcript | yt-dlp + youtube-transcript-api |
| Speech-to-text | Groq Whisper (fallback) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS (cosine similarity) |
| LLM | Groq `llama-3.1-8b-instant` |

---

## Design Decisions

**Why character-bounded chunking?**  
Sentence-count chunking produces unpredictable chunk sizes depending on sentence length. Character-bounded chunking (`~600 chars, 100 overlap`) ensures consistent retrieval quality regardless of content style.

**Why cap conversation history?**  
LLMs have a fixed context window. Sending full history on long conversations causes silent truncation or hard API errors. VideoMind keeps the last 6 message-pairs (~2 400 tokens) and discards older messages.

**Why `verify=False` on subtitle downloads?**  
YouTube's subtitle CDN occasionally presents certificates that fail validation in containerised environments with older CA bundles. The risk is low (read-only request, bad data is discarded by the parser) and is a documented trade-off, not an oversight.

---

## Limitations

- Videos longer than ~90 minutes may hit Groq Whisper's 25 MB audio limit
- Answers are strictly grounded in the transcript — no external knowledge
- Private or age-restricted videos cannot be accessed
