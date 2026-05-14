"""
retriever.py
Responsible for turning a transcript into a searchable vector index
and retrieving relevant chunks for a given query.
"""

import re
import logging
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

CHUNK_SIZE_CHARS = 600     # target characters per chunk
CHUNK_OVERLAP_CHARS = 100  # overlap between consecutive chunks
TOP_K = 3                  # number of chunks to retrieve per query
MAX_CONTEXT_CHARS = 2000   # hard cap on total context sent to LLM (~500 tokens)


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class VectorIndex:
    index: faiss.IndexFlatIP
    chunks: list[str]
    embedder: SentenceTransformer


# ── Chunking ───────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """Rough sentence splitter that handles most punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_transcript(text: str) -> list[str]:
    """
    Split transcript into overlapping character-bounded chunks.

    Uses character count (not sentence count) as the primary unit so that
    chunk sizes are predictable regardless of sentence length distribution.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_chars = 0
    current_sentences: list[str] = []

    for sentence in sentences:
        current_sentences.append(sentence)
        current_chars += len(sentence) + 1  # +1 for space

        if current_chars >= CHUNK_SIZE_CHARS:
            chunk = " ".join(current_sentences).strip()
            if chunk:
                chunks.append(chunk)

            # Retain the last few sentences as overlap for the next chunk
            overlap_chars = 0
            overlap_sentences: list[str] = []
            for s in reversed(current_sentences):
                overlap_chars += len(s) + 1
                overlap_sentences.insert(0, s)
                if overlap_chars >= CHUNK_OVERLAP_CHARS:
                    break

            current_sentences = overlap_sentences
            current_chars = overlap_chars

    # Flush any remaining sentences
    if current_sentences:
        chunk = " ".join(current_sentences).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# ── Indexing & retrieval ───────────────────────────────────────────────────

def build_vector_index(chunks: list[str], embedder: SentenceTransformer) -> VectorIndex:
    """Encode chunks and build a FAISS inner-product index (cosine similarity)."""
    if not chunks:
        raise ValueError("Cannot build index from an empty chunk list.")

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    logger.info("Built FAISS index: %d chunks, dim=%d", len(chunks), embeddings.shape[1])
    return VectorIndex(index=index, chunks=chunks, embedder=embedder)


def retrieve_context(query: str, vector_index: VectorIndex, k: int = TOP_K) -> str:
    """
    Return the top-k most relevant chunks for the query, joined as a string.
    """
    q_embed = vector_index.embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(k, len(vector_index.chunks))
    _, ids = vector_index.index.search(q_embed, k)

    retrieved = [
        vector_index.chunks[i]
        for i in ids[0]
        if i != -1 and i < len(vector_index.chunks)
    ]
    context = "\n---\n".join(retrieved)
    return context[:MAX_CONTEXT_CHARS]