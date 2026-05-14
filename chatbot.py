"""
chatbot.py
Responsible for building prompts and calling the Groq LLM.

Keeps conversation history capped to avoid exceeding the model's context window.
"""

import logging
from groq import Groq

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 900
TEMPERATURE = 0.2

# Maximum number of past message-pairs (user + assistant) to include.
# Each pair is roughly 200-400 tokens, so 6 pairs ≈ 2 400 tokens max history.
MAX_HISTORY_PAIRS = 6


# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are VideoMind, an AI assistant that answers questions \
about a specific YouTube video based solely on its transcript.

Rules:
- Use ONLY the transcript context provided below.
- If the answer is not in the transcript, say so clearly.
- Be specific, concise, and factual.
- Do not invent information beyond what the transcript contains.

Transcript context:
{context}
"""


# ── Public API ─────────────────────────────────────────────────────────────

def build_messages(
    question: str,
    context: str,
    history: list[dict],
) -> list[dict]:
    """
    Assemble the message list for the Groq API call.

    Caps history to MAX_HISTORY_PAIRS most-recent pairs to prevent context
    overflow on long conversations.
    """
    system_msg = {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}

    # history is a flat list: [user, assistant, user, assistant, ...]
    # Keep the last N complete pairs (each pair = 2 messages)
    max_msgs = MAX_HISTORY_PAIRS * 2
    trimmed_history = history[-max_msgs:] if len(history) > max_msgs else history

    user_msg = {"role": "user", "content": question}

    return [system_msg, *trimmed_history, user_msg]


def get_answer(
    question: str,
    context: str,
    history: list[dict],
    groq_client: Groq,
) -> str:
    """
    Send the question + context to the LLM and return the answer string.
    Raises on API errors so the caller can handle them appropriately.
    """
    messages = build_messages(question, context, history)

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content
