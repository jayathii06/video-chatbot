"""
config.py
Centralised configuration and startup validation.

All env-var reads happen here so the rest of the codebase never
calls os.environ directly.
"""

import os
import sys
import logging
from dataclasses import dataclass

from groq import Groq
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDER_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class AppConfig:
    groq_client: Groq
    embedder: SentenceTransformer


def load_config() -> AppConfig:
    """
    Validate required environment variables and initialise shared clients.

    Exits early with a clear error message if configuration is missing,
    rather than letting the app blow up silently mid-request.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "GROQ_API_KEY environment variable is not set. "
            "Set it in your .env file or deployment secrets."
        )
        sys.exit(
            "❌  GROQ_API_KEY is missing.  "
            "Please set it and restart the app."
        )

    groq_client = Groq(api_key=api_key)
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    logger.info("Configuration loaded. Embedder: %s", EMBEDDER_MODEL)
    return AppConfig(groq_client=groq_client, embedder=embedder)
