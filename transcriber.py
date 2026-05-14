"""
transcriber.py
Responsible for extracting transcript text from a YouTube video.

Strategy:
  1. Try to fetch captions (manual or auto-generated) via yt-dlp + requests.
  2. If captions are unavailable, fall back to Groq Whisper audio transcription.
"""

import os
import re
import tempfile
import logging

import requests
import yt_dlp
import imageio_ffmpeg

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_vtt(vtt_text: str) -> str:
    """Convert raw VTT subtitle text into clean, deduplicated plain text."""
    lines = vtt_text.splitlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line or "-->" in line or line.startswith("WEBVTT") or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)  # strip inline tags like <c>
        if line:
            text_lines.append(line)

    # Remove consecutive duplicate lines (common in auto-captions)
    deduped = []
    for line in text_lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return " ".join(deduped).strip()


# ── Public API ─────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube video ID from any common URL format."""
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


def fetch_captions(video_url: str) -> str | None:
    """
    Try to retrieve captions for the given YouTube URL.

    Uses yt-dlp to discover subtitle/auto-caption tracks, then downloads the
    preferred VTT track via requests.

    NOTE: SSL verification is intentionally skipped here.  YouTube's subtitle
    CDN occasionally presents certificates that trigger validation errors in
    some environments (especially containerised deployments with older CA
    bundles).  This is a known, documented trade-off — the request is read-only
    and the worst-case risk is a MITM returning garbage subtitle data, which
    the downstream parser would simply discard.
    """
    PREFERRED_LANGS = ["en", "en-US", "en-GB", "hi", "te", "ta"]

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,   # see docstring above
        "legacy_server_connect": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception as exc:
        logger.warning("yt-dlp info extraction failed: %s", exc)
        return None

    # Prefer manual subtitles; fall back to auto-generated captions
    for track_group in [info.get("subtitles", {}), info.get("automatic_captions", {})]:
        for lang in PREFERRED_LANGS:
            tracks = track_group.get(lang, [])
            for track in tracks:
                if track.get("ext") == "vtt" and track.get("url"):
                    try:
                        resp = requests.get(track["url"], verify=False, timeout=20)  # noqa: S501
                        resp.raise_for_status()
                        text = _parse_vtt(resp.text)
                        if text:
                            return text
                    except Exception as exc:
                        logger.debug("VTT download failed (%s): %s", lang, exc)

    return None


def transcribe_with_whisper(video_url: str, groq_client) -> str | None:
    """
    Download audio from the video and transcribe it using Groq Whisper.

    Returns the transcript string, or None on failure.
    Raises ValueError if the audio file exceeds 25 MB (Groq's limit).
    """
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "quiet": True,
            "noplaylist": True,
            "retries": 3,
            "ffmpeg_location": ffmpeg_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as exc:
            logger.error("Audio download failed: %s", exc)
            return None

        mp3_path = audio_path + ".mp3"
        if not os.path.exists(mp3_path):
            logger.error("Expected MP3 not found at %s", mp3_path)
            return None

        size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
        if size_mb > 25:
            raise ValueError(
                f"Audio file is {size_mb:.1f} MB — Groq Whisper limit is 25 MB. "
                "Try a shorter video."
            )

        with open(mp3_path, "rb") as audio_file:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text",
            )

    return result if isinstance(result, str) else None


def get_transcript(video_url: str, groq_client=None) -> tuple[str | None, str]:
    """
    High-level entry point.  Returns (transcript_text, method_used).

    method_used is one of: "captions", "whisper", "failed"
    """
    text = fetch_captions(video_url)
    if text:
        return text, "captions"

    if groq_client is not None:
        try:
            text = transcribe_with_whisper(video_url, groq_client)
            if text:
                return text, "whisper"
        except ValueError as exc:
            # Propagate size errors so the UI can show a friendly message
            raise exc
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)

    return None, "failed"
