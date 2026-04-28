import re
import os
import tempfile
from urllib.parse import parse_qs, urlparse

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
import whisper
from pydub import AudioSegment


def extract_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pdf_text(file):
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return extract_text(" ".join(pages))


def extract_youtube_video_id(url):
    parsed = urlparse(url.strip())

    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate if re.fullmatch(r"[a-zA-Z0-9_-]{11}", candidate or "") else None

    if "youtube.com" in parsed.netloc:
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id and re.fullmatch(r"[a-zA-Z0-9_-]{11}", query_id):
            return query_id

        path_parts = [part for part in parsed.path.split("/") if part]
        for marker in ("embed", "shorts", "live"):
            if marker in path_parts:
                idx = path_parts.index(marker)
                if idx + 1 < len(path_parts):
                    candidate = path_parts[idx + 1]
                    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", candidate):
                        return candidate

    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def extract_youtube_transcript(url):
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL.")

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        return extract_text(" ".join(entry.text for entry in transcript))
    except Exception:
        raise ValueError(
            "Could not fetch transcript. This video may not have captions enabled."
        )


@st.cache_resource(show_spinner=False)
def load_whisper():
    """Load Whisper model once, cached across all Streamlit reruns."""
    print("Loading Whisper model...")
    return whisper.load_model("base")


def extract_audio_text(file):
    suffix = os.path.splitext(file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.seek(0)
        tmp.write(file.read())
        tmp_path = tmp.name

    wav_path = tmp_path
    try:
        if suffix.lower() != ".wav":
            audio = AudioSegment.from_file(tmp_path)
            wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
            audio.export(wav_path, format="wav")

        model = load_whisper()
        result = model.transcribe(wav_path)
        return extract_text(result["text"])
    finally:
        os.unlink(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            os.unlink(wav_path)


def get_text(source_type, source):
    if source_type == "text":
        return extract_text(source)
    elif source_type == "pdf":
        return extract_pdf_text(source)
    elif source_type == "youtube":
        return extract_youtube_transcript(source)
    elif source_type == "audio":
        return extract_audio_text(source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")
