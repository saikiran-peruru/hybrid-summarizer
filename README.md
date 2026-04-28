# Hybrid Text Summarizer

A Streamlit app that summarizes long-form content with a two-stage NLP pipeline:

1. **TextRank** extracts the most important source sentences.
2. **BART** rewrites those sentences into a fluent abstractive summary.
3. **ROUGE** optionally evaluates the generated summary against a reference summary.

The app supports pasted text, PDFs, YouTube transcripts, and audio files.

## Why this project matters

Most summarization demos send the entire document directly to a transformer model. This project uses a hybrid approach instead: TextRank reduces noisy or very long inputs before the abstractive model runs. That makes the workflow more explainable, easier to compare, and more practical for documents, lectures, and videos.

## Features

- Text, PDF, YouTube, and audio input modes
- Configurable BART model and target summary length
- Automatic or manual TextRank sentence selection
- Side-by-side extractive and abstractive stage comparison
- Compression metrics for original, intermediate, and final summaries
- Markdown and JSON export
- Optional ROUGE-1, ROUGE-2, and ROUGE-L scoring
- Cached model loading for faster Streamlit reruns
- Unit tests for parsing and helper logic

## Tech Stack

- [Streamlit](https://streamlit.io/) for the UI
- [Sumy TextRank](https://github.com/miso-belica/sumy) for extractive summarization
- [BART CNN](https://huggingface.co/facebook/bart-large-cnn) for abstractive summarization
- [Whisper](https://github.com/openai/whisper) for audio transcription
- [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF text extraction
- [ROUGE Score](https://pypi.org/project/rouge-score/) for evaluation

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
streamlit run app.py
```

The first run downloads the selected Hugging Face summarization model. Audio transcription also downloads the Whisper base model the first time it is used.

For non-WAV audio files, install FFmpeg:

```bash
brew install ffmpeg
```

## Usage

1. Pick an input source.
2. Paste text, upload a file, or enter a YouTube URL.
3. Adjust the model, target length, or TextRank sentence count in the sidebar.
4. Generate the summary.
5. Compare the extractive and abstractive stages, export the result, or evaluate it with ROUGE.

## Testing

```bash
pytest
python -m py_compile app.py summarizer.py transcript.py
```

The tests intentionally avoid loading BART or Whisper so they can run quickly in CI.

## Resume Highlights

- Built a full-stack NLP application with Streamlit, Hugging Face Transformers, TextRank, and Whisper.
- Designed a hybrid extractive-abstractive pipeline to improve long-document summarization quality and explainability.
- Added evaluation metrics, configurable inference parameters, export formats, and automated tests.
- Supported multiple real-world content sources: text, PDFs, videos, and audio.

## Future Improvements

- Add a benchmark notebook comparing direct BART vs. hybrid TextRank + BART.
- Add Docker support for one-command deployment.
- Add optional GPU detection and model-size presets.
- Store summary history locally for comparing multiple runs.
