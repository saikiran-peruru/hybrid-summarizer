import html
import json

import streamlit as st
from summarizer import DEFAULT_MODEL_NAME, SummaryConfig, summarize, evaluate_rouge
from transcript import get_text

SAMPLE_TEXT = """Artificial intelligence (AI) has transformed the way businesses operate across nearly every industry. In healthcare, AI algorithms can analyze medical images with accuracy that matches or exceeds human radiologists, enabling earlier detection of diseases like cancer. In finance, machine learning models process vast amounts of transaction data to detect fraudulent activity in real time, saving institutions billions of dollars annually.

The rise of large language models (LLMs) has been particularly impactful. Models like GPT-4, Claude, and Gemini can understand and generate human-like text, enabling applications ranging from customer service chatbots to code generation assistants. These models are trained on massive datasets containing billions of parameters, requiring significant computational resources.

However, the rapid advancement of AI also raises important concerns. Bias in training data can lead to discriminatory outputs, particularly in sensitive areas like hiring and criminal justice. The environmental impact of training large models is substantial, with some estimates suggesting that training a single large language model can produce as much carbon dioxide as five cars over their lifetimes.

Privacy is another major concern. AI systems often require large amounts of personal data to function effectively, raising questions about data ownership and consent. Regulations like the European Union's AI Act and GDPR aim to address these concerns, but the regulatory landscape remains fragmented globally.

Despite these challenges, the potential benefits of AI are enormous. Autonomous vehicles could reduce traffic accidents by up to 90 percent. AI-powered drug discovery is accelerating the development of new medications, potentially saving years of research time. In education, personalized learning systems adapt to individual student needs, improving outcomes for learners of all backgrounds.

The future of AI will likely be shaped by the development of more efficient models that require less data and computing power, advances in explainable AI that make model decisions transparent, and stronger governance frameworks that ensure AI is developed and deployed responsibly. Organizations that successfully integrate AI into their operations while addressing ethical concerns will have a significant competitive advantage in the years ahead."""

st.set_page_config(
    page_title="Hybrid Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Hybrid Text Summarizer")
st.caption("Two-stage pipeline: TextRank (extractive) → BART (abstractive)")

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "Summarization model",
        [DEFAULT_MODEL_NAME, "sshleifer/distilbart-cnn-12-6"],
        index=0,
        help="DistilBART is faster; BART-large is higher quality but heavier.",
    )
    max_summary_words = st.slider("Target summary length", 60, 220, 140, step=20)
    extractive_mode = st.radio("TextRank sentence count", ["Auto", "Manual"], horizontal=True)
    extractive_sentences = None
    if extractive_mode == "Manual":
        extractive_sentences = st.slider("Sentences to pass to BART", 3, 40, 15)

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("Text extraction → TextRank filtering → BART generation → optional ROUGE scoring")

if "result" not in st.session_state:
    st.session_state.result = None
if "source_text" not in st.session_state:
    st.session_state.source_text = None

source_type = st.segmented_control(
    "Input source",
    ["Text", "PDF", "YouTube", "Audio"],
    default="Text",
)

source_type = source_type.lower()
text_input = pdf_file = youtube_url = audio_file = None

if source_type == "text":
    col_sample, col_clear = st.columns([1, 1])
    if col_sample.button("Load Sample Text", use_container_width=True):
        st.session_state.text_value = SAMPLE_TEXT
    if col_clear.button("Clear", use_container_width=True):
        st.session_state.text_value = ""

    text_input = st.text_area(
        "Paste your text here",
        value=st.session_state.get("text_value", ""),
        height=260,
        placeholder="Enter or paste the text you want to summarize...",
    )
    st.session_state.text_value = text_input

elif source_type == "pdf":
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

elif source_type == "youtube":
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

elif source_type == "audio":
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
    )
    st.caption("Supports MP3, WAV, M4A, OGG, FLAC. Requires FFmpeg for non-WAV files.")

config = SummaryConfig(
    model_name=model_name,
    max_summary_words=max_summary_words,
    min_summary_words=max(30, max_summary_words // 3),
    extractive_sentences=extractive_sentences,
)

if st.button("Generate Summary", type="primary", use_container_width=True):
    source = {
        "text": text_input,
        "pdf": pdf_file,
        "youtube": youtube_url,
        "audio": audio_file,
    }[source_type]

    if not source:
        st.warning("Please provide an input before generating a summary.")
    else:
        try:
            with st.spinner("Extracting text..."):
                text = get_text(source_type, source)

            if not text or len(text.strip()) < 50:
                st.error("Could not extract enough text. Please try a different input.")
            else:
                with st.spinner("Running TextRank → BART..."):
                    result = summarize(text, config)

                st.session_state.result = result
                st.session_state.source_text = text

        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.session_state.result:
    result = st.session_state.result

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original", f"{result['original_words']} words")
    col2.metric("After TextRank", f"{result['extractive_words']} words")
    col3.metric("Final Summary", f"{result['final_words']} words")
    col4.metric("Compression", f"{result['compression']}%")
    st.caption(
        f"Model: `{result.get('model', model_name)}` | "
        f"TextRank sentences: `{result.get('extractive_sentences', 'auto')}`"
    )

    st.divider()
    st.subheader("Stage Comparison")

    col_ext, col_abs = st.columns(2)

    with col_ext:
        st.markdown("**Stage 1: TextRank (Extractive)**")
        st.markdown(
            f'<div style="background-color: #1e3a5f; padding: 16px; '
            f'border-radius: 8px; min-height: 200px; font-size: 14px; '
            f'line-height: 1.6;">{html.escape(result["extractive"])}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Important sentences — accurate but choppy")

    with col_abs:
        st.markdown("**Stage 2: BART (Abstractive)**")
        st.markdown(
            f'<div style="background-color: #1a3d2e; padding: 16px; '
            f'border-radius: 8px; min-height: 200px; font-size: 14px; '
            f'line-height: 1.6;">{html.escape(result["final"])}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Fluent generated summary")

    export_text = (
        "# Hybrid Summary\n\n"
        f"{result['final']}\n\n"
        "## Extractive Stage\n\n"
        f"{result['extractive']}\n\n"
        "## Metrics\n\n"
        f"- Original words: {result['original_words']}\n"
        f"- Extractive words: {result['extractive_words']}\n"
        f"- Final words: {result['final_words']}\n"
        f"- Compression: {result['compression']}%\n"
        f"- Model: {result.get('model', model_name)}\n"
    )
    export_json = json.dumps(result, indent=2)

    d1, d2 = st.columns(2)
    d1.download_button("Download Markdown", export_text, "summary.md", use_container_width=True)
    d2.download_button("Download JSON", export_json, "summary.json", use_container_width=True)

    with st.expander("View Original Extracted Text"):
        source_text = st.session_state.source_text
        st.text(source_text[:5000] + ("..." if len(source_text) > 5000 else ""))

    st.divider()
    st.subheader("Evaluate with ROUGE Scores")
    st.caption("Paste a reference summary to measure quality against the generated one.")

    reference_summary = st.text_area(
        "Reference summary (optional)",
        height=100,
        placeholder="Paste a human-written summary here..."
    )

    if reference_summary and st.button("Calculate ROUGE Scores"):
        with st.spinner("Calculating..."):
            rouge_scores = evaluate_rouge(reference_summary, result["final"])

        st.markdown("**ROUGE Scores (F1)**")
        r1, r2, rl = st.columns(3)

        with r1:
            score = rouge_scores["rouge1"]["f1"]
            st.metric("ROUGE-1", f"{score:.3f}", help="Word-level overlap. Good: > 0.35")
            st.progress(min(score, 1.0))

        with r2:
            score = rouge_scores["rouge2"]["f1"]
            st.metric("ROUGE-2", f"{score:.3f}", help="Phrase-level overlap. Good: > 0.15")
            st.progress(min(score, 1.0))

        with rl:
            score = rouge_scores["rougeL"]["f1"]
            st.metric("ROUGE-L", f"{score:.3f}", help="Sentence structure overlap. Good: > 0.30")
            st.progress(min(score, 1.0))

        with st.expander("Detailed Scores (Precision / Recall / F1)"):
            for metric_name, scores in rouge_scores.items():
                st.markdown(f"**{metric_name.upper()}**")
                d1, d2, d3 = st.columns(3)
                d1.metric("Precision", f"{scores['precision']:.3f}")
                d2.metric("Recall", f"{scores['recall']:.3f}")
                d3.metric("F1", f"{scores['f1']:.3f}")
