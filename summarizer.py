import os
import re
from dataclasses import dataclass

import nltk
import streamlit as st
import torch
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024

if not os.path.exists(os.path.expanduser("~/nltk_data/tokenizers/punkt")):
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


@dataclass(frozen=True)
class SummaryConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_summary_words: int = 140
    min_summary_words: int = 45
    extractive_sentences: int | None = None
    num_beams: int = 4


@st.cache_resource(show_spinner=False)
def load_bart(model_name=DEFAULT_MODEL_NAME):
    """Load BART model once, cached across all Streamlit reruns."""
    print(f"Loading summarization model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def clean_text(text):
    """Normalize whitespace while preserving sentence punctuation."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _add_sentence_boundaries(text):
    """
    Adds sentence boundaries to unpunctuated text (e.g. transcripts).
    Splits at natural pause points — conjunctions and comma boundaries —
    so TextRank gets coherent units to rank.
    """
    text = clean_text(text)
    words = text.split()
    sentence_endings = len(re.findall(r"[.!?]", text))

    if not words or sentence_endings / len(words) > 0.01:
        return text

    natural_breaks = re.sub(
        r"\s+\b(but|however|therefore|moreover|furthermore|additionally|"
        r"meanwhile|nevertheless|consequently|otherwise|finally)\b",
        r". \1",
        text,
        flags=re.IGNORECASE,
    )

    parts = natural_breaks.split(", ")
    result = []
    current = ""

    for part in parts:
        current += (", " if current else "") + part
        if len(current.split()) >= 20:
            result.append(current.strip().rstrip(",") + ".")
            current = ""

    if current:
        result.append(current.strip())

    final = " ".join(result)
    final = re.sub(r"\.{2,}", ".", final)
    final = re.sub(r"\.\s*,", ".", final)
    final = re.sub(r"\s+", " ", final)

    return final.strip()


def extractive_summarize(text, num_sentences):
    """Stage 1: TextRank extracts the most important sentences."""
    text = _add_sentence_boundaries(text)
    sentences = sent_tokenize(text)

    print(f"[TextRank] Sentences detected: {len(sentences)}, selecting top {num_sentences}")

    if len(sentences) <= num_sentences:
        return text

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    selected = summarizer(parser.document, min(num_sentences, len(parser.document.sentences)))
    result = " ".join(str(s) for s in selected)

    print(f"[TextRank] Output: {len(result.split())} words")
    return result


def _generation_lengths(token_count, config):
    estimated_words = max(1, int(token_count * 0.75))
    max_words = min(config.max_summary_words, max(config.min_summary_words, estimated_words // 3))
    min_words = min(config.min_summary_words, max(20, max_words // 2))
    max_tokens = max(32, int(max_words / 0.75))
    min_tokens = max(12, int(min_words / 0.75))
    return min_tokens, max_tokens


def _generate_summary(model, tokenizer, text, config):
    tokens = tokenizer.encode(text, truncation=False)
    min_len, max_len = _generation_lengths(len(tokens), config)
    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_INPUT_TOKENS, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=config.num_beams,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def abstractive_summarize(text, config=None):
    """Stage 2: BART generates a fluent abstractive summary."""
    config = config or SummaryConfig()
    model, tokenizer = load_bart(config.model_name)
    tokens = tokenizer.encode(text, truncation=False)

    if len(tokens) <= MAX_INPUT_TOKENS:
        return _generate_summary(model, tokenizer, text, config)

    # Chunk at sentence boundaries for long texts
    text = _add_sentence_boundaries(text)
    sentences = sent_tokenize(text)
    chunks, current = [], ""

    for sentence in sentences:
        candidate = current + " " + sentence
        if len(tokenizer.encode(candidate, truncation=False)) < MAX_INPUT_TOKENS:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())

    summaries = []
    for chunk in chunks:
        if len(chunk.split()) > 20:
            summaries.append(_generate_summary(model, tokenizer, chunk, config))

    return " ".join(summaries)


def evaluate_rouge(reference, generated):
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)

    result = {}
    for metric in ["rouge1", "rouge2", "rougeL"]:
        s = scores[metric]
        result[metric] = {
            "precision": round(s.precision, 3),
            "recall": round(s.recall, 3),
            "f1": round(s.fmeasure, 3),
        }
    return result


def _auto_num_sentences(text):
    """Automatically determine how many sentences to extract based on text length."""
    text = _add_sentence_boundaries(text)
    sentences = sent_tokenize(text)
    n = len(sentences)
    words = len(text.split())

    if n == 0:
        return 10
    if words < 500:
        return max(2, min(n - 1, round(n * 0.6))) if n > 2 else n
    if words < 2000:
        return min(max(5, round(n * 0.45)), n)
    if words < 10000:
        return min(max(8, round(n * 0.35)), n)

    avg_wps = words / n
    target = int(600 / max(avg_wps, 1))
    return max(15, min(40, min(target, n)))


def summarize(text, config=None):
    """Main entry point — runs the two-stage hybrid pipeline."""
    config = config or SummaryConfig()
    text = clean_text(text)

    if not text or len(text.strip()) < 50:
        return {
            "extractive": "",
            "final": "Text is too short to summarize.",
            "original_words": 0,
            "extractive_words": 0,
            "final_words": 0,
            "compression": 0,
        }

    original_words = len(text.split())
    num_sentences = config.extractive_sentences or _auto_num_sentences(text)

    print(f"[Pipeline] Original: {original_words} words, extracting {num_sentences} sentences")

    extractive = extractive_summarize(text, num_sentences)
    extractive_words = len(extractive.split())

    print(f"[Pipeline] Extractive: {extractive_words} words, running BART...")

    final = abstractive_summarize(extractive, config)
    final_words = len(final.split())

    compression = max(0, round((1 - final_words / original_words) * 100, 1))

    print(f"[Pipeline] Final: {final_words} words, compression: {compression}%")

    return {
        "extractive": extractive,
        "final": final,
        "original_words": original_words,
        "extractive_words": extractive_words,
        "final_words": final_words,
        "compression": compression,
        "model": config.model_name,
        "extractive_sentences": num_sentences,
    }
