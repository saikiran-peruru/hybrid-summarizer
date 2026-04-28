from summarizer import SummaryConfig, _generation_lengths, clean_text


def test_clean_text_normalizes_whitespace():
    assert clean_text("  alpha\n\n beta\tgamma  ") == "alpha beta gamma"


def test_generation_lengths_respect_config_bounds():
    config = SummaryConfig(max_summary_words=120, min_summary_words=40)

    min_tokens, max_tokens = _generation_lengths(900, config)

    assert min_tokens < max_tokens
    assert max_tokens <= 160
    assert min_tokens >= 53
