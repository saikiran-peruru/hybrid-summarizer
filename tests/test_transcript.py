from transcript import extract_text, extract_youtube_video_id


def test_extract_text_normalizes_whitespace():
    assert extract_text(" hello\nworld\t ") == "hello world"


def test_extract_youtube_video_id_supports_common_url_formats():
    video_id = "dQw4w9WgXcQ"

    assert extract_youtube_video_id(f"https://www.youtube.com/watch?v={video_id}") == video_id
    assert extract_youtube_video_id(f"https://youtu.be/{video_id}") == video_id
    assert extract_youtube_video_id(f"https://www.youtube.com/embed/{video_id}") == video_id
    assert extract_youtube_video_id(f"https://www.youtube.com/shorts/{video_id}") == video_id
