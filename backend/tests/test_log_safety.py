"""Regression coverage for request-derived values written to server logs."""

from backend import errors


def test_log_text_escapes_line_breaks_and_controls():
    assert errors.safe_log_text("/api/a\r\nforged\tentry\x00") == (
        r"/api/a\r\nforged\tentry\x00"
    )


def test_log_text_is_bounded_after_escaping():
    assert errors.safe_log_text("\n" * 1000, limit=41) == (r"\n" * 21)[:41]
