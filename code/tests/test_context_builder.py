# code/tests/test_context_builder.py
import pytest
from context_builder.compressor import Compressor


def test_short_content_returned_unchanged():
    c = Compressor(max_chars=100)
    summary, ref_id, char_count = c.compress("short")
    assert summary == "short"
    assert ref_id == ""
    assert char_count == 5


def test_long_content_is_truncated():
    c = Compressor(max_chars=10)
    summary, ref_id, char_count = c.compress("x" * 50)
    assert summary.startswith("x" * 10)
    assert "截断" in summary
    assert ref_id.startswith("ref:")
    assert char_count == 50


def test_ref_id_is_deterministic_for_same_content():
    content = "hello world this is long enough"
    c1 = Compressor(max_chars=5)
    c2 = Compressor(max_chars=5)
    _, ref1, _ = c1.compress(content)
    _, ref2, _ = c2.compress(content)
    assert ref1 == ref2


def test_retrieve_returns_original():
    c = Compressor(max_chars=5)
    original = "hello world this is a long string"
    _, ref_id, _ = c.compress(original)
    assert c.retrieve(ref_id) == original


def test_retrieve_missing_returns_none():
    c = Compressor()
    assert c.retrieve("ref:nonexistent") is None


def test_is_compressed_true_for_long():
    c = Compressor(max_chars=5)
    _, ref_id, _ = c.compress("hello world")
    assert c.is_compressed(ref_id) is True


def test_is_compressed_false_for_short():
    c = Compressor(max_chars=100)
    _, ref_id, _ = c.compress("short")
    assert c.is_compressed(ref_id) is False
