"""Tests for the chunker.

The chunker is the most logic-heavy piece of the pipeline that we can
test without hitting any external services, so it gets focused unit
tests here.
"""

from __future__ import annotations

import pytest

from src.chunking import chunk_text, count_tokens


SAMPLE_SOURCE = "sample.pdf"


def test_empty_input_returns_no_chunks() -> None:
    assert chunk_text("", source=SAMPLE_SOURCE) == []
    assert chunk_text("   \n\t  ", source=SAMPLE_SOURCE) == []


def test_short_input_produces_single_chunk() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunk_text(text, source=SAMPLE_SOURCE, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.index == 1
    assert c.source == SAMPLE_SOURCE
    assert c.text.strip() == text
    assert c.n_tokens == count_tokens(text)


def test_long_input_is_split_into_multiple_chunks() -> None:
    # 3000 short words => well over 500 tokens => multiple chunks.
    text = " ".join(["word"] * 3000)
    chunks = chunk_text(text, source=SAMPLE_SOURCE, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    # Indices are 1-based and contiguous.
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
    # Every chunk fits within the configured size.
    assert all(c.n_tokens <= 500 for c in chunks)
    # Every chunk is from the same source.
    assert all(c.source == SAMPLE_SOURCE for c in chunks)


def test_chunks_overlap_by_configured_amount() -> None:
    text = " ".join(f"token{i}" for i in range(2000))
    chunks = chunk_text(text, source=SAMPLE_SOURCE, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    # Sanity check overlap: the tail of chunk N should share at least one
    # word with the head of chunk N+1.
    for a, b in zip(chunks, chunks[1:]):
        tail = set(a.text.split()[-30:])
        head = set(b.text.split()[:30])
        assert tail & head, "expected overlap between consecutive chunks"


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", source=SAMPLE_SOURCE, chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("hello", source=SAMPLE_SOURCE, chunk_size=100, overlap=-1)
    with pytest.raises(ValueError):
        # overlap >= chunk_size would cause infinite loop, so we reject it.
        chunk_text("hello", source=SAMPLE_SOURCE, chunk_size=100, overlap=100)


def test_count_tokens_is_nonzero_for_words() -> None:
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0
