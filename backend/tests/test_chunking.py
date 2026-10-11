import pytest
import tiktoken

from app.chunking import chunk_pages


def test_chunks_preserve_page_and_detect_section_headings() -> None:
    chunks = chunk_pages(
        [
            (
                3,
                "Introduction\n\nOpsMind stores searchable document text.\n\n"
                "2. Access Rules\n\nOnly approved users can upload files.",
            )
        ],
        max_tokens=100,
    )

    assert len(chunks) == 2
    assert chunks[0].page_number == 3
    assert chunks[0].section_title == "Introduction"
    assert "Introduction" in chunks[0].content
    assert chunks[1].section_title == "2. Access Rules"
    assert "Only approved users" in chunks[1].content


def test_chunks_never_exceed_configured_token_limit() -> None:
    text = "Policy details apply to every team. " * 120
    chunks = chunk_pages([(1, text)], max_tokens=80)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk.content)) <= 80 for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_oversized_block_splits_with_configured_token_overlap() -> None:
    text = "0123456789" * 160
    chunks = chunk_pages([(1, text)], max_tokens=800, overlap_tokens=75)

    assert [len(chunk.content) for chunk in chunks] == [800, 800, 150]
    assert chunks[0].content[-75:] == chunks[1].content[:75]
    assert chunks[1].content[-75:] == chunks[2].content[:75]


def test_overlap_must_be_less_than_max_tokens() -> None:
    with pytest.raises(ValueError, match="less than max_tokens"):
        chunk_pages([(1, "text")], max_tokens=10, overlap_tokens=10)


def test_chunks_do_not_combine_pages() -> None:
    chunks = chunk_pages([(1, "First page text."), (2, "Second page text.")])

    assert [chunk.page_number for chunk in chunks] == [1, 2]
