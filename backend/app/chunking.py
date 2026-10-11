import re
from dataclasses import dataclass
from typing import Iterable, Optional

import tiktoken


DEFAULT_MAX_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 75
_HEADING_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+\S")
_SMALL_TITLE_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


@dataclass(frozen=True)
class TextChunk:
    content: str
    page_number: int
    section_title: Optional[str]


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100 or len(line.split()) > 12:
        return False
    if line[-1:] in ".?!;,:":
        return False
    if _HEADING_NUMBER.match(line):
        return True
    letters = [character for character in line if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return True
    words = re.findall(r"[\w’'-]+", line)
    if not words:
        return False
    title_case_words = sum(
        word[0].isupper() or word.casefold() in _SMALL_TITLE_WORDS for word in words
    )
    return len(words) <= 8 and title_case_words >= max(1, len(words) - 2)


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and less than max_tokens")

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[TextChunk] = []

    for page_number, page_text in pages:
        section_title: Optional[str] = None
        pending_heading: Optional[str] = None
        paragraph_lines: list[str] = []
        current_blocks: list[str] = []

        def flush_chunk() -> None:
            nonlocal current_blocks
            if current_blocks:
                chunks.append(
                    TextChunk(
                        content="\n\n".join(current_blocks),
                        page_number=page_number,
                        section_title=section_title,
                    )
                )
                current_blocks = []

        def add_block(block: str) -> None:
            token_ids = encoding.encode(block)
            if len(token_ids) > max_tokens:
                flush_chunk()
                start = 0
                while start < len(token_ids):
                    end = min(start + max_tokens, len(token_ids))
                    part = encoding.decode(token_ids[start:end])
                    while len(encoding.encode(part)) > max_tokens:
                        end -= 1
                        part = encoding.decode(token_ids[start:end])
                    chunks.append(
                        TextChunk(
                            content=part,
                            page_number=page_number,
                            section_title=section_title,
                        )
                    )
                    if end == len(token_ids):
                        break
                    start += max_tokens - overlap_tokens
                return

            candidate = "\n\n".join([*current_blocks, block])
            if current_blocks and len(encoding.encode(candidate)) > max_tokens:
                flush_chunk()
            current_blocks.append(block)

        def flush_paragraph() -> None:
            nonlocal paragraph_lines, pending_heading
            paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
            paragraph_lines = []
            if pending_heading:
                paragraph = f"{pending_heading}\n{paragraph}" if paragraph else pending_heading
                pending_heading = None
            if paragraph:
                add_block(paragraph)

        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                continue
            if _looks_like_heading(line):
                flush_paragraph()
                flush_chunk()
                if pending_heading:
                    add_block(pending_heading)
                section_title = line
                pending_heading = line
                continue
            paragraph_lines.append(line)

        flush_paragraph()
        if pending_heading:
            add_block(pending_heading)
        flush_chunk()

    return chunks
