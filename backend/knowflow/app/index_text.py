"""The exact string budgeted and sent to the document embedding API."""


def chunk_index_text(section: str | None, content: str) -> str:
    if section and section.casefold() not in content.casefold():
        return f"{section}\n{content}"
    return content
