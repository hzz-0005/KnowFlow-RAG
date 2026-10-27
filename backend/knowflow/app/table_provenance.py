"""Verify saved Docling table serialization before assigning logical row/cell ranges.

Ranges address the transformed Block.text, never PDF character offsets. Cell
identities address the saved parser grid; PDF highlights remain table-level.
"""

from knowflow.app.document_processing import _clean

REVISION = "docling-rendered-row-v1"


def table_row_sources(block, row_texts):
    from knowflow.app.structured_parsing import _table_rows

    s = block.structure
    if not {"table_cells", "num_rows", "num_cols"} <= s.keys():
        return None
    pages = {span.get("page_number") for span in block.source_spans}
    if len(block.source_spans) != 1 or len(pages) != 1 or None in pages:
        return None  # No evidence assigning a multi-page table's cells to pages.
    try:
        texts, structure = _table_rows({"data": s})
        caption = "\n".join(filter(None, s.get("captions", [])))
        rendered = [f"{caption}\n{row}".strip() for row in texts]
        if texts != row_texts or "\n".join(rendered) != block.text:
            return None
        rows, cols = structure["num_rows"], structure["num_cols"]
        grid = [[None] * cols for _ in range(rows)]
        cells = s["table_cells"]
        for index, cell in enumerate(cells):
            for r in range(cell["start_row_offset_idx"], cell["end_row_offset_idx"]):
                for c in range(cell["start_col_offset_idx"], cell["end_col_offset_idx"]):
                    if grid[r][c] is not None:
                        return None  # Overlapping parser cells are ambiguous.
                    grid[r][c] = index
        headers = set(structure["header_rows"])

        def value(index):
            return _clean(cells[index].get("text", "")) if index is not None else ""

        body = [r for r in range(rows) if r not in headers and any(map(value, grid[r]))]
        fallback = not body
        if fallback:
            body = [r for r in range(rows) if any(map(value, grid[r]))]
        result, cursor = [], 0
        for r, text, full in zip(body, texts, rendered, strict=True):
            start = cursor + (len(caption) + 1 if caption else 0)
            cursor += len(full) + 1
            pieces, mappings, local = [], [], 0
            for c, index in enumerate(grid[r]):
                val = value(index)
                if not fallback and not val:
                    continue
                if pieces:
                    local += 3  # " | "
                header_groups = {}
                for hr in sorted(headers):
                    hi = grid[hr][c]
                    if htext := value(hi):
                        header_groups.setdefault(htext, []).append(hi)
                header = " / ".join(header_groups) or f"列{c + 1}"
                piece = val if fallback else f"{header}: {val}"
                if not fallback:
                    pos = local
                    for htext, indices in header_groups.items():
                        mappings.append(
                            _mapping(cells, indices, "header", c, start + pos, len(htext))
                        )
                        pos += len(htext) + 3  # " / "
                if val:
                    pos = local + (0 if fallback else len(header) + 2)
                    mappings.append(_mapping(cells, [index], "value", c, start + pos, len(val)))
                pieces.append(piece)
                local += len(piece)
            if " | ".join(pieces) != text:
                return None
            result.append(
                {
                    "text_range": [start, start + len(text)],
                    "row_number": r + 1,
                    "source_mapping": REVISION,
                    "table_cells": mappings,
                }
            )
        return result
    except (KeyError, TypeError, ValueError, IndexError):
        return None  # Legacy/damaged structures remain inspectable but unmapped.


def _mapping(cells, indices, role, column, start, length):
    return {
        "role": role,
        "column_number": column + 1,
        "text_range": [start, start + length],
        "source_cells": [
            {
                "index": i,
                "rows": [cells[i]["start_row_offset_idx"] + 1, cells[i]["end_row_offset_idx"]],
                "columns": [cells[i]["start_col_offset_idx"] + 1, cells[i]["end_col_offset_idx"]],
            }
            for i in dict.fromkeys(indices)
        ],
    }
