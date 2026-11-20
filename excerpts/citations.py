# Excerpt from PolicyLens (private repo): apps/backend/rag/citations.py — published unchanged.
"""Citation extraction and verification.

Parses [Source: <doc>, p.<N>] markers from LLM output and matches them
back to the context chunks that were actually retrieved. Citations that
don't match any retrieved chunk are flagged as unverified.
"""

import re
from dataclasses import dataclass

# Matches: [Source: Some Document Name.pdf, p.12]
_CITATION_PATTERN = re.compile(
    r"\[Source:\s*(?P<doc>[^,\]]+?)\s*,\s*p\.(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)


@dataclass
class Citation:
    document_name: str
    page_number: int
    section_heading: str | None
    citation_type: str  # "direct" | "unverified"
    document_id: str | None = None


def extract_citations(
    answer_text: str,
    context_blocks: list[dict],
) -> list[Citation]:
    """Parse citation markers from answer_text and cross-reference with context_blocks.

    Args:
        answer_text: The LLM-generated answer string.
        context_blocks: The chunks that were passed to the LLM. Each dict has:
            - document_name: str
            - page_number: int
            - section_heading: str | None

    Returns:
        Deduplicated list of Citation objects. Citations not found in context_blocks
        are included but marked citation_type="unverified".
    """
    # Build a lookup set of (normalized_doc_name, page_number) from retrieved context
    retrieved: dict[tuple[str, int], dict] = {}
    for block in context_blocks:
        key = (_normalize(block["document_name"]), int(block["page_number"]))
        retrieved[key] = block

    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []

    for match in _CITATION_PATTERN.finditer(answer_text):
        doc_name = match.group("doc").strip()
        page_number = int(match.group("page"))
        dedup_key = (_normalize(doc_name), page_number)

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        block = retrieved.get(dedup_key)
        if block:
            citations.append(Citation(
                document_name=block["document_name"],  # use canonical name from retrieval
                page_number=page_number,
                section_heading=block.get("section_heading"),
                citation_type="direct",
                document_id=block.get("document_id"),
            ))
        else:
            # LLM cited something not in the retrieved context — surface it but flag it
            citations.append(Citation(
                document_name=doc_name,
                page_number=page_number,
                section_heading=None,
                citation_type="unverified",
            ))

    return citations


def _normalize(name: str) -> str:
    """Normalize a document name for fuzzy deduplication (lowercase, strip whitespace)."""
    return name.strip().lower()
