# Excerpt from PolicyLens (private repo): apps/backend/rag/pipeline.py, the retrieval-to-answer path
# (function body from line 233; the signature and query rewrite helpers are elided).
# One example acronym in a comment was changed; code unchanged.

    from qdrant_client import AsyncQdrantClient

    embeddings = get_embeddings()
    llm = get_llm()

    # Rewrite query using conversation context for better semantic recall
    embed_query = await rewrite_query(query, llm, history=conversation_history)

    # 1. Embed the rewritten query
    query_vector = await embeddings.embed(embed_query)

    collection_name = f"{tenant_id}_documents"
    threshold = settings.rag_confidence_threshold
    qdrant = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key or None,
    )

    try:
        # 2a. Semantic search — nearest vectors
        semantic_hits: list[dict] = []
        try:
            result = await qdrant.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=settings.retrieval_top_k,
                with_payload=True,
            )
            semantic_hits = [
                {"id": str(h.id), "score": h.score, "payload": h.payload or {}}
                for h in result.points
                if h.score >= threshold
            ]
        except Exception:
            pass  # collection doesn't exist yet

        # 2b. Keyword search — literal token match on the ORIGINAL query
        # (not the rewritten one, so "PTO" is searched as "PTO")
        keyword_hits = await _keyword_search(
            qdrant,
            collection_name,
            query,
            limit=settings.retrieval_top_k,
            baseline_score=threshold + 0.05,  # just above threshold
        )

        # 3. Merge: semantic scores take precedence; add keyword-only hits
        seen_ids = {h["id"] for h in semantic_hits}
        merged = list(semantic_hits)
        for kh in keyword_hits:
            if kh["id"] not in seen_ids:
                merged.append(kh)
                seen_ids.add(kh["id"])

        # Sort by score, cap at top-K
        merged.sort(key=lambda x: x["score"], reverse=True)
        merged = merged[: settings.retrieval_top_k]

    finally:
        await qdrant.close()

    if not merged:
        return QueryResult(
            answer=(
                "I couldn't find any relevant information in the available policy documents "
                "to answer your question. Please verify that the relevant documents have been "
                "uploaded and indexed."
            ),
            citations=[],
            confidence=0.0,
            query=query,
            answer_type="no_documents",
        )

    # 4. Build context blocks.
    #    Use parent_text (full section) for LLM generation when available —
    #    this preserves conditional logic and multi-paragraph structure that
    #    gets lost when the LLM only sees a small child chunk.
    #    Deduplicate by parent_text so the same section isn't sent twice when
    #    multiple child chunks from the same section are retrieved.
    seen_parent_texts: set[str] = set()
    context_blocks = []
    for h in merged:
        payload = h["payload"]
        parent_text = payload.get("parent_text")
        generation_text = parent_text or payload.get("text", "")

        # Deduplicate on generation_text — skip if we've already included
        # this section's content (multiple child hits from the same section).
        if generation_text in seen_parent_texts:
            continue
        seen_parent_texts.add(generation_text)

        context_blocks.append({
            "text": generation_text,
            "document_name": payload.get("document_name", "Unknown Document"),
            "page_number": int(payload.get("page_number", 0)),
            "section_heading": payload.get("section_heading"),
            "score": h["score"],
            "document_id": payload.get("document_id"),
        })


    # 5. Build prompts
    user_prompt = build_user_prompt(query, context_blocks, history=conversation_history)

    # 6. Call LLM
    llm_response = await llm.complete(
        system=SYSTEM_PROMPT,
        user=user_prompt,
    )

    # 7. Extract and verify citations
    citations = extract_citations(llm_response.content, context_blocks)

    # 8. Confidence: mean of retrieval scores, capped at 1.0
    confidence = min(sum(b["score"] for b in context_blocks) / len(context_blocks), 1.0)
    confidence = round(confidence, 4)

    # 9. Classify answer type
    #    - "unsupported":    LLM found context but couldn't produce citations
    #    - "low_confidence": retrieval scores are borderline (mean < 0.50)
    #    - "answered":       normal response with adequate confidence
    _LOW_CONFIDENCE_FLOOR = 0.50
    if not citations and context_blocks:
        answer_type = "unsupported"
    elif confidence < _LOW_CONFIDENCE_FLOOR:
        answer_type = "low_confidence"
    else:
        answer_type = "answered"

    return QueryResult(
        answer=llm_response.content,
        citations=citations,
        confidence=confidence,
        query=query,
        answer_type=answer_type,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
    )
