"""Check source fidelity and failures that could mislead the RAG exercise."""

import pytest

from reimplementation.retrieval import TfidfIndex, build_prompt, chunk_document


def test_chunks_keep_original_text_and_cover_document_without_trailing_duplicate():
    text = "zéro un\n  deux trois quatre\ncinq six"
    passages = chunk_document("notes.md", text, size=4, overlap=1)
    assert [(p.start_word, p.end_word) for p in passages] == [(0, 4), (3, 7)]
    assert passages[0].text == "zéro un\n  deux trois"
    assert passages[1].text == "trois quatre\ncinq six"
    assert all(p.source == "notes.md" for p in passages)
    assert chunk_document("empty.md", "  ") == []


def test_retrieval_preserves_source_through_ranking_and_prompt():
    passages = (chunk_document("rope.md", "RoPE encode les positions par rotation.")
                + chunk_document("gqa.md", "GQA partage les projections entre groupes de têtes."))
    hits = TfidfIndex(passages).search("GROUPES de TETES", k=1)
    assert len(hits) == 1 and hits[0][1].source == "gqa.md"
    assert 0 < hits[0][0] <= 1
    prompt = build_prompt("GROUPES de TETES", hits)
    assert "[P1] gqa.md" in prompt
    assert passages[1].text in prompt
    assert "rope.md" not in prompt


def test_unknown_or_empty_query_never_returns_arbitrary_passages():
    index = TfidfIndex(chunk_document("notes.md", "GQA partage les projections."))
    for question in ("", "le la de", "ornithorynque"):
        assert index.search(question) == []
    assert TfidfIndex([]).search("GQA") == []
    assert "Aucun passage retrouvé." in build_prompt("ornithorynque", [])


def test_cosine_is_invariant_to_uniform_document_repetition():
    passages = (chunk_document("short.md", "gqa projections")
                + chunk_document("long.md", "gqa projections gqa projections"))
    hits = TfidfIndex(passages).search("gqa projections", k=2)
    assert [score for score, _ in hits] == pytest.approx([1.0, 1.0])


def test_invalid_chunk_parameters_are_rejected():
    with pytest.raises(ValueError):
        chunk_document("notes.md", "GQA", size=4, overlap=4)
    with pytest.raises(ValueError):
        TfidfIndex([]).search("GQA", k=0)
