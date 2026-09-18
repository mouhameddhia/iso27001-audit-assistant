"""BM25 lexical retriever: tokenizer and ranking, independent from Qdrant/semantic retrieval."""

import pytest

from src.retrieval.bm25 import BM25Retriever, tokenize
from tests.conftest import make_chunk


class TestTokenizer:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A.5.18", ["a.5.18"]),
            ("CLD.6.3.1", ["cld.6.3.1"]),
            ("Clause 9.2", ["clause", "9.2"]),
            ("ISO/IEC 27001:2022", ["iso", "iec", "27001:2022"]),
            ("le contrôle A.8.8 est appliqué", ["le", "contrôle", "a.8.8", "est", "appliqué"]),
            ("d'accès", ["accès"]),  # a lone letter split off by an apostrophe is not a token
        ],
    )
    def test_identifiers_stay_whole_plain_words_split_normally(self, text, expected):
        assert tokenize(text) == expected

    def test_single_letter_tokens_are_dropped(self):
        # otherwise "a" (from French elisions like "n'a") would match almost every chunk and drown
        # out real matches for queries that share no other vocabulary with the corpus (e.g. English).
        assert tokenize("a n'a l'accès à y") == ["accès"]

    def test_case_insensitive(self):
        assert tokenize("Référence A.5.18") == tokenize("référence a.5.18")


class TestRetrieval:
    CHUNKS = [
        make_chunk("doc-a:s05:c00", "Référence A.5.18 : la revue des droits d'accès privilégiés n'a pas été réalisée.",
                  doc_id="doc-a", references=["ISO/IEC 27001 A.5.18"]),
        make_chunk("doc-a:s05:c01", "Référence A.8.8 : trois vulnérabilités critiques restent ouvertes.",
                  doc_id="doc-a", references=["ISO/IEC 27001 A.8.8"]),
        make_chunk("doc-b:s01:c00", "Les sauvegardes sont testées chaque trimestre par une restauration complète.",
                  doc_id="doc-b"),
    ]

    def test_exact_control_id_ranks_its_own_chunk_first(self):
        retriever = BM25Retriever(self.CHUNKS)
        hits = retriever.retrieve("A.5.18", top_k=5)
        assert hits[0].payload["chunk_id"] == "doc-a:s05:c00"

    def test_exact_control_id_does_not_match_an_unrelated_control(self):
        retriever = BM25Retriever(self.CHUNKS)
        hits = retriever.retrieve("A.8.8", top_k=5)
        assert [h.payload["chunk_id"] for h in hits][0] == "doc-a:s05:c01"
        assert "doc-a:s05:c00" not in [h.payload["chunk_id"] for h in hits]

    def test_french_terminology_matches_relevant_chunk(self):
        retriever = BM25Retriever(self.CHUNKS)
        hits = retriever.retrieve("revue des droits d'accès privilégiés", top_k=5)
        assert hits[0].payload["chunk_id"] == "doc-a:s05:c00"

    def test_hit_ids_are_the_chunk_point_ids(self):
        retriever = BM25Retriever(self.CHUNKS)
        hits = retriever.retrieve("A.5.18", top_k=1)
        assert hits[0].id == self.CHUNKS[0].point_id

    def test_query_with_no_lexical_overlap_returns_nothing(self):
        retriever = BM25Retriever(self.CHUNKS)
        assert retriever.retrieve("xylophone marmelade zeppelin", top_k=5) == []

    def test_empty_index_returns_nothing(self):
        assert BM25Retriever([]).retrieve("A.5.18", top_k=5) == []

    def test_top_k_limits_the_number_of_results(self):
        retriever = BM25Retriever(self.CHUNKS)
        assert len(retriever.retrieve("sauvegardes accès vulnérabilités", top_k=1)) == 1

    def test_results_are_sorted_by_score_descending(self):
        retriever = BM25Retriever(self.CHUNKS)
        hits = retriever.retrieve("accès vulnérabilités sauvegardes restauration", top_k=5)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_payload_matches_the_chunk(self):
        retriever = BM25Retriever(self.CHUNKS)
        hit = retriever.retrieve("A.5.18", top_k=1)[0]
        assert hit.payload == self.CHUNKS[0].payload()

    def test_from_settings_indexes_the_real_knowledge_base(self, settings):
        retriever = BM25Retriever.from_settings(settings)
        hits = retriever.retrieve("A.5.18", top_k=5)
        assert hits and any("A.5.18" in h.payload["text"] for h in hits)
