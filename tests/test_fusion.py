"""Reciprocal Rank Fusion and the hybrid retriever that applies it to semantic + BM25 results."""

import pytest

from src.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion
from src.vectorstore import SearchHit


def hit(id_: str, score: float = 1.0, **payload) -> SearchHit:
    return SearchHit(id=id_, score=score, payload={"chunk_id": id_, **payload})


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[SearchHit]]):
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])[:top_k]


class TestReciprocalRankFusion:
    def test_a_candidate_ranked_first_by_both_retrievers_wins(self):
        # "b" is 2nd/3rd, "c" is 3rd/2nd: tied by symmetry, so only "a" (1st in both) is asserted.
        fused = reciprocal_rank_fusion([[hit("a"), hit("b"), hit("c")], [hit("a"), hit("c"), hit("b")]])
        assert [h.id for h in fused][0] == "a"
        assert {h.id for h in fused} == {"a", "b", "c"}

    def test_a_candidate_present_in_only_one_list_still_ranks(self):
        fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("c")]])
        ids = [h.id for h in fused]
        assert set(ids) == {"a", "b", "c"}
        assert ids.index("a") < ids.index("b")  # order within the semantic-only list is preserved

    def test_duplicates_across_lists_are_merged_not_repeated(self):
        fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("b"), hit("a")]])
        assert sorted(h.id for h in fused) == ["a", "b"]
        assert len(fused) == 2

    def test_top_k_limits_the_fused_result(self):
        fused = reciprocal_rank_fusion([[hit("a"), hit("b"), hit("c")]], top_k=2)
        assert len(fused) == 2

    def test_smaller_k_widens_the_gap_between_top_ranks(self):
        rankings = [[hit("a"), hit("b")], [hit("a"), hit("b")]]  # "a" always 1st, "b" always 2nd
        soft = {h.id: h.score for h in reciprocal_rank_fusion(rankings, k=1000)}
        sharp = {h.id: h.score for h in reciprocal_rank_fusion(rankings, k=1)}
        assert (sharp["a"] - sharp["b"]) > (soft["a"] - soft["b"])

    def test_payload_is_preserved(self):
        fused = reciprocal_rank_fusion([[hit("a", text="Le contrôle des accès.")]])
        assert fused[0].payload["text"] == "Le contrôle des accès."

    def test_empty_rankings_produce_no_candidates(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestHybridRetriever:
    def test_fuses_both_retrievers_results(self):
        semantic = FakeRetriever({"q": [hit("a"), hit("b"), hit("c")]})
        lexical = FakeRetriever({"q": [hit("c"), hit("a")]})
        hybrid = HybridRetriever(semantic, lexical)

        hits = hybrid.retrieve("q", top_k=10)

        assert {h.id for h in hits} == {"a", "b", "c"}
        assert hits[0].id == "a"  # ranked first by both retrievers

    def test_asks_each_retriever_for_candidates_per_retriever_not_final_top_k(self):
        semantic = FakeRetriever({"q": []})
        lexical = FakeRetriever({"q": []})
        HybridRetriever(semantic, lexical, candidates_per_retriever=15).retrieve("q", top_k=3)

        assert semantic.calls == [("q", 15)]
        assert lexical.calls == [("q", 15)]

    def test_lexical_only_hit_is_included_even_if_semantic_missed_it(self):
        semantic = FakeRetriever({"A.5.18": [hit("unrelated-1"), hit("unrelated-2")]})
        lexical = FakeRetriever({"A.5.18": [hit("exact-control-id-match")]})
        hybrid = HybridRetriever(semantic, lexical)

        hits = hybrid.retrieve("A.5.18", top_k=10)

        assert "exact-control-id-match" in {h.id for h in hits}

    def test_rrf_k_is_forwarded_to_the_fusion(self):
        semantic = FakeRetriever({"q": [hit("a"), hit("b")]})
        lexical = FakeRetriever({"q": [hit("b"), hit("a")]})
        default_k = HybridRetriever(semantic, lexical, rrf_k=60).retrieve("q", top_k=10)
        tiny_k = HybridRetriever(semantic, lexical, rrf_k=1).retrieve("q", top_k=10)
        assert default_k[0].score != tiny_k[0].score
