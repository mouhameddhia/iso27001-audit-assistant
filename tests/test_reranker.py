"""Cross-encoder reranker: scoring, ranking, configurable top-k, and model loading errors."""

import pytest

from src.retrieval.reranker import CrossEncoderReranker, RerankError
from src.vectorstore import SearchHit
from tests.conftest import FakeCrossEncoder, huggingface_reachable


def hit(id_: str, text: str, heading: str | None = None) -> SearchHit:
    payload = {"chunk_id": id_, "text": text}
    if heading is not None:
        payload["heading"] = heading
    return SearchHit(id=id_, score=0.0, payload=payload)


CANDIDATES = [
    hit("c1", "La revue périodique des droits d'accès privilégiés n'a pas été réalisée."),
    hit("c2", "Les sauvegardes sont testées chaque trimestre par une restauration complète."),
    hit("c3", "Trois vulnérabilités critiques restent non corrigées au-delà du délai."),
]


def make_reranker(model=None) -> CrossEncoderReranker:
    return CrossEncoderReranker("fake-model", model=model or FakeCrossEncoder())


class TestScoringAndRanking:
    def test_candidates_are_reordered_by_relevance_score(self):
        fake = FakeCrossEncoder()
        reranked = make_reranker(fake).rerank("revue des droits d'accès privilégiés", CANDIDATES, top_k=3)
        assert [h.id for h in reranked][0] == "c1"

    def test_scores_are_updated_to_the_cross_encoder_scores(self):
        fake = FakeCrossEncoder()
        reranked = make_reranker(fake).rerank("accès privilégiés", CANDIDATES, top_k=3)
        assert reranked[0].score >= reranked[1].score >= reranked[2].score
        assert reranked[0].score == max(fake.predict([("accès privilégiés", c.payload["text"]) for c in CANDIDATES]))

    def test_query_and_passage_are_paired_and_sent_to_the_model(self):
        fake = FakeCrossEncoder()
        make_reranker(fake).rerank("ma requête", [CANDIDATES[0]], top_k=1)
        assert fake.calls == [[("ma requête", CANDIDATES[0].payload["text"])]]

    def test_heading_is_prefixed_to_the_passage_when_present(self):
        fake = FakeCrossEncoder()
        with_heading = hit("c1", "texte", heading="Doc › Section")
        make_reranker(fake).rerank("q", [with_heading], top_k=1)
        assert fake.calls[0][0][1] == "Doc › Section\n\ntexte"

    def test_payload_is_preserved_after_reranking(self):
        fake = FakeCrossEncoder()
        [reranked] = make_reranker(fake).rerank("accès", [CANDIDATES[0]], top_k=1)
        assert reranked.payload == CANDIDATES[0].payload


class TestTopKAndEdgeCases:
    def test_top_k_limits_the_returned_candidates(self):
        reranked = make_reranker().rerank("accès vulnérabilités sauvegardes", CANDIDATES, top_k=2)
        assert len(reranked) == 2

    def test_top_k_larger_than_candidates_returns_all_of_them(self):
        reranked = make_reranker().rerank("q", CANDIDATES, top_k=100)
        assert len(reranked) == len(CANDIDATES)

    def test_no_candidates_returns_no_candidates(self):
        assert make_reranker().rerank("q", [], top_k=5) == []

    def test_batch_size_is_forwarded_to_the_model(self):
        fake = FakeCrossEncoder()
        CrossEncoderReranker("fake-model", batch_size=2, model=fake).rerank("q", CANDIDATES, top_k=3)
        assert fake.batch_sizes == [2]


class TestErrorHandling:
    def test_scoring_failure_is_wrapped_in_rerank_error(self):
        class BrokenModel:
            def predict(self, pairs, batch_size=None):
                raise RuntimeError("boom")

        with pytest.raises(RerankError, match="Cross-encoder scoring failed"):
            make_reranker(BrokenModel()).rerank("q", CANDIDATES, top_k=3)

    @pytest.mark.integration
    def test_loading_an_unknown_model_raises_rerank_error(self):
        if not huggingface_reachable():
            pytest.skip("huggingface.co not reachable")
        with pytest.raises(RerankError, match="Failed to load cross-encoder"):
            CrossEncoderReranker("this-org/this-model-does-not-exist-at-all")

    def test_missing_sentence_transformers_raises_rerank_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("no module named sentence_transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(RerankError, match="sentence-transformers is required"):
            CrossEncoderReranker("fake-model")


@pytest.mark.integration
def test_live_model_scores_the_relevant_chunk_highest(settings):
    if not huggingface_reachable():
        pytest.skip("huggingface.co not reachable")
    reranker = CrossEncoderReranker.from_settings(settings)
    reranked = reranker.rerank("Le contrôle des accès privilégiés n'est pas revu périodiquement.", CANDIDATES, top_k=3)
    assert reranked[0].id == "c1"
    assert reranked[0].score > reranked[-1].score
