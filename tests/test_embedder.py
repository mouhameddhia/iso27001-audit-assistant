import math
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.config import Settings
from src.embeddings import EmbeddingError, OllamaEmbedder
from tests.conftest import ollama_model_available


def fake_ollama_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def echo_embeddings(*_, json, **__):
    return fake_ollama_response({"embeddings": [[float(len(text)), 1.0] for text in json["input"]]})


def make_embedder(**overrides):
    params = dict(model="nomic-embed-text", base_url="http://ollama:11434/", batch_size=2,
                  document_prefix="search_document: ", query_prefix="search_query: ")
    return OllamaEmbedder(**{**params, **overrides})


def test_documents_are_embedded_in_batches_with_document_prefix():
    embedder = make_embedder()

    with patch("src.embeddings.embedder.requests.post", side_effect=echo_embeddings) as post:
        vectors = embedder.embed_documents(["a", "bb", "ccc"])

    assert post.call_count == 2
    assert post.call_args_list[0].args[0] == "http://ollama:11434/api/embed"
    assert post.call_args_list[0].kwargs["json"] == {
        "model": "nomic-embed-text", "input": ["search_document: a", "search_document: bb"],
    }
    assert post.call_args_list[1].kwargs["json"]["input"] == ["search_document: ccc"]
    assert vectors == [[18.0, 1.0], [19.0, 1.0], [20.0, 1.0]]


def test_query_uses_query_prefix():
    with patch("src.embeddings.embedder.requests.post", side_effect=echo_embeddings) as post:
        vector = make_embedder().embed_query("accès")

    assert post.call_args.kwargs["json"]["input"] == ["search_query: accès"]
    assert vector == [float(len("search_query: accès")), 1.0]


def test_from_settings_uses_configuration():
    settings = Settings(_env_file=None, embedding_model="bge-m3", embedding_base_url="http://gpu-box:11434",
                        embedding_batch_size=4, embedding_document_prefix="", embedding_query_prefix="")
    embedder = OllamaEmbedder.from_settings(settings)
    assert (embedder.model, embedder.base_url, embedder.batch_size, embedder.document_prefix) == (
        "bge-m3", "http://gpu-box:11434", 4, "")


def test_fingerprint_changes_with_model_and_document_prefix():
    assert make_embedder().fingerprint != make_embedder(model="bge-m3").fingerprint
    assert make_embedder().fingerprint != make_embedder(document_prefix="").fingerprint
    assert make_embedder().fingerprint == make_embedder(query_prefix="other: ").fingerprint


def test_wrong_number_of_embeddings_raises():
    with patch("src.embeddings.embedder.requests.post", return_value=fake_ollama_response({"embeddings": [[1.0]]})):
        with pytest.raises(EmbeddingError, match="Expected 2 embeddings"):
            make_embedder().embed_documents(["a", "b"])


def test_http_failure_is_wrapped_with_context():
    with patch("src.embeddings.embedder.requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(EmbeddingError, match="model=nomic-embed-text"):
            make_embedder().embed_query("x")


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.integration
def test_live_ollama_embeddings_capture_meaning(settings):
    if not ollama_model_available(settings):
        pytest.skip(f"Ollama model {settings.embedding_model} not available at {settings.embedding_base_url}")
    embedder = OllamaEmbedder.from_settings(settings)

    access_review, backup = embedder.embed_documents([
        "La revue périodique des droits d'accès privilégiés n'a pas été réalisée.",
        "Les sauvegardes sont testées chaque trimestre par une restauration complète.",
    ])
    access_queries = ["revue des droits des comptes à privilèges", "Privileged access rights are not reviewed regularly."]

    assert len(access_review) == len(backup) > 0
    for text in access_queries:
        query = embedder.embed_query(text)
        assert _cosine(query, access_review) > _cosine(query, backup), text
    query = embedder.embed_query("tests de restauration des sauvegardes")
    assert _cosine(query, backup) > _cosine(query, access_review)
