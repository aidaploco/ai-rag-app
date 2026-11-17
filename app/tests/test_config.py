import pytest
from rag_app.config import Config


def test_blank_qdrant_api_key_is_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    assert Config(_env_file=None).qdrant_api_key is None


def test_qdrant_api_key_is_kept_when_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    assert Config(_env_file=None).qdrant_api_key == "secret"


def test_vector_store_defaults_to_qdrant(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    assert Config(_env_file=None).vector_store == "qdrant"


def test_pgvector_requires_a_dsn(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    with pytest.raises(ValueError, match="POSTGRES_DSN"):
        Config(_env_file=None)
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://u:p@h/db")
    assert Config(_env_file=None).postgres_dsn == "postgresql://u:p@h/db"


def test_unknown_vector_store_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("VECTOR_STORE", "faiss")
    with pytest.raises(ValueError):
        Config(_env_file=None)
