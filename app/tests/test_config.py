from rag_app.config import Config


def test_blank_qdrant_api_key_is_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    assert Config(_env_file=None).qdrant_api_key is None


def test_qdrant_api_key_is_kept_when_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    assert Config(_env_file=None).qdrant_api_key == "secret"
