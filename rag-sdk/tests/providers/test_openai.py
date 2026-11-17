from unittest.mock import AsyncMock, MagicMock

from rag_sdk.providers.base import Prompt
from rag_sdk.providers.openai import OpenAIChatter, OpenAIEmbedder


def _make_embed_response(embeddings: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock(embedding=emb, index=i) for i, emb in enumerate(embeddings)]
    return response


def _make_chat_response(content: str = "answer") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _mock_client(
    embedding: list[float] | None = None,
    chat_content: str = "answer",
) -> MagicMock:
    client = MagicMock()
    if embedding is not None:
        client.embeddings.create = AsyncMock(return_value=_make_embed_response([embedding]))
    client.chat.completions.create = AsyncMock(return_value=_make_chat_response(chat_content))
    return client


async def test_embedder_dimensions_property():
    embedder = OpenAIEmbedder(client=MagicMock(), dimensions=768)
    assert embedder.dimensions == 768


async def test_embed_returns_vector():
    client = _mock_client(embedding=[0.1, 0.2, 0.3])
    embedder = OpenAIEmbedder(client=client, dimensions=3)
    result = await embedder.embed("hello")
    assert result == [0.1, 0.2, 0.3]
    client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input="hello")


async def test_embed_chunks_returns_one_vector_per_text():
    def side_effect(model, input, **kw):
        return _make_embed_response([[0.1, 0.2] for _ in input])

    client = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=side_effect)
    embedder = OpenAIEmbedder(client=client, dimensions=2)
    results = await embedder.embed_chunks(["a", "b", "c"])
    assert len(results) == 3


async def test_embed_chunks_splits_into_batches():
    batches_received: list[list[str]] = []

    def side_effect(model, input, **kw):
        batches_received.append(list(input))
        return _make_embed_response([[0.1] for _ in input])

    client = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=side_effect)
    embedder = OpenAIEmbedder(client=client, dimensions=1)
    await embedder.embed_chunks(["a", "b", "c", "d", "e"], batch_size=2)

    assert batches_received == [["a", "b"], ["c", "d"], ["e"]]


async def test_chatter_returns_response_content():
    client = _mock_client(chat_content="The capital is Paris.")
    chatter = OpenAIChatter(client=client)
    response = await chatter.generate(Prompt(system="be helpful", user="capital of France?"))
    assert response.content == "The capital is Paris."


async def test_chatter_extracts_token_usage():
    client = _mock_client()
    chatter = OpenAIChatter(client=client)
    response = await chatter.generate(Prompt(system="sys", user="usr"))
    assert response.token_usage.prompt_tokens == 10
    assert response.token_usage.completion_tokens == 5
    assert response.token_usage.total_tokens == 15


async def test_chatter_passes_system_and_user_messages():
    client = _mock_client()
    chatter = OpenAIChatter(client=client)
    await chatter.generate(Prompt(system="you are a bot", user="hello"))
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "you are a bot"}
    assert messages[1] == {"role": "user", "content": "hello"}


async def test_chatter_passes_call_options():
    client = _mock_client()
    chatter = OpenAIChatter(client=client, call_options={"temperature": 0.2})
    await chatter.generate(Prompt(system="sys", user="usr"))
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.2
