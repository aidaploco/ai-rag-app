FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY rag-sdk/ ./rag-sdk/
COPY app/ ./app/

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# tiktoken downloads its vocabulary on first use; bake it in so the container works offline.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

EXPOSE 8000

CMD ["uvicorn", "rag_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
