FROM python:3.13-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PATH="/root/.local/bin/:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-group dev --no-group test

FROM python:3.13-slim-bookworm AS runtime

ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    DEBIAN_FRONTEND=noninteractive

RUN useradd -m -u 1000 atheon
USER atheon

COPY --chown=atheon --from=builder /.venv /.venv

COPY --chown=atheon ./app /app

ENV PATH="/.venv/bin:$PATH"
ENV PYTHONPATH="/app:/.venv/lib/python3.13/dist-packages"

WORKDIR /app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--workers", "2", "--host", "0.0.0.0", "--port", "7860"]