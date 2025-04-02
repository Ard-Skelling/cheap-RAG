FROM python:3.12.9-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    vim \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.1.1
ENV POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /cheap-RAG
COPY . .

ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry lock
# RUN poetry install --no-interaction --no-ansi --extras "gpu"
RUN poetry install --no-interaction --no-ansi --extras "cpu"

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache \
    && find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} +
