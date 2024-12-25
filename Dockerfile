FROM ubuntu:24.10

LABEL MAINTAINER="ychsu@iii.org.com"

ARG PYTHON_VERSION=python3.10
ARG UV_VERSION=0.5.7

WORKDIR /app

RUN apt update && \
    apt install texlive-latex-base texlive-xetex texlive-fonts-recommended dvipng cm-super latex-cjk-all

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /bin/uv
RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "src/latex_table_generator/main.py"]
