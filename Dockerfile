FROM ubuntu:24.10

LABEL MAINTAINER="ychsu@iii.org.com"

ARG PYTHON_VERSION=python3.10
ARG UV_VERSION=0.5.7
ARG OS_ARCH=x86_64

WORKDIR /app

COPY ./scripts/install-pandoc.sh ./

RUN apt update && \
    apt install wkhtmltopdf wget \
    ARCH=${OS_ARCH} bash install-pandoc.sh

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /bin/uv
COPY ./src ./pyproject.toml ./README.md ./uv.lock ./
RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "src/latex_table_generator"]
