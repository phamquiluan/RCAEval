ARG BASE_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ARG INSTALLATION_TAG=default

FROM ${BASE_IMAGE}
ARG INSTALLATION_TAG

WORKDIR /app

# Install build-deps, sync production deps, then clean up
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    make \
    build-essential \
    libxml2 \
    libxml2-dev \
    zlib1g-dev \
    python3-tk \
    graphviz \
    git \
    libffi-dev \
    libjpeg-dev \
    default-jdk \
    default-jre && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv

RUN --mount=type=cache,target=/root/.cache/uv uv pip install -e .[${INSTALLATION_TAG}]
RUN --mount=type=cache,target=/root/.cache/uv uv pip install git+https://github.com/salesforce/causalai.git

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${INSTALLATION_TAG}" = "rcd" ]; then \
        uv pip install sfr-pyrca; \
    fi

ENV PATH="/app/.venv/bin:$PATH"
CMD ["uv", "run", "execute_experiments.py"]