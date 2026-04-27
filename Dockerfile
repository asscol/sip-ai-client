FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    | tar -xz -C /usr/local/bin/ --strip-components=1 \
    || echo "Piper will be installed via pip"

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]"

RUN mkdir -p /app/logs /root/.local/share/piper-voices

COPY config.yaml.example config.yaml.example

EXPOSE 5060/udp
EXPOSE 5060/tcp
EXPOSE 10000-20000/udp

ENTRYPOINT ["sip-ai-client"]
CMD ["--config", "config.yaml"]
