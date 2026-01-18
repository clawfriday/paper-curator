FROM lfoppiano/grobid:0.8.0

USER root

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN echo "deb [trusted=yes] http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list.d/bookworm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3-pip nginx \
    && python3.11 -m pip install --no-cache-dir --upgrade pip \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services /app/services
COPY config /app/config
COPY prompts /app/prompts

RUN python3.11 -m pip install --no-cache-dir --break-system-packages \
    fastapi \
    uvicorn \
    requests \
    arxiv \
    pyyaml \
    openai \
    pillow \
    paper-qa \
    eval_type_backport

COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/entrypoint.sh /app/entrypoint.sh

EXPOSE 9070

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
