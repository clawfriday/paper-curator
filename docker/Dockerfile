FROM lfoppiano/grobid:0.8.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services /app/services
COPY config /app/config
COPY prompts /app/prompts

RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    arxiv \
    pyyaml \
    openai \
    paper-qa \
    eval_type_backport

COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/entrypoint.sh /app/entrypoint.sh

EXPOSE 9070

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
