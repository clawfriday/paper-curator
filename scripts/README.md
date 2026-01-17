# Scripts

## Prerequisites
- `paper-curator-service` and GROBID running on `service_base_url` in `config/paperqa.yaml`

## Start GROBID
```bash
bash scripts/docker_mod.sh
bash scripts/docker_run.sh
```

## Endpoint demos (one endpoint per script)
```bash
# Health check (defaults to config service_base_url)
bash scripts/health.sh

# Resolve metadata for a single arXiv ID
bash scripts/arxiv_resolve.sh --arxiv-id 1706.03762

# Download PDF/LaTeX for a single arXiv ID
bash scripts/arxiv_download.sh --arxiv-id 1706.03762

# Extract PDF text (pass pdf_path from the download response)
bash scripts/pdf_extract.sh --pdf-path storage/downloads/1706.03762v7.Attention_Is_All_You_Need.pdf

# If you need the pdf_path from the download JSON:
python services/scripts/extract_pdf_path.py --download-json storage/outputs/arxiv_download.json

# Summarize (uses storage/outputs/extract.json as input)
bash scripts/summarize.sh

# Embed (uses storage/outputs/extract.json as input)
bash scripts/embed.sh

# QA (prepare a plain-text context file first)
docker exec paper-curator-grobid \
  python /app/services/scripts/extract_text.py \
  --extract-json /app/storage/outputs/extract.json \
  --output-file /app/storage/outputs/qa_context.txt

bash scripts/qa.sh --context-file storage/outputs/qa_context.txt \
  --question "What is the main contribution?"
```
