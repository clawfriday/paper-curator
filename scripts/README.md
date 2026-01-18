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

# Summarize (PDF-first; falls back to extract.json if no --pdf-path)
bash scripts/summarize.sh --pdf-path storage/downloads/1706.03762v7.Attention_Is_All_You_Need.pdf

# Embed (uses storage/outputs/extract.json as input)
bash scripts/embed.sh

# QA (PDF-first; context file still supported)
bash scripts/qa.sh --pdf-path storage/downloads/1706.03762v7.Attention_Is_All_You_Need.pdf \
  --question "What is the main contribution?"
```
