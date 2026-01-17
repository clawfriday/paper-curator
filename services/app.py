from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
from functools import lru_cache
from typing import Any, Optional

import arxiv
import requests
import yaml
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

try:
    from paperqa import Docs
    from paperqa.settings import Settings, make_default_litellm_model_list_settings
    from paperqa.types import Doc, Text
except Exception:  # noqa: BLE001
    Docs = None
    Settings = None
    make_default_litellm_model_list_settings = None
    Doc = None
    Text = None


app = FastAPI(title="paper-curator-service")


class ArxivResolveRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: Optional[str] = Field(default=None, description="arXiv URL")


class ArxivDownloadRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: Optional[str] = Field(default=None, description="arXiv URL")
    output_dir: Optional[str] = Field(default=None, description="Directory to store downloads")


class PdfExtractRequest(BaseModel):
    pdf_path: str = Field(description="Local PDF file path")
    grobid_url: Optional[str] = Field(default=None, description="Override GROBID URL")


class SummarizeRequest(BaseModel):
    text: str = Field(description="Full paper text or extracted sections")


class EmbedRequest(BaseModel):
    text: str = Field(description="Text to embed")


class QaRequest(BaseModel):
    context: str = Field(description="Context text to answer from")
    question: str = Field(description="Question to answer")


def _require_identifier(arxiv_id: Optional[str], url: Optional[str]) -> str:
    if arxiv_id:
        return arxiv_id
    if url:
        return url
    raise HTTPException(status_code=400, detail="Provide arxiv_id or url.")


def _get_grobid_url(override: Optional[str]) -> str:
    if override:
        return override
    config = _load_config()
    return config.get("service_base_url", "http://localhost:9070")


def _load_prompt() -> tuple[str, str, str]:
    prompt_path = pathlib.Path("prompts/paper_summary.md")
    if not prompt_path.exists():
        raise HTTPException(status_code=500, detail="Prompt file not found.")
    content = prompt_path.read_text(encoding="utf-8").strip()
    first_line, _, rest = content.partition("\n")
    if not first_line.startswith("ID: "):
        raise HTTPException(status_code=500, detail="Prompt ID missing.")
    prompt_id = first_line.replace("ID: ", "").strip()
    prompt_body = rest.strip()
    prompt_hash = hashlib.sha256(prompt_body.encode("utf-8")).hexdigest()
    return prompt_id, prompt_hash, prompt_body


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    config_path = pathlib.Path("config/paperqa.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=500, detail="Config file not found: config/paperqa.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = (
        "openai_api_base",
        "openai_api_base2",
        "openai_api_base3",
        "openai_api_key",
        "service_base_url",
    )
    for key in required:
        if key not in config:
            raise HTTPException(status_code=500, detail=f"Missing '{key}' in config.")
    return config


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=3)
def _resolve_model(base_url: str, api_key: str) -> str:
    client = _get_openai_client(base_url, api_key)
    try:
        models = client.models.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Model list failed: {exc}") from exc
    model_ids = sorted([model.id for model in models.data if getattr(model, "id", None)])
    if not model_ids:
        raise HTTPException(status_code=502, detail="No models returned by OpenAI-compatible endpoint.")
    return model_ids[0]


def _chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text[:max_chars]]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        candidate_len = length + len(paragraph) + (2 if current else 0)
        if candidate_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            length = len(paragraph)
        else:
            current.append(paragraph)
            length = candidate_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _paperqa_answer(
    text: str,
    question: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
    llm_model: str,
    embed_model: str,
) -> str:
    if Docs is None:
        raise HTTPException(status_code=500, detail="paper-qa is not installed.")
    if Doc is None or Text is None or Settings is None or make_default_litellm_model_list_settings is None:
        raise HTTPException(status_code=500, detail="paper-qa types are not available.")
    os.environ["OPENAI_API_BASE"] = llm_base_url
    os.environ["OPENAI_API_KEY"] = api_key

    async def _run() -> Any:
        docs = Docs()
        chunks = _chunk_text(text)
        doc = Doc(docname="paper", dockey="paper", citation="Local text")
        texts = [Text(text=chunk, name=f"chunk-{idx}", doc=doc) for idx, chunk in enumerate(chunks)]
        llm_config = make_default_litellm_model_list_settings(llm_model)
        llm_config["model_list"][0]["litellm_params"].update(
            {"api_base": llm_base_url, "api_key": api_key}
        )
        summary_llm_config = make_default_litellm_model_list_settings(llm_model)
        summary_llm_config["model_list"][0]["litellm_params"].update(
            {"api_base": llm_base_url, "api_key": api_key}
        )
        settings = Settings(
            llm=llm_model,
            llm_config=llm_config,
            summary_llm=llm_model,
            summary_llm_config=summary_llm_config,
            embedding=embed_model,
            embedding_config={
                "kwargs": {
                    "api_base": embed_base_url,
                    "api_key": api_key,
                    "encoding_format": "float",
                },
            },
        )
        await docs.aadd_texts(texts=texts, doc=doc, settings=settings)
        return await docs.aquery(question, settings=settings)

    try:
        result = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"PaperQA request failed: {exc}") from exc

    if hasattr(result, "answer"):
        return str(result.answer)
    return str(result)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/arxiv/resolve")
def arxiv_resolve(payload: ArxivResolveRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
    if not results:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
    result = results[0]
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "published": result.published.isoformat() if result.published else None,
        "summary": result.summary,
        "pdf_url": result.pdf_url,
        "entry_id": result.entry_id,
        "comment": result.comment,
    }


@app.post("/arxiv/download")
def arxiv_download(payload: ArxivDownloadRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    output_dir = payload.output_dir or os.getenv("ARXIV_DOWNLOAD_DIR", "storage/downloads")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
    if not results:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
    result = results[0]

    pdf_path = result.download_pdf(dirpath=output_dir)
    latex_path = None
    source_error = None
    try:
        latex_path = result.download_source(dirpath=output_dir)
    except Exception as exc:  # noqa: BLE001 - propagate as response field
        source_error = str(exc)

    return {
        "arxiv_id": result.get_short_id(),
        "pdf_path": pdf_path,
        "latex_path": latex_path,
        "latex_error": source_error,
    }


@app.post("/pdf/extract")
def pdf_extract(payload: PdfExtractRequest) -> dict[str, Any]:
    pdf_path = pathlib.Path(payload.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")

    grobid_url = _get_grobid_url(payload.grobid_url)
    endpoint = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"

    with pdf_path.open("rb") as handle:
        files = {"input": (pdf_path.name, handle, "application/pdf")}
        response = requests.post(endpoint, files=files, timeout=120)

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"GROBID error: {response.status_code} {response.text[:200]}",
        )

    tei_xml = response.text
    parsed = _parse_tei(tei_xml)
    return {"tei_xml": tei_xml, **parsed}


@app.post("/summarize")
def summarize(payload: SummarizeRequest) -> dict[str, Any]:
    prompt_id, prompt_hash, prompt_body = _load_prompt()
    config = _load_config()
    base_url = config["openai_api_base"]
    embed_base_url = config["openai_api_base3"]
    api_key = config["openai_api_key"]
    model = f"openai/{_resolve_model(base_url, api_key)}"
    embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    summary = _paperqa_answer(
        payload.text,
        prompt_body,
        base_url,
        embed_base_url,
        api_key,
        model,
        embed_model,
    )
    return {
        "summary": summary.strip(),
        "prompt_id": prompt_id,
        "prompt_hash": prompt_hash,
        "model": model,
    }


@app.post("/embed")
def embed(payload: EmbedRequest) -> dict[str, Any]:
    config = _load_config()
    base_url = config["openai_api_base3"]
    api_key = config["openai_api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_openai_client(base_url, api_key)
    try:
        response = client.embeddings.create(model=model, input=payload.text)
        vector = response.data[0].embedding
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc
    return {"embedding": vector, "model": model}


@app.post("/qa")
def qa(payload: QaRequest) -> dict[str, Any]:
    config = _load_config()
    base_url = config["openai_api_base"]
    embed_base_url = config["openai_api_base3"]
    api_key = config["openai_api_key"]
    model = f"openai/{_resolve_model(base_url, api_key)}"
    embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    answer = _paperqa_answer(
        payload.context,
        payload.question,
        base_url,
        embed_base_url,
        api_key,
        model,
        embed_model,
    )
    return {"answer": answer}


def _parse_tei(tei_xml: str) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(tei_xml)

    title_el = root.find(".//tei:titleStmt/tei:title", ns)
    abstract_el = root.find(".//tei:abstract", ns)

    sections = []
    for div in root.findall(".//tei:text/tei:body/tei:div", ns):
        head = div.find("tei:head", ns)
        paragraphs = [p.text for p in div.findall("tei:p", ns) if p.text]
        if paragraphs:
            sections.append(
                {
                    "title": head.text if head is not None else None,
                    "text": "\n".join(paragraphs),
                }
            )

    references = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", ns):
        title = bibl.find(".//tei:title", ns)
        authors = [a.text for a in bibl.findall(".//tei:author/tei:persName/tei:surname", ns) if a.text]
        references.append(
            {
                "title": title.text if title is not None else None,
                "authors": authors,
            }
        )

    return {
        "title": title_el.text if title_el is not None else None,
        "abstract": "".join(abstract_el.itertext()).strip() if abstract_el is not None else None,
        "sections": sections,
        "references": references,
    }
