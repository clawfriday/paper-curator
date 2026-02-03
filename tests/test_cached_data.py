"""Test cached-data endpoint and structured summary persistence."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add src/backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

import db


def test_cached_data_structured_summary(client: TestClient):
    """Ensure structured summary is saved and restored via cached-data endpoint."""
    arxiv_id = "test.structured.summary"
    paper_id = db.create_paper(
        arxiv_id=arxiv_id,
        title="Structured Summary Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
    )
    structured_summary = {
        "components": ["Component A"],
        "sections": [
            {
                "component": "Component A",
                "steps": "Step A",
                "benefits": "Benefit A",
                "rationale": "Rationale A",
                "results": "Result A",
            }
        ],
        "model": "test-model",
    }
    db.update_paper_structured_summary(paper_id, structured_summary)

    # Use require_embedding=false since test paper has no embedding
    response = client.get(f"/papers/{arxiv_id}/cached-data?require_embedding=false")
    assert response.status_code == 200
    data = response.json()
    assert data["arxiv_id"] == arxiv_id
    assert data["structured_summary"] == structured_summary

    cleanup = client.delete(f"/papers/{arxiv_id}")
    assert cleanup.status_code in [200, 404]
