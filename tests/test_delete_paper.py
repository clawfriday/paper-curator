"""Test paper deletion endpoint."""
import sys
from pathlib import Path

# Add src/backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

import db


def test_delete_paper_endpoint(client):
    """Create a paper, delete via endpoint, then ensure cached-data is gone."""
    arxiv_id = "test.delete.paper"
    paper_id = db.create_paper(
        arxiv_id=arxiv_id,
        title="Delete Paper Test",
        authors=["Test Author"],
        abstract="Test abstract",
    )
    assert paper_id is not None

    delete_res = client.delete(f"/papers/{arxiv_id}")
    assert delete_res.status_code in [200, 404]

    cached_res = client.get(f"/papers/{arxiv_id}/cached-data")
    assert cached_res.status_code == 404
