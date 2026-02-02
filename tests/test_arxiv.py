"""Test arXiv endpoints."""


def test_arxiv_resolve_with_id(client, sample_arxiv_id):
    """Test resolving arXiv paper by ID."""
    response = client.post("/arxiv/resolve", json={"arxiv_id": sample_arxiv_id})
    assert response.status_code == 200
    data = response.json()
    assert "arxiv_id" in data
    assert "title" in data
    assert "authors" in data
    assert "summary" in data
    assert "pdf_url" in data
    # Verify it's the "Attention Is All You Need" paper
    assert "Attention" in data["title"]


def test_arxiv_resolve_no_identifier(client):
    """Test that missing identifier returns 400 error."""
    response = client.post("/arxiv/resolve", json={})
    assert response.status_code == 400
    assert "Provide arxiv_id or url" in response.json()["detail"]
