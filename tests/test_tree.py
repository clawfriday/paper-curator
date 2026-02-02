"""Test tree endpoint."""


def test_tree_endpoint(client):
    """Request GET /tree and verify basic structure."""
    response = client.get("/tree")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "children" in data
    assert isinstance(data["children"], list)
