"""Hierarchical clustering for paper classification using document embeddings."""

from __future__ import annotations

import hashlib
import json
import numpy as np
from typing import Any
from sklearn.cluster import KMeans

import db


# =============================================================================
# Configuration
# =============================================================================

def _get_clustering_config() -> dict[str, Any]:
    """Get clustering configuration from config file.
    
    Returns:
        Dictionary with clustering parameters:
        - branching_factor: Maximum children before branching
        - clustering_method: Clustering algorithm (currently only "divisive")
    """
    import yaml
    from pathlib import Path
    
    # Try multiple paths
    config_paths = [
        Path("config/paperqa.yaml"),
        Path("../config/paperqa.yaml"),
        Path("../../config/paperqa.yaml"),
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break
    
    if not config_path:
        # Default config
        return {
            "branching_factor": 3,
            "clustering_method": "divisive",
        }
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    classification = config.get("classification", {})
    return {
        "branching_factor": int(classification.get("branching_factor", 3)),
        "clustering_method": classification.get("clustering_method", "divisive"),
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_embedding_to_numpy(embedding) -> np.ndarray:
    """Convert pgvector embedding to numpy array.
    
    Args:
        embedding: Embedding from database (pgvector type)
        
    Returns:
        Numpy array of shape (embedding_dim,)
    """
    if hasattr(embedding, 'tolist'):
        emb_list = embedding.tolist()
    elif isinstance(embedding, (list, tuple)):
        emb_list = list(embedding)
    else:
        emb_list = list(embedding)
    
    return np.array(emb_list, dtype=np.float32)


def _generate_cluster_id(paper_ids: list[int], prefix: str = "") -> str:
    """Generate a deterministic cluster ID from sorted paper IDs using stable hash.
    
    Args:
        paper_ids: List of paper IDs in this cluster (will be sorted for stability)
        prefix: Optional prefix for the cluster ID
        
    Returns:
        Deterministic cluster ID string
    """
    # Sort paper IDs for deterministic hashing
    sorted_ids = sorted(paper_ids)
    # Create stable hash from sorted paper IDs
    id_string = ",".join(str(pid) for pid in sorted_ids)
    hash_obj = hashlib.sha256(id_string.encode())
    unique_suffix = hash_obj.hexdigest()[:8]
    return f"cluster_{unique_suffix}" if not prefix else f"{prefix}_{unique_suffix}"


# =============================================================================
# Clustering Algorithm Helpers
# =============================================================================

def _check_embeddings_valid(embeddings: np.ndarray, k: int) -> bool:
    """Check if embeddings are valid for k-means clustering.
    
    Args:
        embeddings: Embeddings array
        k: Number of clusters to create
        
    Returns:
        True if valid, False if should fallback to leaf
    """
    # Check if enough papers for k clusters
    if embeddings.shape[0] < k:
        return False
    
    # Check if all embeddings are identical (would cause k-means to fail)
    if embeddings.shape[0] > 1:
        embedding_std = np.std(embeddings, axis=0)
        if np.all(embedding_std == 0):
            return False
    
    return True


def _compute_cosine_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute cosine-based silhouette score for clustering quality.
    
    Args:
        embeddings: Embeddings array
        labels: Cluster labels from k-means
        
    Returns:
        Silhouette score (higher is better, range: -1 to 1)
    """
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    
    # Compute cosine distance matrix (1 - cosine similarity)
    cosine_similarity = np.dot(normalized_embeddings, normalized_embeddings.T)
    cosine_distance = 1 - cosine_similarity
    
    # Compute silhouette score using cosine distance
    n_samples = len(labels)
    if n_samples < 2:
        return 0.0
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    silhouette_scores = []
    for i in range(n_samples):
        label_i = labels[i]
        
        # Compute average distance to same cluster (a_i)
        same_cluster_mask = labels == label_i
        same_cluster_distances = cosine_distance[i, same_cluster_mask]
        if len(same_cluster_distances) > 1:
            a_i = np.mean(same_cluster_distances[same_cluster_distances > 0])
        else:
            a_i = 0.0
        
        # Compute minimum average distance to other clusters (b_i)
        other_clusters = unique_labels[unique_labels != label_i]
        b_i_values = []
        for other_label in other_clusters:
            other_cluster_mask = labels == other_label
            other_cluster_distances = cosine_distance[i, other_cluster_mask]
            if len(other_cluster_distances) > 0:
                b_i_values.append(np.mean(other_cluster_distances))
        
        if b_i_values:
            b_i = min(b_i_values)
            # Silhouette score for this sample
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores) if silhouette_scores else 0.0


def _select_optimal_k(
    embeddings: np.ndarray,
    branching_factor: int,
    n_papers: int
) -> int:
    """Select optimal k for k-means clustering using quality scoring.
    
    Tries k = 2, 3, 4, 5 (up to branching_factor, but not exceeding n_papers)
    and selects the smallest k whose score is close to the best score.
    
    Args:
        embeddings: Embeddings array
        branching_factor: Maximum children before branching
        n_papers: Number of papers
        
    Returns:
        Optimal k value
    """
    # Determine candidate k values
    max_k = min(branching_factor, n_papers)
    candidate_ks = [k for k in range(2, max_k + 1)]
    
    if len(candidate_ks) == 0:
        return 2
    
    if len(candidate_ks) == 1:
        return candidate_ks[0]
    
    # Try each k and compute quality score
    k_scores = {}
    for k in candidate_ks:
        if not _check_embeddings_valid(embeddings, k):
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute cosine silhouette score
        score = _compute_cosine_silhouette_score(embeddings, labels)
        k_scores[k] = score
    
    if not k_scores:
        return 2
    
    # Find best score
    best_k = max(k_scores, key=k_scores.get)
    best_score = k_scores[best_k]
    
    # Tolerance: within 10% of best score (relative tolerance)
    # This ensures we prefer smaller k when scores are similar
    tolerance = max(0.05, best_score * 0.10)  # At least 0.05 absolute, or 10% relative
    
    # Find smallest k whose score is within tolerance of best score
    for k in sorted(candidate_ks):
        if k in k_scores:
            score = k_scores[k]
            if score >= best_score - tolerance:
                return k
    
    # Fallback to best k
    return best_k


# =============================================================================
# Tree Builder Class
# =============================================================================

class TreeBuilder:
    """Class-based tree builder that separates tree creation from wrapping logic.
    
    Tree structure: {node_id: [child_node_ids or paper_ids]}
    - Paper IDs are prefixed with "paper_" (e.g., "paper_1", "paper_2")
    - Node IDs are prefixed with "node_" (e.g., "node_abc123")
    - Leaf nodes contain only paper IDs (strings starting with "paper_")
    - Intermediate nodes contain only child node IDs (strings starting with "node_")
    """
    
    def __init__(self, embeddings_dict: dict[int, np.ndarray], branching_factor: int):
        """Initialize tree builder.
        
        Args:
            embeddings_dict: Mapping of paper_id (int) to embedding vector
            branching_factor: Maximum children before a node becomes a leaf
        """
        self.embeddings_dict = embeddings_dict
        self.branching_factor = branching_factor
        self.tree: dict[str, list[str]] = {}  # {node_id: [child_node_ids or paper_ids]}
        self._node_id_cache: dict[tuple, str] = {}  # Cache for deterministic node IDs
    
    def build_tree(self, paper_ids: list[int]) -> dict[str, list[str]]:
        """Build tree structure starting with given paper IDs.
        
        Args:
            paper_ids: List of integer paper IDs
            
        Returns:
            Tree dictionary: {node_id: [child_node_ids or paper_ids]}
        """
        # Convert integer paper IDs to prefixed strings
        prefixed_paper_ids = [f"paper_{pid}" for pid in paper_ids]
        
        # Create root node
        root_id = self._generate_node_id(paper_ids)
        self.tree[root_id] = prefixed_paper_ids
        
        # Recursively split root node
        self._split_node(root_id, paper_ids)
        
        return self.tree
    
    def _split_node(self, node_id: str, paper_ids: list[int]) -> None:
        """Recursively split a node if it contains too many papers.
        
        Args:
            node_id: Current node ID to split
            paper_ids: List of integer paper IDs in this node
        """
        # Base case: if node is small enough, it's a leaf
        if len(paper_ids) <= self.branching_factor or len(paper_ids) < 2:
            # Ensure node value contains prefixed paper IDs
            self.tree[node_id] = [f"paper_{pid}" for pid in paper_ids]
            return
        
        # Get embeddings for these papers
        embeddings_list = [self.embeddings_dict[pid] for pid in paper_ids]
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # Select optimal k using quality scoring
        k = _select_optimal_k(embeddings_array, self.branching_factor, len(paper_ids))
        
        # Validate embeddings
        if not _check_embeddings_valid(embeddings_array, k):
            # Can't cluster, make it a leaf
            self.tree[node_id] = [f"paper_{pid}" for pid in paper_ids]
            return
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Group papers by cluster label
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(paper_ids[idx])
        
        # Filter out empty clusters
        clusters = {label: papers for label, papers in clusters.items() if len(papers) > 0}
        
        if len(clusters) == 0:
            # No clusters created, make it a leaf
            self.tree[node_id] = [f"paper_{pid}" for pid in paper_ids]
            return
        
        # Create child nodes for each cluster
        child_node_ids = []
        for label, cluster_paper_ids in clusters.items():
            child_id = self._generate_node_id(cluster_paper_ids)
            # Initialize child node with prefixed paper IDs
            self.tree[child_id] = [f"paper_{pid}" for pid in cluster_paper_ids]
            child_node_ids.append(child_id)
            
            # Recursively split child node
            self._split_node(child_id, cluster_paper_ids)
        
        # Replace node value with child node IDs
        self.tree[node_id] = child_node_ids
    
    def _generate_node_id(self, paper_ids: list[int]) -> str:
        """Generate deterministic node ID from paper IDs.
        
        Args:
            paper_ids: List of integer paper IDs
            
        Returns:
            Node ID string (e.g., "node_abc12345")
        """
        # Use cache for deterministic IDs
        key = tuple(sorted(paper_ids))
        if key in self._node_id_cache:
            return self._node_id_cache[key]
        
        # Sort paper IDs for deterministic hashing
        sorted_ids = sorted(paper_ids)
        id_string = ",".join(str(pid) for pid in sorted_ids)
        hash_obj = hashlib.sha256(id_string.encode())
        unique_suffix = hash_obj.hexdigest()[:12]  # Use 12 chars to reduce collision risk
        node_id = f"node_{unique_suffix}"
        
        # Cache the result
        self._node_id_cache[key] = node_id
        return node_id


def _wrap_tree_for_db(tree: dict[str, list[str]], root_node_id: str, node_names: dict[str, str] | None = None) -> dict[str, Any]:
    """Convert tree structure to frontend format with children as list.
    
    This function:
    1. Traverses the tree structure
    2. Removes single-child intermediate nodes (unwraps them)
    3. Creates frontend-compatible nested structure with children as list
    
    Args:
        tree: Tree dictionary {node_id: [child_node_ids or paper_ids]}
        root_node_id: Root node ID to start traversal
        node_names: Optional dictionary mapping cluster_id to node name
        
    Returns:
        Frontend-compatible structure:
        {
            "name": "AI Papers",
            "children": [
                {
                    "name": "Category Name",
                    "node_id": "cluster_abc",
                    "node_type": "category",
                    "children": [...]
                },
                {
                    "name": "Paper Name",
                    "node_id": "cluster_def",
                    "node_type": "paper",
                    "paper_id": 123
                }
            ]
        }
    """
    def _wrap_node(node_id: str) -> dict[str, Any] | None:
        """Recursively wrap a node into frontend format."""
        node_value = tree.get(node_id, [])
        
        # Check if this is a leaf node (contains only paper IDs)
        paper_ids = []
        child_node_ids = []
        
        for item in node_value:
            if item.startswith("paper_"):
                try:
                    paper_id = int(item[6:])
                    paper_ids.append(paper_id)
                except ValueError:
                    continue
            elif item.startswith("node_"):
                child_node_ids.append(item)
        
        # If node contains paper IDs and no child nodes, it's a leaf
        if paper_ids and not child_node_ids:
            # Leaf node - create paper node(s)
            cluster_id = _generate_cluster_id(paper_ids)
            if len(paper_ids) == 1:
                # Single paper node
                name = node_names.get(cluster_id, f"Paper {paper_ids[0]}") if node_names else f"Paper {paper_ids[0]}"
                return {
                    "name": name,
                    "node_id": cluster_id,
                    "node_type": "paper",
                    "paper_id": paper_ids[0],
                }
            else:
                # Multiple papers - create category node
                name = node_names.get(cluster_id, f"Cluster {cluster_id}") if node_names else f"Cluster {cluster_id}"
                return {
                    "name": name,
                    "node_id": cluster_id,
                    "node_type": "category",
                    "children": [
                        {
                            "name": node_names.get(_generate_cluster_id([pid]), f"Paper {pid}") if node_names else f"Paper {pid}",
                            "node_id": _generate_cluster_id([pid]),
                            "node_type": "paper",
                            "paper_id": pid,
                        }
                        for pid in paper_ids
                    ],
                }
        
        # Intermediate node - wrap children recursively
        wrapped_children = []
        for child_id in child_node_ids:
            child_wrapped = _wrap_node(child_id)
            if child_wrapped is not None:
                wrapped_children.append(child_wrapped)
        
        # Remove single-child intermediates: if only one child after unwrapping, return it directly
        if len(wrapped_children) == 0:
            return None
        elif len(wrapped_children) == 1:
            # Single child - unwrap: return the child directly, don't create intermediate
            return wrapped_children[0]
        
        # Multiple children - create intermediate category node
        # Collect all paper IDs from children for deterministic ID generation
        all_paper_ids = []
        def collect_papers(children_list):
            for child in children_list:
                if child.get("node_type") == "paper":
                    all_paper_ids.append(child.get("paper_id"))
                elif child.get("node_type") == "category" and child.get("children"):
                    collect_papers(child["children"])
        collect_papers(wrapped_children)
        
        cluster_id = _generate_cluster_id(all_paper_ids)
        name = node_names.get(cluster_id, f"Cluster {cluster_id}") if node_names else f"Cluster {cluster_id}"
        
        return {
            "name": name,
            "node_id": cluster_id,
            "node_type": "category",
            "children": wrapped_children,
        }
    
    root_wrapped = _wrap_node(root_node_id)
    
    if root_wrapped is None:
        return {"name": "AI Papers", "children": []}
    
    # If root is a single category, unwrap it
    if root_wrapped.get("node_type") == "category" and root_wrapped.get("children"):
        return {
            "name": "AI Papers",
            "children": root_wrapped["children"],
        }
    
    # Root is a single paper (edge case)
    return {
        "name": "AI Papers",
        "children": [root_wrapped],
    }


# Obsolete functions removed - tree is now stored directly in frontend format as JSONB


def write_tree_to_database(tree_structure: dict[str, Any], node_names: dict[str, str] | None = None) -> int:
    """Write frontend format tree structure to database as JSONB.
    
    Args:
        tree_structure: Frontend format tree structure {name, children: [...]}
        node_names: Optional dictionary mapping cluster_id to node name
    
    Returns:
        Number of nodes in tree
    """
    # Extract node_names from tree if not provided
    if node_names is None:
        node_names = {}
        def extract_names(node: dict[str, Any]):
            if node.get("node_id"):
                node_names[node["node_id"]] = node.get("name", "")
            if node.get("children"):
                for child in node["children"]:
                    extract_names(child)
        extract_names(tree_structure)
    
    # Count nodes
    def count_nodes(node: dict[str, Any]) -> int:
        count = 1
        if node.get("children"):
            for child in node["children"]:
                count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(tree_structure)
    
    # Save to database
    db.save_tree(tree_structure, node_names)
    
    return total_nodes


# =============================================================================
# Main API
# =============================================================================

def _extract_papers_with_embeddings() -> tuple[list[dict], list[np.ndarray], list[int]]:
    """Extract papers with embeddings from database.
    
    Returns:
        Tuple of:
        - List of paper dictionaries
        - List of numpy embedding arrays
        - List of paper IDs
    """
    papers = db.get_all_papers()
    
    papers_with_embeddings = []
    embeddings_list = []
    paper_ids_list = []
    
    for paper in papers:
        emb = paper.get("embedding")
        if emb is not None:
            papers_with_embeddings.append(paper)
            embeddings_list.append(_convert_embedding_to_numpy(emb))
            paper_ids_list.append(paper["id"])
    
    return papers_with_embeddings, embeddings_list, paper_ids_list


def build_tree_from_clusters() -> dict[str, Any]:
    """Build tree structure from all papers using hierarchical clustering.
    
    This is the main entry point for clustering. It:
    1. Fetches papers with embeddings from database
    2. Creates embeddings dictionary mapping
    3. Uses TreeBuilder class to create tree structure
    4. Wraps tree for database/rendering
    5. Returns cluster tree structure
    
    Returns:
        Dictionary with tree structure and metadata:
        {
            "clusters": [list of cluster dictionaries],
            "total_papers": int,
            "total_clusters": int
        }
    """
    # Extract papers with embeddings
    papers_with_embeddings, embeddings_list, paper_ids_list = _extract_papers_with_embeddings()
    
    if len(papers_with_embeddings) < 2:
        return {
            "clusters": [],
            "total_papers": len(papers_with_embeddings),
            "total_clusters": 0,
            "message": f"Need at least 2 papers with embeddings, found {len(papers_with_embeddings)}",
        }
    
    # Create embeddings dictionary: {paper_id: embedding_array}
    embeddings_dict = {
        paper_id: embedding
        for paper_id, embedding in zip(paper_ids_list, embeddings_list)
    }
    
    # Get clustering config
    config = _get_clustering_config()
    branching_factor = config["branching_factor"]
    
    # Phase 1: Create raw tree structure using TreeBuilder
    builder = TreeBuilder(embeddings_dict, branching_factor)
    raw_tree = builder.build_tree(paper_ids_list)
    
    # Find root node ID (the one that contains all initial papers)
    root_node_id = builder._generate_node_id(paper_ids_list)
    
    # Phase 2: Wrap tree for frontend format (removes single-child intermediates)
    tree_structure = _wrap_tree_for_db(raw_tree, root_node_id, node_names=None)
    
    # Count total nodes in tree
    def count_nodes(node: dict[str, Any]) -> int:
        count = 1  # Count this node
        if node.get("children"):
            for child in node["children"]:
                count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(tree_structure)
    
    return {
        **tree_structure,  # Frontend format tree: {name, children: [...]}
        "total_papers": len(papers_with_embeddings),
        "total_clusters": total_nodes,
    }


# =============================================================================
# Main Block - Standalone Execution
# =============================================================================

if __name__ == "__main__":
    """Sample calling script to run clustering as standalone function.
    
    Usage:
        python clustering.py
        
    This will:
        1. Load all papers with embeddings from the database
        2. Build a hierarchical clustering tree
        3. Print the tree structure as JSON
    """
    print("=" * 60)
    print("Running hierarchical clustering on all papers...")
    print("=" * 60)
    
    # Build tree from all papers
    result = build_tree_from_clusters()
    
    # Print summary
    print(f"\n✓ Clustering complete!")
    print(f"  - Total papers processed: {result.get('total_papers', 0)}")
    print(f"  - Top-level clusters: {result.get('total_clusters', 0)}")
    
    if result.get('message'):
        print(f"  - Note: {result['message']}")
    
    # Print tree structure as JSON
    print("\n" + "=" * 60)
    print("Tree structure (nested format):")
    print("=" * 60)
    with open("tree.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


    # Tree is already in frontend format - can be saved directly to database
