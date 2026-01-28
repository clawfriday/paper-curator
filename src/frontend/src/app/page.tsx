"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import type { RawNodeDatum, CustomNodeElementProps } from "react-d3-tree";

const Tree = dynamic(() => import("react-d3-tree"), { ssr: false });

interface PaperNode {
  name: string;
  node_id?: string;
  node_type?: string;
  children?: PaperNode[];
  attributes?: {
    arxivId?: string;
    title?: string;
    authors?: string[];
    summary?: string;
    pdfPath?: string;
    category?: string;
  };
}

interface IngestionStep {
  name: string;
  status: "pending" | "running" | "done" | "error";
  message?: string;
}

interface RepoResult {
  source: string;
  repo_url: string;
  repo_name: string;
  stars: number;
  is_official: boolean;
}

interface Reference {
  id: number;
  cited_title: string;
  cited_arxiv_id?: string;
  cited_authors?: string[];
  cited_year?: number;
  explanation?: string;
}

interface SimilarPaper {
  arxiv_id?: string;
  title: string;
  similarity?: number;
  authors?: string[];
  abstract?: string;
  year?: number;
  citation_count?: number;
  url?: string;
  source?: string;
}

interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  node: PaperNode | null;
}

interface UIConfig {
  hover_debounce_ms: number;
  max_similar_papers: number;
  tree_auto_save_interval_ms: number;
}

// Collapsible section component for structured summaries
function CollapsibleSection({ 
  title, 
  children, 
  defaultOpen = false 
}: { 
  title: string; 
  children: React.ReactNode; 
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  return (
    <div style={{ 
      marginBottom: "0.5rem", 
      border: "1px solid #e5e7eb", 
      borderRadius: "6px",
      overflow: "hidden",
    }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          width: "100%",
          padding: "0.5rem 0.75rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          backgroundColor: "#f9fafb",
          border: "none",
          cursor: "pointer",
          fontSize: "0.8rem",
          fontWeight: 600,
          color: "#374151",
          textAlign: "left",
        }}
      >
        <span>{title}</span>
        <span style={{ fontSize: "0.7rem", color: "#6b7280" }}>
          {isOpen ? "▼" : "▶"}
        </span>
      </button>
      {isOpen && (
        <div style={{ 
          padding: "0.75rem", 
          backgroundColor: "#fff",
          borderTop: "1px solid #e5e7eb",
        }}>
          {children}
        </div>
      )}
    </div>
  );
}

// Convert PaperNode to react-d3-tree compatible format
function toTreeData(node: PaperNode): RawNodeDatum {
  return {
    name: node.name,
    attributes: node.attributes
      ? {
          arxivId: node.attributes.arxivId || "",
          title: node.attributes.title || "",
          authors: node.attributes.authors?.join(", ") || "",
          category: node.attributes.category || "",
        }
      : undefined,
    children: node.children?.map(toTreeData),
  };
}

const initialTaxonomy: PaperNode = {
  name: "AI Papers",
  children: [],
};

export default function Home() {
  const [arxivUrl, setArxivUrl] = useState("");
  const [taxonomy, setTaxonomy] = useState<PaperNode>(initialTaxonomy);
  const [selectedNode, setSelectedNode] = useState<PaperNode | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [steps, setSteps] = useState<IngestionStep[]>([]);
  const [uiConfig, setUiConfig] = useState<UIConfig | null>(null);
  
  // Context menu state
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({
    visible: false,
    x: 0,
    y: 0,
    node: null,
  });
  
  // Feature panel states
  const [activePanel, setActivePanel] = useState<"details" | "repos" | "references" | "similar" | "query">("details");
  const [repos, setRepos] = useState<RepoResult[]>([]);
  const [queryInput, setQueryInput] = useState("");
  const [queryAnswer, setQueryAnswer] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [isReabbreviating, setIsReabbreviating] = useState(false);
  const [isLoadingStructured, setIsLoadingStructured] = useState(false);
  const [structuredAnalysis, setStructuredAnalysis] = useState<{
    components: string[];
    sections: Array<{
      component: string;
      steps: string;
      benefits: string;
      rationale: string;
      results: string;
    }>;
  } | null>(null);
  const [references, setReferences] = useState<Reference[]>([]);
  const [similarPapers, setSimilarPapers] = useState<SimilarPaper[]>([]);
  const [queryHistory, setQueryHistory] = useState<Array<{id: number; question: string; answer: string; created_at: string}>>([]);
  const [isLoadingFeature, setIsLoadingFeature] = useState(false);
  const [isLoadingCachedData, setIsLoadingCachedData] = useState(false);
  const [hoveredRefId, setHoveredRefId] = useState<number | null>(null);
  const [refExplanations, setRefExplanations] = useState<Record<number, string>>({});
  const [featureLog, setFeatureLog] = useState<string[]>([]);
  const [ingestLog, setIngestLog] = useState<string[]>([]);
  
  // Batch ingest state
  const [batchDirectory, setBatchDirectory] = useState("");
  const [isBatchIngesting, setIsBatchIngesting] = useState(false);
  const [batchResults, setBatchResults] = useState<{
    total: number;
    success: number;
    skipped: number;
    errors: number;
    results: Array<{ file: string; status: string; reason?: string; title?: string; category?: string }>;
  } | null>(null);
  
  // Rebalance state
  const [isRebalancing, setIsRebalancing] = useState(false);
  const [rebalanceResult, setRebalanceResult] = useState<{
    message: string;
    categories_processed?: string[];
    reclassified: Array<{ arxiv_id: string; old_category: string; new_category: string }>;
  } | null>(null);
  
  // Collapsible sections
  const [isIngestExpanded, setIsIngestExpanded] = useState(true);
  const [isDetailsExpanded, setIsDetailsExpanded] = useState(true);
  
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Helper to add to feature log
  const logFeature = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setFeatureLog((prev) => [...prev, `[${timestamp}] ${message}`]);
  };
  
  const clearFeatureLog = () => setFeatureLog([]);
  
  // Helper to add to ingest log
  const logIngest = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setIngestLog((prev) => [...prev, `[${timestamp}] ${message}`]);
  };
  
  const clearIngestLog = () => setIngestLog([]);

  // Load config and tree on mount
  useEffect(() => {
    const loadInitialData = async () => {
      // Load UI config
      try {
        const configRes = await fetch("/api/config");
        if (configRes.ok) {
          const config = await configRes.json();
          setUiConfig(config);
        }
      } catch (e) {
        console.error("Failed to load config:", e);
      }
      
      // Load tree from database
      try {
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          if (treeData.children && treeData.children.length > 0) {
            setTaxonomy(treeData);
          }
        }
      } catch (e) {
        console.error("Failed to load tree:", e);
      }
    };
    
    loadInitialData();
  }, []);

  // Close context menu on click outside
  useEffect(() => {
    const handleClick = () => setContextMenu((prev) => ({ ...prev, visible: false }));
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  const updateStep = (index: number, update: Partial<IngestionStep>) => {
    setSteps((prev) => prev.map((s, i) => (i === index ? { ...s, ...update } : s)));
  };

  // Get existing categories from the tree
  const existingCategories = useMemo(() => {
    return taxonomy.children?.map((c) => c.name) || [];
  }, [taxonomy]);

  const addPaperToTree = useCallback(
    (paper: {
      arxivId: string;
      title: string;
      authors: string[];
      summary: string;
      pdfPath?: string;
      category: string;
      abbreviation?: string;
    }) => {
      setTaxonomy((prev) => {
        const newTree = JSON.parse(JSON.stringify(prev)) as PaperNode;
        let categoryNode = newTree.children?.find((c) => c.name === paper.category);
        if (!categoryNode) {
          categoryNode = { name: paper.category, children: [], node_type: "category" };
          newTree.children = newTree.children || [];
          newTree.children.push(categoryNode);
        }
        categoryNode.children = categoryNode.children || [];
        // Use abbreviation if provided, otherwise truncate title
        const displayName = paper.abbreviation || (paper.title.length > 20 ? paper.title.slice(0, 18) + ".." : paper.title);
        categoryNode.children.push({
          name: displayName,
          node_type: "paper",
          attributes: {
            arxivId: paper.arxivId,
            title: paper.title,
            authors: paper.authors,
            summary: paper.summary,
            pdfPath: paper.pdfPath,
            category: paper.category,
          },
        });
        return newTree;
      });
    },
    []
  );

  const handleRebalanceCategories = async () => {
    setIsRebalancing(true);
    setRebalanceResult(null);
    clearIngestLog();
    logIngest("Rebalancing crowded categories...");
    
    try {
      const res = await fetch("/api/categories/rebalance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      
      if (res.ok) {
        const data = await res.json();
        setRebalanceResult(data);
        logIngest(`✓ ${data.message}`);
        if (data.reclassified?.length > 0) {
          logIngest(`Moved ${data.reclassified.length} papers to new categories`);
          // Reload tree
          const treeRes = await fetch("/api/tree");
          if (treeRes.ok) {
            const treeData = await treeRes.json();
            setTaxonomy(treeData.tree);
          }
        }
      } else {
        const errText = await res.text();
        logIngest(`Error: HTTP ${res.status}`);
        logIngest(`Details: ${errText.slice(0, 200)}`);
      }
    } catch (e) {
      logIngest(`Error: ${e}`);
    } finally {
      setIsRebalancing(false);
      logIngest("Done");
    }
  };

  const handleBatchIngest = async () => {
    if (!batchDirectory.trim()) return;
    
    setIsBatchIngesting(true);
    setBatchResults(null);
    clearIngestLog();
    logIngest(`Starting batch ingest from: ${batchDirectory.trim()}`);
    
    try {
      const res = await fetch("/api/papers/batch-ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ directory: batchDirectory.trim(), skip_existing: true }),
      });
      
      if (!res.ok) {
        const errText = await res.text();
        logIngest(`Error: HTTP ${res.status}`);
        try {
          const errJson = JSON.parse(errText);
          logIngest(`Details: ${errJson.detail || errText}`);
        } catch {
          logIngest(`Details: ${errText.slice(0, 200)}`);
        }
        setIsBatchIngesting(false);
        return;
      }
      
      const data = await res.json();
      setBatchResults(data);
      logIngest(`Completed: ${data.success} success, ${data.skipped} skipped, ${data.errors} errors`);
      
      // Refresh tree
      const treeRes = await fetch("/api/tree");
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        setTaxonomy(treeData);
        logIngest("Tree refreshed");
      }
    } catch (err) {
      logIngest(`Error: ${err}`);
    } finally {
      setIsBatchIngesting(false);
    }
  };

  const handleIngest = async () => {
    if (!arxivUrl.trim()) return;

    setIsIngesting(true);
    clearIngestLog();
    logIngest(`Starting ingestion for: ${arxivUrl.trim()}`);
    setSteps([
      { name: "Resolve arXiv", status: "pending" },
      { name: "Download PDF", status: "pending" },
      { name: "Extract text", status: "pending" },
      { name: "Classify (LLM)", status: "pending" },
      { name: "Abbreviate (LLM)", status: "pending" },
      { name: "Summarize (LLM)", status: "pending" },
      { name: "Save", status: "pending" },
    ]);

    let arxivId = "";
    let title = "";
    let authors: string[] = [];
    let abstract = "";
    let pdfPath = "";
    let latexPath = "";
    let pdfUrl = "";
    let published = "";
    let category = "";
    let abbreviation = "";
    let summary = "";

    // Step 1: Resolve
    updateStep(0, { status: "running" });
    logIngest("Resolving arXiv metadata...");
    const resolveRes = await fetch("/api/arxiv/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: arxivUrl }),
    });
    if (!resolveRes.ok) {
      const errText = await resolveRes.text();
      logIngest(`Error: HTTP ${resolveRes.status} - ${errText}`);
      updateStep(0, { status: "error", message: `HTTP ${resolveRes.status}` });
      setIsIngesting(false);
      return;
    }
    const resolveData = await resolveRes.json();
    arxivId = resolveData.arxiv_id;
    title = resolveData.title;
    authors = resolveData.authors;
    abstract = resolveData.summary;
    pdfUrl = resolveData.pdf_url;
    published = resolveData.published;
    logIngest(`Found: ${title}`);
    logIngest(`arXiv ID: ${arxivId}`);
    updateStep(0, { status: "done", message: title });

    // Step 2: Download
    updateStep(1, { status: "running" });
    logIngest("Downloading PDF...");
    const downloadRes = await fetch("/api/arxiv/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: arxivId }),
    });
    if (!downloadRes.ok) {
      const errText = await downloadRes.text();
      logIngest(`Error: HTTP ${downloadRes.status} - ${errText}`);
      updateStep(1, { status: "error", message: `HTTP ${downloadRes.status}` });
      setIsIngesting(false);
      return;
    }
    const downloadData = await downloadRes.json();
    pdfPath = downloadData.pdf_path;
    latexPath = downloadData.latex_path;
    logIngest(`PDF saved to: ${pdfPath}`);
    updateStep(1, { status: "done", message: "PDF downloaded" });

    // Steps 3-5: Extract + Classify + Abbreviate in PARALLEL
    // These are independent: extract uses PDF, classify/abbreviate use title/abstract
    logIngest("Running Extract + Classify + Abbreviate in parallel...");
    updateStep(2, { status: "running" });
    updateStep(3, { status: "running", message: "Determining category..." });
    updateStep(4, { status: "running", message: "Creating short name..." });

    const [extractRes, classifyRes, abbreviateRes] = await Promise.all([
      fetch("/api/pdf/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pdf_path: pdfPath }),
      }),
      fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          abstract,
          existing_categories: existingCategories,
        }),
      }),
      fetch("/api/abbreviate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }),
    ]);

    // Handle Extract result
    if (!extractRes.ok) {
      updateStep(2, { status: "error", message: `HTTP ${extractRes.status}` });
      setIsIngesting(false);
      return;
    }
    updateStep(2, { status: "done", message: "Text extracted" });

    // Handle Classify result
    if (!classifyRes.ok) {
      updateStep(3, { status: "error", message: `HTTP ${classifyRes.status}` });
      setIsIngesting(false);
      return;
    }
    const classifyData = await classifyRes.json();
    category = classifyData.category;
    updateStep(3, { status: "done", message: category });

    // Handle Abbreviate result
    if (abbreviateRes.ok) {
      const abbreviateData = await abbreviateRes.json();
      abbreviation = abbreviateData.abbreviation;
      updateStep(4, { status: "done", message: abbreviation });
    } else {
      // Fallback to truncated title
      abbreviation = title.length > 20 ? title.slice(0, 18) + ".." : title;
      updateStep(4, { status: "done", message: abbreviation });
    }

    // Step 6: Summarize (quick single query, also indexes PDF for QA)
    logIngest("Summarizing paper (may take ~30s)...");
    updateStep(5, { status: "running", message: "May take a minute..." });
    const summarizeRes = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdf_path: pdfPath, arxiv_id: arxivId }),
    });
    if (!summarizeRes.ok) {
      const errText = await summarizeRes.text();
      logIngest(`Error: HTTP ${summarizeRes.status}`);
      try {
        const errJson = JSON.parse(errText);
        logIngest(`Details: ${errJson.detail || errText}`);
      } catch {
        logIngest(`Details: ${errText.slice(0, 200)}`);
      }
      updateStep(5, { status: "error", message: `HTTP ${summarizeRes.status}` });
      setIsIngesting(false);
      return;
    }
    const summarizeData = await summarizeRes.json();
    summary = summarizeData.summary;
    logIngest("Summary generated successfully");
    updateStep(5, { status: "done", message: "Done" });

    // Step 7: Save to database
    updateStep(6, { status: "running" });
    const saveRes = await fetch("/api/papers/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        arxiv_id: arxivId,
        title,
        authors,
        abstract,
        summary,
        pdf_path: pdfPath,
        latex_path: latexPath,
        pdf_url: pdfUrl,
        published_at: published,
        category,
        abbreviation,
      }),
    });
    if (!saveRes.ok) {
      updateStep(6, { status: "error", message: `HTTP ${saveRes.status}` });
    } else {
      updateStep(6, { status: "done", message: "Saved" });
    }

    // Add to tree
    addPaperToTree({
      arxivId,
      title,
      authors,
      summary,
      pdfPath,
      category,
      abbreviation,
    });

    // Prefetch auxiliary data in background (don't wait)
    fetch("/api/papers/prefetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: arxivId, title }),
    }).catch(() => {}); // Ignore errors, this is optional

    setArxivUrl("");
    setIsIngesting(false);
  };

  // Convert taxonomy to tree-compatible format
  const treeData = useMemo(() => toTreeData(taxonomy), [taxonomy]);

  // Find original node by name for details display
  const findNode = useCallback((tree: PaperNode, name: string): PaperNode | null => {
    if (tree.name === name) return tree;
    for (const child of tree.children || []) {
      const found = findNode(child, name);
      if (found) return found;
    }
    return null;
  }, []);

  const handleNodeClick = useCallback(async (nodeName: string) => {
    const node = findNode(taxonomy, nodeName);
    setSelectedNode(node);
    setActivePanel("details");
    setStructuredAnalysis(null);
    
    // Reset states
    setRepos([]);
    setReferences([]);
    setSimilarPapers([]);
    setQueryHistory([]);
    setRefExplanations({});
    
    // Load cached data if this is a paper node
    if (node?.attributes?.arxivId) {
      setIsLoadingCachedData(true);
      try {
        const res = await fetch("/api/papers/cached-data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ arxiv_id: node.attributes.arxivId }),
        });
        if (res.ok) {
          const data = await res.json();
          // Restore cached data
          if (data.repos?.length > 0) setRepos(data.repos);
          if (data.references?.length > 0) {
            setReferences(data.references);
            // Pre-populate cached explanations
            const cached: Record<number, string> = {};
            for (const ref of data.references) {
              if (ref.explanation) cached[ref.id] = ref.explanation;
            }
            setRefExplanations(cached);
          }
          if (data.similar_papers?.length > 0) setSimilarPapers(data.similar_papers);
          if (data.queries?.length > 0) setQueryHistory(data.queries);
          // Restore structured analysis if available
          if (data.structured_summary) {
            setStructuredAnalysis(data.structured_summary);
          }
        }
      } catch (e) {
        console.error("Failed to load cached data:", e);
      } finally {
        setIsLoadingCachedData(false);
      }
    }
  }, [taxonomy, findNode]);

  const handleNodeRightClick = useCallback((event: React.MouseEvent, nodeName: string) => {
    event.preventDefault();
    event.stopPropagation();
    const node = findNode(taxonomy, nodeName);
    if (node?.attributes?.arxivId) {
      setContextMenu({
        visible: true,
        x: event.clientX,
        y: event.clientY,
        node,
      });
    }
  }, [taxonomy, findNode]);

  // Feature handlers
  const handleFindRepos = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) {
      logFeature("Error: No arXiv ID found for this paper");
      return;
    }
    clearFeatureLog();
    setSelectedNode(node);
    setActivePanel("repos");
    setIsLoadingFeature(true);
    setRepos([]);
    
    const arxivId = node.attributes.arxivId;
    const title = node.attributes.title || node.name;
    
    logFeature(`Starting GitHub repo search for: ${arxivId}`);
    logFeature(`Paper title: ${title}`);
    logFeature("Step 1: Querying Papers With Code API...");
    
    try {
      const res = await fetch("/api/repos/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId, title }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const repos = data.repos || [];
        setRepos(repos);
        
        if (data.from_cache) {
          logFeature("Retrieved from cache");
        } else {
          logFeature("Step 2: Querying GitHub Search API...");
        }
        
        if (repos.length > 0) {
          logFeature(`✓ Found ${repos.length} repositories`);
          const official = repos.filter((r: RepoResult) => r.is_official);
          if (official.length > 0) {
            logFeature(`  - ${official.length} official repo(s)`);
          }
        } else {
          logFeature("✗ No repositories found");
        }
      } else {
        logFeature(`✗ Error: HTTP ${res.status}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      console.error("Failed to fetch repos:", e);
    } finally {
      setIsLoadingFeature(false);
      logFeature("Done");
    }
  };

  const handleFetchReferences = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) {
      logFeature("Error: No arXiv ID found for this paper");
      return;
    }
    clearFeatureLog();
    setSelectedNode(node);
    setActivePanel("references");
    setIsLoadingFeature(true);
    setReferences([]);
    setRefExplanations({});
    
    const arxivId = node.attributes.arxivId;
    logFeature(`Starting reference extraction for: ${arxivId}`);
    logFeature("Step 1: Querying Semantic Scholar API...");
    
    try {
      const res = await fetch("/api/references/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const refs = data.references || [];
        setReferences(refs);
        
        if (data.from_cache) {
          logFeature("Retrieved from cache");
        }
        
        // Pre-populate cached explanations
        const cached: Record<number, string> = {};
        for (const ref of refs) {
          if (ref.explanation) {
            cached[ref.id] = ref.explanation;
          }
        }
        setRefExplanations(cached);
        
        if (refs.length > 0) {
          logFeature(`✓ Found ${refs.length} references`);
          const withArxiv = refs.filter((r: Reference) => r.cited_arxiv_id);
          logFeature(`  - ${withArxiv.length} with arXiv IDs`);
        } else {
          logFeature("✗ No references found");
        }
      } else {
        logFeature(`✗ Error: HTTP ${res.status}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      console.error("Failed to fetch references:", e);
    } finally {
      setIsLoadingFeature(false);
      logFeature("Hover over references for LLM explanations");
      logFeature("Done");
    }
  };

  const handleFindSimilar = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) {
      logFeature("Error: No arXiv ID found for this paper");
      return;
    }
    clearFeatureLog();
    setSelectedNode(node);
    setActivePanel("similar");
    setIsLoadingFeature(true);
    setSimilarPapers([]);
    
    const arxivId = node.attributes.arxivId;
    logFeature(`Starting similar paper search for: ${arxivId}`);
    logFeature("Step 1: Querying Semantic Scholar Recommendations API...");
    logFeature("Searching 200M+ papers on the internet...");
    
    try {
      const res = await fetch("/api/papers/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const papers = data.similar_papers || [];
        setSimilarPapers(papers);
        
        if (data.from_cache) {
          logFeature("Retrieved from cache");
        }
        
        if (papers.length > 0) {
          logFeature(`✓ Found ${papers.length} similar papers`);
          const withArxiv = papers.filter((p: SimilarPaper) => p.arxiv_id);
          logFeature(`  - ${withArxiv.length} available on arXiv`);
        } else {
          logFeature("✗ No similar papers found");
        }
      } else {
        logFeature(`✗ Error: HTTP ${res.status}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      console.error("Failed to find similar papers:", e);
    } finally {
      setIsLoadingFeature(false);
      logFeature("Done");
    }
  };

  // Reference hover handler with debounce
  const handleRefHover = (ref: Reference) => {
    if (refExplanations[ref.id]) {
      setHoveredRefId(ref.id);
      return;
    }
    
    const debounceMs = uiConfig?.hover_debounce_ms || 500;
    
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    
    hoverTimeoutRef.current = setTimeout(async () => {
      setHoveredRefId(ref.id);
      
      try {
        const res = await fetch("/api/references/explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            reference_id: ref.id,
            source_paper_title: selectedNode?.attributes?.title || "",
            cited_title: ref.cited_title,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          setRefExplanations((prev) => ({ ...prev, [ref.id]: data.explanation }));
        }
      } catch (e) {
        console.error("Failed to get explanation:", e);
      }
    }, debounceMs);
  };

  const handleRefLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    setHoveredRefId(null);
  };

  const handleAddSimilarPaper = async (paper: SimilarPaper) => {
    if (!paper.arxiv_id) return;
    // Auto-trigger ingestion
    setArxivUrl(paper.arxiv_id);
    setActivePanel("details");
    // Trigger ingestion after a short delay to allow state update
    setTimeout(() => {
      const ingestBtn = document.getElementById("ingest-btn");
      if (ingestBtn) ingestBtn.click();
    }, 100);
  };

  const handleAddReference = async (ref: Reference) => {
    if (!ref.cited_arxiv_id) return;
    // Auto-trigger ingestion
    setArxivUrl(ref.cited_arxiv_id);
    setActivePanel("details");
    // Trigger ingestion after a short delay to allow state update
    setTimeout(() => {
      const ingestBtn = document.getElementById("ingest-btn");
      if (ingestBtn) ingestBtn.click();
    }, 100);
  };

  const handleQuery = async () => {
    if (!selectedNode?.attributes?.arxivId || !queryInput.trim()) return;
    setIsQuerying(true);
    setQueryAnswer("");
    clearFeatureLog();
    const question = queryInput.trim();
    logFeature(`Querying: "${question}"`);
    logFeature("Searching paper for answer...");
    
    try {
      const res = await fetch("/api/qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arxiv_id: selectedNode.attributes.arxivId,
          pdf_path: selectedNode.attributes.pdfPath || `storage/downloads/${selectedNode.attributes.arxivId}.pdf`,
          question: question,
          context: "",
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const answer = data.answer || "No answer found.";
        setQueryAnswer(answer);
        logFeature(`✓ Answer generated${data.used_cache ? " (cached index)" : ""}`);
        
        // Add to query history (backend already persisted it)
        setQueryHistory(prev => [{
          id: Date.now(), // Temporary ID, actual ID is in DB
          question: question,
          answer: answer,
          created_at: new Date().toISOString(),
        }, ...prev]);
        
        // Clear input for next question
        setQueryInput("");
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status}`);
        setQueryAnswer(`Error: ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      setQueryAnswer(`Error: ${e}`);
    } finally {
      setIsQuerying(false);
      logFeature("Done");
    }
  };

  const handleReabbreviate = async () => {
    if (!selectedNode?.attributes?.arxivId) return;
    setIsReabbreviating(true);
    clearFeatureLog();
    logFeature(`Re-generating abbreviation for ${selectedNode.attributes.arxivId}...`);
    
    try {
      const res = await fetch("/api/papers/reabbreviate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: selectedNode.attributes.arxivId }),
      });
      if (res.ok) {
        const data = await res.json();
        logFeature(`✓ New abbreviation: ${data.abbreviation}`);
        // Reload tree to reflect changes
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          setTaxonomy(treeData);
        }
      } else {
        logFeature(`✗ Error: ${res.status}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    } finally {
      setIsReabbreviating(false);
      logFeature("Done");
    }
  };

  const handleStructuredAnalysis = async () => {
    if (!selectedNode?.attributes?.arxivId) return;
    setIsLoadingStructured(true);
    setStructuredAnalysis(null);
    clearFeatureLog();
    logFeature(`Running detailed analysis for ${selectedNode.attributes.arxivId}...`);
    logFeature("This may take several minutes...");
    
    try {
      const res = await fetch("/api/qa/structured", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          arxiv_id: selectedNode.attributes.arxivId,
          pdf_path: selectedNode.attributes.pdfPath,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setStructuredAnalysis(data);
        logFeature(`✓ Analysis complete: ${data.components.length} components`);
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status}`);
        try {
          const errJson = JSON.parse(errText);
          logFeature(`Details: ${errJson.detail || errText}`);
        } catch {
          logFeature(`Details: ${errText.slice(0, 200)}`);
        }
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    } finally {
      setIsLoadingStructured(false);
      logFeature("Done");
    }
  };

  const getStepIcon = (status: IngestionStep["status"]) => {
    switch (status) {
      case "pending": return "○";
      case "running": return "◐";
      case "done": return "●";
      case "error": return "✕";
    }
  };

  const getStepColor = (status: IngestionStep["status"]) => {
    switch (status) {
      case "pending": return "#999";
      case "running": return "#0070f3";
      case "done": return "#10b981";
      case "error": return "#ef4444";
    }
  };

  // Custom node renderer with right-click support - compact display
  const renderCustomNode = useCallback(({ nodeDatum }: CustomNodeElementProps) => {
    const hasArxivId = nodeDatum.attributes && nodeDatum.attributes.arxivId;
    const isCategory = !nodeDatum.attributes || !nodeDatum.attributes.arxivId;
    const isRoot = nodeDatum.name === "AI Papers";
    
    // Show full name, wrap into multiple lines if needed
    const name = nodeDatum.name;
    const maxCharsPerLine = 14;
    const lines: string[] = [];
    
    // Simple word wrap
    const words = name.split(/\s+/);
    let currentLine = "";
    for (const word of words) {
      if ((currentLine + " " + word).trim().length <= maxCharsPerLine) {
        currentLine = (currentLine + " " + word).trim();
      } else {
        if (currentLine) lines.push(currentLine);
        currentLine = word.length > maxCharsPerLine ? word.slice(0, maxCharsPerLine - 2) + ".." : word;
      }
    }
    if (currentLine) lines.push(currentLine);
    
    // Limit to 2 lines max
    if (lines.length > 2) {
      lines.splice(2);
      lines[1] = lines[1].slice(0, maxCharsPerLine - 2) + "..";
    }
    
    // Node sizing based on content
    const fontSize = 11;
    const lineHeight = 15;
    const paddingX = 14;
    const paddingY = 8;
    const nodeWidth = Math.max(90, Math.min(130, Math.max(...lines.map(l => l.length)) * 7.5 + paddingX * 2));
    const nodeHeight = lines.length * lineHeight + paddingY * 2;
    
    // Colors - high contrast: light backgrounds with dark text
    const fillColor = isRoot ? "#1f2937" : (isCategory ? "#e0e7ff" : "#dbeafe");
    const strokeColor = isRoot ? "#111827" : (isCategory ? "#6366f1" : "#3b82f6");
    const textColor = isRoot ? "#ffffff" : (isCategory ? "#312e81" : "#1e3a8a");
    
    return (
      <g
        style={{ cursor: "pointer" }}
        onClick={() => handleNodeClick(nodeDatum.name)}
        onContextMenu={(e) => {
          if (hasArxivId) {
            handleNodeRightClick(e, nodeDatum.name);
          }
        }}
      >
        {/* Rounded rectangle node */}
        <rect
          x={-nodeWidth / 2}
          y={-nodeHeight / 2}
          width={nodeWidth}
          height={nodeHeight}
          rx={8}
          ry={8}
          fill={fillColor}
          stroke={strokeColor}
          strokeWidth={2}
        />
        {/* Multi-line text with sharp rendering */}
        {lines.map((line, i) => (
          <text
            key={i}
            fill={textColor}
            textAnchor="middle"
            x={0}
            y={-((lines.length - 1) * lineHeight) / 2 + i * lineHeight + 4}
            style={{ 
              fontSize: `${fontSize}px`, 
              fontWeight: 500,
              fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
              pointerEvents: "none",
              textRendering: "optimizeLegibility",
            }}
          >
            {line}
          </text>
        ))}
      </g>
    );
  }, [handleNodeClick, handleNodeRightClick]);

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Left panel: Tree visualization */}
      <div style={{ flex: 2, borderRight: "1px solid #e5e5e5", display: "flex", flexDirection: "column" }}>
        <style>{`
          .tree-link {
            stroke: #94a3b8 !important;
            stroke-width: 1.5px !important;
            fill: none !important;
          }
          .rd3t-tree-container {
            background: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%);
          }
          .rd3t-tree-container svg text {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            text-rendering: optimizeLegibility;
          }
          .tree-container {
            overflow: auto !important;
          }
          .tree-container::-webkit-scrollbar {
            width: 10px;
            height: 10px;
          }
          .tree-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 5px;
          }
          .tree-container::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 5px;
          }
          .tree-container::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
          }
        `}</style>
        <div style={{ padding: "1rem", borderBottom: "1px solid #e5e5e5", backgroundColor: "#fafafa" }}>
          <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 600 }}>Paper Curator</h1>
          <p style={{ margin: "0.25rem 0 0", fontSize: "0.875rem", color: "#666" }}>
            {taxonomy.children?.length || 0} categories, {" "}
            {taxonomy.children?.reduce((acc, c) => acc + (c.children?.length || 0), 0) || 0} papers
          </p>
        </div>
        <div 
          className="tree-container"
          style={{ flex: 1, position: "relative", overflow: "auto" }}
          onContextMenu={(e) => e.preventDefault()}
        >
          {taxonomy.children && taxonomy.children.length > 0 ? (
            <Tree
              data={treeData}
              orientation="vertical"
              pathFunc="diagonal"
              translate={{ x: 300, y: 50 }}
              nodeSize={{ x: 140, y: 80 }}
              separation={{ siblings: 1.0, nonSiblings: 1.3 }}
              renderCustomNodeElement={renderCustomNode}
              pathClassFunc={() => "tree-link"}
              zoomable={true}
              scaleExtent={{ min: 0.1, max: 3 }}
              draggable={true}
            />
          ) : (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#999" }}>
              <p>No papers yet. Add one using the panel on the right.</p>
            </div>
          )}
        </div>
      </div>

      {/* Context Menu */}
      {contextMenu.visible && contextMenu.node && (
        <div
          style={{
            position: "fixed",
            top: contextMenu.y,
            left: contextMenu.x,
            backgroundColor: "white",
            border: "1px solid #e5e5e5",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            zIndex: 1000,
            minWidth: "200px",
            overflow: "hidden",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{ padding: "0.5rem 1rem", borderBottom: "1px solid #eee", backgroundColor: "#f9f9f9" }}>
            <strong style={{ fontSize: "0.75rem", color: "#666" }}>
              {contextMenu.node.attributes?.arxivId}
            </strong>
          </div>
          <div
            style={{ padding: "0.75rem 1rem", cursor: "pointer", display: "flex", alignItems: "center", gap: "0.5rem" }}
            onClick={() => { handleFindRepos(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f5f5f5")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            <span>🔗</span> Find GitHub Repo
          </div>
          <div
            style={{ padding: "0.75rem 1rem", cursor: "pointer", display: "flex", alignItems: "center", gap: "0.5rem" }}
            onClick={() => { handleFetchReferences(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f5f5f5")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            <span>📚</span> Explain References
          </div>
          <div
            style={{ padding: "0.75rem 1rem", cursor: "pointer", display: "flex", alignItems: "center", gap: "0.5rem" }}
            onClick={() => { handleFindSimilar(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f5f5f5")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            <span>🔍</span> Find Similar Papers
          </div>
        </div>
      )}

      {/* Right panel: Details and ingest */}
      <div style={{ flex: 1, padding: "1.5rem", display: "flex", flexDirection: "column", backgroundColor: "#fafafa", overflowY: "auto" }}>
        {/* Ingest section - Collapsible */}
        <div style={{ marginBottom: "1rem", backgroundColor: "white", borderRadius: "8px", border: "1px solid #e5e5e5", overflow: "hidden" }}>
          <div 
            onClick={() => setIsIngestExpanded(!isIngestExpanded)}
            style={{ 
              padding: "0.75rem 1rem", 
              display: "flex", 
              justifyContent: "space-between", 
              alignItems: "center",
              cursor: "pointer",
              backgroundColor: "#f9f9f9",
              borderBottom: isIngestExpanded ? "1px solid #e5e5e5" : "none",
            }}
          >
            <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 600 }}>📥 Ingest Paper</h2>
            <span style={{ fontSize: "0.875rem", color: "#666" }}>{isIngestExpanded ? "▼" : "▶"}</span>
          </div>
          {isIngestExpanded && (
            <div style={{ padding: "1rem" }}>
              <input
                type="text"
                value={arxivUrl}
                onChange={(e) => setArxivUrl(e.target.value)}
                placeholder="arXiv URL or ID (e.g., 1706.03762)"
                disabled={isIngesting}
                style={{
                  width: "100%",
                  padding: "0.5rem",
                  marginBottom: "0.5rem",
                  boxSizing: "border-box",
                  border: "1px solid #ddd",
                  borderRadius: "4px",
                  fontSize: "0.875rem",
                }}
              />
              <button
                id="ingest-btn"
                onClick={handleIngest}
                disabled={isIngesting || !arxivUrl.trim()}
                style={{
                  width: "100%",
                  padding: "0.5rem",
                  backgroundColor: isIngesting ? "#ccc" : "#0070f3",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: isIngesting ? "not-allowed" : "pointer",
                  fontSize: "0.875rem",
                  fontWeight: 500,
                }}
              >
                {isIngesting ? "Ingesting..." : "Ingest"}
              </button>

              {/* Progress steps */}
              {steps.length > 0 && (
                <div style={{ marginTop: "0.75rem" }}>
                  {steps.map((step, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "flex-start", marginBottom: "0.25rem" }}>
                      <span style={{ color: getStepColor(step.status), marginRight: "0.375rem", fontSize: "0.75rem" }}>
                        {getStepIcon(step.status)}
                      </span>
                      <div style={{ flex: 1 }}>
                        <span style={{ fontSize: "0.75rem", color: step.status === "error" ? "#ef4444" : "#333" }}>
                          {step.name}
                        </span>
                        {step.message && (
                          <p style={{ margin: "0.125rem 0 0", fontSize: "0.625rem", color: step.status === "error" ? "#ef4444" : "#666" }}>
                            {step.message}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {/* Batch Ingest */}
              <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid #e5e5e5" }}>
                <h3 style={{ margin: "0 0 0.5rem", fontSize: "0.875rem", fontWeight: 500, color: "#666" }}>
                  📁 Batch Ingest (Local PDFs)
                </h3>
                <input
                  type="text"
                  value={batchDirectory}
                  onChange={(e) => setBatchDirectory(e.target.value)}
                  placeholder="Directory path (e.g., /Users/you/papers)"
                  disabled={isBatchIngesting}
                  style={{
                    width: "100%",
                    padding: "0.5rem",
                    marginBottom: "0.5rem",
                    boxSizing: "border-box",
                    border: "1px solid #ddd",
                    borderRadius: "4px",
                    fontSize: "0.75rem",
                  }}
                />
                <button
                  onClick={handleBatchIngest}
                  disabled={isBatchIngesting || !batchDirectory.trim()}
                  style={{
                    width: "100%",
                    padding: "0.5rem",
                    backgroundColor: isBatchIngesting ? "#ccc" : "#10b981",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: isBatchIngesting ? "not-allowed" : "pointer",
                    fontSize: "0.75rem",
                    fontWeight: 500,
                  }}
                >
                  {isBatchIngesting ? "Processing..." : "Batch Ingest"}
                </button>
                
                {/* Batch results */}
                {batchResults && (
                  <div style={{ marginTop: "0.5rem", fontSize: "0.7rem" }}>
                    <div style={{ color: "#10b981" }}>✓ {batchResults.success} ingested</div>
                    {batchResults.skipped > 0 && (
                      <div style={{ color: "#f59e0b" }}>⏭ {batchResults.skipped} skipped (existing)</div>
                    )}
                    {batchResults.errors > 0 && (
                      <div style={{ color: "#ef4444" }}>✗ {batchResults.errors} errors</div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Rebalance Categories */}
              <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid #e5e5e5" }}>
                <h3 style={{ margin: "0 0 0.5rem", fontSize: "0.875rem", fontWeight: 500, color: "#666" }}>
                  ⚖️ Rebalance Categories
                </h3>
                <p style={{ fontSize: "0.7rem", color: "#888", marginBottom: "0.5rem" }}>
                  Reclassify papers in crowded categories (10+ papers)
                </p>
                <button
                  onClick={handleRebalanceCategories}
                  disabled={isRebalancing}
                  style={{
                    width: "100%",
                    padding: "0.5rem",
                    backgroundColor: isRebalancing ? "#ccc" : "#8b5cf6",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: isRebalancing ? "not-allowed" : "pointer",
                    fontSize: "0.75rem",
                    fontWeight: 500,
                  }}
                >
                  {isRebalancing ? "Rebalancing..." : "Rebalance Now"}
                </button>
                
                {/* Rebalance results */}
                {rebalanceResult && (
                  <div style={{ marginTop: "0.5rem", fontSize: "0.7rem" }}>
                    <div style={{ color: "#8b5cf6" }}>{rebalanceResult.message}</div>
                    {rebalanceResult.reclassified?.length > 0 && (
                      <div style={{ color: "#10b981", marginTop: "0.25rem" }}>
                        ✓ Moved {rebalanceResult.reclassified.length} papers
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Ingest log textbox */}
              {ingestLog.length > 0 && (
                <div
                  style={{
                    marginTop: "0.75rem",
                    backgroundColor: "#1e1e1e",
                    borderRadius: "4px",
                    padding: "0.5rem",
                    maxHeight: "150px",
                    overflowY: "auto",
                    fontFamily: "monospace",
                    fontSize: "0.7rem",
                  }}
                >
                  {ingestLog.map((log, i) => (
                    <div key={i} style={{ color: log.includes("Error") ? "#f87171" : "#a3e635", marginBottom: "2px" }}>
                      {log}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Paper Details Section - Collapsible */}
        <div style={{ flex: 1, backgroundColor: "white", borderRadius: "8px", border: "1px solid #e5e5e5", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div 
            onClick={() => setIsDetailsExpanded(!isDetailsExpanded)}
            style={{ 
              padding: "0.75rem 1rem", 
              display: "flex", 
              justifyContent: "space-between", 
              alignItems: "center",
              cursor: "pointer",
              backgroundColor: "#f9f9f9",
              borderBottom: isDetailsExpanded ? "1px solid #e5e5e5" : "none",
            }}
          >
            <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 600 }}>
              📄 {selectedNode ? (selectedNode.attributes?.title ? selectedNode.name : selectedNode.name) : "Paper Details"}
            </h2>
            <span style={{ fontSize: "0.875rem", color: "#666" }}>{isDetailsExpanded ? "▼" : "▶"}</span>
          </div>
          
          {isDetailsExpanded && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              {/* Panel tabs */}
              {selectedNode && (
                <div style={{ display: "flex", padding: "0.5rem", gap: "0.25rem", borderBottom: "1px solid #eee", flexWrap: "wrap" }}>
                  {["details", "repos", "refs", "similar", "query"].map((panel) => (
                    <button
                      key={panel}
                      onClick={() => setActivePanel(panel === "refs" ? "references" : panel as any)}
                      style={{
                        flex: 1,
                        padding: "0.375rem",
                        backgroundColor: (activePanel === panel || (panel === "refs" && activePanel === "references")) ? "#0070f3" : "#e5e5e5",
                        color: (activePanel === panel || (panel === "refs" && activePanel === "references")) ? "white" : "#333",
                        border: "none",
                        borderRadius: "4px",
                        cursor: "pointer",
                        fontSize: "0.625rem",
                        textTransform: "capitalize",
                      }}
                    >
                      {panel}
                    </button>
                  ))}
                </div>
              )}

              {/* Feature Progress Log */}
              {featureLog.length > 0 && (
                <div style={{ 
                  margin: "0.5rem", 
                  backgroundColor: "#1a1a2e", 
                  padding: "0.5rem", 
                  borderRadius: "6px", 
                  fontFamily: "monospace",
                  fontSize: "0.625rem",
                  maxHeight: "100px",
                  overflowY: "auto",
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" }}>
                    <span style={{ color: "#10b981", fontWeight: 600 }}>Log</span>
                    <button 
                      onClick={clearFeatureLog}
                      style={{ 
                        background: "none", 
                        border: "none", 
                        color: "#666", 
                        cursor: "pointer",
                        fontSize: "0.5rem",
                      }}
                    >
                      Clear
                    </button>
                  </div>
                  {featureLog.map((log, i) => (
                    <div key={i} style={{ 
                      color: log.includes("✓") ? "#10b981" : log.includes("✗") ? "#ef4444" : log.includes("Error") ? "#ef4444" : "#e5e5e5",
                      lineHeight: 1.4,
                    }}>
                      {log}
                    </div>
                  ))}
                </div>
              )}

              {/* Dynamic panel content */}
              <div style={{ flex: 1, padding: "0.75rem", overflowY: "auto" }}>
          {/* Details Panel */}
          {activePanel === "details" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>Paper Details</h2>
              {selectedNode?.attributes ? (
                <div>
                  <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>{selectedNode.attributes.title || selectedNode.name}</h3>
                  {selectedNode.attributes.arxivId && (
                    <p style={{ fontSize: "0.875rem", color: "#666", margin: "0 0 0.5rem" }}>
                      <strong>arXiv:</strong>{" "}
                      <a href={`https://arxiv.org/abs/${selectedNode.attributes.arxivId}`} target="_blank" rel="noopener noreferrer" style={{ color: "#0070f3" }}>
                        {selectedNode.attributes.arxivId}
                      </a>
                    </p>
                  )}
                  {selectedNode.attributes.authors && (
                    <p style={{ fontSize: "0.875rem", color: "#666", margin: "0 0 0.5rem" }}>
                      <strong>Authors:</strong> {selectedNode.attributes.authors.slice(0, 3).join(", ")}
                      {selectedNode.attributes.authors.length > 3 && ` +${selectedNode.attributes.authors.length - 3} more`}
                    </p>
                  )}
                  {selectedNode.attributes.category && (
                    <p style={{ fontSize: "0.875rem", color: "#666", margin: "0 0 1rem" }}>
                      <strong>Category:</strong> {selectedNode.attributes.category}
                    </p>
                  )}
                  {selectedNode.attributes.summary && (() => {
                    // Try to parse as structured summary
                    let structured: { type: string; components: string[]; sections: Array<{
                      component: string;
                      steps: string;
                      benefits: string;
                      rationale: string;
                      results: string;
                    }> } | null = null;
                    try {
                      const parsed = JSON.parse(selectedNode.attributes.summary);
                      if (parsed.type === "structured" && parsed.sections) {
                        structured = parsed;
                      }
                    } catch {
                      // Not JSON, use as plain text
                    }
                    
                    if (structured) {
                      // Render structured summary with collapsible sections
                      return (
                        <div>
                          <h4 style={{ fontSize: "0.875rem", marginBottom: "0.5rem" }}>
                            Summary ({structured.components.length} components)
                          </h4>
                          {structured.sections.map((section, idx) => (
                            <CollapsibleSection
                              key={idx}
                              title={section.component}
                              defaultOpen={idx === 0}
                            >
                              <div style={{ fontSize: "0.8rem", lineHeight: 1.5 }}>
                                <div style={{ marginBottom: "0.5rem" }}>
                                  <strong style={{ color: "#4f46e5" }}>Steps:</strong>
                                  <div style={{ whiteSpace: "pre-wrap", marginTop: "0.25rem" }}>{section.steps}</div>
                                </div>
                                <div style={{ marginBottom: "0.5rem" }}>
                                  <strong style={{ color: "#059669" }}>Benefits:</strong>
                                  <div style={{ marginTop: "0.25rem" }}>{section.benefits}</div>
                                </div>
                                <div style={{ marginBottom: "0.5rem" }}>
                                  <strong style={{ color: "#d97706" }}>Rationale:</strong>
                                  <div style={{ marginTop: "0.25rem" }}>{section.rationale}</div>
                                </div>
                                <div>
                                  <strong style={{ color: "#dc2626" }}>Results:</strong>
                                  <div style={{ whiteSpace: "pre-wrap", marginTop: "0.25rem" }}>{section.results}</div>
                                </div>
                              </div>
                            </CollapsibleSection>
                          ))}
                        </div>
                      );
                    } else {
                      // Render plain text summary (legacy format)
                      return (
                        <div>
                          <h4 style={{ fontSize: "0.875rem", marginBottom: "0.5rem" }}>Summary</h4>
                          <p style={{ fontSize: "0.875rem", lineHeight: 1.6, color: "#333", whiteSpace: "pre-wrap" }}>
                            {selectedNode.attributes.summary}
                          </p>
                        </div>
                      );
                    }
                  })()}
                  {/* Action buttons */}
                  <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem", flexWrap: "wrap" }}>
                    <button
                      onClick={handleReabbreviate}
                      disabled={isReabbreviating}
                      style={{
                        padding: "0.5rem 1rem",
                        fontSize: "0.75rem",
                        backgroundColor: isReabbreviating ? "#ccc" : "#6366f1",
                        color: "white",
                        border: "none",
                        borderRadius: "4px",
                        cursor: isReabbreviating ? "not-allowed" : "pointer",
                      }}
                    >
                      {isReabbreviating ? "Updating..." : "🔄 Re-abbreviate"}
                    </button>
                    <button
                      onClick={handleStructuredAnalysis}
                      disabled={isLoadingStructured}
                      style={{
                        padding: "0.5rem 1rem",
                        fontSize: "0.75rem",
                        backgroundColor: isLoadingStructured ? "#ccc" : "#059669",
                        color: "white",
                        border: "none",
                        borderRadius: "4px",
                        cursor: isLoadingStructured ? "not-allowed" : "pointer",
                      }}
                    >
                      {isLoadingStructured ? "Analyzing..." : "🔍 Detailed Analysis"}
                    </button>
                  </div>
                  
                  {/* Structured Analysis Results */}
                  {structuredAnalysis && (
                    <div style={{ marginTop: "1rem" }}>
                      <h4 style={{ fontSize: "0.875rem", marginBottom: "0.5rem" }}>
                        Detailed Analysis ({structuredAnalysis.components.length} components)
                      </h4>
                      {structuredAnalysis.sections.map((section, idx) => (
                        <CollapsibleSection
                          key={idx}
                          title={section.component}
                          defaultOpen={idx === 0}
                        >
                          <div style={{ fontSize: "0.8rem", lineHeight: 1.5 }}>
                            <div style={{ marginBottom: "0.5rem" }}>
                              <strong style={{ color: "#4f46e5" }}>Steps:</strong>
                              <div style={{ whiteSpace: "pre-wrap", marginTop: "0.25rem" }}>{section.steps}</div>
                            </div>
                            <div style={{ marginBottom: "0.5rem" }}>
                              <strong style={{ color: "#059669" }}>Benefits:</strong>
                              <div style={{ marginTop: "0.25rem" }}>{section.benefits}</div>
                            </div>
                            <div style={{ marginBottom: "0.5rem" }}>
                              <strong style={{ color: "#d97706" }}>Rationale:</strong>
                              <div style={{ marginTop: "0.25rem" }}>{section.rationale}</div>
                            </div>
                            <div>
                              <strong style={{ color: "#dc2626" }}>Results:</strong>
                              <div style={{ whiteSpace: "pre-wrap", marginTop: "0.25rem" }}>{section.results}</div>
                            </div>
                          </div>
                        </CollapsibleSection>
                      ))}
                    </div>
                  )}
                </div>
              ) : selectedNode ? (
                <div>
                  <h3 style={{ fontSize: "1rem" }}>{selectedNode.name}</h3>
                  {selectedNode.children && (
                    <p style={{ fontSize: "0.875rem", color: "#666" }}>
                      {selectedNode.children.length} paper{selectedNode.children.length !== 1 ? "s" : ""} in this category
                    </p>
                  )}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>Click on a node in the tree to see details</p>
              )}
            </>
          )}

          {/* Repos Panel */}
          {activePanel === "repos" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>GitHub Repositories</h2>
              {isLoadingFeature ? (
                <p style={{ color: "#666" }}>Searching...</p>
              ) : repos.length > 0 ? (
                <div>
                  {repos.map((repo, i) => (
                    <div key={i} style={{ padding: "0.75rem", borderBottom: "1px solid #eee" }}>
                      <a href={repo.repo_url} target="_blank" rel="noopener noreferrer" style={{ color: "#0070f3", fontWeight: 500 }}>
                        {repo.repo_name}
                      </a>
                      <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>
                        {repo.is_official && <span style={{ color: "#10b981", marginRight: "0.5rem" }}>✓ Official</span>}
                        <span>⭐ {repo.stars || 0}</span>
                        <span style={{ marginLeft: "0.5rem" }}>via {repo.source}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>No repositories found. Right-click on a paper and select "Find GitHub Repo".</p>
              )}
            </>
          )}

          {/* References Panel */}
          {activePanel === "references" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>References</h2>
              {isLoadingFeature ? (
                <p style={{ color: "#666" }}>Loading references...</p>
              ) : references.length > 0 ? (
                <div>
                  {references.map((ref) => (
                    <div
                      key={ref.id}
                      style={{ padding: "0.75rem", borderBottom: "1px solid #eee", position: "relative" }}
                      onMouseEnter={() => handleRefHover(ref)}
                      onMouseLeave={handleRefLeave}
                    >
                      <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>{ref.cited_title}</div>
                      <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>
                        {ref.cited_authors?.slice(0, 2).join(", ")}
                        {ref.cited_authors && ref.cited_authors.length > 2 && " et al."}
                        {ref.cited_year && ` (${ref.cited_year})`}
                      </div>
                      {ref.cited_arxiv_id && (
                        <button
                          onClick={() => handleAddReference(ref)}
                          style={{
                            marginTop: "0.5rem",
                            padding: "0.25rem 0.5rem",
                            fontSize: "0.75rem",
                            backgroundColor: "#f0f0f0",
                            border: "none",
                            borderRadius: "4px",
                            cursor: "pointer",
                          }}
                        >
                          + Add to tree
                        </button>
                      )}
                      
                      {/* Hover tooltip */}
                      {hoveredRefId === ref.id && (
                        <div
                          style={{
                            position: "absolute",
                            left: "100%",
                            top: 0,
                            marginLeft: "0.5rem",
                            width: "300px",
                            padding: "0.75rem",
                            backgroundColor: "#333",
                            color: "white",
                            borderRadius: "4px",
                            fontSize: "0.75rem",
                            lineHeight: 1.5,
                            zIndex: 100,
                            boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
                          }}
                        >
                          {refExplanations[ref.id] || "Loading explanation..."}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>No references loaded. Right-click on a paper and select "Explain References".</p>
              )}
            </>
          )}

          {/* Similar Papers Panel */}
          {activePanel === "similar" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>Similar Papers</h2>
              <p style={{ fontSize: "0.75rem", color: "#666", marginBottom: "1rem" }}>
                Recommended papers from Semantic Scholar (200M+ papers)
              </p>
              {isLoadingFeature ? (
                <p style={{ color: "#666" }}>Searching the internet for similar papers...</p>
              ) : similarPapers.length > 0 ? (
                <div>
                  {similarPapers.map((paper, i) => (
                    <div key={i} style={{ padding: "0.75rem", borderBottom: "1px solid #eee" }}>
                      <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>
                        {paper.url ? (
                          <a href={paper.url} target="_blank" rel="noopener noreferrer" style={{ color: "#0070f3" }}>
                            {paper.title}
                          </a>
                        ) : paper.title}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>
                        {paper.year && <span>Year: {paper.year}</span>}
                        {paper.citation_count !== undefined && (
                          <span style={{ marginLeft: "0.5rem" }}>Citations: {paper.citation_count}</span>
                        )}
                        {paper.arxiv_id && (
                          <span style={{ marginLeft: "0.5rem" }}>arXiv: {paper.arxiv_id}</span>
                        )}
                      </div>
                      {paper.authors && paper.authors.length > 0 && (
                        <div style={{ fontSize: "0.75rem", color: "#888", marginTop: "0.25rem" }}>
                          {paper.authors.slice(0, 3).join(", ")}
                          {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                        </div>
                      )}
                      {paper.arxiv_id && (
                        <button
                          onClick={() => handleAddSimilarPaper(paper)}
                          style={{
                            marginTop: "0.5rem",
                            padding: "0.25rem 0.5rem",
                            fontSize: "0.75rem",
                            backgroundColor: "#f0f0f0",
                            border: "none",
                            borderRadius: "4px",
                            cursor: "pointer",
                          }}
                        >
                          + Add to tree
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>Right-click a paper to find similar.</p>
              )}
            </>
          )}

          {/* Query Panel */}
          {activePanel === "query" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1rem", fontWeight: 600 }}>Ask a Question</h2>
              {selectedNode?.attributes?.arxivId ? (
                <div>
                  <p style={{ fontSize: "0.75rem", color: "#666", marginBottom: "0.75rem" }}>
                    Ask questions about: {selectedNode.attributes.title || selectedNode.name}
                  </p>
                  <textarea
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    placeholder="e.g., What is the main contribution of this paper?"
                    disabled={isQuerying}
                    style={{
                      width: "100%",
                      height: "80px",
                      padding: "0.5rem",
                      fontSize: "0.875rem",
                      border: "1px solid #ddd",
                      borderRadius: "4px",
                      resize: "vertical",
                      boxSizing: "border-box",
                    }}
                  />
                  <button
                    onClick={handleQuery}
                    disabled={isQuerying || !queryInput.trim()}
                    style={{
                      marginTop: "0.5rem",
                      width: "100%",
                      padding: "0.5rem",
                      fontSize: "0.875rem",
                      backgroundColor: isQuerying ? "#ccc" : "#0070f3",
                      color: "white",
                      border: "none",
                      borderRadius: "4px",
                      cursor: isQuerying ? "not-allowed" : "pointer",
                    }}
                  >
                    {isQuerying ? "Searching..." : "Ask"}
                  </button>
                  {queryAnswer && (
                    <div style={{ marginTop: "1rem", padding: "0.75rem", backgroundColor: "#f9f9f9", borderRadius: "4px" }}>
                      <h4 style={{ fontSize: "0.75rem", marginTop: 0, marginBottom: "0.5rem", color: "#666" }}>Answer:</h4>
                      <p style={{ fontSize: "0.875rem", lineHeight: 1.6, whiteSpace: "pre-wrap", margin: 0 }}>
                        {queryAnswer}
                      </p>
                    </div>
                  )}
                  
                  {/* Query History */}
                  {queryHistory.length > 0 && (
                    <div style={{ marginTop: "1.5rem" }}>
                      <h3 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem", color: "#333" }}>
                        Query History ({queryHistory.length})
                      </h3>
                      {queryHistory.map((q) => (
                        <div key={q.id} style={{ 
                          marginBottom: "0.75rem", 
                          padding: "0.75rem", 
                          backgroundColor: "#f5f5f5", 
                          borderRadius: "4px",
                          borderLeft: "3px solid #0070f3"
                        }}>
                          <div style={{ fontSize: "0.75rem", color: "#666", marginBottom: "0.25rem" }}>
                            {new Date(q.created_at).toLocaleString()}
                          </div>
                          <div style={{ fontSize: "0.8125rem", fontWeight: 500, marginBottom: "0.5rem" }}>
                            Q: {q.question}
                          </div>
                          <div style={{ fontSize: "0.8125rem", color: "#444", whiteSpace: "pre-wrap" }}>
                            A: {q.answer}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>Select a paper to ask questions.</p>
              )}
            </>
          )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
