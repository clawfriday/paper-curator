"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import type { RawNodeDatum, CustomNodeElementProps } from "react-d3-tree";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

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

// Component to render text with LaTeX formulas
function TextWithMath({ text, style }: { text: string; style?: React.CSSProperties }) {
  // Split by lines first to preserve line breaks
  const lines = text.split('\n');
  
  return (
    <span style={style}>
      {lines.map((line, lineIdx) => {
        // Split each line by LaTeX patterns: \(...\) for inline and \[...\] for block
        const parts: (string | { type: "inline" | "block"; content: string })[] = [];
        let lastIndex = 0;
        
        // Match both inline \(...\) and block \[...\] math
        const mathRegex = /(\\(?:\[|\())(.*?)(\\(?:\)|\]))/gs;
        let match;
        
        while ((match = mathRegex.exec(line)) !== null) {
          // Add text before the match
          if (match.index > lastIndex) {
            parts.push(line.substring(lastIndex, match.index));
          }
          
          // Add the math content
          const isBlock = match[1] === "\\[";
          parts.push({
            type: isBlock ? "block" : "inline",
            content: match[2],
          });
          
          lastIndex = match.index + match[0].length;
        }
        
        // Add remaining text
        if (lastIndex < line.length) {
          parts.push(line.substring(lastIndex));
        }
        
        // If no math found, return plain text
        if (parts.length === 1 && typeof parts[0] === "string") {
          return (
            <span key={lineIdx}>
              {parts[0]}
              {lineIdx < lines.length - 1 && <br />}
            </span>
          );
        }
        
        return (
          <span key={lineIdx}>
            {parts.map((part, idx) => {
              if (typeof part === "string") {
                return <span key={idx}>{part}</span>;
              } else if (part.type === "block") {
                return <BlockMath key={idx} math={part.content} />;
              } else {
                return <InlineMath key={idx} math={part.content} />;
              }
            })}
            {lineIdx < lines.length - 1 && <br />}
          </span>
        );
      })}
    </span>
  );
}

// Note: CollapsibleSection replaced with Shadcn/ui Accordion component

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
  
  // Query selection for merge feature
  const [selectedQueryIds, setSelectedQueryIds] = useState<Set<number>>(new Set());
  const [isMergingQueries, setIsMergingQueries] = useState(false);
  const [isDedupingSummary, setIsDedupingSummary] = useState(false);
  
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
    // Detect if input is URL or plain arXiv ID
    const inputValue = arxivUrl.trim();
    const isUrl = inputValue.includes("arxiv.org") || inputValue.startsWith("http");
    const resolveRes = await fetch("/api/arxiv/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(isUrl ? { url: inputValue } : { arxiv_id: inputValue }),
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
    
    // Check for duplicate
    const checkDupRes = await fetch("/api/papers/cached-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: arxivId }),
    });
    if (checkDupRes.ok) {
      logIngest(`⚠️ Paper already exists in database!`);
      logIngest(`Skipping ingestion - paper ${arxivId} was previously ingested.`);
      updateStep(0, { status: "done", message: `${title} (already exists)` });
      // Mark remaining steps as skipped
      for (let i = 1; i < 7; i++) {
        updateStep(i, { status: "done", message: "Skipped (duplicate)" });
      }
      setIsIngesting(false);
      return;
    }
    
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

  const handleRemoveNode = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) return;
    
    const arxivId = node.attributes.arxivId;
    const confirmed = window.confirm(
      `Are you sure you want to remove "${node.name}"?\n\nThis will permanently delete:\n- The paper from the database\n- All associated references, queries, and similar papers\n- The tree node\n\nThis action cannot be undone.`
    );
    
    if (!confirmed) return;
    
    clearFeatureLog();
    logFeature(`Removing paper: ${arxivId}...`);
    
    try {
      const res = await fetch("/api/papers/delete", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId }),
      });
      
      if (res.ok) {
        logFeature(`✓ Paper ${arxivId} removed successfully`);
        // Reload tree to reflect changes
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          setTaxonomy(treeData);
        }
        // Clear selection if we just deleted the selected node
        if (selectedNode?.attributes?.arxivId === arxivId) {
          setSelectedNode(null);
          setActivePanel("details");
        }
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status} - ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    }
    logFeature("Done");
  };

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

  const toggleQuerySelection = (queryId: number) => {
    setSelectedQueryIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(queryId)) {
        newSet.delete(queryId);
      } else {
        newSet.add(queryId);
      }
      return newSet;
    });
  };

  const handleMergeQueries = async () => {
    if (!selectedNode?.attributes?.arxivId || selectedQueryIds.size === 0) return;
    setIsMergingQueries(true);
    clearFeatureLog();
    logFeature(`Merging ${selectedQueryIds.size} Q&A pair(s) into summary...`);
    
    // Get selected Q&A pairs
    const selectedQa = queryHistory
      .filter(q => selectedQueryIds.has(q.id))
      .map(q => ({ question: q.question, answer: q.answer }));
    
    try {
      const res = await fetch("/api/summary/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arxiv_id: selectedNode.attributes.arxivId,
          selected_qa: selectedQa,
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        logFeature(`✓ Merged ${data.merged_count} Q&A pair(s) into summary`);
        // Update the selected node's summary
        if (selectedNode) {
          setSelectedNode({
            ...selectedNode,
            attributes: {
              ...selectedNode.attributes,
              summary: data.summary,
            }
          });
        }
        // Clear selection
        setSelectedQueryIds(new Set());
        // Switch to details panel to show updated summary
        setActivePanel("details");
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status} - ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    } finally {
      setIsMergingQueries(false);
      logFeature("Done");
    }
  };

  const handleDedupSummary = async () => {
    if (!selectedNode?.attributes?.arxivId) return;
    setIsDedupingSummary(true);
    clearFeatureLog();
    logFeature("Deduplicating summary...");
    
    try {
      const res = await fetch("/api/summary/dedup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arxiv_id: selectedNode.attributes.arxivId,
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const reduction = data.original_length - data.new_length;
        const percent = Math.round((reduction / data.original_length) * 100);
        logFeature(`✓ Removed duplicates (${reduction > 0 ? `-${percent}%` : "no changes"})`);
        // Update the selected node's summary
        if (selectedNode) {
          setSelectedNode({
            ...selectedNode,
            attributes: {
              ...selectedNode.attributes,
              summary: data.summary,
            }
          });
        }
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status} - ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    } finally {
      setIsDedupingSummary(false);
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
    <TooltipProvider>
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
          <div
            style={{ 
              padding: "0.75rem 1rem", 
              cursor: "pointer", 
              display: "flex", 
              alignItems: "center", 
              gap: "0.5rem",
              borderTop: "1px solid #eee",
              color: "#dc2626",
            }}
            onClick={() => { handleRemoveNode(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#fef2f2")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            <span>🗑️</span> Remove Paper
          </div>
        </div>
      )}

      {/* Right panel: Details and ingest */}
      <div className="flex-1 p-6 flex flex-col bg-gray-50 overflow-y-auto">
        {/* Ingest section - Accordion */}
        <Card className="mb-4">
          <Accordion type="single" collapsible defaultValue="ingest" className="w-full">
            <AccordionItem value="ingest" className="border-0">
              <AccordionTrigger className="px-4 py-3 hover:no-underline">
                <h2 className="text-base font-semibold m-0">📥 Ingest Paper</h2>
              </AccordionTrigger>
              <AccordionContent>
                <div className="p-4">
                  <input
                    type="text"
                    value={arxivUrl}
                    onChange={(e) => setArxivUrl(e.target.value)}
                    placeholder="arXiv URL or ID (e.g., 1706.03762)"
                    disabled={isIngesting}
                    className="w-full px-2 py-2 mb-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                  />
                  <button
                    id="ingest-btn"
                    onClick={handleIngest}
                    disabled={isIngesting || !arxivUrl.trim()}
                    className="w-full px-2 py-2 bg-blue-600 text-white rounded text-sm font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                  >
                    {isIngesting ? "Ingesting..." : "Ingest"}
                  </button>

                  {/* Progress steps */}
                  {steps.length > 0 && (
                    <div className="mt-3">
                      {steps.map((step, i) => (
                        <div key={i} className="flex items-start mb-1">
                          <span className="mr-1.5 text-xs" style={{ color: getStepColor(step.status) }}>
                            {getStepIcon(step.status)}
                          </span>
                          <div className="flex-1">
                            <span className={`text-xs ${step.status === "error" ? "text-red-500" : "text-gray-800"}`}>
                              {step.name}
                            </span>
                            {step.message && (
                              <p className={`mt-0.5 text-[10px] ${step.status === "error" ? "text-red-500" : "text-gray-600"}`}>
                                {step.message}
                              </p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {/* Batch Ingest */}
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h3 className="m-0 mb-2 text-sm font-medium text-gray-600">
                      📁 Batch Ingest (Local PDFs)
                    </h3>
                    <input
                      type="text"
                      value={batchDirectory}
                      onChange={(e) => setBatchDirectory(e.target.value)}
                      placeholder="Directory path (e.g., /Users/you/papers)"
                      disabled={isBatchIngesting}
                      className="w-full px-2 py-2 mb-2 border border-gray-300 rounded text-xs focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-gray-100"
                    />
                    <button
                      onClick={handleBatchIngest}
                      disabled={isBatchIngesting || !batchDirectory.trim()}
                      className="w-full px-2 py-2 bg-green-600 text-white rounded text-xs font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-green-700 transition-colors"
                    >
                      {isBatchIngesting ? "Processing..." : "Batch Ingest"}
                    </button>
                    
                    {/* Batch results */}
                    {batchResults && (
                      <div className="mt-2 text-[11px]">
                        <div className="text-green-600">✓ {batchResults.success} ingested</div>
                        {batchResults.skipped > 0 && (
                          <div className="text-amber-500">⏭ {batchResults.skipped} skipped (existing)</div>
                        )}
                        {batchResults.errors > 0 && (
                          <div className="text-red-500">✗ {batchResults.errors} errors</div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Rebalance Categories */}
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h3 className="m-0 mb-2 text-sm font-medium text-gray-600">
                      ⚖️ Rebalance Categories
                    </h3>
                    <p className="text-[11px] text-gray-500 mb-2">
                      Reclassify papers in crowded categories (10+ papers)
                    </p>
                    <button
                      onClick={handleRebalanceCategories}
                      disabled={isRebalancing}
                      className="w-full px-2 py-2 bg-purple-600 text-white rounded text-xs font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-purple-700 transition-colors"
                    >
                      {isRebalancing ? "Rebalancing..." : "Rebalance Now"}
                    </button>
                    
                    {/* Rebalance results */}
                    {rebalanceResult && (
                      <div className="mt-2 text-[11px]">
                        <div className="text-purple-600">{rebalanceResult.message}</div>
                        {rebalanceResult.reclassified?.length > 0 && (
                          <div className="text-green-600 mt-1">
                            ✓ Moved {rebalanceResult.reclassified.length} papers
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Ingest log textbox */}
                  {ingestLog.length > 0 && (
                    <div className="mt-3 bg-gray-900 rounded p-2 max-h-[150px] overflow-y-auto font-mono text-[11px]">
                      {ingestLog.map((log, i) => (
                        <div key={i} className={`mb-0.5 ${log.includes("Error") ? "text-red-400" : "text-green-400"}`}>
                          {log}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </Card>

        {/* Paper Details Section - Accordion */}
        <Card className="flex-1 flex flex-col">
          <Accordion type="single" collapsible defaultValue="details" className="w-full flex-1 flex flex-col">
            <AccordionItem value="details" className="border-0 flex-1 flex flex-col">
              <AccordionTrigger className="px-4 py-3 hover:no-underline">
                <h2 className="text-base font-semibold m-0">
                  📄 {selectedNode ? (selectedNode.attributes?.title ? selectedNode.name : selectedNode.name) : "Paper Details"}
                </h2>
              </AccordionTrigger>
              <AccordionContent className="flex-1 flex flex-col overflow-hidden">
                <div className="flex-1 flex flex-col overflow-hidden">
                  {/* Panel tabs */}
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
                    {selectedNode && (
                  <Tabs 
                    value={activePanel === "references" ? "refs" : activePanel} 
                    onValueChange={(value) => setActivePanel(value === "refs" ? "references" : value as any)}
                    className="w-full"
                  >
                    <TabsList className="grid w-full grid-cols-5 h-9 mb-4">
                      <TabsTrigger value="details" className="text-xs">Details</TabsTrigger>
                      <TabsTrigger value="repos" className="text-xs">Repos</TabsTrigger>
                      <TabsTrigger value="refs" className="text-xs">Refs</TabsTrigger>
                      <TabsTrigger value="similar" className="text-xs">Similar</TabsTrigger>
                      <TabsTrigger value="query" className="text-xs">Query</TabsTrigger>
                    </TabsList>
                    
                    {/* Details Panel */}
                    <TabsContent value="details" className="mt-0">
                      <h2 className="text-lg font-semibold mb-4">Paper Details</h2>
                      {selectedNode?.attributes ? (
                        <div>
                          <h3 className="text-base font-semibold mb-2">{selectedNode.attributes.title || selectedNode.name}</h3>
                          {selectedNode.attributes.arxivId && (
                            <p className="text-sm text-gray-600 mb-2">
                              <strong>arXiv:</strong>{" "}
                              <a href={`https://arxiv.org/abs/${selectedNode.attributes.arxivId}`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                                {selectedNode.attributes.arxivId}
                              </a>
                            </p>
                          )}
                          {selectedNode.attributes.authors && (
                            <p className="text-sm text-gray-600 mb-2">
                              <strong>Authors:</strong> {selectedNode.attributes.authors.slice(0, 3).join(", ")}
                              {selectedNode.attributes.authors.length > 3 && ` +${selectedNode.attributes.authors.length - 3} more`}
                            </p>
                          )}
                          {selectedNode.attributes.category && (
                            <p className="text-sm text-gray-600 mb-4">
                              <strong>Category:</strong> {selectedNode.attributes.category}
                            </p>
                          )}
                          {/* Show structured analysis if available (from cached data), otherwise show regular summary */}
                          {structuredAnalysis ? (
                    <Card className="mt-4">
                      <CardHeader>
                        <CardTitle className="text-sm">
                          Detailed Analysis ({structuredAnalysis.components.length} components)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <Accordion type="single" collapsible defaultValue={structuredAnalysis.sections[0]?.component} className="w-full">
                          {structuredAnalysis.sections.map((section, idx) => (
                            <AccordionItem key={idx} value={section.component}>
                              <AccordionTrigger className="text-sm font-semibold">{section.component}</AccordionTrigger>
                              <AccordionContent>
                                <div className="space-y-3 text-sm">
                                  <div>
                                    <strong className="text-indigo-600">Steps:</strong>
                                    <div className="mt-1 whitespace-pre-wrap">
                                      <TextWithMath text={section.steps} />
                                    </div>
                                  </div>
                                  <div>
                                    <strong className="text-emerald-600">Benefits:</strong>
                                    <div className="mt-1">
                                      <TextWithMath text={section.benefits} />
                                    </div>
                                  </div>
                                  <div>
                                    <strong className="text-amber-600">Rationale:</strong>
                                    <div className="mt-1">
                                      <TextWithMath text={section.rationale} />
                                    </div>
                                  </div>
                                  <div>
                                    <strong className="text-red-600">Results:</strong>
                                    <div className="mt-1 whitespace-pre-wrap">
                                      <TextWithMath text={section.results} />
                                    </div>
                                  </div>
                                </div>
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </CardContent>
                    </Card>
                  ) : selectedNode.attributes.summary ? (() => {
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
                      // Render structured summary with accordion sections
                      return (
                        <Card className="mt-4">
                          <CardHeader>
                            <CardTitle className="text-sm">
                              Summary ({structured.components.length} components)
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <Accordion type="single" collapsible defaultValue={structured.sections[0]?.component} className="w-full">
                              {structured.sections.map((section, idx) => (
                                <AccordionItem key={idx} value={section.component}>
                                  <AccordionTrigger className="text-sm font-semibold">{section.component}</AccordionTrigger>
                                  <AccordionContent>
                                    <div className="space-y-3 text-sm">
                                      <div>
                                        <strong className="text-indigo-600">Steps:</strong>
                                        <div className="mt-1 whitespace-pre-wrap">
                                          <TextWithMath text={section.steps} />
                                        </div>
                                      </div>
                                      <div>
                                        <strong className="text-emerald-600">Benefits:</strong>
                                        <div className="mt-1">
                                          <TextWithMath text={section.benefits} />
                                        </div>
                                      </div>
                                      <div>
                                        <strong className="text-amber-600">Rationale:</strong>
                                        <div className="mt-1">
                                          <TextWithMath text={section.rationale} />
                                        </div>
                                      </div>
                                      <div>
                                        <strong className="text-red-600">Results:</strong>
                                        <div className="mt-1 whitespace-pre-wrap">
                                          <TextWithMath text={section.results} />
                                        </div>
                                      </div>
                                    </div>
                                  </AccordionContent>
                                </AccordionItem>
                              ))}
                            </Accordion>
                          </CardContent>
                        </Card>
                      );
                    } else {
                      // Render plain text summary (legacy format)
                      return (
                        <Card className="mt-4">
                          <CardHeader>
                            <CardTitle className="text-sm">Summary</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-sm leading-relaxed text-gray-700 whitespace-pre-wrap">
                              <TextWithMath text={selectedNode.attributes.summary} />
                            </p>
                          </CardContent>
                        </Card>
                      );
                    }
                  })() : null}
                          {/* Action buttons */}
                          <div className="flex gap-2 mt-4 flex-wrap">
                            <button
                              onClick={handleReabbreviate}
                              disabled={isReabbreviating}
                              className="px-4 py-2 text-xs bg-indigo-600 text-white rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-indigo-700"
                            >
                              {isReabbreviating ? "Updating..." : "🔄 Re-abbreviate"}
                            </button>
                            <button
                              onClick={handleStructuredAnalysis}
                              disabled={isLoadingStructured}
                              className="px-4 py-2 text-xs bg-emerald-600 text-white rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-emerald-700"
                            >
                              {isLoadingStructured ? "Analyzing..." : "🔍 Detailed Analysis"}
                            </button>
                            <button
                              onClick={handleDedupSummary}
                              disabled={isDedupingSummary}
                              className="px-4 py-2 text-xs bg-amber-600 text-white rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-amber-700"
                            >
                              {isDedupingSummary ? "Deduping..." : "🧹 Dedup"}
                            </button>
                          </div>
                        </div>
                        ) : selectedNode ? (
                          <div>
                            <h3 className="text-base font-semibold">{selectedNode.name}</h3>
                            {selectedNode.children && (
                              <p className="text-sm text-gray-600 mt-1">
                                {selectedNode.children.length} paper{selectedNode.children.length !== 1 ? "s" : ""} in this category
                              </p>
                            )}
                          </div>
                        ) : (
                          <p className="text-gray-500 text-sm">Click on a node in the tree to see details</p>
                        )}
                    </TabsContent>

                    {/* Repos Panel */}
                    <TabsContent value="repos" className="mt-3">
                      <h2 className="text-lg font-semibold mb-4">GitHub Repositories</h2>
                      {isLoadingFeature ? (
                        <p className="text-gray-600">Searching...</p>
                      ) : repos.length > 0 ? (
                        <div>
                          {repos.map((repo, i) => (
                            <div key={i} className="p-3 border-b border-gray-200">
                              <a href={repo.repo_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 font-medium hover:underline">
                                {repo.repo_name}
                              </a>
                              <div className="text-xs text-gray-600 mt-1">
                                {repo.is_official && <span className="text-emerald-600 mr-2">✓ Official</span>}
                                <span>⭐ {repo.stars || 0}</span>
                                <span className="ml-2">via {repo.source}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-500 text-sm">No repositories found. Right-click on a paper and select "Find GitHub Repo".</p>
                      )}
                    </TabsContent>

                    {/* References Panel */}
                    <TabsContent value="refs" className="mt-3">
                      <h2 className="text-lg font-semibold mb-4">References</h2>
                      {isLoadingFeature ? (
                        <p className="text-gray-600">Loading references...</p>
                      ) : references.length > 0 ? (
                        <div>
                          {references.map((ref) => (
                            <div
                              key={ref.id}
                              className="p-3 border-b border-gray-200 relative"
                              onMouseEnter={() => handleRefHover(ref)}
                              onMouseLeave={handleRefLeave}
                            >
                              <div className="text-sm font-medium">{ref.cited_title}</div>
                              <div className="text-xs text-gray-600 mt-1">
                                {ref.cited_authors?.slice(0, 2).join(", ")}
                                {ref.cited_authors && ref.cited_authors.length > 2 && " et al."}
                                {ref.cited_year && ` (${ref.cited_year})`}
                              </div>
                              {ref.cited_arxiv_id && (
                                <button
                                  onClick={() => handleAddReference(ref)}
                                  className="mt-2 px-2 py-1 text-xs bg-gray-100 border-none rounded cursor-pointer hover:bg-gray-200"
                                >
                                  + Add to tree
                                </button>
                              )}
                              
                              {/* Hover tooltip */}
                              {hoveredRefId === ref.id && (
                                <div className="absolute left-full top-0 ml-2 w-[300px] p-3 bg-gray-800 text-white rounded text-xs leading-relaxed z-50 shadow-lg">
                                  {refExplanations[ref.id] || "Loading explanation..."}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
              ) : (
                <p className="text-gray-500 text-sm">No references loaded. Right-click on a paper and select "Explain References".</p>
              )}
                    </TabsContent>

                    {/* Similar Papers Panel */}
                    <TabsContent value="similar" className="mt-3">
                      <h2 className="text-lg font-semibold mb-4">Similar Papers</h2>
                      <p className="text-xs text-gray-600 mb-4">
                        Recommended papers from Semantic Scholar (200M+ papers)
                      </p>
                      {isLoadingFeature ? (
                        <p className="text-gray-600">Searching the internet for similar papers...</p>
                      ) : similarPapers.length > 0 ? (
                        <div>
                          {similarPapers.map((paper, i) => (
                            <div key={i} className="p-3 border-b border-gray-200">
                              <div className="text-sm font-medium">
                                {paper.url ? (
                                  <a href={paper.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                                    {paper.title}
                                  </a>
                                ) : paper.title}
                              </div>
                              <div className="text-xs text-gray-600 mt-1">
                                {paper.year && <span>Year: {paper.year}</span>}
                                {paper.citation_count !== undefined && (
                                  <span className="ml-2">Citations: {paper.citation_count}</span>
                                )}
                                {paper.arxiv_id && (
                                  <span className="ml-2">arXiv: {paper.arxiv_id}</span>
                                )}
                              </div>
                              {paper.authors && paper.authors.length > 0 && (
                                <div className="text-xs text-gray-500 mt-1">
                                  {paper.authors.slice(0, 3).join(", ")}
                                  {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                                </div>
                              )}
                              {paper.arxiv_id && (
                                <button
                                  onClick={() => handleAddSimilarPaper(paper)}
                                  className="mt-2 px-2 py-1 text-xs bg-gray-100 border-none rounded cursor-pointer hover:bg-gray-200"
                                >
                                  + Add to tree
                                </button>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-500 text-sm">Right-click a paper to find similar.</p>
                      )}
                    </TabsContent>

                    {/* Query Panel */}
                    <TabsContent value="query" className="mt-3">
                      <h2 className="text-base font-semibold mb-4">Ask a Question</h2>
                      {selectedNode?.attributes?.arxivId ? (
                        <div>
                          <p className="text-xs text-gray-600 mb-3">
                            Ask questions about: {selectedNode.attributes.title || selectedNode.name}
                          </p>
                          <textarea
                            value={queryInput}
                            onChange={(e) => setQueryInput(e.target.value)}
                            placeholder="e.g., What is the main contribution of this paper?"
                            disabled={isQuerying}
                            className="w-full h-20 p-2 text-sm border border-gray-300 rounded resize-y box-border"
                          />
                          <button
                            onClick={handleQuery}
                            disabled={isQuerying || !queryInput.trim()}
                            className="mt-2 w-full py-2 text-sm bg-blue-600 text-white border-none rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700"
                          >
                            {isQuerying ? "Searching..." : "Ask"}
                          </button>
                          {queryAnswer && (
                            <Card className="mt-4">
                              <CardContent className="pt-4">
                                <h4 className="text-xs mb-2 text-gray-600">Answer:</h4>
                                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                  <TextWithMath text={queryAnswer} />
                                </p>
                              </CardContent>
                            </Card>
                          )}
                  
                          {/* Query History */}
                          {queryHistory.length > 0 && (
                            <div className="mt-6">
                              <h3 className="text-sm font-semibold mb-3 text-gray-800">
                                Query History ({queryHistory.length})
                                {selectedQueryIds.size > 0 && (
                                  <span className="font-normal text-blue-600 ml-2">
                                    ({selectedQueryIds.size} selected)
                                  </span>
                                )}
                              </h3>
                              {queryHistory.map((q) => (
                                <Card 
                                  key={q.id} 
                                  onClick={() => toggleQuerySelection(q.id)}
                                  className={`mb-3 cursor-pointer transition-all ${
                                    selectedQueryIds.has(q.id) ? "bg-blue-50 border-blue-300" : "bg-gray-50"
                                  }`}
                                >
                                  <CardContent className="pt-4">
                                    <div className="flex items-start gap-2">
                                      <input
                                        type="checkbox"
                                        checked={selectedQueryIds.has(q.id)}
                                        onChange={() => toggleQuerySelection(q.id)}
                                        onClick={(e) => e.stopPropagation()}
                                        className="mt-0.5 cursor-pointer"
                                      />
                                      <div className="flex-1">
                                        <div className="text-xs text-gray-600 mb-1">
                                          {new Date(q.created_at).toLocaleString()}
                                        </div>
                                        <div className="text-sm font-medium mb-2">
                                          Q: {q.question}
                                        </div>
                                        <div className="text-sm text-gray-700 whitespace-pre-wrap">
                                          A: <TextWithMath text={q.answer} />
                                        </div>
                                      </div>
                                    </div>
                                  </CardContent>
                                </Card>
                              ))}
                              
                              {/* Add to Details button */}
                              <button
                                onClick={handleMergeQueries}
                                disabled={selectedQueryIds.size === 0 || isMergingQueries}
                                className="mt-2 w-full py-2 text-sm bg-green-600 text-white border-none rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-green-700 font-medium transition-colors"
                              >
                                {isMergingQueries ? "Merging..." : `Add to Details${selectedQueryIds.size > 0 ? ` (${selectedQueryIds.size})` : ""}`}
                              </button>
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-gray-500 text-sm">Select a paper to ask questions.</p>
                      )}
                    </TabsContent>
                  </Tabs>
                    )}
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </Card>
      </div>
    </div>
    </TooltipProvider>
  );
}
