"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { hierarchy, tree as d3tree } from "d3-hierarchy";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Search } from "lucide-react";

interface PaperNode {
  name: string;
  node_id?: string;
  node_type?: string;
  paper_id?: number;
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

interface TreeLayout {
  nodes: Array<{
    x: number;
    y: number;
    data: PaperNode;
    nodeId: string;
    isPaper: boolean;
  }>;
  links: Array<{
    source: { x: number; y: number; data: PaperNode };
    target: { x: number; y: number; data: PaperNode };
    isPaper: boolean;
  }>;
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
  linesById: Record<string, string[]>;
  dimsById: Record<string, { width: number; height: number; lineHeight: number; paddingY: number }>;
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
        const mathRegex = /(\\(?:\[|\())([\s\S]*?)(\\(?:\)|\]))/g;
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

const initialTaxonomy: PaperNode = {
  name: "AI Papers",
  children: [],
};

export default function Home() {
  const [unifiedIngestInput, setUnifiedIngestInput] = useState("");
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
  const [activePanel, setActivePanel] = useState<"explorer" | "details" | "repos" | "references" | "similar" | "query">("explorer");
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
  
  // Unified ingest state (replaces separate arxivUrl and batchDirectory)
  const [isBatchIngesting, setIsBatchIngesting] = useState(false);
  const [batchResults, setBatchResults] = useState<{
    total: number;
    success: number;
    skipped: number;
    errors: number;
    results: Array<{ file: string; status: string; reason?: string; title?: string; category?: string }>;
    progress_log?: string[];
  } | null>(null);
  
  // Slack credential prompt state
  const [showSlackCredentialPrompt, setShowSlackCredentialPrompt] = useState(false);
  const [pendingSlackChannel, setPendingSlackChannel] = useState("");
  const [slackToken, setSlackToken] = useState("");
  
  // Re-classify state (replaces old rebalance)
  const [isReclassifying, setIsReclassifying] = useState(false);
  const [reclassifyResult, setReclassifyResult] = useState<{
    message: string;
    papers_classified: number;
    clusters_created: number;
    nodes_named: number;
    levels_processed: number;
  } | null>(null);
  
  // Collapsible sections
  
  // Query selection for merge feature
  const [selectedQueryIds, setSelectedQueryIds] = useState<Set<number>>(new Set());
  const [isMergingQueries, setIsMergingQueries] = useState(false);
  const [isDedupingSummary, setIsDedupingSummary] = useState(false);
  
  // Tree diagram state - temporarily disabled during rebuild
  // const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());
  // const [nodes, setNodes, onNodesChange] = useNodesState([]);
  // const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  // Phase 3: Navigation & Interaction states (only search kept)
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PaperNode[]>([]);
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  
  // Phase 4: Responsive Design & Polish states
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1024);
  const [rightPanelWidth, setRightPanelWidth] = useState(50); // Percentage
  const [isRightCollapsed, setIsRightCollapsed] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState<"tree" | "details" | null>(null);
  const [isDebugPanelOpen, setIsDebugPanelOpen] = useState(false);
  
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

  const handleReclassifyPapers = async () => {
    setIsReclassifying(true);
    setReclassifyResult(null);
    clearIngestLog();
    logIngest("Re-classifying papers using embedding-based clustering...");
    
    try {
      const res = await fetch("/api/papers/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
        setIsReclassifying(false);
        return;
      }
      
      const data = await res.json();
      setReclassifyResult(data);
      logIngest(`✓ ${data.message}`);
      logIngest(`  Papers classified: ${data.papers_classified}`);
      logIngest(`  Clusters created: ${data.clusters_created}`);
      logIngest(`  Nodes named: ${data.nodes_named}`);
      
      // Refresh tree
      const treeRes = await fetch("/api/tree");
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        setTaxonomy(treeData);
        logIngest("Tree refreshed");
      }
    } catch (e) {
      logIngest(`Error: ${e}`);
    } finally {
      setIsReclassifying(false);
      logIngest("Done");
    }
  };

  // Unified ingest handler with auto-detection
  const handleUnifiedIngest = async () => {
    if (!unifiedIngestInput.trim()) return;
    
    const input = unifiedIngestInput.trim();
    
    // Detect input type
    const isUrl = input.startsWith("http://") || input.startsWith("https://") || input.includes("arxiv.org");
    const isArxivId = /^\d{4}\.\d{4,5}(v\d+)?$/.test(input) || /^arxiv:\/\/(abs|pdf)\/\d{4}\.\d{4,5}(v\d+)?/.test(input);
    // Slack channel: starts with #, contains slack.com, or is a Slack channel ID (starts with C followed by alphanumeric)
    const isSlackChannel = input.startsWith("#") || input.includes("slack.com") || /^C[A-Z0-9]{8,}$/.test(input);
    
    // Check if it's a directory path (local folder)
    // Simple heuristic: if it starts with / or contains \ and doesn't look like a URL
    const isDirectory = (input.startsWith("/") || input.includes("\\")) && !isUrl && !isArxivId && !isSlackChannel;
    
    if (isSlackChannel) {
      // Show credential prompt for Slack channel
      setPendingSlackChannel(input);
      setShowSlackCredentialPrompt(true);
      return;
    }
    
    if (isDirectory) {
      // Batch ingest
      setIsBatchIngesting(true);
      setBatchResults(null);
      clearIngestLog();
      logIngest(`Starting batch ingest from: ${input}`);
      
      try {
        const res = await fetch("/api/papers/batch-ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ directory: input, skip_existing: true }),
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
        logIngest("Done");
      }
    } else {
      // Single ingest (URL or arXiv ID)
      setIsIngesting(true);
      clearIngestLog();
      logIngest(`Starting ingestion for: ${input}`);
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
      const inputValue = input;
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

      setUnifiedIngestInput("");
      setIsIngesting(false);
    }
  };

  const handleSlackIngest = async () => {
    if (!slackToken.trim()) {
      logIngest("Error: Slack token is required");
      return;
    }

    setIsBatchIngesting(true);
    setBatchResults(null);
    clearIngestLog();
    setShowSlackCredentialPrompt(false);
    logIngest(`Starting Slack channel ingest from: ${pendingSlackChannel}`);

    try {
      const res = await fetch("/api/papers/batch-ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slack_channel: pendingSlackChannel,
          slack_token: slackToken,
          skip_existing: true,
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        logIngest(`Error: HTTP ${res.status}`);
        try {
          const errJson = JSON.parse(errText);
          // Display progress_log if available (for Slack ingestion errors)
          if (errJson.detail?.progress_log && Array.isArray(errJson.detail.progress_log)) {
            errJson.detail.progress_log.forEach((logMessage: string) => {
              logIngest(logMessage);
            });
          }
          // Display error message
          const errorMsg = errJson.detail?.message || errJson.detail || errText;
          logIngest(`Details: ${errorMsg}`);
        } catch {
          logIngest(`Details: ${errText.slice(0, 200)}`);
        }
        setIsBatchIngesting(false);
        return;
      }

      const data = await res.json();
      
      // Display progress log if available (for Slack ingestion)
      if (data.progress_log && Array.isArray(data.progress_log)) {
        data.progress_log.forEach((logMessage: string) => {
          logIngest(logMessage);
        });
      }
      
      setBatchResults(data);
      logIngest(`Completed: ${data.success} success, ${data.skipped} skipped, ${data.errors} errors`);

      // Refresh tree
      const treeRes = await fetch("/api/tree");
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        setTaxonomy(treeData);
        logIngest("Tree refreshed");
      }

      // Clear token and pending channel
      setSlackToken("");
      setPendingSlackChannel("");
    } catch (err) {
      logIngest(`Error: ${err}`);
    } finally {
      setIsBatchIngesting(false);
      logIngest("Done");
    }
  };

  // Convert taxonomy to tree-compatible format

  // Find original node by node_id (or name for root) for details display
  const findNode = useCallback((tree: PaperNode, nodeId: string): PaperNode | null => {
    // Match by node_id if available, otherwise by name (for root node)
    if (tree.node_id === nodeId || (!tree.node_id && tree.name === nodeId)) {
      return tree;
    }
    for (const child of tree.children || []) {
      const found = findNode(child, nodeId);
      if (found) return found;
    }
    return null;
  }, []);
  
  // Search function
  const performSearch = useCallback((query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    const results: PaperNode[] = [];
    const searchLower = query.toLowerCase();
    
    const searchTree = (node: PaperNode) => {
      // Search in paper nodes (not categories)
      if (node.attributes?.arxivId) {
        const titleMatch = node.attributes.title?.toLowerCase().includes(searchLower);
        const nameMatch = node.name.toLowerCase().includes(searchLower);
        const authorMatch = node.attributes.authors?.some(author => 
          author.toLowerCase().includes(searchLower)
        );
        
        if (titleMatch || nameMatch || authorMatch) {
          results.push(node);
        }
      }
      
      // Recursively search children
      if (node.children) {
        node.children.forEach(child => searchTree(child));
      }
    };
    
    if (taxonomy.children) {
      taxonomy.children.forEach(category => {
        if (category.children) {
          category.children.forEach(paper => searchTree(paper));
        }
      });
    }
    
    setSearchResults(results.slice(0, 10)); // Limit to 10 results
  }, [taxonomy]);

  const handleNodeClick = useCallback(async (nodeId: string) => {
    const node = findNode(taxonomy, nodeId);
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

  const handleNodeRightClick = useCallback((event: React.MouseEvent, nodeId: string) => {
    event.preventDefault();
    event.stopPropagation();
    const node = findNode(taxonomy, nodeId);
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
    setUnifiedIngestInput(paper.arxiv_id);
    setActivePanel("explorer");
    // Trigger ingestion after a short delay to allow state update
    setTimeout(() => {
      handleUnifiedIngest();
    }, 100);
  };

  const handleAddReference = async (ref: Reference) => {
    if (!ref.cited_arxiv_id) return;
    // Auto-trigger ingestion
    setUnifiedIngestInput(ref.cited_arxiv_id);
    setActivePanel("explorer");
    // Trigger ingestion after a short delay to allow state update
    setTimeout(() => {
      handleUnifiedIngest();
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

  // Handle window resize for responsive design
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  
  // Determine responsive layout
  const isMobile = windowWidth < 768;
  const isTablet = windowWidth >= 768 && windowWidth < 1024;
  const isDesktop = windowWidth >= 1024;
  
  // Calculate panel widths based on responsive state and fullscreen mode
  const getPanelStyles = () => {
    if (isFullscreen === "tree") {
      return {
        left: { flex: "1 1 100%", display: "flex" },
        right: { flex: "0 0 0", display: "none" },
      };
    }
    if (isFullscreen === "details") {
      return {
        left: { flex: "0 0 0", display: "none" },
        right: { flex: "1 1 100%", display: "flex" },
      };
    }
    if (isMobile) {
      // Mobile: stack vertically, show one at a time
      return {
        left: { 
          flex: isFullscreen === null ? "1 1 50%" : "0 0 0", 
          display: isFullscreen === "details" ? "none" : "flex",
          minHeight: isFullscreen === null ? "50vh" : "100vh",
        },
        right: { 
          flex: isFullscreen === null ? "1 1 50%" : "1 1 100%", 
          display: "flex",
          minHeight: isFullscreen === null ? "50vh" : "100vh",
        },
      };
    }
    // Desktop/Tablet: side by side with resizable panels
    const rightPercent = isRightCollapsed ? 0 : rightPanelWidth;
    const leftPercent = 100 - rightPercent;
    return {
      left: { flex: `0 0 ${leftPercent}%`, display: "flex" },
      right: { flex: `0 0 ${rightPercent}%`, display: isRightCollapsed ? "none" : "flex" },
    };
  };
  
  const panelStyles = getPanelStyles();
  const toggleRightPanel = () => {
    if (isRightCollapsed) {
      setRightPanelWidth(50);
      setIsRightCollapsed(false);
    } else {
      setIsRightCollapsed(true);
    }
  };

  const treeLayout: TreeLayout | null = useMemo(() => {
    if (!taxonomy) return null;

    const wrapText = (text: string, maxChars: number) => {
      const words = text.split(" ");
      const lines: string[] = [];
      let current = "";
      for (const word of words) {
        if (!current) {
          current = word;
          continue;
        }
        if ((current + " " + word).length <= maxChars) {
          current += " " + word;
        } else {
          lines.push(current);
          current = word;
        }
      }
      if (current) lines.push(current);
      return lines.length ? lines : [text];
    };

    const dimsById: Record<string, { width: number; height: number; lineHeight: number; paddingY: number }> = {};
    const linesById: Record<string, string[]> = {};
    let maxCategoryHeight = 0;
    let maxPaperHeight = 0;
    let maxPaperColumnHeight = 0;

    const getNodeId = (node: PaperNode) => node.node_id || node.name;

    const isPaperNode = (node: PaperNode) =>
      node.node_type === "paper" ||
      !!node.paper_id ||
      (!!node.attributes?.arxivId && !(node.children && node.children.length));

    const buildDims = (node: PaperNode) => {
      const isPaper = isPaperNode(node);
      const nodeId = getNodeId(node);
      const maxChars = isPaper ? 22 : 20;
      const lineHeight = isPaper ? 16 : 18;
      const paddingY = isPaper ? 14 : 16;
      const width = isPaper ? 200 : 240;
      const lines = wrapText(node.name || "", maxChars);
      const height = Math.max(isPaper ? 44 : 52, lines.length * lineHeight + paddingY * 2);
      dimsById[nodeId] = { width, height, lineHeight, paddingY };
      linesById[nodeId] = lines;
      if (isPaper) {
        maxPaperHeight = Math.max(maxPaperHeight, height);
      } else {
        maxCategoryHeight = Math.max(maxCategoryHeight, height);
      }
    };

    const countPaperColumn = (node: PaperNode) => {
      const children = node.children || [];
      if (!children.length) return 0;
      if (children.every((c) => isPaperNode(c))) {
        const gap = 18;
        return children.length * (maxPaperHeight + gap);
      }
      return 0;
    };

    const collect = (node: PaperNode) => {
      buildDims(node);
      for (const child of node.children || []) {
        collect(child);
      }
    };

    collect(taxonomy);

    const collectPaperColumnHeights = (node: PaperNode) => {
      const columnHeight = countPaperColumn(node);
      maxPaperColumnHeight = Math.max(maxPaperColumnHeight, columnHeight);
      for (const child of node.children || []) {
        if (child.node_type === "category") {
          collectPaperColumnHeights(child);
        }
      }
    };

    collectPaperColumnHeights(taxonomy);

    const buildCategoryTree = (node: PaperNode): PaperNode => {
      const children = (node.children || []).filter((c) => !isPaperNode(c));
      return {
        ...node,
        children: children.map(buildCategoryTree),
      };
    };

    // Build lookup map from node_id to original taxonomy node (with paper children)
    const originalNodeById: Record<string, PaperNode> = {};
    const buildLookup = (node: PaperNode) => {
      const nodeId = getNodeId(node);
      originalNodeById[nodeId] = node;
      for (const child of node.children || []) {
        buildLookup(child);
      }
    };
    buildLookup(taxonomy);

    const categoryRoot = buildCategoryTree(taxonomy);
    const root = hierarchy(categoryRoot, (d) => d.children ?? []);

    const depthGap = maxCategoryHeight + Math.max(80, maxPaperColumnHeight) + 80;
    const layout = d3tree<PaperNode>()
      .nodeSize([260, depthGap])
      .separation((a, b) => (a.parent === b.parent ? 1.2 : 1.6));

    const laidOut = layout(root);
    const categoryNodes = laidOut.descendants();
    const categoryLinks = laidOut.links();

    const nodes: TreeLayout["nodes"] = [];
    const links: TreeLayout["links"] = [];

    for (const node of categoryNodes) {
      const original = node.data;
      const nodeId = getNodeId(original);
      nodes.push({
        x: node.x,
        y: node.y,
        data: original,
        nodeId,
        isPaper: false,
      });
    }

    for (const link of categoryLinks) {
      links.push({
        source: link.source,
        target: link.target,
        isPaper: false,
      });
    }

    const paperGap = 18;
    for (const catNode of categoryNodes) {
      const catId = getNodeId(catNode.data);
      // Use original taxonomy node (has paper children), not filtered catNode.data
      const originalNode = originalNodeById[catId];
      if (!originalNode) continue;
      const children = originalNode.children || [];
      if (!children.length) continue;
      const paperChildren = children.filter((c) => isPaperNode(c));
      if (!paperChildren.length) continue;
      const catDims = dimsById[catId];
      const startY = catNode.y + catDims.height / 2 + paperGap + maxPaperHeight / 2;
      paperChildren.forEach((paper, idx) => {
        const paperId = getNodeId(paper);
        const y = startY + idx * (maxPaperHeight + paperGap);
        nodes.push({
          x: catNode.x,
          y,
          data: paper,
          nodeId: paperId,
          isPaper: true,
        });
        links.push({
          source: { x: catNode.x, y: catNode.y, data: originalNode },
          target: { x: catNode.x, y, data: paper },
          isPaper: true,
        });
      });
    }

    if (!nodes.length) return null;
    const padding = 60;
    const minX = Math.min(...nodes.map((n) => n.x - dimsById[n.nodeId].width / 2));
    const maxX = Math.max(...nodes.map((n) => n.x + dimsById[n.nodeId].width / 2));
    const minY = Math.min(...nodes.map((n) => n.y - dimsById[n.nodeId].height / 2));
    const maxY = Math.max(...nodes.map((n) => n.y + dimsById[n.nodeId].height / 2));

    return {
      nodes,
      links,
      width: maxX - minX + padding * 2,
      height: maxY - minY + padding * 2,
      offsetX: padding - minX,
      offsetY: padding - minY,
      linesById,
      dimsById,
    };
  }, [taxonomy]);

  return (
    <TooltipProvider>
      <div style={{ display: "flex", height: "100vh", flexDirection: isMobile && !isFullscreen ? "column" : "row" }}>
      {/* Left panel: Tree visualization */}
      <div style={{ ...panelStyles.left, borderRight: isMobile ? "none" : "1px solid #e5e5e5", borderBottom: isMobile && !isFullscreen ? "1px solid #e5e5e5" : "none", display: "flex", flexDirection: "column", position: "relative" }}>
        <div style={{ padding: isMobile ? "0.75rem" : "1rem", borderBottom: "1px solid #e5e5e5", backgroundColor: "#fafafa" }}>
          <div className="flex items-center justify-between mb-2">
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="flex items-center gap-2">
                <h1 style={{ margin: 0, fontSize: isMobile ? "1.25rem" : "1.5rem", fontWeight: 600 }}>Paper Curator</h1>
                {/* Fullscreen toggle buttons */}
                {!isMobile && (
                  <>
                    <button
                      onClick={() => setIsFullscreen(isFullscreen === "tree" ? null : "tree")}
                      className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
                      title="Toggle tree fullscreen"
                    >
                      {isFullscreen === "tree" ? "↗" : "⛶"}
                    </button>
                    <button
                      onClick={() => setIsFullscreen(isFullscreen === "details" ? null : "details")}
                      className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
                      title="Toggle details fullscreen"
                    >
                      {isFullscreen === "details" ? "↗" : "⛶"}
                    </button>
                    {isRightCollapsed && (
                      <button
                        onClick={toggleRightPanel}
                        className="px-2 py-1 text-xs bg-blue-500 hover:bg-blue-600 text-white rounded"
                        title="Show details panel"
                      >
                        Show Panel →
                      </button>
                    )}
                  </>
                )}
              </div>
              <p style={{ margin: "0.25rem 0 0", fontSize: isMobile ? "0.75rem" : "0.875rem", color: "#666" }}>
                {taxonomy.children?.length || 0} categories, {" "}
                {taxonomy.children?.reduce((acc, c) => acc + (c.children?.length || 0), 0) || 0} papers
              </p>
            </div>
            {/* Global Search Bar */}
            <div className={`relative flex-1 ${isMobile ? "max-w-full mt-2" : "max-w-md ml-4"}`}>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    performSearch(e.target.value);
                  }}
                  onFocus={() => setIsSearchFocused(true)}
                  onBlur={() => setTimeout(() => setIsSearchFocused(false), 200)}
                  placeholder="Search papers by title or author..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              {isSearchFocused && searchResults.length > 0 && (
                <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                  {searchResults.map((result, idx) => (
                    <div
                      key={idx}
                      onClick={() => {
                        handleNodeClick(result.node_id || result.name);
                        setSearchQuery("");
                        setSearchResults([]);
                      }}
                      className="px-4 py-2 hover:bg-gray-100 cursor-pointer border-b border-gray-100 last:border-b-0"
                    >
                      <div className="font-medium text-sm">{result.name}</div>
                      {result.attributes?.title && (
                        <div className="text-xs text-gray-600 mt-1">{result.attributes.title}</div>
                      )}
                      {result.attributes?.authors && result.attributes.authors.length > 0 && (
                        <div className="text-xs text-gray-500 mt-1">
                          {result.attributes.authors.slice(0, 3).join(", ")}
                          {result.attributes.authors.length > 3 && ` +${result.attributes.authors.length - 3} more`}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
        {/* Tree Diagram Container */}
        <div
          style={{
            flex: 1,
            position: "relative",
            overflow: "auto",
            backgroundColor: "#f8fafc",
          }}
        >
          {!treeLayout ? (
            <div
              style={{
                textAlign: "center",
                color: "#64748b",
                padding: "40px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
              }}
            >
              <div>
                <div style={{ fontSize: "48px", marginBottom: "16px" }}>🌳</div>
                <h2 style={{ fontSize: "24px", fontWeight: 600, marginBottom: "8px", color: "#334155" }}>
                  No tree data loaded yet
                </h2>
                <p style={{ fontSize: "14px", maxWidth: "420px", lineHeight: 1.6 }}>
                  Ingest papers to build the tree, then re-classify to update the visualization.
                </p>
              </div>
            </div>
          ) : (
            <svg width={treeLayout.width} height={treeLayout.height}>
              <g transform={`translate(${treeLayout.offsetX}, ${treeLayout.offsetY})`}>
                {treeLayout.links.map((link, idx) => {
                  const sourceId = link.source.data.node_id || link.source.data.name;
                  const targetId = link.target.data.node_id || link.target.data.name;
                  const sourceDims = treeLayout.dimsById[sourceId];
                  const targetDims = treeLayout.dimsById[targetId];
                  const x1 = link.source.x;
                  const y1 = link.source.y + sourceDims.height / 2;
                  const x2 = link.target.x;
                  const y2 = link.target.y - targetDims.height / 2;
                  const midY = (y1 + y2) / 2;
                  const d = link.isPaper
                    ? `M ${x1},${y1} L ${x2},${y2}`
                    : `M ${x1},${y1} C ${x1},${midY} ${x2},${midY} ${x2},${y2}`;
                  return (
                    <path
                      key={`link-${idx}`}
                      d={d}
                      fill="none"
                      stroke="#94a3b8"
                      strokeWidth={1.5}
                    />
                  );
                })}
                {treeLayout.nodes.map((node) => {
                  const dims = treeLayout.dimsById[node.nodeId];
                  const x = node.x - dims.width / 2;
                  const y = node.y - dims.height / 2;
                  const isPaper = node.data.node_type === "paper";
                  const nodeId = node.nodeId;
                  const lines = treeLayout.linesById[nodeId] || [node.data.name];
                  return (
                    <g
                      key={nodeId}
                      transform={`translate(${x}, ${y})`}
                      onClick={() => handleNodeClick(nodeId)}
                      onContextMenu={(e) => handleNodeRightClick(e, nodeId)}
                      style={{ cursor: "pointer" }}
                    >
                      <rect
                        width={dims.width}
                        height={dims.height}
                        rx={10}
                        ry={10}
                        fill={isPaper ? "#e2e8f0" : "#1d4ed8"}
                        stroke={isPaper ? "#cbd5f5" : "#1e3a8a"}
                        strokeWidth={1}
                      />
                      <text
                        x={dims.width / 2}
                        y={dims.paddingY + dims.lineHeight}
                        textAnchor="middle"
                        fontSize={isPaper ? 12 : 13}
                        fontWeight={isPaper ? 500 : 600}
                        fill={isPaper ? "#334155" : "#ffffff"}
                      >
                        {lines.map((line, idx) => (
                          <tspan key={idx} x={dims.width / 2} dy={idx === 0 ? 0 : dims.lineHeight}>
                            {line}
                          </tspan>
                        ))}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          )}
        </div>
      </div>

      {/* Context Menu - Only Remove Paper option remains */}
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
              {contextMenu.node.attributes?.arxivId || contextMenu.node.name}
            </strong>
          </div>
          <div
            style={{ 
              padding: "0.75rem 1rem", 
              cursor: "pointer", 
              display: "flex", 
              alignItems: "center", 
              gap: "0.5rem",
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

      {/* Slack Credential Prompt Modal */}
      {showSlackCredentialPrompt && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2000,
          }}
          onClick={() => {
            setShowSlackCredentialPrompt(false);
            setPendingSlackChannel("");
            setSlackToken("");
          }}
        >
          <div
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "24px",
              maxWidth: "500px",
              width: "90%",
              boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-semibold mb-4">Slack Channel Access</h2>
            <p className="text-sm text-gray-600 mb-4">
              Enter your Slack User OAuth Token to access the channel:
            </p>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Channel: <span className="font-mono text-xs">{pendingSlackChannel}</span>
              </label>
              <label className="block text-sm font-medium mb-2">
                Slack Token (xoxp-...)
              </label>
              <input
                type="password"
                value={slackToken}
                onChange={(e) => setSlackToken(e.target.value)}
                placeholder="xoxp-..."
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && slackToken.trim()) {
                    handleSlackIngest();
                  }
                  if (e.key === "Escape") {
                    setShowSlackCredentialPrompt(false);
                    setPendingSlackChannel("");
                    setSlackToken("");
                  }
                }}
                autoFocus
              />
              <p className="text-xs text-gray-500 mt-2">
                Token is not persisted and will be cleared after use. Get your token from{" "}
                <a
                  href="https://api.slack.com/apps"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  api.slack.com/apps
                </a>
              </p>
            </div>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => {
                  setShowSlackCredentialPrompt(false);
                  setPendingSlackChannel("");
                  setSlackToken("");
                }}
                className="px-4 py-2 text-sm border border-gray-300 rounded hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSlackIngest}
                disabled={!slackToken.trim() || isBatchIngesting}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
              >
                {isBatchIngesting ? "Processing..." : "Ingest"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Resizer handle for desktop/tablet */}
      {!isMobile && !isFullscreen && !isRightCollapsed && (
        <div
          style={{
            width: "4px",
            backgroundColor: "#e5e5e5",
            cursor: "col-resize",
            position: "relative",
          }}
          onMouseDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startRightWidth = rightPanelWidth;
            
            const handleMouseMove = (moveEvent: MouseEvent) => {
              const deltaX = moveEvent.clientX - startX;
              const deltaPercent = (deltaX / windowWidth) * 100;
              const newRightWidth = Math.max(20, Math.min(80, startRightWidth - deltaPercent));
              setRightPanelWidth(newRightWidth);
            };
            
            const handleMouseUp = () => {
              document.removeEventListener("mousemove", handleMouseMove);
              document.removeEventListener("mouseup", handleMouseUp);
            };
            
            document.addEventListener("mousemove", handleMouseMove);
            document.addEventListener("mouseup", handleMouseUp);
          }}
        />
      )}
      
      {/* Right panel: Details and ingest */}
      <div style={{ ...panelStyles.right, padding: isMobile ? "1rem" : "1.5rem", display: "flex", flexDirection: "column", backgroundColor: "#f9fafb", overflowY: "auto", position: "relative" }}>
        {!isMobile && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "0.5rem" }}>
            <button
              onClick={toggleRightPanel}
              style={{
                padding: "4px 8px",
                borderRadius: "6px",
                border: "1px solid #e5e7eb",
                background: "#ffffff",
                fontSize: "12px",
                cursor: "pointer",
              }}
              title={isRightCollapsed ? "Expand panel" : "Collapse panel"}
            >
              {isRightCollapsed ? "Expand Panel" : "Collapse Panel"}
            </button>
          </div>
        )}

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

                  {/* Dynamic panel content */}
                  <div style={{ flex: 1, padding: "0.75rem", overflowY: "auto" }}>
                    {/* Loading skeleton for cached data */}
                    {isLoadingCachedData && (
                      <div className="space-y-4">
                        <div className="animate-pulse">
                          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                        </div>
                      </div>
                    )}
                    {selectedNode && !isLoadingCachedData && (
                  <Tabs 
                    value={activePanel === "references" ? "refs" : activePanel} 
                    onValueChange={(value) => {
                      if (value === "refs") {
                        setActivePanel("references");
                      } else {
                        setActivePanel(value as any);
                      }
                    }}
                    className="w-full"
                  >
                    <TabsList className="grid w-full grid-cols-6 h-9 mb-4">
                      <TabsTrigger value="explorer" className="text-xs">Explorer</TabsTrigger>
                      <TabsTrigger value="details" className="text-xs">Details</TabsTrigger>
                      <TabsTrigger value="repos" className="text-xs">Repos</TabsTrigger>
                      <TabsTrigger value="refs" className="text-xs">Refs</TabsTrigger>
                      <TabsTrigger value="similar" className="text-xs">Similar</TabsTrigger>
                      <TabsTrigger value="query" className="text-xs">Query</TabsTrigger>
                    </TabsList>
                    
                    {/* Explorer Panel */}
                    <TabsContent value="explorer" className="mt-0">
                      <h2 className="text-lg font-semibold mb-4">Explorer</h2>
                      
                      {/* Unified Ingest Input */}
                      <div className="mb-4">
                        <label className="block text-sm font-medium mb-2">
                          📥 Ingest Paper
                        </label>
                        <input
                          type="text"
                          value={unifiedIngestInput}
                          onChange={(e) => setUnifiedIngestInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !isIngesting && !isBatchIngesting && unifiedIngestInput.trim()) {
                              handleUnifiedIngest();
                            }
                          }}
                          placeholder="arXiv URL/ID, local file path, or folder path"
                          disabled={isIngesting || isBatchIngesting}
                          className="w-full px-3 py-2 mb-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                        />
                        <button
                          onClick={handleUnifiedIngest}
                          disabled={isIngesting || isBatchIngesting || !unifiedIngestInput.trim()}
                          className="w-full px-3 py-2 bg-blue-600 text-white rounded text-sm font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                        >
                          {(isIngesting || isBatchIngesting) ? "Processing..." : "Ingest"}
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
                        
                        {/* Batch results */}
                        {batchResults && (
                          <div className="mt-2 text-sm">
                            <div className="text-green-600">✓ {batchResults.success} ingested</div>
                            {batchResults.skipped > 0 && (
                              <div className="text-amber-500">⏭ {batchResults.skipped} skipped (existing)</div>
                            )}
                            {batchResults.errors > 0 && (
                              <div className="mt-2">
                                <div className="text-red-500 font-medium">✗ {batchResults.errors} errors</div>
                                <Accordion type="single" collapsible className="w-full mt-2">
                                  <AccordionItem value="error-details" className="border-0">
                                    <AccordionTrigger className="px-0 py-1 hover:no-underline text-xs text-red-600">
                                      Show error details
                                    </AccordionTrigger>
                                    <AccordionContent className="px-0">
                                      <div className="bg-red-50 rounded p-2 max-h-60 overflow-y-auto text-xs space-y-2">
                                        {batchResults.results
                                          .filter((r) => r.status === "error")
                                          .map((result, idx) => (
                                            <div key={idx} className="border-b border-red-200 pb-2 last:border-0 last:pb-0">
                                              <div className="font-medium text-red-800">{result.file}</div>
                                              <div className="text-red-600 mt-0.5">{result.reason || "Unknown error"}</div>
                                            </div>
                                          ))}
                                      </div>
                                    </AccordionContent>
                                  </AccordionItem>
                                </Accordion>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      
                      {/* Re-classify Button */}
                      <div className="mb-4 pt-4 border-t border-gray-200">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="text-sm font-medium text-gray-700 mb-1">
                              🔄 Re-classify Papers
                            </h3>
                            <p className="text-xs text-gray-500">
                              Rebuild tree using embedding-based hierarchical clustering
                            </p>
                          </div>
                          <button
                            onClick={handleReclassifyPapers}
                            disabled={isReclassifying}
                            className="px-3 py-1.5 bg-purple-600 text-white rounded text-xs font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-purple-700 transition-colors whitespace-nowrap"
                          >
                            {isReclassifying ? "Re-classifying..." : "Re-classify"}
                          </button>
                        </div>
                        
                        {/* Re-classify results */}
                        {reclassifyResult && (
                          <div className="mt-2 text-xs space-y-1">
                            <div className="text-purple-600">{reclassifyResult.message}</div>
                            <div className="text-gray-600">
                              ✓ {reclassifyResult.papers_classified} papers classified
                            </div>
                            <div className="text-gray-600">
                              ✓ {reclassifyResult.clusters_created} clusters created
                            </div>
                            <div className="text-gray-600">
                              ✓ {reclassifyResult.nodes_named} nodes named across {reclassifyResult.levels_processed} levels
                            </div>
                          </div>
                        )}
                      </div>
                      
                    </TabsContent>
                    
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
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-semibold">GitHub Repositories</h2>
                        {selectedNode && selectedNode.attributes?.arxivId && (
                          <button
                            onClick={() => handleFindRepos(selectedNode)}
                            disabled={isLoadingFeature}
                            className="px-3 py-1.5 bg-blue-600 text-white rounded text-xs font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                          >
                            🔗 Find Repos
                          </button>
                        )}
                      </div>
                      {isLoadingFeature ? (
                        <div className="space-y-3 animate-pulse">
                          <div className="h-16 bg-gray-200 rounded"></div>
                          <div className="h-16 bg-gray-200 rounded"></div>
                          <div className="h-16 bg-gray-200 rounded"></div>
                        </div>
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
                        <p className="text-gray-500 text-sm">Click "Find Repos" to search for GitHub repositories related to this paper.</p>
                      )}
                    </TabsContent>

                    {/* References Panel */}
                    <TabsContent value="refs" className="mt-3">
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-semibold">References</h2>
                        {selectedNode && selectedNode.attributes?.arxivId && (
                          <button
                            onClick={() => handleFetchReferences(selectedNode)}
                            disabled={isLoadingFeature}
                            className="px-3 py-1.5 bg-blue-600 text-white rounded text-xs font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                          >
                            📚 Explain References
                          </button>
                        )}
                      </div>
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
                <p className="text-gray-500 text-sm">Click "Explain References" to load and explain references from this paper.</p>
              )}
                    </TabsContent>

                    {/* Similar Papers Panel */}
                    <TabsContent value="similar" className="mt-3">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h2 className="text-lg font-semibold">Similar Papers</h2>
                          <p className="text-xs text-gray-600 mt-1">
                            Recommended papers from Semantic Scholar (200M+ papers)
                          </p>
                        </div>
                        {selectedNode && selectedNode.attributes?.arxivId && (
                          <button
                            onClick={() => handleFindSimilar(selectedNode)}
                            disabled={isLoadingFeature}
                            className="px-3 py-1.5 bg-blue-600 text-white rounded text-xs font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors whitespace-nowrap"
                          >
                            🔍 Find Similar
                          </button>
                        )}
                      </div>
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
        
        {/* Debug Panel - Collapsible */}
        {(featureLog.length > 0 || ingestLog.length > 0) && (
          <Card className="mt-4">
            <Accordion type="single" collapsible value={isDebugPanelOpen ? "debug" : undefined} onValueChange={(value) => setIsDebugPanelOpen(value === "debug")}>
              <AccordionItem value="debug" className="border-0">
                <AccordionTrigger className="px-4 py-2 hover:no-underline">
                  <h3 className="text-sm font-semibold m-0">🐛 Debug Logs</h3>
                  {(featureLog.length > 0 || ingestLog.length > 0) && (
                    <span className="ml-2 text-xs text-gray-500">
                      ({featureLog.length + ingestLog.length} entries)
                    </span>
                  )}
                </AccordionTrigger>
                <AccordionContent>
                  <div className="p-4 space-y-4">
                    {/* Feature Log */}
                    {featureLog.length > 0 && (
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-xs font-semibold text-gray-700">Feature Log</span>
                          <button 
                            onClick={clearFeatureLog}
                            className="text-xs text-gray-500 hover:text-gray-700 cursor-pointer"
                          >
                            Clear
                          </button>
                        </div>
                        <div className="bg-gray-900 rounded p-2 max-h-[200px] overflow-y-auto font-mono text-[11px]">
                          {featureLog.map((log, i) => (
                            <div key={i} className={`mb-0.5 ${log.includes("✓") ? "text-green-400" : log.includes("✗") ? "text-red-400" : log.includes("Error") ? "text-red-400" : "text-gray-300"}`}>
                              {log}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Ingest Log */}
                    {ingestLog.length > 0 && (
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-xs font-semibold text-gray-700">Ingest Log</span>
                          <button 
                            onClick={clearIngestLog}
                            className="text-xs text-gray-500 hover:text-gray-700 cursor-pointer"
                          >
                            Clear
                          </button>
                        </div>
                        <div className="bg-gray-900 rounded p-2 max-h-[200px] overflow-y-auto font-mono text-[11px]">
                          {ingestLog.map((log, i) => (
                            <div key={i} className={`mb-0.5 ${log.includes("Error") ? "text-red-400" : "text-green-400"}`}>
                              {log}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </Card>
        )}
      </div>
    </div>
    </TooltipProvider>
  );
}
