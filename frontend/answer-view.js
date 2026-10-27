import createDOMPurify from "dompurify";
import { marked } from "marked";
export { renderRetrievalEvidence, RELEVANCE_NAMES } from "./retrieval-view.js";
export { renderGraphEvidence, renderQualityReport } from "./validation-view.js";
export { understandingBudgetLines } from "./understanding-budget-view.js";
export { roundBudgetLines } from "./round-budget-view.js";
export { installWorkspaceLayout } from "./workspace-layout.js";

const purify = createDOMPurify(window);
let diagramLibrary;
let nextDiagramId = 0;
const DIAGRAM_LIMIT = 6;
const MAX_DIAGRAM_LENGTH = 10000;

async function renderDiagram(source) {
  if (source.length > MAX_DIAGRAM_LENGTH) throw new Error("Diagram too large");
  if (/%%\s*\{|^\s*---|^\s*click\s/im.test(source)) {
    throw new Error("Unsupported diagram configuration");
  }
  if (!/^\s*(?:flowchart\b|graph\b|sequenceDiagram\b|stateDiagram(?:-v2)?\b|classDiagram\b|erDiagram\b|pie\b|xychart(?:-beta)?\b|mindmap\b|timeline\b)/.test(source)) {
    throw new Error("Unsupported diagram type");
  }
  if (!diagramLibrary) {
    diagramLibrary = import("mermaid").then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        htmlLabels: false,
        suppressErrorRendering: true,
        maxTextSize: MAX_DIAGRAM_LENGTH,
        theme: "neutral",
        fontFamily: document.defaultView?.getComputedStyle(document.documentElement).getPropertyValue("--font-ui").trim() || '"Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", sans-serif',
      });
      return mermaid;
    }).catch((error) => { diagramLibrary = undefined; throw error; });
  }
  const mermaid = await diagramLibrary;
  await mermaid.parse(source);
  const { svg } = await mermaid.render(`answer-diagram-${++nextDiagramId}`, source);
  return svg;
}

export async function renderRichAnswer(container, text, diagramRenderer = renderDiagram) {
  const html = marked.parse(String(text ?? ""), { gfm: true, breaks: false, async: false });
  container.innerHTML = purify.sanitize(html, {
    ALLOWED_TAGS: ["p", "br", "strong", "em", "del", "ul", "ol", "li", "blockquote", "pre", "code", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "table", "thead", "tbody", "tr", "th", "td", "a"],
    ALLOWED_ATTR: ["href", "title", "class", "start", "align"],
    ALLOW_DATA_ATTR: false,
  });
  for (const link of container.querySelectorAll("a")) {
    const href = link.getAttribute("href") || "";
    // GFM mistakes metrics such as mAP@0.5-0.95 for email addresses.
    if (/^mailto:[^@]+@[0-9.-]+$/i.test(href)) {
      link.replaceWith(...link.childNodes);
      continue;
    }
    if (!/^(https?:|mailto:|#)/i.test(href)) link.removeAttribute("href");
    link.setAttribute("rel", "noopener noreferrer");
  }
  for (const table of container.querySelectorAll("table")) {
    const wrapper = document.createElement("div");
    wrapper.className = "answer-table-scroll";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "回答中的表格，可横向滚动");
    table.replaceWith(wrapper);
    wrapper.append(table);
  }
  const diagrams = [...container.querySelectorAll("pre > code.language-mermaid")];
  for (const [index, code] of diagrams.entries()) {
    const pre = code.parentElement;
    const source = code.textContent;
    const figure = document.createElement("figure");
    figure.className = "answer-diagram";
    figure.setAttribute("aria-label", "回答中的图表");
    figure.setAttribute("aria-busy", "true");
    figure.textContent = "正在绘制图表…";
    pre.replaceWith(figure);
    try {
      if (index >= DIAGRAM_LIMIT) throw new Error("Too many diagrams");
      const svg = await diagramRenderer(source);
      figure.innerHTML = purify.sanitize(svg, {
        USE_PROFILES: { svg: true, svgFilters: true },
        FORBID_TAGS: ["foreignObject", "image", "a", "script"],
        FORBID_ATTR: ["href", "xlink:href"],
      });
      if (!figure.querySelector("svg")) throw new Error("No chart output");
    } catch {
      figure.replaceChildren();
      const note = document.createElement("p");
      note.textContent = "这张图表暂时无法绘制，请参考上方文字说明。";
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "查看图表描述";
      details.append(summary, pre);
      figure.append(note, details);
    } finally {
      figure.removeAttribute("aria-busy");
    }
  }
}
export { renderParsingReport, renderSourcePage, tabularLocations } from "./source-view.js";
export { renderSegments } from "./segment-view.js";
export { renderOcrReview } from "./ocr-view.js";
export { renderDocumentReferences } from "./reference-view.js";
export { renderBatches } from "./batch-view.js";
export { renderDiagnostics } from "./diagnostics-view.js";
export { renderBatchDiagnostics } from "./batch-diagnostics-view.js";
export {renderTaskAlignment} from "./task-alignment-view.js";
