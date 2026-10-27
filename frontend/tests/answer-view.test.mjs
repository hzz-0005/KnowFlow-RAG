import assert from "node:assert/strict";
import { test } from "node:test";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.CSSStyleSheet = dom.window.CSSStyleSheet;
const { renderRichAnswer } = await import("../answer-view.js");

test("renders conversational paragraphs, tables, code and citations", async () => {
  const host = document.createElement("div");
  await renderRichAnswer(host, "答案是 **余弦距离**。[1]\n\n| 运算符 | 含义 |\n| --- | --- |\n| `<=>` | 余弦距离 |\n\n```sql\nSELECT 1;\n```");
  assert.equal(host.querySelector("strong").textContent, "余弦距离");
  assert.equal(host.querySelector("td code").textContent, "<=>");
  assert.equal(host.querySelectorAll("table").length, 1);
  assert.equal(host.querySelector(".answer-table-scroll").tabIndex, 0);
  assert.match(host.textContent, /\[1\]/);
  assert.match(host.querySelector("pre").textContent, /SELECT 1;/);
});

test("untrusted model output cannot inject scripts, handlers or remote images", async () => {
  const host = document.createElement("div");
  await renderRichAnswer(host, '<script>alert(1)</script>\n\n<img src="https://evil.invalid/log" onerror="alert(2)">\n\n[unsafe](javascript:alert%281%29)\n\n<iframe src="https://evil.invalid"></iframe>\n\n<a href="https://example.com" onclick="alert(3)">safe</a>');
  assert.equal(host.querySelectorAll("script,img,iframe,[onclick],[onerror]").length, 0);
  assert.equal(host.querySelector('a[href^="javascript:"]'), null);
  assert.equal(host.querySelector('a[href="https://example.com"]').getAttribute("rel"), "noopener noreferrer");
});

test("metric IoU ranges stay plain text while real email links remain usable", async () => {
  const host = document.createElement("div");
  await renderRichAnswer(host, "mAP@0.5-0.95 为 67；联系 reader@example.com。");
  assert.match(host.textContent, /mAP@0.5-0.95/);
  assert.equal(host.querySelector('a[href="mailto:mAP@0.5-0.95"]'), null);
  assert.equal(host.querySelector('a[href="mailto:reader@example.com"]').textContent, "reader@example.com");
});

test("renders a sanitized SVG chart and retains its textual explanation", async () => {
  const host = document.createElement("div");
  await renderRichAnswer(host, '关系如下。[2]\n\n```mermaid\nflowchart LR\n A["网关"] --> B["索引"]\n```', async () => '<svg xmlns="http://www.w3.org/2000/svg"><text>网关到索引</text><script>alert(1)</script><image href="https://evil.invalid/log"/><foreignObject>bad</foreignObject></svg>');
  assert.equal(host.querySelectorAll("figure svg").length, 1);
  assert.equal(host.querySelectorAll("script,image,foreignObject").length, 0);
  assert.equal(host.querySelector("figure").hasAttribute("aria-busy"), false);
  assert.match(host.textContent, /关系如下。\[2\]/);
});

test("invalid diagram falls back without hiding the answer or displaying source by default", async () => {
  const host = document.createElement("div");
  await renderRichAnswer(host, "有依据的文字。[1]\n\n```mermaid\nbroken syntax\n```", async () => { throw new Error("Invalid diagram"); });
  assert.match(host.querySelector("p").textContent, /有依据的文字/);
  assert.equal(host.querySelector("details").open, false);
  assert.match(host.querySelector("details code").textContent, /broken syntax/);
  assert.equal(document.body.querySelector("svg"), null);
});

test("limits chart work without discarding explanatory text", async () => {
  const host = document.createElement("div");
  let calls = 0;
  await renderRichAnswer(host, "说明。[1]\n\n" + '```mermaid\nflowchart LR\n A --> B\n```\n\n'.repeat(8), async () => {
    calls += 1;
    return '<svg xmlns="http://www.w3.org/2000/svg"><text>A to B</text></svg>';
  });
  assert.equal(calls, 6);
  assert.equal(host.querySelectorAll("figure").length, 8);
  assert.equal(host.querySelectorAll("details").length, 2);
});

test("real Mermaid output retains readable labels after SVG sanitization", async () => {
  // jsdom has no SVG layout engine. Stub geometry only; use the real parser,
  // renderer and sanitizer to check labels across their integration boundary.
  dom.window.SVGElement.prototype.getBBox = () => ({ x: 0, y: 0, width: 150, height: 30 });
  dom.window.SVGElement.prototype.getComputedTextLength = () => 150;
  for (const [source, label] of [
    ['flowchart LR\n A["文档"] --> B["词条"]', '文档'],
    ['xychart\n title "索引维度上限"\n x-axis [vector, halfvec]\n y-axis "维度" 0 --> 4000\n bar [2000, 4000]', '索引维度上限'],
    ['pie title 示例占比\n "甲" : 3\n "乙" : 7', '示例占比'],
  ]) {
    const host = document.createElement("div");
    document.body.append(host);
    await renderRichAnswer(host, `说明。[1]\n\n\`\`\`mermaid\n${source}\n\`\`\``);
    assert.equal(host.querySelectorAll("details").length, 0, host.textContent);
    const svg = host.querySelector("figure svg");
    assert.ok(svg, "Expected a rendered chart");
    assert.ok([...svg.querySelectorAll("text")].some(node => node.textContent.includes(label)), "Chart labels must survive sanitization");
    assert.equal(svg.querySelectorAll("foreignObject").length, 0);
    host.remove();
  }
});
