import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderGraphEvidence, renderQualityReport } from "../validation-view.js";

function root() { globalThis.document = new JSDOM("<main></main>").window.document; return document.querySelector("main"); }

test("graph inspector shows actual directed relations, raw source and unverified semantics safely", () => {
  const host = root(); let page;
  renderGraphEvidence(host, { node_count: 2, filtered_edge_count: 1, note: "候选关系", edges: [
    { source: "<script>A</script>", target: "B", predicate: "depends_on", description: "关联", evidence: "A uses B.", source_text: "A uses B.", filename: "manual.pdf", page_number: 2, quote_in_source: true, endpoints_in_source: true }
  ] }, (row) => page = row.page_number);
  assert.equal(host.querySelector("script"), null);
  assert.match(host.textContent, /→ depends_on → B/);
  assert.match(host.textContent, /关系语义未验证/);
  host.querySelector("button").click(); assert.equal(page, 2);
});

test("quality report exposes rank, missing context, model estimates and graph regressions", () => {
  const host = root();
  renderQualityReport(host, { status: "incomplete", dataset_version: "fixture-v1", top_k: 3, note: "不是总体准确率", variants: {
    hybrid: { metrics: { "recall@3": 1, context_complete_evidence: 0, faithfulness_estimate: null }, denominators: { "recall@3": 1 },
      cases: [{ id: "case", question: "<img src=x>", answer: "答案", evidence_checks: [{ filename: "a.md", contains: "标准原文", stage: "assessment_or_context_filter", ranks: { base: [2], context: [], citations: [] } }], answer_review: { status: "unavailable" } }] }
  }, comparisons: { graph_hybrid: [{ id: "case", status: "regressed", mrr_before: 1, mrr_after: .5 }] } });
  assert.equal(host.querySelector("img"), null);
  assert.match(host.textContent, /基础召回排名：2/);
  assert.match(host.textContent, /进入上下文：未进入/);
  assert.match(host.textContent, /未知 \/ 不适用/);
  assert.match(host.textContent, /上下文证据退化/);
});

test("graph review exposes excluded and unknown relations without presenting them as usable", () => {
  const host = root();
  renderGraphEvidence(host, {node_count:3, filtered_edge_count:2, eligible_edge_count:0, review_counts:{unsupported:1,unknown:1}, edges:[
    {source:"A",target:"B",predicate:"waits",filename:"a.md",evidence:"A waits.",source_text:"A waits.",semantic_correctness:"unsupported",retrieval_eligible:false,semantic_review:{reason:"<script>predicate mismatch</script>"}},
    {source:"A",target:"C",predicate:"uses",filename:"a.md",evidence:"A uses C.",source_text:"A uses C.",semantic_correctness:"unknown",retrieval_eligible:false},
  ]});
  assert.equal(host.querySelector('script'),null);
  assert.match(host.textContent,/可用于图谱检索 0 条/);
  assert.match(host.textContent,/模型判缺乏依据/);
  assert.match(host.textContent,/审核未知/);
  assert.equal((host.textContent.match(/不参与图谱检索/g)||[]).length,2);
});

test("quality report separates valid source use from literal gold excerpt matching", () => {
  const host = root();
  renderQualityReport(host, {status:"completed", dataset_version:"fixture", top_k:3, variants:{hybrid:{
    metrics:{citation_sources_complete_evidence:1,citations_complete_evidence:0},
    denominators:{citation_sources_complete_evidence:1,citations_complete_evidence:1},
    cases:[{id:"short",question:"Timeout?",answer:"30 ms",evidence_checks:[{
      filename:"policy.md",contains:"Full policy sentence",stage:"citation_excerpt_shorter_than_gold",
      ranks:{base:[1],context:[1],citation_sources:[1],citations:[]}
    }]}]
  }}});
  assert.match(host.textContent,/最终引用来源完整覆盖率/);
  assert.match(host.textContent,/最终引文金标全文匹配率/);
  assert.match(host.textContent,/引用来源：1；金标全文匹配引文：未全文匹配/);
  assert.match(host.textContent,/它不证明节选支持答案/);
});

test("replayed quality metrics preserve original-run provenance and avoid a pass claim", () => {
  const host = root();
  renderQualityReport(host,{status:"completed",dataset_version:"fixture",top_k:3,
    metrics_version:"v3",metric_replay_of:"original-run",source_snapshot_unchanged:true,
    models:{generator:"old-model",complex_generator:"old-complex",judge:"old-judge"},variants:{}});
  assert.match(host.textContent,/原运行 original-run/);
  assert.match(host.textContent,/沿用原回答与原模型审核/);
  assert.match(host.textContent,/不表示所有质量指标通过/);
  assert.match(host.textContent,/不代表当前服务或固定模型权重/);
});

test("a legacy complete audit with sampled coverage is visibly labeled as a sample", () => {
  const host = root();
  renderQualityReport(host,{status:"completed",dataset_version:"external",top_k:5,variants:{},
    graph_review:{status:"complete",sampled:true,reviewed_edges:40,total_edges:340,
      relation_support_estimate:.95,units:[]}});
  assert.match(host.textContent,/40\/340/);
  assert.match(host.textContent,/仅完成抽样审核，不能视为全量通过/);
  assert.match(host.textContent,/支持度仅针对已复查关系/);
});
