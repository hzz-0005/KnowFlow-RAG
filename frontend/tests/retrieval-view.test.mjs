import assert from "node:assert/strict";
import { test } from "node:test";
import { JSDOM } from "jsdom";
import { renderRetrievalEvidence } from "../retrieval-view.js";

const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.document = dom.window.document;

test("summary citation counts use legacy dropped counts without implying semantic review", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{answer_validation:{version:"document-summary-citation-blocks-v1",status:"citation_checked",kept_blocks:1,dropped_blocks:0,note:"只检查标记"}});
  assert.match(host.textContent,/总结引用标记检查：保留 1\/1 块 · 未做语义复核/);
  assert.doesNotMatch(host.textContent,/保留 1\/0|回答依据复核/);
});

test("repair trigger explains presentation-only skip without calling it a quality pass", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{answer_validation:{status:"filtered",timing_contract:"answer-guard-stages-v1",
    repair_trigger:{version:"evidence-driven-repair-v1",decision:"skipped",reason:"nonfactual_prose_only"},stages:[]}});
  assert.match(host.textContent,/未额外生成纠错或再次审核/);
  assert.match(host.textContent,/不证明模型语义分类正确/);
});

test("adopted repair shows its original trigger while legacy traces are not reinterpreted", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{answer_validation:{status:"complete",timing_contract:"answer-guard-stages-v1",
    initial_review:{repair_trigger:{version:"evidence-driven-repair-v1",reason:"filtered_or_rejected_requires_attention"}},stages:[]}});
  assert.match(host.textContent,/按原策略最多纠错一次并再次审核/);
  renderRetrievalEvidence(host,{answer_validation:{status:"filtered",timing_contract:"answer-guard-stages-v1",stages:[]}});
  assert.doesNotMatch(host.textContent,/纠错触发判断/);
});

test("guard stages separate estimated budgets, actual usage and incomplete repair candidates", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{answer_validation:{status:"complete",timing_contract:"answer-guard-stages-v1",stages:[
    {stage:"initial_review",status:"filtered",context_budget:{version:"answer-guard-request-window-v1",
      input_tokens_estimate:900,output_reserved_tokens:8192,safety_tokens:1024,estimated_total_tokens:10116,configured_total_tokens:32768,
      request_attempted:true,response_received:true,provider_usage:{prompt_tokens:930,completion_tokens:50},reason_code:"validated_response"}},
    {stage:"repair_generation",status:"completed",context_budget:{version:"answer-guard-request-window-v1",
      reason_code:"completed_candidate",provider_usage:null}},
    {stage:"repair_review",status:"unavailable",context_budget:{version:"answer-guard-request-window-v1",
      reason_code:"output_limit_reached",request_attempted:true,response_received:true}}
  ]}});
  assert.match(host.textContent,/完整输入估算 900 token/);
  assert.match(host.textContent,/实报输入 930、输出 50/);
  assert.match(host.textContent,/候选已生成，仍须再次审核/);
  assert.match(host.textContent,/输出触顶，未采用截断结果/);
  assert.match(host.textContent,/实报输入 未记录、输出 未记录/);
  assert.match(host.textContent,/不删待审单元、条件或证据/);
});

test("guard no-call budget and old missing traces remain explicit unknown and inert text", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{answer_validation:{status:"budget_exceeded",timing_contract:"answer-guard-stages-v1",stages:[
    {stage:"initial_review",status:"budget_exceeded",context_budget:{version:"answer-guard-request-window-v1",
      reason_code:"full_input_exceeds_budget",request_attempted:false,response_received:false}},
    {stage:"repair_review",status:"unavailable"}
  ]}});
  assert.match(host.textContent,/完整输入超预算，未发起请求/);
  assert.match(host.textContent,/再次审核未记录完整请求预算，保持未知/);
  assert.equal(host.querySelectorAll("script,img,[onerror]").length,0);
});

for (const version of ["relevance-window-whole-passages-v1", "relevance-window-source-metadata-v2"]) test(`relevance budget distinguishes observations and unknown use: ${version}`, () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{retrieval_diagnostics:{status:"partial",note:"模型判断",context_budget:{
    version,selected_count:2,skipped_count:1,observed_count:1,unknown_count:2,
    input_tokens_estimate:900,fixed_input_tokens_estimate:700,evidence_tokens_estimate:200,
    output_reserved_tokens:4096,safety_tokens:1024,estimated_total_tokens:6020,configured_total_tokens:32768,
    request_attempted:true,response_received:true,reason_code:"validated_response",provider_usage:null}}});
  assert.match(host.textContent,/装配 2 块 \/ 未装配 1 块/);
  assert.match(host.textContent,/取得可用判断 1 个身份，未核验 2 个身份/);
  assert.match(host.textContent,/实报输入 未记录、输出 未记录/);
  assert.match(host.textContent,/未核验不等于不相关/);
});

test("relevance nonfinal, no-call and historical traces do not invent success or zero cost", () => {
  const host=document.createElement("div");
  const trace={version:"relevance-window-whole-passages-v1",reason_code:"output_limit_reached",request_attempted:true};
  renderRetrievalEvidence(host,{retrieval_diagnostics:{status:"unavailable",context_budget:trace}});
  assert.match(host.textContent,/输出触顶，未采用截断判断/);
  assert.match(host.textContent,/实报输入 未记录/);
  trace.reason_code="fixed_input_exhausted";trace.request_attempted=false;
  renderRetrievalEvidence(host,{retrieval_diagnostics:{status:"unavailable",context_budget:trace}});
  assert.match(host.textContent,/未调用检测模型/);
  renderRetrievalEvidence(host,{retrieval_diagnostics:{status:"complete"}});
  assert.match(host.textContent,/历史记录未保存本阶段完整请求预算/);
  assert.doesNotMatch(host.textContent,/装配 0|实报输入 0/);
});

test("relevance diagnostic untrusted fields remain inert text", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{retrieval_diagnostics:{status:"unavailable",context_budget:{
    version:"relevance-window-whole-passages-v1",reason_code:'<img src=x onerror="bad()">',input_tokens_estimate:"<script>x</script>"}}});
  assert.equal(host.querySelectorAll("img,script,[onerror]").length,0);
  assert.match(host.textContent,/状态原因未记录/);
});

test("answer route explains raw-intent selection without treating it as a quality score", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{context_budget:{status:"ready",selected_count:1,skipped_count:0},
    generation_route:{policy_version:"task-route-authoritative-input-v2",model:"simple-fixture",
      thinking:"disabled",input_basis:"raw_user_intent",signals:{scope_or_negation:false,long_task:false,multi_source:false}}});
  assert.match(host.textContent,/判断输入：当前及引用的用户原话/);
  assert.match(host.textContent,/未命中现有复杂规则/);
  assert.match(host.textContent,/不是问题难度或答案质量评分/);
  assert.match(host.textContent,/辅助提示仍完整发送/);
});

test("complex route keeps condition and multi-source reasons and safely renders the model", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{context_budget:{status:"ready"},generation_route:{
    policy_version:"task-route-authoritative-input-v2",model:'<img src=x onerror="bad()">',
    thinking:"enabled",input_basis:"question_text",signals:{scope_or_negation:true,multi_source:true}}});
  assert.match(host.textContent,/条件\/范围\/否定词、多来源/);
  assert.match(host.textContent,/思考开启/);
  assert.equal(host.querySelectorAll("img,script,[onerror]").length,0);
});

test("query failure details distinguish validation from provider usage and never invent historical reasons", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",trace_contract:"query-rewrite-diagnostics-v1",
    phase:"validation",reason_code:"numeric_constraints_changed",request_attempted:true,response_received:true,
    original_only:true,fallback_used:true,provider_usage:{prompt_tokens:92,completion_tokens:17},search_queries:[]}});
  assert.match(host.textContent,/改写增加或遗漏了原问题中的数字/);
  assert.match(host.textContent,/阶段：内容校验/);
  assert.match(host.textContent,/输入 92、输出 17/);
  assert.match(host.textContent,/不证明同义、否定或范围在语义上完全保真/);
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",latency_ms:3229}});
  assert.match(host.textContent,/历史记录未保存具体失败原因/);
  assert.doesNotMatch(host.textContent,/达到本地改写时限|数字，整组未采用/);
});

test("query timeout usage stays unknown; same-query completion is not presented as expansion", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",trace_contract:"query-rewrite-diagnostics-v1",
    phase:"request",reason_code:"local_timeout",request_attempted:true,response_received:false,
    original_only:true,fallback_used:true,provider_usage:null}});
  assert.match(host.textContent,/达到本地改写时限/);
  assert.match(host.textContent,/实报输入 未记录、输出 未记录/);
  renderRetrievalEvidence(host,{query_plan:{status:"complete",reason_code:"no_effective_queries"}});
  assert.match(host.querySelector("summary").textContent,/未增加检索式/);
  assert.doesNotMatch(host.querySelector("summary").textContent,/不可用|已完成/);
});

test("untrusted diagnostic reason stays text and cannot insert markup", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",trace_contract:"query-rewrite-diagnostics-v1",
    phase:"<svg>",reason_code:'<img src=x onerror="bad()">',http_status:"<script>bad()</script>"}});
  assert.match(host.textContent,/未知原因/);
  assert.equal(host.querySelectorAll("img,script,svg,[onerror]").length,0);
});

test("protected numeric input and atomic rejection remain distinct from semantic success", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",trace_contract:"query-rewrite-diagnostics-v1",
    phase:"validation",reason_code:"numeric_constraints_changed",protected_numeric_token_count:3,
    validation_failure:{query_index:2,missing_numeric_count:1,added_numeric_count:0}}});
  assert.match(host.textContent,/显式保护 3 个不同数字字面值/);
  assert.match(host.textContent,/第 2 条检索式：遗漏 1 个、新增 0 个/);
  assert.match(host.textContent,/未自动补写遗漏值/);
});

test("diagnostic groups have distinct named stages and source binding stays a review fact", () => {
  const host = document.createElement("div");
  renderRetrievalEvidence(host, {query_plan:{status:"complete"},
    answer_validation:{status:"filtered",citation_binding:{added:[{block_id:"b1"}]}},
    reference_expansion:{enabled:false},
    retrieval_diagnostics:{status:"complete",rerank:{status:"completed",candidate_count:0}},
    retrieval_evidence:[]});
  assert.equal(host.querySelectorAll(":scope > .diagnostic-stage").length,5);
  for (const name of ["query","review","reference","ranking","evidence"]) {
    assert.ok(host.querySelector(`.diagnostic-${name} > summary .diagnostic-stage-label`));
  }
  assert.match(host.textContent,/1 个保留块补充/);
  assert.match(host.textContent,/不代表独立语义标注/);
});

test("draft budget is visible without retrieval and separates estimates from unknown provider usage", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{context_budget:{status:"blocked",selected_count:0,skipped_count:1,
    input_tokens_estimate:2000,fixed_input_tokens_estimate:2000,evidence_tokens_estimate:0,
    evidence_cap_tokens:6000,output_reserved_tokens:8192,safety_tokens:1024,
    estimated_total_tokens:11216,configured_total_tokens:10000,provider_usage:null,
    decisions:[{id:"<img src=x>",decision:"skipped",reason:"fixed_input_exhausted"}]}});
  assert.match(host.querySelector(".diagnostic-budget > summary").textContent,/预算不足/);
  assert.match(host.textContent,/供应商实报输入 未记录/);
  assert.match(host.textContent,/原始条件不截断/);
  assert.match(host.textContent,/仅覆盖首稿生成/);
  assert.equal(host.querySelectorAll("img").length,0);
});

test("observed usage and output exhaustion are distinct from an estimated fit", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{context_budget:{status:"output_limit_reached",selected_count:2,
    skipped_count:0,provider_usage:{prompt_tokens:123,completion_tokens:8192},
    output_limit_scope:"reasoning_and_answer"}});
  assert.match(host.querySelector(".diagnostic-budget > summary").textContent,/生成达到上限/);
  assert.match(host.textContent,/供应商实报输入 123、输出 8192/);
  assert.match(host.textContent,/思考和正文合计/);
});

test("parsing warnings belong to their own expandable frame rather than the rerank group", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{retrieval_diagnostics:{status:"complete",
    rerank:{status:"completed",candidate_count:1},parsing_warnings:["来源映射待核对","<img src=x>"]}});
  assert.equal(host.querySelector(".diagnostic-warning").querySelectorAll("p").length,2);
  assert.match(host.querySelector(".diagnostic-warning > summary").textContent,/2 项待核对/);
  assert.doesNotMatch(host.querySelector(".diagnostic-ranking").textContent,/来源映射待核对/);
  assert.equal(host.querySelectorAll("img").length,0);
});

test("English stemming additions stay distinguishable from base top-k", () => {
  const host = document.createElement("div");
  renderRetrievalEvidence(host, {
    retrieval_diagnostics:{status:"complete"},
    retrieval_evidence:[{rank:6,filename:"paper.pdf",quote:"Tables were filtered.",
      retrieval_stage:"text_expansion",english_keyword_rank:3,
      assessment:{verdict:"partial"}}],
  });
  assert.match(host.textContent, /基础 k 之外/);
  assert.match(host.textContent, /英文词干排名 3/);
});

test("graph additions expose their separate budget and actual candidate origin", () => {
  const host = document.createElement("div");
  renderRetrievalEvidence(host, {
    graph_expansion: { base_count: 3, added_count: 2, max_additions: 3, shared_count: 1,
      note: "基础 Recall@k 不含新增候选；扩展收益不是同候选数量的对照。" },
    retrieval_diagnostics: { status: "complete" },
    retrieval_evidence: [{ rank: 4, filename: "path.md", quote: "source",
      retrieval_stage: "graph_expansion", assessment: { verdict: "partial" } }],
  });
  assert.match(host.textContent, /保留 3 条文本基础候选/);
  assert.match(host.textContent, /新增 2 条图谱候选/);
  assert.match(host.textContent, /不是同候选数量/);
  assert.match(host.textContent, /图谱补充候选/);
});

test("shows each candidate's relevance and usage without inventing confidence percentages", () => {
  const host = document.createElement("div");
  renderRetrievalEvidence(host, {
    refused: true, retrieval_diagnostics: { status: "complete", note: "未经概率校准" },
    retrieval_evidence: [
      { rank: 1, filename: "guide.pdf", quote: "原文依据", cosine_similarity: 0.88,
        assessment: { verdict: "direct", reason: "给出所问事实" }, included_in_context: true },
      { rank: 2, filename: "other.pdf", quote: "另一版本", assessment: { verdict: "irrelevant", reason: "版本不符" } },
    ],
  });
  assert.equal(host.querySelectorAll(".retrieval-row").length, 2);
  assert.equal(host.querySelector(".retrieval-check").open, true);
  assert.match(host.textContent, /直接相关/);
  assert.match(host.textContent, /不相关/);
  assert.match(host.textContent, /已提供给模型/);
  assert.match(host.textContent, /未采用/);
  assert.match(host.textContent, /余弦相似度 0.880/);
  assert.doesNotMatch(host.textContent, /88%|88％|置信度 88/);
});

test("assessment text and source metadata cannot inject HTML", () => {
  const host = document.createElement("div");
  renderRetrievalEvidence(host, {
    query_plan: {status:"complete", search_queries:['<img src=x onerror="bad()">'], note:"<script>bad()</script>"},
    answer_validation: {status:"filtered", total_blocks:2, kept_blocks:1, reviewed_blocks:[{verdict:"unsupported", text:"<svg onload=bad()>", reason:"<img src=x>"}]},
    retrieval_diagnostics: { status: "partial", note: '<img src=x onerror="alert(1)">' },
    retrieval_evidence: [{ rank: 1, filename: "<script>bad()</script>", quote: "<img src=x>",
      assessment: { verdict: 'unknown" onclick="bad()', reason: "<svg onload=bad()>" } }],
  });
  assert.equal(host.querySelectorAll("script,img,svg,[onclick]").length, 0);
  assert.equal(host.querySelector(".relevance-unknown").textContent, "未核验");
});


test("rerank trace exposes same-pool movement without a quality or version claim", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{retrieval_diagnostics:{status:"empty",rerank:{status:"completed",candidate_count:2,latency_ms:12,model:"<img src=x>",model_revision_pin:"unknown",fallback_used:false,requested_top_k:1,candidate_pool_sha256:"abc",note:"同池执行记录，不是质量提升",ranking:[{chunk_id:"id",before_rank:2,after_rank:1}]}}});
  assert.match(host.textContent,/同池 2 条/);
  assert.match(host.textContent,/原排名 2 → 重排 1/);
  assert.match(host.textContent,/修订固定：unknown/);
  assert.match(host.textContent,/不是质量提升/);
  assert.equal(host.querySelectorAll("img").length,0);
});


test("answer review phases separate repair work and retain unknown unit coverage", () => {
  const host=document.createElement("div");
  renderRetrievalEvidence(host,{query_plan:{status:"complete"},answer_validation:{status:"complete",kept_blocks:1,total_blocks:1,latency_ms:80,timing_contract:"answer-guard-stages-v1",stages:[
    {stage:"initial_review",status:"filtered",latency_ms:30,total_units:2,unknown_units:1,timing:{review_ms:28,local_validation_ms:1}},
    {stage:"repair_generation",status:"completed",latency_ms:20},
    {stage:"repair_review",status:"complete",latency_ms:29,total_units:2,unknown_units:0}
  ]}});
  assert.match(host.textContent,/首轮审核：部分块未保留 · 30 ms · 2 个单元 \/ 未知 1/);
  assert.match(host.textContent,/纠错生成：完成 · 20 ms/);
  assert.match(host.textContent,/再次审核：复核完成 · 29 ms/);
  assert.match(host.textContent,/不可与总耗时重复相加/);
  assert.match(host.textContent,/查询改写/);
});
