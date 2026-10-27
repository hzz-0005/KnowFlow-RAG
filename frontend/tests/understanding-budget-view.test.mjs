import assert from "node:assert/strict";
import {test} from "node:test";
import {JSDOM} from "jsdom";
import {understandingBudgetLines} from "../understanding-budget-view.js";
import {renderRetrievalEvidence} from "../retrieval-view.js";

test("understanding budget distinguishes local estimate, actual usage and unknown",()=>{
  const text=understandingBudgetLines({version:"understanding-whole-request-v1",input_tokens_estimate:900,
    output_reserved_tokens:2048,safety_tokens:512,estimated_total_tokens:3460,configured_total_tokens:8192,
    estimated_feasible:true,provider_usage:{prompt_tokens:930}}).join(" ");
  assert.match(text,/输入估算 900/);assert.match(text,/实报输入 930、输出 未记录/);
  assert.match(text,/不是整条查询的累计预算/);
  assert.match(understandingBudgetLines(null).join(" "),/保持未知/);
});

test("blocked rewrite remains original-only and displays no-call budget without HTML injection",()=>{
  const dom=new JSDOM("<div id='host'></div>");globalThis.document=dom.window.document;
  const host=document.querySelector("#host");
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",trace_contract:"query-rewrite-diagnostics-v1",
    phase:"budget",reason_code:"request_budget_exceeded",original_only:true,fallback_used:true,
    request_budget:{version:"understanding-whole-request-v1",estimated_feasible:false,
      input_tokens_estimate:"<img src=x onerror=bad()>",provider_usage:null}},retrieval_diagnostics:{status:"empty"}});
  assert.match(host.textContent,/完整请求超预算/);assert.match(host.textContent,/预算不足，未调用模型/);
  assert.equal(host.querySelector("img"),null);dom.window.close();
});
