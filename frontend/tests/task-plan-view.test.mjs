import assert from "node:assert/strict";
import {test} from "node:test";
import {JSDOM} from "jsdom";
import {renderRetrievalEvidence} from "../retrieval-view.js";
import {taskPlanLines} from "../task-plan-view.js";

test("v2 source-reference diagnostics expose counts without inventing the invalid value",()=>{
  assert.match(taskPlanLines({version:"quoted-constraints-static-plan-v2-source-ids",goal:"goal"}).join(" "),/任务目标/);
  const dom=new JSDOM("<div id='host'></div>");globalThis.document=dom.window.document;
  const host=document.querySelector("#host");
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",planning_mode:"quoted-constraints-static-plan-v2-source-ids",
    trace_contract:"query-rewrite-diagnostics-v1",reason_code:"invalid_step_sources",phase:"validation",
    validation_failure:{step_id:"s1",source_ref_count:1,unknown_source_ref_count:1,has_dependency:false}}});
  assert.match(host.textContent,/步骤 s1：声明 1 个/);assert.match(host.textContent,/未保存无效引用的原值/);
  dom.window.close();
});

test("plan details distinguish independent hints from waiting dependencies and safe goal text",()=>{
  const dom=new JSDOM("<div id='host'></div>");globalThis.document=dom.window.document;
  const host=document.querySelector("#host");
  renderRetrievalEvidence(host,{query_plan:{status:"complete",planning_mode:"quoted-constraints-static-plan-v1",
    task_plan:{version:"quoted-constraints-static-plan-v1",goal:"<img src=x onerror=bad()>",
      constraints:[{id:"c1",scope:"global",source_id:"current",quote:"不要生产"}],steps:[
        {id:"s1",kind:"lookup",question:"查测试",depends_on:[],constraint_refs:["c1"],status:"planned_lookup"},
        {id:"s2",kind:"lookup",question:"根据结果再查",depends_on:["s1"],constraint_refs:["c1"],status:"waiting_on_evidence"}],
      retrieval_status:"shared_retriever_returned",shared_candidate_count:2}}});
  assert.match(host.textContent,/问题提炼与拆解/);assert.match(host.textContent,/待依赖证据，未执行/);
  assert.match(host.textContent,/不代表每步都有证据/);assert.match(host.textContent,/不证明提取完整或语义正确/);
  assert.equal(host.querySelector("img"),null);dom.window.close();
});

test("invalid task plan keeps explicit fallback reason and does not invent subtask completion",()=>{
  const dom=new JSDOM("<div id='host'></div>");globalThis.document=dom.window.document;
  const host=document.querySelector("#host");
  renderRetrievalEvidence(host,{query_plan:{status:"unavailable",planning_mode:"quoted-constraints-static-plan-v1",
    trace_contract:"query-rewrite-diagnostics-v1",phase:"validation",reason_code:"invalid_task_dependency",fallback_used:true,original_only:true}});
  assert.match(host.textContent,/前向\/循环引用/);assert.doesNotMatch(host.textContent,/任务目标/);
  dom.window.close();
});
