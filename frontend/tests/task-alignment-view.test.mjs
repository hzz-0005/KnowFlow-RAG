import assert from "node:assert/strict";
import {test} from "node:test";
import {JSDOM} from "jsdom";
import {renderTaskAlignment} from "../task-alignment-view.js";

test("task mismatch is separate from source support and escaped, with no success score",()=>{
  const dom=new JSDOM('<div id="host"></div>');globalThis.document=dom.window.document;
  const host=document.querySelector('#host');
  renderTaskAlignment(host,[{request_id:"r1",status:"complete",outcome:"mismatch_detected",result:{model:"judge",same_model_as_generator:false,
    requirements:[{id:"r1",source_id:"task_goal",request_quote:"production only",answer_quote:"<img src=x onerror=bad()>",verdict:"violated",reason:"scope differs"},
      {id:"r2",source_id:"current",request_quote:"continue",answer_quote:"",verdict:"unknown",reason:"model explanation",quote_validation:"failed_unknown"}]}}],async()=>{});
  assert.match(host.textContent,/模型发现任务偏离/);assert.match(host.textContent,/不是任务成功率/);
  assert.match(host.textContent,/模型修订未固定/);assert.equal(host.querySelector('img'),null);
  assert.match(host.textContent,/模型理由不能代替有效节选/);
  assert.equal(host.querySelector('.task-requirement').dataset.verdict,"violated");dom.window.close();
});

test("unknown, reconstruction and full request budget are explicit",()=>{
  const dom=new JSDOM('<div id="host"></div>');globalThis.document=dom.window.document;
  const host=document.querySelector('#host');
  renderTaskAlignment(host,[{status:"unavailable",outcome:"unknown",input_snapshot:{goal_basis:"reconstructed_from_task_source_turn_only"},
    result:{context_budget:{version:"answer-guard-request-window-v1",input_tokens_estimate:200,output_reserved_tokens:8192,safety_tokens:1024,estimated_total_tokens:9416,configured_total_tokens:32768,provider_usage:null}}}],async()=>{});
  assert.match(host.textContent,/结果未知/);assert.match(host.textContent,/未使用后来的任务更新/);
  assert.match(host.textContent,/9416 \/ 32768/);assert.match(host.textContent,/实报输入 未知/);dom.window.close();
});

test("running check disables repeated calls while an empty panel is opt-in",()=>{
  const dom=new JSDOM('<div id="host"></div>');globalThis.document=dom.window.document;
  const host=document.querySelector('#host');let calls=0;
  renderTaskAlignment(host,[],async()=>{calls++;});
  assert.match(host.textContent,/尚未运行/);assert.match(host.textContent,/额外模型调用/);assert.equal(calls,0);
  host.replaceChildren();renderTaskAlignment(host,[{status:"running",outcome:"unknown"}],async()=>{calls++;});
  assert.equal(host.querySelector('button').disabled,true);host.querySelector('button').click();assert.equal(calls,0);dom.window.close();
});

test("expired review can be retried explicitly and never starts a call by rendering",()=>{
  const dom=new JSDOM('<div id="host"></div>');globalThis.document=dom.window.document;
  const host=document.querySelector('#host');let calls=0;
  renderTaskAlignment(host,[{status:"running",outcome:"unknown",lease_until:"2020-01-01T00:00:00Z"}],async()=>{calls++;});
  assert.equal(calls,0);assert.match(host.textContent,/核对等待已过期，结果未知/);
  assert.equal(host.querySelector('button').disabled,false);dom.window.close();
});
