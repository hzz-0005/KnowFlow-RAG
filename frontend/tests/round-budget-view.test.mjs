import assert from "node:assert/strict";
import {test} from "node:test";
import {JSDOM} from "jsdom";
import {roundBudgetLines} from "../round-budget-view.js";
import {renderRetrievalEvidence} from "../retrieval-view.js";

test("round totals preserve unknown usage and distinguish admission from response quality",()=>{
  const lines=roundBudgetLines({version:"online-round-admission-v1",status:"stopped",attempted_calls:3,
    call_limit:3,blocked_calls:1,committed_reserved_tokens:1234,token_reservation_limit:2000,
    observed_total_tokens:null,known_reported_tokens_subtotal:99,usage_unknown_calls:1,
    calls:[{stage:"answer_review",kind:"chat",status:"blocked",reason:"provider_calls_exhausted",reserved_tokens:1000}]}).join(" ");
  assert.match(lines,/实报总token 未知，已知小计 99/);assert.match(lines,/已阻止，未发送/);
  assert.match(lines,/收到响应不代表答案正确/);assert.match(lines,/未计算账单金额/);
  assert.deepEqual(roundBudgetLines(null),[]);
});

test("round budget details render safely and old records do not get a fake ledger",()=>{
  const dom=new JSDOM("<div id='host'></div>");globalThis.document=dom.window.document;
  const host=document.querySelector("#host");
  renderRetrievalEvidence(host,{round_budget:{version:"online-round-admission-v1",calls:[{
    stage:"<img src=x onerror=bad()>",kind:"chat",status:"failed",reason:"<script>bad()</script>"}]}});
  assert.match(host.textContent,/整轮执行预算/);assert.equal(host.querySelector("img,script"),null);
  renderRetrievalEvidence(host,{});assert.doesNotMatch(host.textContent,/整轮执行预算/);
  dom.window.close();
});
