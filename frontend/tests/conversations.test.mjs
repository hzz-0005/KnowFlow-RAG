import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {randomUUID} from "node:crypto";
import {test} from "node:test";
import {JSDOM} from "jsdom";
import {understandingBudgetLines} from "../understanding-budget-view.js";
import {roundBudgetLines} from "../round-budget-view.js";
import {renderTaskAlignment} from "../task-alignment-view.js";

async function harness(fetch) {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const source = await readFile(new URL("../app.js", import.meta.url), "utf8");
  const dom = new JSDOM(html, {url:"http://localhost:8080/", runScripts:"outside-only"});
  const {window} = dom;
  window.Headers = Headers; window.crypto.randomUUID = randomUUID;
  window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute("open", ""); };
  window.HTMLDialogElement.prototype.close = function () { this.removeAttribute("open"); };
  window.renderRichAnswer = async (node,text) => { node.textContent=text; };
  window.understandingBudgetLines = understandingBudgetLines;
  window.roundBudgetLines = roundBudgetLines;
  window.renderTaskAlignment = (...args) => {
    globalThis.document=window.document;return renderTaskAlignment(...args);
  };
  window.fetch = async (url, options) => {
    const result = await fetch(url, options);
    return {ok:true,status:200,json:async()=>result};
  };
  window.eval(source.replace(/^import[^\n]*\n/, "") + `
    renderAnswer = async response => {
      state.lastAnswer = response;
      document.querySelector('#answer-panel').textContent = response.answer;
      document.querySelector('#answer-panel').prepend(document.createElement('h2'));
    };
    globalThis.harness = {state,loadConversations,loadConversation,renderKnowledgeBases,refreshWorkspace,signOut};
  `);
  const {state} = window.harness;
  state.token = "offline-test";
  state.currentKnowledgeBase = {id:"alpha",name:"Alpha"};
  state.knowledgeBases = [state.currentKnowledgeBase];
  window.harness.renderKnowledgeBases();
  return {dom, window, ...window.harness};
}

const tick = () => new Promise(resolve=>setImmediate(resolve));
const turn = (sequence, question, answer) => ({id:`turn${sequence}`, sequence, question,
  status:"completed", response:{answer,citations:[]}, resolution:{status:"standalone"}});

test("review export is explicit GET, downloads a local package and never invokes a judge",async()=>{
  const calls=[],downloads=[],blobs=[];
  const h=await harness((url,options)=>{calls.push([url,options]);if(url.endsWith('/review-export'))return {schema_version:'conversation-review-bundle-v1',snapshot_sha256:'offline'};return {id:'c1',revision:1,turns:[turn(1,'question','answer')],has_more:false};});
  try{
    h.window.URL.createObjectURL=blob=>{blobs.push(blob);return 'blob:offline';};h.window.URL.revokeObjectURL=()=>{};
    h.window.HTMLAnchorElement.prototype.click=function(){downloads.push({href:this.href,name:this.download});};
    await h.loadConversation('c1',h.state.conversationEpoch);
    h.window.document.querySelector('[data-turn-id]').click();await tick();
    assert.equal(calls.filter(([u])=>u.endsWith('/review-export')).length,0);
    h.window.document.querySelector('[data-export-review]').click();await tick();await tick();
    assert.equal(downloads.length,1);assert.match(downloads[0].name,/conversation-review-c1-turn-1.json/);
    assert.equal(blobs[0].type,'application/json;charset=utf-8');
    assert.ok(calls.every(([,o])=>!o.method));
    assert.equal(h.state.turns[0].response.answer,'answer');
  }finally{h.dom.window.close();}
});

test("late review export cannot download after switching to another chat",async()=>{
  let release;const downloads=[];
  const h=await harness(url=>url.endsWith('/review-export')?new Promise(resolve=>{release=resolve;}):{id:'c1',revision:1,turns:[turn(1,'question','answer')],has_more:false});
  try{
    h.window.URL.createObjectURL=()=>{downloads.push('unexpected');return 'blob:bad';};
    await h.loadConversation('c1',h.state.conversationEpoch);h.window.document.querySelector('[data-turn-id]').click();await tick();
    h.window.document.querySelector('[data-export-review]').click();await tick();
    h.window.document.querySelector('[data-nav-action="new"]').click();release({schema_version:'conversation-review-bundle-v1'});await tick();await tick();
    assert.equal(downloads.length,0);assert.equal(h.state.conversation,null);
  }finally{h.dom.window.close();}
});

test("sidebar disclosure reads another library without switching or creating a conversation",async()=>{
  const calls=[];
  const h=await harness((url,options)=>{calls.push([url,options]);return {items:[{id:'b1',title:'<img src=x onerror=bad()> A long chat'}]};});
  try{
    h.state.knowledgeBases.push({id:'beta',name:'Beta <unsafe>'});h.renderKnowledgeBases();
    h.window.document.querySelector('[data-nav-action="toggle"][data-kb-id="beta"]').click();await tick();await tick();
    assert.equal(h.state.currentKnowledgeBase.id,'alpha');assert.equal(h.state.conversation,null);
    assert.equal(calls.length,1);assert.equal(calls[0][1].method,undefined);
    const disclosure=h.window.document.querySelector('[data-nav-action="toggle"][data-kb-id="beta"]');
    assert.equal(disclosure.getAttribute('aria-expanded'),'true');
    assert.equal(h.window.document.querySelector('#chats-beta').hidden,false);
    assert.equal(h.window.document.querySelector('#kb-list img'),null);
    assert.match(h.window.document.querySelector('#chats-beta').textContent,/<img/);
    disclosure.click();assert.equal(h.window.document.querySelector('#chats-beta').hidden,true);
    assert.equal(h.state.currentKnowledgeBase.id,'alpha');assert.equal(calls.length,1);
  }finally{h.dom.window.close();}
});

test("sidebar chat selection restores history, highlights the selected chat and new chat is an unsent draft",async()=>{
  const calls=[];
  const h=await harness((url,options)=>{calls.push([url,options]);return {id:'c2',revision:1,turns:[turn(1,'second question','second answer')],has_more:false};});
  try{
    h.state.conversations=[{id:'c1',title:'First'},{id:'c2',title:'Second'}];h.state.expandedKnowledgeBases.add('alpha');h.renderKnowledgeBases();
    h.window.document.querySelector('[data-chat-id="c2"]').click();await tick();await tick();
    assert.match(h.window.document.querySelector('#conversation-history').textContent,/second answer/);
    assert.equal(h.window.document.querySelector('[data-chat-id="c2"]').getAttribute('aria-current'),'true');
    assert.equal(h.window.sessionStorage.getItem('knowflow_conversation_alpha'),'c2');
    h.window.document.querySelector('[data-nav-action="new"]').click();await tick();
    assert.equal(h.state.conversation,null);assert.equal(h.state.turns.length,0);
    assert.equal(h.window.sessionStorage.getItem('knowflow_conversation_alpha'),null);
    assert.match(h.window.document.querySelector('#chats-alpha').textContent,/尚未发送/);
    assert.equal(calls.length,1);assert.ok(calls.every(([,o])=>!o.method));
    assert.equal(h.window.document.querySelector('.main-column #conversation-select'),null);
    assert.equal(h.window.document.querySelector('#refresh-conversation'),null);
  }finally{h.dom.window.close();}
});

test("library actions are scoped to their own library and cancel never deletes",async()=>{
  const calls=[];
  const h=await harness((url,options)=>{calls.push([url,options]);return {};});
  try{
    h.state.knowledgeBases.push({id:'beta',name:'Beta'});h.renderKnowledgeBases();let confirmation;
    h.window.confirm=text=>{confirmation=text;return false;};
    h.window.document.querySelector('[data-nav-action="menu"][data-kb-id="beta"]').click();
    const betaDelete=h.window.document.querySelector('[data-kb-context-action="delete"]');
    assert.ok(betaDelete);betaDelete.click();await tick();
    assert.match(confirmation,/Beta.*文档、索引和聊天记录/);assert.equal(calls.length,0);
    assert.equal(h.state.currentKnowledgeBase.id,'alpha');
    assert.ok(h.window.document.querySelector('[data-nav-action="new"][data-kb-id="alpha"]'));
    assert.ok(h.window.document.querySelector('[data-nav-action="menu"][data-kb-id="alpha"]'));
    assert.equal(h.window.document.querySelector('#kb-context-menu').hidden,true);
  }finally{h.dom.window.close();}
});

test("late library list after sign out cannot restore data even when a new session reuses its id",async()=>{
  let release;
  const h=await harness(()=>new Promise(resolve=>{release=resolve;}));
  try{
    h.window.document.querySelector('[data-nav-action="toggle"]').click();await tick();
    h.signOut(false);h.state.token='new-session';h.state.knowledgeBases=[{id:'alpha',name:'New Alpha'}];h.state.currentKnowledgeBase=h.state.knowledgeBases[0];h.renderKnowledgeBases();
    release({items:[{id:'secret-old',title:'Previous account history'}]});await tick();await tick();
    assert.equal(h.state.sidebarConversations.size,0);assert.equal(h.state.conversations.length,0);
    assert.doesNotMatch(h.window.document.querySelector('#kb-list').textContent,/Previous account/);
  }finally{h.dom.window.close();}
});

test("record recovery only GETs the failed target and never repeats a paid turn",async()=>{
  let fail=true;const calls=[];
  const h=await harness((url,options)=>{calls.push([url,options]);if(fail)throw new Error('connection lost');return {id:'c1',revision:1,turns:[turn(1,'saved','preserved')],has_more:false};});
  try{
    h.state.conversations=[{id:'c1',title:'Saved'}];h.renderKnowledgeBases();
    h.window.document.querySelector('[data-chat-id="c1"]').click();await tick();await tick();
    assert.equal(h.window.document.querySelector('#reload-records').classList.contains('is-hidden'),false);
    fail=false;h.window.document.querySelector('#reload-records').click();await tick();await tick();
    assert.match(h.window.document.querySelector('#conversation-history').textContent,/preserved/);
    assert.equal(h.window.document.querySelector('#reload-records').classList.contains('is-hidden'),true);
    assert.deepEqual(calls.map(([url])=>url),['/api/v1/conversations/c1','/api/v1/conversations/c1']);
    assert.ok(calls.every(([,o])=>!o.method));
  }finally{h.dom.window.close();}
});

test("new chat from another library switches scope without creating a paid turn",async()=>{
  const calls=[],kbs=[{id:'alpha',name:'Alpha'},{id:'beta',name:'Beta'}];
  const h=await harness((url,options)=>{calls.push([url,options]);if(url==='/api/v1/knowledge-bases')return kbs;if(url.endsWith('/conversations'))return {items:[{id:'old-beta',title:'Beta history'}]};return [];});
  try{
    h.state.knowledgeBases=kbs;h.state.conversation={id:'old-alpha',revision:1};h.window.sessionStorage.setItem('knowflow_conversation_beta','old-beta');h.renderKnowledgeBases();
    h.window.document.querySelector('[data-nav-action="new"][data-kb-id="beta"]').click();await tick();await tick();await tick();
    assert.equal(h.state.currentKnowledgeBase.id,'beta');assert.equal(h.state.conversation,null);
    assert.equal(h.window.sessionStorage.getItem('knowflow_conversation_beta'),null);
    assert.match(h.window.document.querySelector('#chats-beta').textContent,/Beta history/);
    assert.ok(calls.every(([,o])=>!o.method));
    h.window.document.querySelector('#mobile-menu').click();
    assert.equal(h.window.document.querySelector('#mobile-menu').getAttribute('aria-expanded'),'true');
    h.window.document.querySelector('#close-sidebar').click();
    assert.equal(h.window.document.querySelector('.sidebar').classList.contains('mobile-open'),false);
  }finally{h.dom.window.close();}
});

test("confirmed deletion targets only its own library and retains the currently open chat",async()=>{
  const calls=[];
  const h=await harness((url,options)=>{calls.push([url,options]);if(options.method==='DELETE')return null;if(url==='/api/v1/knowledge-bases')return [{id:'alpha',name:'Alpha'}];return [];});
  try{
    h.state.knowledgeBases.push({id:'beta',name:'Beta'});h.state.conversation={id:'keep-alpha',revision:1};h.state.contentKnowledgeBaseId='alpha';h.renderKnowledgeBases();h.window.confirm=()=>true;
    h.window.document.querySelector('[data-nav-action="menu"][data-kb-id="beta"]').click();
    h.window.document.querySelector('[data-kb-context-action="delete"]').click();await tick();await tick();await tick();
    assert.equal(calls.find(([,o])=>o.method==='DELETE')[0],'/api/v1/knowledge-bases/beta');
    assert.equal(h.state.currentKnowledgeBase.id,'alpha');assert.equal(h.state.conversation.id,'keep-alpha');
    assert.equal(h.window.document.querySelector('[data-kb-id="beta"]'),null);
  }finally{h.dom.window.close();}
});

test("task review is explicit, updates its own turn and keeps the original answer",async()=>{
  const row=turn(1,"production only","original answer");row.conversation_id="c1";
  const calls=[];
  const h=await harness((url,options)=>{
    if(url.endsWith('/task-review')){calls.push([url,options]);return {cached:false,check:{request_id:"review1",status:"complete",outcome:"mismatch_detected",result:{requirements:[]}}};}
    return {id:"c1",revision:1,turns:[row],has_more:false};
  });
  try{
    await h.loadConversation("c1",h.state.conversationEpoch);
    h.window.document.querySelector('[data-turn-id]').click();await tick();
    assert.equal(calls.length,0);
    h.window.document.querySelector('.task-alignment button').click();await tick();await tick();
    assert.equal(calls.length,1);assert.match(calls[0][0],/c1\/turns\/turn1\/task-review$/);
    assert.ok(JSON.parse(calls[0][1].body).request_id);
    assert.equal(h.state.turns[0].response.answer,"original answer");
    assert.match(h.window.document.querySelector('.task-alignment').textContent,/模型发现任务偏离/);
    assert.equal(h.window.document.querySelector('.task-alignment').open,true);
  }finally{h.dom.window.close();}
});

test("memory manager escapes content and sends explicit activation with revision",async()=>{
  const calls=[];const row={id:"m1",namespace:"kb:alpha",key:"my archive",kind:"alias",content:"<img src=x onerror=bad()>",status:"candidate",revision:2,source:{type:"explicit_user_entry"}};
  const h=await harness((url,options)=>{calls.push([url,options]);return {items:[row]};});
  try{
    h.window.HTMLElement.prototype.scrollIntoView=function(){};
    h.window.document.querySelector("#open-memories").click();await tick();await tick();
    assert.equal(h.window.document.querySelector("#memory-list img"),null);
    assert.match(h.window.document.querySelector("#memory-list").textContent,/候选/);
    h.window.document.querySelector("[data-edit-memory]").click();
    assert.equal(h.window.document.querySelector("#memory-scope").disabled,true);
    h.window.document.querySelector("#memory-status").value="active";
    h.window.document.querySelector("#memory-form").dispatchEvent(new h.window.Event("submit",{cancelable:true}));
    await tick();await tick();
    const request=calls.find(([,o])=>o.method==="PUT");
    const body=JSON.parse(request[1].body);assert.equal(body.expected_revision,2);assert.equal(body.status,"active");
    assert.equal(h.window.document.querySelector("#memory-save").disabled,false);
  }finally{h.dom.window.close();}
});

test("task resume starts a new chat and submits task revision, not the old answer",async()=>{
  const calls=[];const task={id:"task1",title:"A long task",revision:4,status:"ready",updated_at:new Date().toISOString(),state:{goal:"Original goal",note:"fresh retrieval"},attempts:[]};
  const h=await harness((url,options)=>{
    calls.push([url,options]);
    if(url.includes("/tasks?"))return {items:[task]};
    if(url==="/api/v1/tasks/task1")return task;
    if(url.endsWith("/conversations"))return {id:"new",revision:0};
    if(url.endsWith("/turns"))return {};
    return {id:"new",revision:1,turns:[],has_more:false};
  });
  try{
    h.state.conversation={id:"old",revision:9};
    h.window.document.querySelector("#open-tasks").click();await tick();await tick();
    h.window.document.querySelector("[data-resume-task]").click();await tick();await tick();
    assert.equal(h.state.conversation,null);assert.equal(h.state.resumeTask.id,"task1");
    assert.match(h.window.document.querySelector("#resume-task-bar").textContent,/恢复任务/);
    h.window.document.querySelector("#query-form").dispatchEvent(new h.window.Event("submit",{cancelable:true}));
    await tick();await tick();
    const body=JSON.parse(calls.find(([u,o])=>u.endsWith("/turns")&&o.method==="POST")[1].body);
    assert.equal(body.task_id,"task1");assert.equal(body.task_revision,4);assert.equal(body.expected_revision,0);
    assert.equal(h.state.resumeTask,null);
  }finally{h.dom.window.close();}
});

test("turn details expose memory identity and task restart without claiming factual memory",async()=>{
  const row=turn(1,"Question","Answer");
  row.resolution.memory_context={selected:[{id:"m1",revision:3,namespace:"user"}],selected_tokens:48,budget_tokens:800,conflicts:[],skipped:[]};
  row.resolution.task_resume={task_id:"t1",source_versions_changed:["doc"]};
  row.resolution.status="resolved";row.resolution.intent_context={current_user_input:"Continue",active_task:{goal:{user_input:"Use production"}},referenced_user_inputs:[{turn_id:"memory:1:0",question:"saved staging clue"}]};
  const h=await harness(()=>({id:"c1",revision:1,turns:[row],has_more:false}));
  try{
    await h.loadConversation("c1",h.state.conversationEpoch);
    h.window.document.querySelector("[data-turn-id]").click();await tick();
    const text=h.window.document.querySelector("#answer-resolution").textContent;
    assert.match(text,/用户线索，不是文档证据/);assert.match(text,/来源变更\/缺失 1 项/);assert.match(text,/修订 3/);
    assert.match(text,/正在恢复的任务目标：Use production/);assert.match(text,/仅帮助识别，不覆盖任务条件/);
  }finally{h.dom.window.close();}
});

test("failed turn retains budget summary without exposing an unchecked answer",async()=>{
  const row=turn(1,"Q","UNVERIFIED-DRAFT");row.status="failed";row.response=null;
  row.error={message:"本轮未完成"};row.resolution.round_budget={version:"online-round-admission-v1",
    status:"stopped",attempted_calls:2,call_limit:12,blocked_calls:1,usage_unknown_calls:1};
  const h=await harness(()=>({id:"c1",revision:1,turns:[row],has_more:false}));
  try{
    await h.loadConversation("c1",h.state.conversationEpoch);
    const text=h.window.document.querySelector("#conversation-history").textContent;
    assert.match(text,/已尝试 2 \/ 12 次/);assert.match(text,/已触及限制/);
    assert.doesNotMatch(text,/UNVERIFIED-DRAFT/);
  }finally{h.dom.window.close();}
});

test("context details show whole-request budget and preserve historical unknowns", async () => {
  const all=[turn(1,"历史问题","旧回答"),turn(2,"那多久？","新回答")];
  all[0].resolution={version:"conversation-reference-v6-topic-boundary",status:"standalone",request_attempted:true};
  all[1].resolution={version:"conversation-reference-v6-topic-boundary",status:"resolved",request_attempted:true,
    request_budget:{version:"understanding-whole-request-v1",input_tokens_estimate:950,output_reserved_tokens:2048,
      safety_tokens:512,estimated_total_tokens:3510,configured_total_tokens:8192,estimated_feasible:true,
      provider_usage:{prompt_tokens:960,completion_tokens:0}}};
  const h=await harness(()=>({id:"c1",revision:2,turns:all,has_more:false}));
  try {
    await h.loadConversation("c1",h.state.conversationEpoch);
    h.window.document.querySelector('[data-turn-id="turn2"]').click();await tick();
    const panel=h.window.document.querySelector("#answer-resolution");
    assert.match(panel.textContent,/输入估算 950/);assert.match(panel.textContent,/实报输入 960、输出 0/);
    h.window.document.querySelector('[data-turn-id="turn1"]').click();await tick();
    assert.match(panel.textContent,/未记录本阶段完整请求预算，保持未知/);
    assert.doesNotMatch(panel.textContent,/输入估算 950/);
  } finally {h.dom.window.close();}
});

test("composer grows to four lines, scrolls beyond the cap and shrinks after content removal", async () => {
  const h = await harness(() => ({}));
  const {window,dom} = h;
  try {
    const input=window.document.querySelector("#question");
    input.style.lineHeight="24px"; input.style.padding="8px 6px"; input.style.border="0px";
    let measuredContent=40;
    Object.defineProperty(input,"scrollHeight",{get:()=>measuredContent});
    const resize=(text,height)=>{
      input.value=text; measuredContent=height;
      input.dispatchEvent(new window.Event("input"));
    };
    resize("one",40); assert.equal(input.style.height,"40px");
    resize("one\ntwo\nthree",88); assert.equal(input.style.height,"88px");
    resize("one\ntwo\nthree\nfour",112);
    assert.equal(input.style.height,"112px"); assert.equal(input.style.overflowY,"hidden");
    resize("pasted long content",256);
    assert.equal(input.style.height,"112px"); assert.equal(input.style.overflowY,"auto");
    resize("",40); assert.equal(input.style.height,"40px"); assert.equal(input.style.overflowY,"hidden");
  } finally {dom.window.close();}
});

test("restores scoped history, safely displays resolution, loads older turns and starts empty", async () => {
  const all = [turn(1,"old question","old answer"), turn(2,"它是什么？","fresh answer")];
  all[1].resolution.status = "resolved";
  all[1].resolved_question = '<img src=x onerror="bad()">独立问题';
  const h = await harness(url => {
    if (url.endsWith("/conversations")) return {items:[{id:"c1",title:"Saved <script>"}]};
    if (url.endsWith("?before=2")) return {id:"c1",revision:2,turns:[all[0]],has_more:false};
    return {id:"c1",title:"Saved",revision:2,turns:[all[1]],has_more:true};
  });
  const {window,state,dom} = h;
  try {
    window.sessionStorage.setItem("knowflow_conversation_alpha", "c1");
    await h.loadConversations();
    assert.equal(state.conversation.id,"c1");
    assert.equal(window.document.querySelector("#answer-dialog").open,false);
    assert.match(window.document.querySelector(".chat-assistant .chat-answer").textContent,/fresh answer/);
    window.document.querySelector('[data-turn-id="turn2"]').click(); await tick();
    assert.equal(window.document.querySelector("#answer-dialog").open,true);
    assert.equal(window.document.querySelector("#answer-panel").textContent,"fresh answer");
    assert.match(window.document.querySelector("#answer-resolution").textContent,/独立问题/);
    assert.equal(window.document.querySelector("#answer-resolution img"),null);
    window.document.querySelector("#older-turns").click(); await tick(); await tick();
    assert.equal(state.turns.length,2);
    window.document.querySelector('[data-turn-id="turn1"]').click(); await tick();
    assert.equal(state.lastAnswer.answer,"old answer");
    window.document.querySelector('[data-nav-action="new"][data-kb-id="alpha"]').click();
    assert.equal(state.conversation,null); assert.equal(state.lastAnswer,null);
    assert.equal(state.turns.length,0);
    assert.match(window.document.querySelector("#conversation-history").textContent,/从一个问题开始/);
    assert.equal(window.sessionStorage.getItem("knowflow_conversation_alpha"),null);
  } finally { dom.window.close(); }
});

test("details distinguish retrieval hints from escaped original constraints and clear on older turn", async () => {
  const all = [turn(1,"old question","old answer"),turn(2,"改为2025，不是2024。","fresh answer")];
  all[1].resolution = {status:"resolved",intent_context:{
    current_user_input:'改为2025，不是2024。<img src=x onerror="bad()">',
    referenced_user_inputs:[{question:"Only staging, exclude production.",
      supplements:['<script>bad()</script>']}],
  }};
  all[1].resolved_question="policy 2025, excluding 2024";
  const h = await harness(() => ({id:"c1",revision:2,turns:all,has_more:false}));
  const {window,state,dom} = h;
  try {
    await h.loadConversation("c1",state.conversationEpoch);
    window.document.querySelector('[data-turn-id="turn2"]').click(); await tick();
    const panel=window.document.querySelector("#answer-resolution");
    assert.match(panel.textContent,/检索问题与原始条件/);
    assert.match(panel.textContent,/改为2025，不是2024/);
    assert.match(panel.textContent,/Only staging, exclude production/);
    assert.match(panel.textContent,/本轮更正优先/);
    assert.equal(panel.querySelector("img,script"),null);
    window.document.querySelector('[data-turn-id="turn1"]').click(); await tick();
    assert.equal(panel.querySelector(".resolved-intent:not(.task-alignment)"),null);
    assert.match(panel.textContent,/本轮任务核对：尚未运行/);
  } finally {dom.window.close();}
});

test("history-aware topic switches expose the classifier call and discard prior constraints", async () => {
  const all=[turn(1,"old","old"),turn(2,"New topic","new")];
  all[1].resolution={version:"conversation-reference-v4-history-aware",status:"standalone",
    routing_reason:"history_present",request_attempted:true,latency_ms:123,
    provider_usage:{prompt_tokens:210,completion_tokens:29}};
  const h=await harness(()=>({id:"c1",revision:2,turns:all,has_more:false}));
  const {window,state,dom}=h;
  try {
    await h.loadConversation("c1",state.conversationEpoch);
    window.document.querySelector('[data-turn-id="turn2"]').click();await tick();
    const panel=window.document.querySelector("#answer-resolution");
    assert.match(panel.textContent,/上下文判断：独立问题/);
    assert.match(panel.textContent,/实报输入 210、输出 29/);
    assert.match(panel.textContent,/没有继承旧话题的条件/);
    assert.doesNotMatch(panel.textContent,/引用的历史用户条件/);
    window.document.querySelector('[data-turn-id="turn1"]').click();await tick();
    assert.equal(panel.querySelector(".resolved-intent:not(.task-alignment)"),null);
    assert.match(panel.textContent,/本轮任务核对：尚未运行/);
  }finally{dom.window.close();}
});

test("budget-blocked context clarifies without presenting an older topic or a model charge", async () => {
  const all=[turn(1,"How long?","Please restate")];
  all[0].status="needs_clarification";
  all[0].resolution={version:"conversation-reference-v4-history-aware",status:"clarify",
    routing_reason:"history_present",request_attempted:false,
    required_recent_intent_omitted:{turn_id:'<img src=x onerror="bad()">'}};
  const h=await harness(()=>({id:"c1",revision:1,turns:all,has_more:false}));
  const {window,state,dom}=h;
  try {
    await h.loadConversation("c1",state.conversationEpoch);
    window.document.querySelector('[data-turn-id="turn1"]').click();await tick();
    const panel=window.document.querySelector("#answer-resolution");
    assert.match(panel.textContent,/未发起上下文解析调用/);
    assert.match(panel.textContent,/未采用更早话题代替/);
    assert.equal(panel.querySelector("img,script"),null);
  }finally{dom.window.close();}
});

test("validated active topic trace explains omitted loaded intents without claiming deletion", async () => {
  const all=[turn(1,"How long?","14 days")];
  all[0].resolution={version:"conversation-reference-v6-topic-boundary",status:"resolved",
    routing_reason:"history_present",request_attempted:true,
    topic_scope:{older_loaded_turns_omitted:3,boundary_turn_id:'<img src=x>'}};
  const h=await harness(()=>({id:"c1",revision:1,turns:all,has_more:false}));
  const {window,state,dom}=h;
  try{
    await h.loadConversation("c1",state.conversationEpoch);
    window.document.querySelector('[data-turn-id="turn1"]').click();await tick();
    const panel=window.document.querySelector("#answer-resolution");
    assert.match(panel.textContent,/本次已读历史中未装入 3 条更早意图/);
    assert.match(panel.textContent,/历史记录仍保留/);
    assert.match(panel.textContent,/请明确对象和条件/);
    assert.equal(panel.querySelectorAll("img,script").length,0);
  }finally{dom.window.close();}
});

test("late history and submitted answer cannot populate a newly started conversation", async () => {
  let releaseHistory, releaseTurn;
  const h = await harness((url,options) => {
    if (url.endsWith("/turns")) return new Promise(resolve=>{ releaseTurn=resolve; });
    return new Promise(resolve=>{releaseHistory=resolve;});
  });
  const {window,state,dom} = h;
  try {
    const oldLoad = h.loadConversation("old",state.conversationEpoch);
    window.document.querySelector('[data-nav-action="new"][data-kb-id="alpha"]').click();
    releaseHistory({id:"old",revision:1,turns:[turn(1,"old","old")],has_more:false});
    await oldLoad;
    assert.equal(state.conversation,null);
    state.conversation={id:"c1",revision:0};
    window.document.querySelector("#question").value="first question";
    window.document.querySelector("#query-form").dispatchEvent(new window.Event("submit",{cancelable:true}));
    await tick();
    assert.equal(state.conversationBusy,true);
    assert.match(window.document.querySelector(".chat-user .chat-bubble").textContent,/first question/);
    assert.match(window.document.querySelector(".chat-assistant").textContent,/正在发送/);
    assert.equal(window.document.querySelector("#question").value,"");
    window.document.querySelector('[data-nav-action="new"][data-kb-id="alpha"]').click();
    window.document.querySelector("#question").value="new draft";
    releaseTurn(turn(1,"first question","late answer")); await tick(); await tick();
    assert.equal(state.conversation,null);
    assert.equal(window.document.querySelector("#question").value,"new draft");
    assert.equal(window.document.querySelector("#answer-panel").textContent,"");
    assert.equal(state.conversationBusy,false);
  } finally {dom.window.close();}
});

test("failed network retry reuses request key while changed question gets a different key", async () => {
  const requests=[];
  const h = await harness((url, options) => {
    if (url.endsWith("/turns")) { requests.push(JSON.parse(options.body)); throw new Error("network lost"); }
    return {};
  });
  const {window,state,dom} = h;
  try {
    state.conversation={id:"c1",revision:0};
    const send = async value => {
      window.document.querySelector("#question").value=value;
      window.document.querySelector("#query-form").dispatchEvent(new window.Event("submit",{cancelable:true}));
      await tick(); await tick();
    };
    await send("question A"); await send("question A"); await send("question B");
    assert.equal(requests.length,3);
    assert.equal(requests[0].request_id,requests[1].request_id);
    assert.notEqual(requests[1].request_id,requests[2].request_id);
    assert.match(window.document.querySelector("#query-error").textContent,/重新加载服务器记录/);
    assert.equal(window.document.querySelector("#reload-records").classList.contains("is-hidden"),false);
  } finally {dom.window.close();}
});

test("a failed newest turn clears the previously displayed answer and loading blocks submissions", async () => {
  let release;
  const h = await harness(() => new Promise(resolve=>{release=resolve;}));
  const {window,state,dom} = h;
  try {
    state.conversation={id:"c1",revision:1};
    state.lastAnswer={answer:"old answer"};
    window.document.querySelector("#answer-panel").textContent="old answer";
    const loading=h.loadConversation("c1",state.conversationEpoch);
    assert.equal(state.conversationBusy,true);
    assert.equal(window.document.querySelector("#ask-button").disabled,true);
    release({id:"c1",revision:2,turns:[{id:"failed",sequence:2,question:"new question",
      status:"failed",error:{message:"未完成"}}],has_more:false});
    await loading;
    assert.equal(state.lastAnswer,null);
    assert.equal(window.document.querySelector("#answer-panel").textContent,"");
    assert.match(window.document.querySelector("#conversation-history").textContent,/未完成/);
    assert.equal(state.conversationBusy,false);
  } finally {dom.window.close();}
});

test("a chat row deletes only itself after confirmation and removes its node",async()=>{
  const calls=[];
  const h=await harness((url,options)=>{calls.push([url,options]);if(options.method==='DELETE')return null;return {};});
  try{
    h.state.conversations=[{id:'c1',title:'First'},{id:'c2',title:'Second'}];
    h.state.expandedKnowledgeBases.add('alpha');h.renderKnowledgeBases();
    h.window.confirm=()=>true;
    assert.ok(h.window.document.querySelector('[data-nav-action="delete-chat"][data-chat-id="c2"]'));
    h.window.document.querySelector('[data-nav-action="delete-chat"][data-chat-id="c2"]').click();
    await tick();await tick();
    const removed=calls.find(([,options])=>options.method==='DELETE');
    assert.equal(removed[0],'/api/v1/conversations/c2');
    assert.deepEqual(h.state.conversations.map(row=>row.id),['c1']);
    assert.equal(h.window.document.querySelector('[data-chat-id="c2"]'),null);
    assert.ok(h.window.document.querySelector('[data-chat-id="c1"]'));
  }finally{h.dom.window.close();}
});

test("right-clicking a chat opens a one-item delete menu that Escape closes",async()=>{
  const h=await harness(()=>({}));
  try{
    h.state.conversations=[{id:'c1',title:'First'}];h.state.expandedKnowledgeBases.add('alpha');h.renderKnowledgeBases();
    const row=h.window.document.querySelector('[data-nav-action="chat"][data-chat-id="c1"]');
    row.dispatchEvent(new h.window.MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:20,clientY:20}));
    const menu=h.window.document.querySelector('#chat-context-menu');
    assert.equal(menu.hidden,false);
    assert.equal(menu.textContent.trim(),'删除对话');
    h.window.document.dispatchEvent(new h.window.KeyboardEvent('keydown',{key:'Escape'}));
    assert.equal(menu.hidden,true);
  }finally{h.dom.window.close();}
});

test("each sidebar row owns new/menu icons with compact hover reveal",async()=>{
  const css=await readFile(new URL("../styles.css",import.meta.url),"utf8");
  const h=await harness(()=>({}));
  try{
    h.state.knowledgeBases=[{id:"alpha",name:"Alpha"},{id:"beta",name:"Beta"}];
    h.state.currentKnowledgeBase=h.state.knowledgeBases[0];
    h.state.conversations=[{id:"c1",title:"First"},{id:"c2",title:"Second"}];
    h.state.expandedKnowledgeBases.add("alpha");h.renderKnowledgeBases();

    assert.equal(h.window.document.querySelectorAll(".kb-group .kb-actions").length,2);
    assert.equal(h.window.document.querySelectorAll('[data-nav-action="delete-chat"]').length,2);
    assert.ok(h.window.document.querySelector('[data-nav-action="new"] svg'));
    assert.ok(h.window.document.querySelector('[data-nav-action="menu"] svg'));
    assert.ok(h.window.document.querySelector('[data-nav-action="delete-chat"] svg'));

    // Compact mode uses hover or keyboard focus; wide mode exposes both icons.
    assert.match(css,/\.kb-row:hover \.kb-actions[^{]*\{[^}]*opacity:1/);
    assert.doesNotMatch(css,/\.kb-row:focus-within \.kb-actions/);
    assert.match(css,/\.chat-nav-row:hover \.chat-nav-delete[^{]*\{[^}]*opacity:1/);
    assert.doesNotMatch(css,/\.chat-nav-row:focus-within \.chat-nav-delete/);
    assert.match(css,/\.kb-actions[^{]*\{[^}]*opacity:0/);
    assert.match(css,/data-sidebar-compact="true"/);
  }finally{h.dom.window.close();}
});

test("knowledge-base menu opens on hover without deleting and Escape restores focus",async()=>{
  const calls=[],h=await harness((url)=>{calls.push(url);return {};});
  try{
    const doc=h.window.document,button=doc.querySelector('[data-nav-action="menu"]');
    button.dispatchEvent(new h.window.MouseEvent('pointerover',{bubbles:true}));
    const menu=doc.querySelector('#kb-context-menu');
    assert.equal(menu.hidden,false);assert.match(menu.textContent,/展开聊天/);assert.match(menu.textContent,/删除知识库/);
    assert.equal(calls.length,0);
    doc.dispatchEvent(new h.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
    assert.equal(menu.hidden,true);assert.equal(doc.activeElement,button);
    button.click();assert.equal(menu.hidden,false);
    doc.querySelector('#workspace-title').click();assert.equal(menu.hidden,true);
    assert.equal(calls.length,0);
  }finally{h.dom.window.close();}
});
