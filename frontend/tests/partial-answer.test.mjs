import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {test} from "node:test";
import {JSDOM} from "jsdom";

test("partial answers keep their unfinished scope, escape source text and reload without sending", async()=>{
  const html=await readFile(new URL("../index.html",import.meta.url),"utf8");
  const source=await readFile(new URL("../app.js",import.meta.url),"utf8");
  const dom=new JSDOM(html,{url:"http://localhost:8080/",runScripts:"outside-only"});
  const {window}=dom,calls=[];
  window.Headers=Headers;
  window.HTMLElement.prototype.scrollIntoView=()=>{};
  window.renderRichAnswer=async(node,text)=>{node.textContent=text;};
  window.renderRetrievalEvidence=()=>{};
  const response={answer:"A checked statement. Still needs clarification.",strategy:"keyword",refused:false,citations:[],
    partial_answer:{version:"quoted-partial-lookup-v1",status:"answered",full_request_complete:false,
      question:'<img src=x onerror="bad()">',current_scope:"current constraints",blocked_scopes:["Missing entity"]}};
  const row={id:"t",sequence:1,question:"Composite question",status:"needs_clarification",response};
  window.fetch=async(url,o)=>{calls.push(o?.method||"GET");return {ok:true,status:200,json:async()=>({id:"c",revision:1,turns:[row],has_more:false})};};
  window.eval(source.replace(/^import[^\n]*\n/,"")+"\nglobalThis.partialTest={state,loadConversation,renderAnswer,partialAnswerHtml};");
  const h=window.partialTest;
  h.state.token="offline";h.state.currentKnowledgeBase={id:"kb",name:"KB"};
  try{
    await h.loadConversation("c",h.state.conversationEpoch);
    const scope=window.document.querySelector('#conversation-history [aria-label="部分回答范围"]');
    assert.match(scope.textContent,/整道问题尚未完成/);
    assert.match(scope.textContent,/核验仅对应这一部分/);
    assert.equal(scope.querySelector("details").open,false);
    assert.match(scope.textContent,/Missing entity/);
    assert.match(scope.textContent,/<img/);
    assert.equal(scope.querySelector("img"),null);
    assert.ok(calls.every(m=>m==="GET"));
    await h.renderAnswer(response);
    assert.match(window.document.querySelector('#answer-panel [aria-label="部分回答范围"]').textContent,/仍需补充/);
    assert.match(h.partialAnswerHtml({...response.partial_answer,status:"failed"}),/部分查询未返回答案/);
  }finally{dom.window.close();}
});
