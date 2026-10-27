import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {test} from "node:test";
import {JSDOM} from "jsdom";

test("referent choices disclose bounded scope and unconfirmed binding, with escaped witnesses",async()=>{
  const html=await readFile(new URL("../index.html",import.meta.url),"utf8");
  const source=await readFile(new URL("../app.js",import.meta.url),"utf8");
  const dom=new JSDOM(html,{url:"http://localhost:8080/",runScripts:"outside-only"});
  const {window}=dom;
  window.Headers=Headers;
  const calls=[];window.fetch=async(...args)=>{calls.push(args);return {ok:true,status:200,json:async()=>[]};};
  window.eval(source.replace(/^import[^\n]*\n/,"")+"\nglobalThis.optionView={referentOptionsHtml,partialAnswerHtml};");
  try{
    const host=window.document.createElement("div");
    const info={version:"grounded-referent-options-v1",status:"conditional_unique",coverage:"all_processed_text_supplied",
      options:[{name:'<img src=x onerror="bad()">',filename:"rules.txt",quote:"An exact <script>untrusted</script> witness"}]};
    host.innerHTML=window.optionView.referentOptionsHtml(info);
    assert.equal(host.querySelector("img,script"),null);
    assert.match(host.textContent,/尚未由你确认/);
    assert.match(host.textContent,/<img/);
    assert.match(host.textContent,/穷尽性仍未验证/);
    host.innerHTML=window.optionView.referentOptionsHtml({...info,coverage:"retrieved_subset",status:"confirm_options"});
    assert.match(host.textContent,/不能据此断言全库唯一/);
    assert.equal(host.querySelector("button,input,a"),null);
    assert.match(window.optionView.partialAnswerHtml({version:info.version,status:"answered",blocked_scopes:[]}),/整道问题尚未完成/);
    assert.equal(calls.length,0);
    host.innerHTML=window.optionView.referentOptionsHtml({...info,
      version:"grounded-referent-options-v4-actor-roles",coverage:"retrieved_subset",status:"conditional_candidate"});
    assert.match(host.textContent,/尚未由你确认/);
    assert.match(host.textContent,/不能据此断言全库唯一/);
    assert.equal(host.querySelector("img,script"),null);
    assert.match(window.optionView.partialAnswerHtml({version:"grounded-referent-options-v4-actor-roles",status:"answered"}),/条件回答，指代待确认/);
  }finally{dom.window.close();}
});
