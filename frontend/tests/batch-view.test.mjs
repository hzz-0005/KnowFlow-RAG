import test from 'node:test';
import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';
import {renderBatches} from '../batch-view.js';

function setup() { const dom=new JSDOM('<main></main>'); globalThis.document=dom.window.document; return document.querySelector('main'); }
function batch() { return {id:'batch',created_at:'2026-10-23T12:00:00Z',total:4,active:1,can_retry:true,
  counts:{completed:1,failed:1,rejected:1,skipped:0},items:[
    {id:'j1',filename:'<img src=x>.pdf',status:'failed',stage:'parsing',attempt_count:1,max_attempts:3,error_message:'bad file',failure_pages_known:false},
    {id:'j2',filename:'page.pdf',status:'retry_wait',stage:'parsing',attempt_count:2,max_attempts:3,error_message:'timeout',failure_pages_known:true,failure_pages:[{page_number:4,reason:'page_reason_unknown'}]},
    {id:'j3',filename:'cache.xlsx',status:'completed',stage:'completed',attempt_count:2,max_attempts:3,cache_reused:true},
    {id:'j4',filename:'empty.md',status:'rejected',stage:'admission',attempt_count:0,max_attempts:3,error_message:'empty'}]}; }

test('batch view keeps rejected/failed rows, page uncertainty and safe text rather than invented success',()=>{
  const host=setup(); renderBatches(host,[batch()],()=>{});
  assert.match(host.textContent,/完成仅表示执行完成/);
  assert.match(host.textContent,/失败页未知/);
  assert.match(host.textContent,/4（页级原因未知）/);
  assert.match(host.textContent,/已复用校验通过的解析缓存/);
  assert.equal(host.querySelector('img'),null);
  assert.equal(host.querySelectorAll('td').length,16);
});

test('retry errors remain visible and attempt histories retain filename and lineage',async()=>{
  const host=setup();
  renderBatches(host,[batch()],async()=>{throw new Error('queue full');}, async()=>({items:batch().items,attempts:[
    {job_id:'j3',number:2,status:'completed',stage:'completed',id:'a2',retry_of_attempt_id:'a1',execution_config:{embedding_model:'fixture'}}]}));
  [...host.querySelectorAll('button')].find(n=>n.textContent==='重试失败文件').click();
  await new Promise(r=>setTimeout(r,0)); assert.match(host.textContent,/queue full/);
  [...host.querySelectorAll('button')].find(n=>n.textContent==='查看尝试记录').click();
  await new Promise(r=>setTimeout(r,0)); assert.match(host.textContent,/cache.xlsx · 尝试 2/); assert.match(host.textContent,/重试自 a1/);
  host.querySelector('details').open=true; renderBatches(host,[batch()],()=>{});
  assert.equal(host.querySelector('details').open,true);
});

test('batch diagnosis delegates the selected batch without submitting a retry',async()=>{
  const host=setup(); let checked; let retries=0;
  renderBatches(host,[batch()],()=>{retries++;},null,async id=>{checked=id;});
  [...host.querySelectorAll('button')].find(n=>n.textContent==='检查整批来源与切片').click();
  await new Promise(r=>setTimeout(r,0));
  assert.equal(checked,'batch'); assert.equal(retries,0);
});
