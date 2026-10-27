const STATES = {queued:"排队",running:"处理中",retry_wait:"等待自动重试",completed:"已完成",failed:"失败",skipped:"重复跳过",rejected:"拒收",interrupted:"已中断"};
const STAGES = {admission:"接收检查",uploaded:"已保存",reading:"读取校验",parsing:"解析",chunking:"分块",embedding:"向量化",publishing:"提交索引",completed:"已完成"};
function el(tag, text, cls) { const n=document.createElement(tag); if(text !== undefined) n.textContent=text; if(cls) n.className=cls; return n; }

export function renderBatches(host, batches, onRetry, onHistory, onDiagnostics) {
  const opened = new Set([...host.querySelectorAll('details[open]')].map(n=>n.dataset.batchId));
  const wasEmpty = !host.children.length;
  host.replaceChildren();
  if (!batches.length) { host.append(el('p','暂无批任务。上传可一次选择 1–20 个文件；关闭页面后服务仍会继续处理。')); return; }
  host.append(el('p','最近 10 批。完成仅表示执行完成，不表示文档或回答质量通过。','retrieval-note'));
  for (const batch of batches) {
    const card=el('details',undefined,'batch-card'); card.dataset.batchId=batch.id;
    card.open=opened.has(batch.id) || (wasEmpty && batches[0]===batch);
    const c=batch.counts;
    card.append(el('summary',`${new Date(batch.created_at).toLocaleString('zh-CN')} · ${batch.total} 文件 · 完成 ${c.completed} · 处理中 ${batch.active} · 失败 ${c.failed} · 拒收 ${c.rejected} · 跳过 ${c.skipped}`));
    const table=el('table',undefined,'batch-table');
    const head=el('tr'); for(const h of ['文件','状态 / 阶段','尝试','结果与失败范围']) head.append(el('th',h)); table.append(head);
    for(const item of batch.items) {
      const tr=el('tr');
      let note=item.error_message || (item.cache_reused ? '已复用校验通过的解析缓存' : '');
      if(['failed','retry_wait','rejected'].includes(item.status)) note+=item.failure_pages_known ? `；失败页：${item.failure_pages.map(p=>`${p.page_number}（${p.reason === 'page_reason_unknown' ? '页级原因未知' : p.reason}）`).join('、')}` : '；失败页未知（只有文件/阶段级诊断）';
      tr.append(el('td',item.filename), el('td',`${STATES[item.status] || item.status} / ${STAGES[item.stage] || item.stage}`),
        el('td',`${item.attempt_count} / ${item.max_attempts}`), el('td',note)); table.append(tr);
    }
    card.append(table);
    const actions=el('div',undefined,'segment-actions');
    const message=el('p',undefined,'inline-error'); message.setAttribute('role','alert');
    if(onDiagnostics) {
      const diagnose=el('button','检查整批来源与切片','button button-small button-outline'); diagnose.type='button';
      diagnose.addEventListener('click',async()=>{ diagnose.disabled=true; try {await onDiagnostics(batch.id);} catch(e){message.textContent=e.message;} finally {diagnose.disabled=false;} });
      actions.append(diagnose);
    }
    if(batch.can_retry && onRetry) {
      const retry=el('button','重试失败文件','button button-small button-outline'); retry.type='button';
      retry.addEventListener('click',async()=>{ retry.disabled=true; try {await onRetry(batch.id);} catch(e){message.textContent=e.message;} finally {retry.disabled=false;} }); actions.append(retry);
    }
    if(onHistory) {
      const history=el('button','查看尝试记录','text-button'); history.type='button';
      const detail=el('div');
      history.addEventListener('click',async()=>{
        history.disabled=true; detail.textContent='加载尝试记录…';
        try {
          const result=await onHistory(batch.id); detail.replaceChildren();
          if(!result.attempts.length) detail.append(el('p','尚未开始执行；拒收/重复跳过不会创建模型调用尝试。'));
          for(const a of result.attempts) detail.append(el('p',`${result.items?.find(j=>j.id===a.job_id)?.filename || a.job_id} · 尝试 ${a.number} · ${STATES[a.status] || a.status} / ${STAGES[a.stage] || a.stage} · ${a.id}${a.retry_of_attempt_id ? ` · 重试自 ${a.retry_of_attempt_id}` : ''}${a.error_message ? ` · ${a.error_message}` : ''}${a.execution_config?.embedding_model ? ` · embedding ${a.execution_config.embedding_model}` : ''}`,'retrieval-note'));
        } catch(e){detail.textContent=e.message;} finally {history.disabled=false;}
      });
      actions.append(history); card.append(detail);
    }
    card.append(actions,message); host.append(card);
  }
}
