import { renderTaskAlignment, roundBudgetLines, understandingBudgetLines, renderRichAnswer, renderRetrievalEvidence, RELEVANCE_NAMES, renderParsingReport, renderSourcePage, renderSegments, renderOcrReview, renderDocumentReferences, renderGraphEvidence, renderQualityReport, tabularLocations, renderBatches, renderDiagnostics, renderBatchDiagnostics, installWorkspaceLayout } from "/dist/answer-view.js?v=20261026-workspace-panes-v2";

const API = "/api/v1";
const TOKEN_KEY = "knowflow_access_token";
const STRATEGY_NAMES = {
  vector: "语义检索",
  keyword: "关键词检索",
  hybrid: "混合检索",
  hybrid_rerank: "混合 + 重排序",
  graph: "图谱检索",
  graph_hybrid: "图谱 + 文本",
};
const STATUS_NAMES = {
  pending: "等待处理",
  processing: "正在处理",
  processed: "已就绪",
  failed: "处理失败",
  not_requested: "尚未构建图谱",
};

const state = {
  token: sessionStorage.getItem(TOKEN_KEY),
  authMode: "login",
  knowledgeBases: [],
  currentKnowledgeBase: null,
  documents: [],
  evaluations: [],
  batches: [],
  pollTimer: null,
  workspaceRevision: 0,
  contentKnowledgeBaseId: null,
  conversations: [], conversation: null, turns: [], conversationEpoch: 0,
  conversationTimer: null, conversationBusy: false, pendingRequest: null,
  conversationLoadId: 0,
  selectedTurnId: null,
  expandedKnowledgeBases: new Set(), sidebarConversations: new Map(),
  sidebarEpoch: 0, sidebarInitialized: false, openKnowledgeBaseMenu: null, navigationRequestId: 0,
  deletingKnowledgeBases: new Set(), conversationLoadTarget: null,
  resumeTask: null, memories: [], tasks: [], memoryEdit: null,
};

const $ = (selector) => document.querySelector(selector);
const authView = $("#auth-view");
const appView = $("#app-view");
if (typeof installWorkspaceLayout === "function") installWorkspaceLayout(appView);
const authForm = $("#auth-form");
const authError = $("#auth-error");
const kbDialog = $("#kb-dialog");
const batchDialog = $("#batch-dialog");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("visible"), 3200);
}

function setError(element, message = "") {
  element.textContent = message;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  let body = options.body;
  if (body instanceof FormData || body instanceof URLSearchParams) {
    // Let the browser choose the multipart boundary or form encoding.
  } else if (body !== undefined && body !== null) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(body);
  }
  const response = await fetch(`${API}${path}`, { ...options, headers, body });
  const payload = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401 && state.token) signOut(false);
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message ? `${detail.message}${detail.rerank?.reason ? `（${detail.rerank.reason}）` : ""}` : `请求失败（${response.status}）`);
  }
  return payload;
}

function setAuthMode(mode) {
  state.authMode = mode;
  const registering = mode === "register";
  $("#auth-title").innerHTML = registering ? "建立你的<br />知识工作区" : "让每个答案<br />都能回到原文";
  $("#auth-submit").textContent = registering ? "创建账号" : "登录";
  $("#auth-switch-copy").textContent = registering ? "已经有账号？" : "还没有账号？";
  $("#auth-switch").textContent = registering ? "返回登录" : "创建账号";
  $("#auth-password").autocomplete = registering ? "new-password" : "current-password";
  setError(authError);
}

function showAuthenticatedApp() {
  authView.classList.add("is-hidden");
  appView.classList.remove("is-hidden");
}

function showAuth() {
  appView.classList.add("is-hidden");
  authView.classList.remove("is-hidden");
}

function signOut(notify = true) {
  closeKbMenu();
  state.sidebarEpoch++;
  state.sidebarConversations.clear(); state.expandedKnowledgeBases.clear();
  state.sidebarInitialized = false; state.openKnowledgeBaseMenu = null;
  state.knowledgeBases = [];
  resetConversation();
  state.workspaceRevision += 1;
  state.token = null;
  state.currentKnowledgeBase = null;
  state.batches = [];
  if (batchDialog.open) batchDialog.close();
  $("#batch-list").replaceChildren();
  delete $("#batch-list").dataset.snapshot;
  sessionStorage.removeItem(TOKEN_KEY);
  if (state.pollTimer) window.clearInterval(state.pollTimer);
  state.pollTimer = null;
  showAuth();
  if (notify) showToast("已退出登录");
}

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError(authError);
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  const submit = $("#auth-submit");
  submit.disabled = true;
  try {
    const result = state.authMode === "register"
      ? await api("/auth/register", { method: "POST", body: { email, password } })
      : await api("/auth/token", {
        method: "POST",
        body: new URLSearchParams({ username: email, password }),
      });
    state.token = result.access_token;
    sessionStorage.setItem(TOKEN_KEY, state.token);
    showAuthenticatedApp();
    await refreshWorkspace();
  } catch (error) {
    setError(authError, error.message);
  } finally {
    submit.disabled = false;
  }
});

$("#auth-switch").addEventListener("click", () => {
  setAuthMode(state.authMode === "login" ? "register" : "login");
});
$("#logout").addEventListener("click", () => signOut());
function setMobileSidebar(open) {
  $(".sidebar").classList.toggle("mobile-open", open);
  $("#mobile-menu").setAttribute("aria-expanded", String(open));
  if (open) $("#close-sidebar").focus();
}
$("#mobile-menu").addEventListener("click", () => setMobileSidebar(!$(".sidebar").classList.contains("mobile-open")));
$("#close-sidebar").addEventListener("click", () => {setMobileSidebar(false); $("#mobile-menu").focus();});

function renderKnowledgeBases() {
  closeKbMenu();
  const host = $("#kb-list");
  const focusedKey = host.contains(document.activeElement) ? document.activeElement.dataset.navKey : null;
  host.innerHTML = state.knowledgeBases.map(kb => {
    const id = escapeHtml(kb.id), name = escapeHtml(kb.name), expanded = state.expandedKnowledgeBases.has(kb.id);
    const current = kb.id === state.currentKnowledgeBase?.id, cache = state.sidebarConversations.get(kb.id);
    const deleting = state.deletingKnowledgeBases.has(kb.id);
    const rows = current ? state.conversations : cache?.items || [];
    return `<section class="kb-group" data-kb-group="${id}">
      <div class="kb-row ${current ? 'active' : ''}">
        <button type="button" class="kb-disclosure" data-nav-action="toggle" data-kb-id="${id}" data-nav-key="toggle-${id}" aria-expanded="${expanded}" aria-controls="chats-${id}" aria-label="${expanded ? '收起' : '展开'} ${name}" title="${expanded ? '收起' : '展开'}聊天">${expanded ? '▾' : '▸'}</button>
        <button type="button" class="kb-item" data-nav-action="select" data-kb-id="${id}" data-nav-key="kb-${id}" title="${name}" ${current ? 'aria-current="true"' : ''}><span class="kb-glyph" aria-hidden="true">▤</span><span class="kb-label"><span class="nav-label-text">${name}</span></span></button>
        <div class="kb-actions"><button type="button" class="kb-action" data-nav-action="new" data-kb-id="${id}" data-nav-key="new-${id}" title="新建对话" aria-label="新建对话 · ${name}"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 13v3h3L16 7l-3-3-9 9zM11 6l3 3"/></svg></button><button type="button" class="kb-action" data-nav-action="menu" data-kb-id="${id}" data-nav-key="menu-${id}" title="更多操作" aria-label="更多操作 · ${name}" aria-haspopup="menu" aria-expanded="false"><svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="4" cy="10" r="1"/><circle cx="10" cy="10" r="1"/><circle cx="16" cy="10" r="1"/></svg></button></div>
      </div>
      <div id="chats-${id}" class="kb-chats" ${expanded ? '' : 'hidden'}>
        ${current && !state.conversation && !state.conversationLoadTarget ? '<span class="chat-nav-draft">新对话 · 尚未发送</span>' : ''}
        ${cache?.loading && !rows.length ? '<p class="chat-nav-note" role="status">正在加载聊天…</p>' : ''}
        ${rows.map(c => {
          const label = escapeHtml(c.title || '未命名对话'), selected = current && c.id === state.conversation?.id;
          return `<div class="chat-nav-row${selected ? ' active' : ''}"><button type="button" class="chat-nav-item ${selected ? 'active' : ''}" data-nav-action="chat" data-kb-id="${id}" data-chat-id="${escapeHtml(c.id)}" data-nav-key="chat-${escapeHtml(c.id)}" title="${label}" ${selected ? 'aria-current="true"' : ''}><span aria-hidden="true">◦</span><span class="chat-nav-label"><span class="nav-label-text">${label}</span></span></button><button type="button" class="chat-nav-delete" data-nav-action="delete-chat" data-kb-id="${id}" data-chat-id="${escapeHtml(c.id)}" data-nav-key="del-${escapeHtml(c.id)}" title="删除对话" aria-label="删除对话 · ${label}"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 6h12M8 6V4h4v2M6 6l1 10h6l1-10M9 9v4M11 9v4"/></svg>删除</button></div>`;
        }).join('')}
        ${cache?.error ? `<p class="chat-nav-note" role="alert">聊天列表加载失败</p><button type="button" class="chat-nav-retry" data-nav-action="retry" data-kb-id="${id}" data-nav-key="retry-${id}">重新加载列表</button>` : !cache?.loading && !rows.length ? '<p class="chat-nav-note">还没有聊天</p>' : ''}
        ${rows.length >= 50 ? '<p class="chat-nav-note">显示最近 50 个对话</p>' : ''}
      </div></section>`;
  }).join('');
  if (focusedKey) [...host.querySelectorAll('[data-nav-key]')].find(n => n.dataset.navKey === focusedKey)?.focus({preventScroll:true});
}

async function loadSidebarConversations(kb, force = false) {
  const previous = state.sidebarConversations.get(kb);
  if (!force && previous?.promise) return previous.promise;
  const epoch = state.sidebarEpoch, token = state.token, conversationEpoch = state.conversationEpoch;
  const cache = {items:previous?.items || [], loading:true, error:null};
  state.sidebarConversations.set(kb, cache); renderKnowledgeBases();
  const valid = () => epoch === state.sidebarEpoch && token === state.token && state.sidebarConversations.get(kb) === cache && state.knowledgeBases.some(k => k.id === kb);
  cache.promise = (async () => {
    try {
      const data = await api(`/knowledge-bases/${encodeURIComponent(kb)}/conversations`);
      if (!valid()) return null;
      cache.items = data.items;
      if (kb === state.currentKnowledgeBase?.id && conversationEpoch === state.conversationEpoch && !state.conversationBusy) {
        state.conversations = data.items;
        if (state.conversation && !state.conversations.some(c => c.id === state.conversation.id)) state.conversations.unshift(state.conversation);
      }
      return data.items;
    } catch (error) {
      if (valid()) cache.error = error.message;
      throw error;
    } finally {
      if (valid()) { cache.loading = false; cache.promise = null; renderKnowledgeBases(); }
    }
  })();
  return cache.promise;
}

function showEmptyWorkspace() {
  $("#workspace").classList.add("is-hidden");
  $("#right-rail").classList.add("is-hidden");
  $("#welcome-state").classList.remove("is-hidden");
  $("#current-kb-name").textContent = "选择知识库";
  if (state.pollTimer) window.clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function refreshWorkspace() {
  const revision = ++state.workspaceRevision;
  try {
    const knowledgeBases = await api("/knowledge-bases");
    if (revision !== state.workspaceRevision || !state.token) return;
    state.knowledgeBases = knowledgeBases;
  } catch (error) {
    if (state.token && revision === state.workspaceRevision) showToast(error.message, true);
    return;
  }
  const selectedId = state.currentKnowledgeBase?.id || sessionStorage.getItem("knowflow_selected_kb");
  const selected = state.knowledgeBases.find((kb) => kb.id === selectedId);
  state.currentKnowledgeBase = selected || state.knowledgeBases[0] || null;
  for (const id of state.sidebarConversations.keys()) if (!state.knowledgeBases.some(kb => kb.id === id)) {
    state.sidebarConversations.delete(id); state.expandedKnowledgeBases.delete(id);
  }
  if (!state.sidebarInitialized && state.currentKnowledgeBase) {
    state.expandedKnowledgeBases.add(state.currentKnowledgeBase.id); state.sidebarInitialized = true;
  }
  renderKnowledgeBases();
  if (!state.currentKnowledgeBase) {
    showEmptyWorkspace();
    return;
  }
  $("#welcome-state").classList.add("is-hidden");
  $("#workspace").classList.remove("is-hidden");
  $("#right-rail").classList.remove("is-hidden");
  $("#current-kb-name").textContent = state.currentKnowledgeBase.name;
  sessionStorage.setItem("knowflow_selected_kb", state.currentKnowledgeBase.id);
  const kbId = encodeURIComponent(state.currentKnowledgeBase.id);
  if (state.contentKnowledgeBaseId !== state.currentKnowledgeBase.id) clearWorkspaceContent();
  try {
    const [documents, evaluations, batches] = await Promise.all([
      api(`/knowledge-bases/${kbId}/documents`),
      api(`/knowledge-bases/${kbId}/evaluations`),
      api(`/knowledge-bases/${kbId}/ingestion-batches`),
    ]);
    if (revision !== state.workspaceRevision || kbId !== encodeURIComponent(state.currentKnowledgeBase?.id)) return;
    state.documents = documents;
    state.evaluations = evaluations;
    state.batches = batches;
    state.contentKnowledgeBaseId = state.currentKnowledgeBase.id;
    $("#open-quality").disabled = false;
    $("#open-graph").disabled = false;
    renderDocuments();
    renderEvaluations();
    renderBatchPanel();
    schedulePolling();
    if (!state.conversation && !state.conversationBusy) await loadConversations();
    if (qualityRunId && $("#quality-dialog").open) await loadQualityReport();
  } catch (error) {
    if (revision === state.workspaceRevision) {
      if (!state.contentKnowledgeBaseId) $("#document-count").textContent = "加载失败，请重新选择知识库";
      showToast(error.message, true);
    }
  }
}

function clearWorkspaceContent() {
  resetConversation();
  state.documents = []; state.evaluations = []; state.batches = []; state.contentKnowledgeBaseId = null;
  state.lastAnswer = null;
  $("#answer-panel").replaceChildren();
  $("#answer-panel").classList.add("is-hidden");
  if (batchDialog.open) batchDialog.close();
  $("#batch-list").replaceChildren();
  $("#batch-overview").textContent = "正在加载当前知识库…";
  renderDocuments(); renderEvaluations();
  $("#empty-documents").classList.add("is-hidden");
  $("#empty-evaluations").classList.add("is-hidden");
  $("#document-count").textContent = "正在加载当前知识库…";
  $("#open-quality").disabled = true; $("#open-graph").disabled = true;
}

$("#kb-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-nav-action]");
  if (!button) return;
  const kb = state.knowledgeBases.find(kb => kb.id === button.dataset.kbId);
  if (!kb) return;
  const action = button.dataset.navAction;
  if (action === 'menu') {
    openKbMenu(button, true); return;
  }
  closeKbMenu();
  try {
    if (action === 'toggle') {
      if (state.expandedKnowledgeBases.has(kb.id)) state.expandedKnowledgeBases.delete(kb.id);
      else state.expandedKnowledgeBases.add(kb.id);
      renderKnowledgeBases();
      if (state.expandedKnowledgeBases.has(kb.id)) await loadSidebarConversations(kb.id);
      return;
    }
    if (action === 'retry') { await loadSidebarConversations(kb.id, true); return; }
    if (action === 'delete') { await deleteKnowledgeBase(kb); return; }
    if (action === 'delete-chat') { await deleteChat(kb, button.dataset.chatId); return; }
    const epoch = state.sidebarEpoch, navigation = ++state.navigationRequestId;
    if (action === 'new') sessionStorage.removeItem(`knowflow_conversation_${kb.id}`);
    if (action === 'chat') sessionStorage.setItem(`knowflow_conversation_${kb.id}`, button.dataset.chatId);
    if (state.currentKnowledgeBase?.id !== kb.id) {
      state.currentKnowledgeBase = kb; state.expandedKnowledgeBases.add(kb.id);
      clearWorkspaceContent(); await refreshWorkspace();
      if (epoch !== state.sidebarEpoch || state.currentKnowledgeBase?.id !== kb.id || navigation !== state.navigationRequestId) return;
    } else if (action === 'chat') {
      await selectConversation(button.dataset.chatId);
    } else if (action === 'select') {
      state.expandedKnowledgeBases.add(kb.id); renderKnowledgeBases();
      await loadSidebarConversations(kb.id);
    }
    if (action === 'new') startNewConversation();
    setMobileSidebar(false);
  } catch (error) { if (state.token) showToast(error.message, true); }
});

const chatContextMenu = $("#chat-context-menu");
let chatContextTarget = null;
$("#kb-list").addEventListener("contextmenu", event => {
  const row = event.target.closest("[data-nav-action='chat']");
  if (!row) return;
  event.preventDefault();
  chatContextTarget = { kbId: row.dataset.kbId, chatId: row.dataset.chatId };
  chatContextMenu.hidden = false;
  const box = chatContextMenu.getBoundingClientRect();
  chatContextMenu.style.left = `${Math.min(event.clientX, window.innerWidth - box.width - 8)}px`;
  chatContextMenu.style.top = `${Math.min(event.clientY, window.innerHeight - box.height - 8)}px`;
  chatContextMenu.querySelector("button")?.focus({ preventScroll: true });
});
chatContextMenu.addEventListener("click", async event => {
  const action = event.target.closest("[data-chat-context-action]")?.dataset.chatContextAction;
  chatContextMenu.hidden = true;
  if (action !== "delete" || !chatContextTarget) return;
  const kb = state.knowledgeBases.find(row => row.id === chatContextTarget.kbId);
  if (kb) await deleteChat(kb, chatContextTarget.chatId);
  chatContextTarget = null;
});
document.addEventListener("click", () => { chatContextMenu.hidden = true; });
document.addEventListener("keydown", event => {
  if (event.key === "Escape") chatContextMenu.hidden = true;
});
window.addEventListener("blur", () => { chatContextMenu.hidden = true; });

const kbContextMenu = $("#kb-context-menu");
let kbMenuTimer;
function closeKbMenu(restoreFocus=false) {
  window.clearTimeout(kbMenuTimer);
  const id=state.openKnowledgeBaseMenu;
  state.openKnowledgeBaseMenu=null;
  if(typeof kbContextMenu !== "undefined") kbContextMenu.hidden=true;
  for(const button of $('#kb-list').querySelectorAll('[data-nav-action="menu"]')) {
    button.setAttribute('aria-expanded','false');
    if(restoreFocus && button.dataset.kbId===id) button.focus({preventScroll:true});
  }
}
function openKbMenu(button,focus=false) {
  closeKbMenu(); chatContextMenu.hidden=true;
  state.openKnowledgeBaseMenu=button.dataset.kbId;
  button.setAttribute('aria-expanded','true');
  kbContextMenu.querySelector('[data-kb-context-action="toggle"]').textContent=
    state.expandedKnowledgeBases.has(button.dataset.kbId)?'收起聊天':'展开聊天';
  kbContextMenu.hidden=false;
  const anchor=button.getBoundingClientRect(), box=kbContextMenu.getBoundingClientRect();
  kbContextMenu.style.left=`${Math.max(8,Math.min(anchor.right-box.width,window.innerWidth-box.width-8))}px`;
  kbContextMenu.style.top=`${Math.max(8,Math.min(anchor.bottom+4,window.innerHeight-box.height-8))}px`;
  if(focus) kbContextMenu.querySelector('button').focus({preventScroll:true});
}
function scheduleKbMenuClose() {window.clearTimeout(kbMenuTimer);kbMenuTimer=window.setTimeout(()=>closeKbMenu(),250);}
$('#kb-list').addEventListener('pointerover',event=>{
  const button=event.target.closest('[data-nav-action="menu"]');
  if(button && !button.contains(event.relatedTarget)) openKbMenu(button);
});
$('#kb-list').addEventListener('pointerout',event=>{
  const button=event.target.closest('[data-nav-action="menu"]');
  if(button && !button.contains(event.relatedTarget) && !kbContextMenu.contains(event.relatedTarget)) scheduleKbMenuClose();
});
kbContextMenu.addEventListener('pointerenter',()=>window.clearTimeout(kbMenuTimer));
kbContextMenu.addEventListener('pointerleave',scheduleKbMenuClose);
kbContextMenu.addEventListener('focusin',()=>window.clearTimeout(kbMenuTimer));
kbContextMenu.addEventListener('focusout',event=>{if(!kbContextMenu.contains(event.relatedTarget))closeKbMenu();});
kbContextMenu.addEventListener('click',async event=>{
  const action=event.target.closest('[data-kb-context-action]')?.dataset.kbContextAction;
  const id=state.openKnowledgeBaseMenu, kb=state.knowledgeBases.find(k=>k.id===id);
  closeKbMenu();if(!kb)return;
  if(action==='toggle') [...$('#kb-list').querySelectorAll('[data-nav-action="toggle"]')].find(b=>b.dataset.kbId===id)?.click();
  if(action==='delete') {try{await deleteKnowledgeBase(kb);}catch(error){showToast(error.message,true);}}
});
document.addEventListener('click',event=>{
  if(!event.target.closest('#kb-context-menu,[data-nav-action="menu"]')) closeKbMenu();
});
document.addEventListener('keydown',event=>{
  if(event.key==='Escape' && state.openKnowledgeBaseMenu){event.preventDefault();closeKbMenu(true);}
});
$('#kb-list').addEventListener('keydown',event=>{
  const button=event.target.closest('[data-nav-action="menu"]');
  if(button && event.key==='ArrowDown'){event.preventDefault();openKbMenu(button,true);}
});
kbContextMenu.addEventListener('keydown',event=>{
  if(!['ArrowDown','ArrowUp','Home','End'].includes(event.key)) return;
  event.preventDefault();const buttons=[...kbContextMenu.querySelectorAll('button')];
  const index=buttons.indexOf(document.activeElement), last=buttons.length-1;
  buttons[event.key==='Home'?0:event.key==='End'?last:(index+(event.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length].focus();
});
window.addEventListener('blur',()=>closeKbMenu());
window.addEventListener('resize',()=>closeKbMenu());
$('#kb-list').addEventListener('scroll',()=>closeKbMenu());

function openKnowledgeBaseDialog() {
  $("#kb-name").value = "";
  setError($("#kb-error"));
  kbDialog.showModal();
  window.setTimeout(() => $("#kb-name").focus(), 0);
}
$("#new-kb").addEventListener("click", openKnowledgeBaseDialog);
$("[data-action='new-kb']").addEventListener("click", openKnowledgeBaseDialog);
$("#cancel-kb").addEventListener("click", () => kbDialog.close());
$("#close-kb").addEventListener("click", () => kbDialog.close());
$("#kb-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = $("#kb-name").value.trim();
  if (!name) return setError($("#kb-error"), "请输入知识库名称。");
  try {
    const created = await api("/knowledge-bases", { method: "POST", body: { name } });
    state.currentKnowledgeBase = created;
    state.expandedKnowledgeBases.add(created.id);
    kbDialog.close();
    await refreshWorkspace();
    showToast("知识库已创建");
  } catch (error) {
    setError($("#kb-error"), error.message);
  }
});

async function deleteChat(kb, chatId) {
  if (!chatId) return;
  const known = [...(state.sidebarConversations.get(kb.id)?.items || []), ...state.conversations]
    .find(row => row.id === chatId);
  const title = known?.title || "未命名对话";
  const confirmed = window.confirm(
    `删除对话“${title}”及其全部轮次？由这些轮次保存的检查点会一并删除，此操作无法撤销。`
  );
  if (!confirmed) return;
  const cache = state.sidebarConversations.get(kb.id);
  const wasCurrent = state.conversation?.id === chatId;
  try {
    await api(`/conversations/${encodeURIComponent(chatId)}`, { method: "DELETE" });
    if (cache?.items) cache.items = cache.items.filter(row => row.id !== chatId);
    state.conversations = (state.conversations || []).filter(row => row.id !== chatId);
    if (wasCurrent && state.currentKnowledgeBase?.id === kb.id) startNewConversation();
    else renderKnowledgeBases();
    showToast("对话已删除");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function deleteKnowledgeBase(kb) {
  if (state.deletingKnowledgeBases.has(kb.id)) return;
  const confirmed = window.confirm(`删除“${kb.name}”及其中的文档、索引和聊天记录？此操作无法撤销。`);
  if (!confirmed) { renderKnowledgeBases(); return; }
  const epoch = state.sidebarEpoch;
  state.deletingKnowledgeBases.add(kb.id); renderKnowledgeBases();
  try {
    await api(`/knowledge-bases/${encodeURIComponent(kb.id)}`, { method: "DELETE" });
    if (epoch !== state.sidebarEpoch) return;
    state.sidebarConversations.delete(kb.id); state.expandedKnowledgeBases.delete(kb.id);
    sessionStorage.removeItem(`knowflow_conversation_${kb.id}`);
    if (state.currentKnowledgeBase?.id === kb.id) {state.currentKnowledgeBase = null; clearWorkspaceContent();}
    await refreshWorkspace();
    showToast("知识库已删除");
  } catch (error) {
    if (epoch === state.sidebarEpoch) showToast(error.message, true);
  } finally {
    state.deletingKnowledgeBases.delete(kb.id); renderKnowledgeBases();
  }
}

function fileKind(filename) {
  const extension = filename.split(".").pop()?.toUpperCase();
  return ["PDF", "DOCX", "MD", "TXT", "CSV", "XLSX"].includes(extension) ? extension : "FILE";
}

function renderDocuments() {
  const host = $("#document-list");
  $("#document-count").textContent = state.documents.length
    ? `${state.documents.length} 份文档`
    : "尚未添加文档";
  $("#empty-documents").classList.toggle("is-hidden", state.documents.length > 0);
  host.innerHTML = state.documents.map((doc) => {
    const status = doc.status;
    const semantic = doc.parsing_report?.graph_index?.semantic_review;
    const graphLabel = doc.graph_status === "processed"
      ? semantic ? `图谱可检索 ${semantic.eligible_relations}/${semantic.reviewed_relations} 条` : "图谱待语义审核（需重建）"
      : doc.graph_status === "failed"
        ? "图谱构建失败"
        : STATUS_NAMES[doc.graph_status] || "图谱待构建";
    const graphError = doc.graph_error ? ` title="${escapeHtml(doc.graph_error)}"` : "";
    return `
      <article class="document-row" data-document-id="${escapeHtml(doc.id)}">
        <div class="doc-main">
          <span class="file-glyph">${escapeHtml(fileKind(doc.filename))}</span>
          <div class="doc-copy">
            <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
            <div class="doc-sub">${formatBytes(doc.file_size)} · ${formatDate(doc.created_at)}</div>
            <div class="doc-status status-${escapeHtml(status)}">${escapeHtml(STATUS_NAMES[status] || status)}
              <span class="graph-status"${graphError}>${escapeHtml(graphLabel)}</span>
            </div>
            ${doc.error_message ? `<div class="doc-sub status-failed">${escapeHtml(doc.error_message)}</div>` : ""}
            ${doc.parsing_report?.status === "needs_review" ? '<div class="doc-sub parse-warning">解析需核对，请查看报告</div>' : ""}
          </div>
        </div>
        <div class="doc-actions">
          <button type="button" data-doc-action="report" data-doc-id="${escapeHtml(doc.id)}">解析报告</button>
          <button type="button" data-doc-action="diagnostics" data-doc-id="${escapeHtml(doc.id)}">自动诊断</button>
          <button type="button" data-doc-action="segments" data-doc-id="${escapeHtml(doc.id)}">查看切片</button>
          <button type="button" data-doc-action="references" data-doc-id="${escapeHtml(doc.id)}">文内引用</button>
          ${doc.filename.toLowerCase().endsWith(".pdf") && status === "processed" ? `<button type="button" data-doc-action="ocr" data-doc-id="${escapeHtml(doc.id)}">OCR 对照</button>` : ""}
          ${["failed", "processed"].includes(status) ? `<button type="button" data-doc-action="reprocess" data-doc-id="${escapeHtml(doc.id)}">重新解析</button>` : ""}
          ${["processed", "failed"].includes(status) && doc.parsing_report?.parse_artifact?.available ? `<button type="button" data-doc-action="rechunk" data-doc-id="${escapeHtml(doc.id)}">重新分块</button>` : ""}
          ${status === "processed" && !["pending", "processing"].includes(doc.graph_status) ? `<button type="button" data-doc-action="graph" data-doc-id="${escapeHtml(doc.id)}">${doc.graph_status === "processed" ? "重建图谱" : "构建图谱"}</button>` : ""}
          <button class="delete-doc" type="button" data-doc-action="delete" data-doc-id="${escapeHtml(doc.id)}">删除</button>
        </div>
      </article>`;
  }).join("");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric" }).format(new Date(value));
}

function schedulePolling() {
  const active = state.documents.some((doc) => ["pending", "processing"].includes(doc.status)
    || ["pending", "processing"].includes(doc.graph_status))
    || state.evaluations.some((run) => run.status === "running")
    || state.batches.some((batch) => batch.active > 0);
  if (active && !state.pollTimer) {
    state.pollTimer = window.setInterval(() => refreshWorkspace(), 2400);
  } else if (!active && state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

$("#file-input").addEventListener("change", async (event) => {
  const files = Array.from(event.target.files || []);
  event.target.value = "";
  if (!files.length || !state.currentKnowledgeBase) return;
  setError($("#upload-error"));
  if (files.length > 20) return setError($("#upload-error"), "每批最多选择 20 个文件。");
  const kbId = state.currentKnowledgeBase.id;
  event.target.disabled = true;
  const body = new FormData();
  for (const file of files) body.append("files", file);
  try {
    await api(`/knowledge-bases/${encodeURIComponent(kbId)}/ingestion-batches`, {
      method: "POST", body, headers: {"Idempotency-Key": crypto.randomUUID()},
    });
    if (state.token && state.currentKnowledgeBase?.id === kbId) {
      await refreshWorkspace();
      showToast("批任务已保存；可在“查看批任务”中检查进度和失败原因");
    }
  } catch (error) {
    if (state.currentKnowledgeBase?.id === kbId) setError($("#upload-error"), error.message);
  } finally {
    event.target.disabled = false;
  }
});

function renderBatchPanel() {
  const active = state.batches.reduce((n,b)=>n+b.active,0);
  $("#batch-overview").textContent = `${state.batches.length} 批最近记录 · ${active} 个文件处理中`;
  if (!batchDialog.open) return;
  const batchHost = $("#batch-list");
  const snapshot = JSON.stringify(state.batches);
  if (batchHost.children.length && batchHost.dataset.snapshot === snapshot) return;
  batchHost.dataset.snapshot = snapshot;
  const kbId = state.currentKnowledgeBase?.id;
  renderBatches($("#batch-list"), state.batches, async (id) => {
    await api(`/ingestion-batches/${encodeURIComponent(id)}/retry`, {method:"POST"});
    if (state.currentKnowledgeBase?.id === kbId) await refreshWorkspace();
  }, async (id) => {
    const result = await api(`/ingestion-batches/${encodeURIComponent(id)}`);
    if (state.currentKnowledgeBase?.id !== kbId) throw new Error("已切换知识库，请重新打开批任务。");
    return result;
  }, showBatchDiagnostics);
}
$("#open-batches").addEventListener("click", () => { batchDialog.showModal(); renderBatchPanel(); });
$("#close-batches").addEventListener("click", () => batchDialog.close());

$("#document-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-doc-action]");
  if (!button) return;
  const { docId, docAction } = button.dataset;
  if (docAction === "delete" && !window.confirm("删除这份文档及其文本和图谱索引？")) return;
  button.disabled = true;
  try {
    if (docAction === "references") {
      await showReferences(docId);
      return;
    } else if (docAction === "ocr") {
      await showOcrReview(docId);
      return;
    } else if (docAction === "segments") {
      await showSegments(docId);
      return;
    } else if (docAction === "diagnostics") {
      await showDiagnostics(docId);
      return;
    } else if (docAction === "report") {
      await showParsingReport(docId);
      return;
    } else if (docAction === "delete") {
      await api(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
      showToast("文档已删除");
    } else if (docAction === "reprocess") {
      await api(`/documents/${encodeURIComponent(docId)}/reprocess`, { method: "POST" });
      showToast("已重新开始处理");
    } else if (docAction === "rechunk") {
      await api(`/documents/${encodeURIComponent(docId)}/rechunk`, { method: "POST" });
      showToast("已复用解析产物开始分块，不重复 OCR；完成后需重建图谱");
    } else if (docAction === "graph") {
      await api(`/documents/${encodeURIComponent(docId)}/graph-index`, { method: "POST" });
      showToast("图谱构建已开始");
    }
    await refreshWorkspace();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
});

function resizeQuestion() {
  const input = $("#question");
  const style = getComputedStyle(input);
  const line = Number.parseFloat(style.lineHeight) || 24;
  const padding = (Number.parseFloat(style.paddingTop) || 0) + (Number.parseFloat(style.paddingBottom) || 0);
  const border = (Number.parseFloat(style.borderTopWidth) || 0) + (Number.parseFloat(style.borderBottomWidth) || 0);
  const minimum = line + padding + border, maximum = 4 * line + padding + border;
  input.style.height = "auto";
  const needed = input.scrollHeight + border;
  input.style.height = `${Math.min(maximum, Math.max(minimum, needed))}px`;
  input.style.overflowY = needed > maximum ? "auto" : "hidden";
}
$("#question").addEventListener("input", resizeQuestion);
window.addEventListener("resize", resizeQuestion);
resizeQuestion();

$("#query-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.currentKnowledgeBase || state.conversationBusy || state.conversation?.active_turn_id) return;
  const question = $("#question").value.trim();
  if (!question) return setError($("#query-error"), "先写下你想查的问题。");
  setError($("#query-error"));
  const askButton = $("#ask-button");
  const requestedKb = state.currentKnowledgeBase.id;
  const epoch = state.conversationEpoch;
  const parameters = { question, strategy: $("#strategy").value, top_k: Number($("#top-k").value), expand_references: $("#expand-references").checked };
  if (state.resumeTask) Object.assign(parameters, {task_id:state.resumeTask.id, task_revision:state.resumeTask.revision});
  state.conversationBusy = true;
  state.turns = state.turns.filter(t => t.id !== "local-pending");
  state.turns.push({id:"local-pending", sequence:(state.conversation?.revision || 0)+1, question, status:"sending"});
  renderConversationHistory();
  $("#question").value = "";
  resizeQuestion();
  state.lastAnswer = null;
  $("#answer-panel").replaceChildren(); $("#answer-panel").classList.add("is-hidden");
  $("#conversation-status").textContent = "正在处理本轮问题…";
  askButton.disabled = true;
  askButton.querySelector("span").textContent = "回答中…";
  try {
    if (!state.conversation) {
      const created = await api(`/knowledge-bases/${encodeURIComponent(requestedKb)}/conversations`, {method:"POST"});
      if (epoch !== state.conversationEpoch) return;
      state.conversation = created;
      state.conversations.unshift(created);
      sessionStorage.setItem(`knowflow_conversation_${requestedKb}`, created.id);
      renderConversationOptions();
    }
    const cid = state.conversation.id;
    // A network retry of the identical request reuses its key; changed input gets a new key.
    const signature = JSON.stringify({cid, ...parameters});
    const requestId = state.pendingRequest?.signature === signature ? state.pendingRequest.id : crypto.randomUUID();
    state.pendingRequest = {signature, id:requestId};
    await api(`/conversations/${encodeURIComponent(cid)}/turns`, {
      method: "POST", body: {...parameters, request_id:requestId, expected_revision:state.conversation.revision},
    });
    if (epoch !== state.conversationEpoch || state.currentKnowledgeBase?.id !== requestedKb) return;
    state.pendingRequest = null;
    state.resumeTask = null;
    $("#resume-task-bar")?.remove();
    await loadConversation(cid, epoch);
  } catch (error) {
    if (epoch === state.conversationEpoch) {
      const local = state.turns.find(t => t.id === "local-pending");
      if (local) {local.status="connection_unknown"; local.error={message:"连接未确认，可重新加载记录查看服务器状态。"}; renderConversationHistory();}
      showRecordError(error.message);
    }
  } finally {
    if (epoch === state.conversationEpoch) {
      state.conversationBusy = false;
      askButton.disabled = Boolean(state.conversation?.active_turn_id);
      askButton.querySelector("span").textContent = "发送";
    }
  }
});

function resetConversation() {
  state.conversationEpoch++;
  clearTimeout(state.conversationTimer);
  state.conversations = []; state.conversation = null; state.turns = [];
  state.conversationBusy = false; state.pendingRequest = null;
  state.selectedTurnId = null;
  state.conversationLoadTarget = null;
  state.resumeTask = null;
  $("#resume-task-bar")?.remove();
  for (const id of ["#memory-dialog", "#tasks-dialog"]) if ($(id).open) $(id).close();
  state.memoryEdit = null;
  if ($("#answer-dialog").open) $("#answer-dialog").close();
  renderConversationHistory();
  $("#conversation-status").textContent = "";
  $("#older-turns").classList.add("is-hidden");
  $("#ask-button").disabled = false;
  $("#ask-button span").textContent = "发送";
  $("#question").value = "";
  resizeQuestion();
  setError($("#query-error"));
  $("#reload-records").classList.add("is-hidden");
  renderConversationOptions();
}

function renderConversationOptions() {
  const kb = state.currentKnowledgeBase?.id;
  if (kb && state.conversations.length) {
    const cache = state.sidebarConversations.get(kb);
    if (cache) cache.items = state.conversations;
    else state.sidebarConversations.set(kb, {items:state.conversations,loading:false,error:null});
  }
  renderKnowledgeBases();
}

async function loadConversations() {
  const kb = state.currentKnowledgeBase?.id, epoch = state.conversationEpoch;
  if (!kb) return;
  const items = await loadSidebarConversations(kb);
  if (epoch !== state.conversationEpoch || kb !== state.currentKnowledgeBase?.id || !items) return;
  state.conversations = items;
  renderConversationOptions();
  const saved = sessionStorage.getItem(`knowflow_conversation_${kb}`);
  if (saved && items.some(c => c.id === saved)) await loadConversation(saved, epoch);
}

async function loadConversation(cid, epoch, older = false) {
  const loadId = ++state.conversationLoadId;
  state.conversationLoadTarget = cid;
  state.conversationBusy = true;
  $("#ask-button").disabled = true;
  try {
  const before = older && state.turns.length ? `?before=${state.turns[0].sequence}` : "";
  const data = await api(`/conversations/${encodeURIComponent(cid)}${before}`);
  if (epoch !== state.conversationEpoch || loadId !== state.conversationLoadId) return;
  state.conversation = data;
  state.turns = older ? [...data.turns, ...state.turns] : data.turns;
  const item = state.conversations.find(c => c.id === cid);
  if (item) Object.assign(item, data);
  renderConversationOptions();
  setError($("#query-error")); $("#reload-records").classList.add("is-hidden");
  $("#older-turns").classList.toggle("is-hidden", !data.has_more);
  renderConversationHistory(older);
  const active = Boolean(data.active_turn_id);
  const expired = active && Date.parse(data.active_until) < Date.now();
  $("#conversation-status").textContent = expired ? "上轮等待已过期，发送新问题可恢复对话；旧轮结果仍标为未知。" : active ? "本轮正在处理；刷新页面也能恢复记录。" : "";
  $("#ask-button").disabled = active && !expired;
  // Expired claims are recovered by the next request, never silently retried.
  if (expired) state.conversation.active_turn_id = null;
  clearTimeout(state.conversationTimer);
  if (active && !expired) state.conversationTimer = setTimeout(() => loadConversation(cid, epoch).catch(error => {
    if (epoch === state.conversationEpoch) showRecordError(error.message);
  }), 3000);
  if (!older) {
    if (!$("#answer-dialog").open) {
      state.lastAnswer = null;
      $("#answer-panel").replaceChildren(); $("#answer-panel").classList.add("is-hidden");
    }
  }
  } finally {
    if (epoch === state.conversationEpoch && loadId === state.conversationLoadId) {
      state.conversationBusy = false;
      $("#ask-button").disabled = Boolean(state.conversation?.active_turn_id);
    }
  }
}

function clarificationHintsHtml(hints) {
  if (hints?.version !== "clarification-hints-v1" || hints.kind !== "available_documents_not_relevance_ranked") return "";
  const docs = Array.isArray(hints.documents) ? hints.documents.slice(0,5) : [];
  return `<aside class="clarification-hints" aria-label="补充问题的文档提示"><strong>可用文档</strong>
    <p>${docs.length ? "当前知识库已处理的文档，未按相关性筛选：" : "当前知识库暂无已处理文档。"}</p>
    ${docs.length ? `<ul>${docs.map(doc=>`<li>${escapeHtml(doc.filename)}</li>`).join("")}</ul>` : ""}
    ${hints.has_more ? "<p>仅显示前 5 份；其余文档可在右侧栏查看。</p>" : ""}
    <p>请在输入框补充文件名和具体对象，或重新写出完整问题。文件名本身不能确定文档中的对象。</p></aside>`;
}

function partialAnswerHtml(partial) {
  if (!["quoted-partial-lookup-v1","quoted-partial-lookup-v2-lossless-scope","grounded-referent-options-v1","grounded-referent-options-v2-name-only-query","grounded-referent-options-v3-adaptive-discovery","grounded-referent-options-v4-actor-roles","grounded-referent-options-v5-document-coverage"].includes(partial?.version)) return "";
  const blocked = Array.isArray(partial.blocked_scopes) ? partial.blocked_scopes : [];
  const conditional = ["grounded-referent-options-v1","grounded-referent-options-v2-name-only-query","grounded-referent-options-v3-adaptive-discovery","grounded-referent-options-v4-actor-roles","grounded-referent-options-v5-document-coverage"].includes(partial.version);
  return `<aside class="clarification-hints" aria-label="部分回答范围"><strong>${partial.status === "answered" ? conditional ? "条件回答，指代待确认" : "部分回答，仍需补充" : "部分查询未返回答案，仍需补充"}</strong>
    <p>${conditional ? "本轮按候选指代成立的假设检索，整道问题尚未完成；引用核验不能替代指代确认。" : "本轮只处理可独立确认的部分，整道问题尚未完成；引用和答案核验仅对应这一部分。"}</p>
    <details><summary>查看本轮范围</summary><p>独立查询：${escapeHtml(partial.question)}</p><p>本轮原话范围：</p><p>${escapeHtml(partial.current_scope)}</p>
    <p>尚未查询的部分：</p><ul>${blocked.map(text=>`<li>${escapeHtml(text)}</li>`).join("")}</ul></details></aside>`;
}

function referentOptionsHtml(info) {
  if (!["grounded-referent-options-v1","grounded-referent-options-v2-name-only-query","grounded-referent-options-v3-adaptive-discovery","grounded-referent-options-v4-actor-roles","grounded-referent-options-v5-document-coverage"].includes(info?.version) || !info.options?.length) return "";
  return `<aside class="clarification-hints" aria-label="指代候选与来源"><strong>${["conditional_unique","conditional_candidate"].includes(info.status) ? "本轮暂按下列候选理解，尚未由你确认" : "请确认所指对象"}</strong>
    ${info.target_phrase ? `<p>先按“${escapeHtml(info.target_phrase)}”发现对象候选，再按候选名称查本轮事实；候选原句不作为检索问题中的答案。</p>` : ""}
    <p>${info.coverage === "all_processed_text_supplied" ? "已提供当前已处理文本供模型核对；实体提取的穷尽性仍未验证。" : "来自有界检索，未覆盖全部材料；不能据此断言全库唯一。"}</p>
    ${info.options.map(row=>`<div><strong>${escapeHtml(row.name)}</strong><p>${escapeHtml(row.filename)}</p><blockquote>${escapeHtml(row.quote)}</blockquote></div>`).join("")}
    <p>候选身份判断不代替回答依据核验。若对象不符，请输入正确名称。</p></aside>`;
}

function coverageHtml(coverage) {
  if (!coverage || coverage.mode !== "sweep" || !Number(coverage.checked)) return "";
  const documents = Array.isArray(coverage.documents) ? coverage.documents : [];
  const unsampled = documents.filter((doc) => !Number(doc.evidence_chunks || 0)).map((doc) => doc.filename);
  const uncited = documents.filter((doc) => doc.cited_in_answer === false).map((doc) => doc.filename);
  const sampled = documents.filter((doc) => doc.distance_gate_fallback).map((doc) => doc.filename);
  const text = (label, names) => (names.length ? `${label}：${names.join("、")}` : "");
  const title = [
    text("检索未取到片段", unsampled),
    text("答案未引用（不能判断内容是否缺失）", uncited),
    text("仅取到最近取样片段（未过相似度阈值）", sampled),
  ].filter(Boolean).join(" ｜ ");
  return `<div class="coverage-note" title="${escapeHtml(title)}" aria-label="跨文档覆盖范围">
    <span class="summary-scope-title">覆盖</span>
    <span>片段抽样 ${Number(coverage.checked)}/${Number(coverage.eligible_documents ?? coverage.checked)} 份文件 · 答案引用 ${Number(coverage.cited_documents ?? coverage.answered ?? 0)} 份</span>
    ${Number(coverage.omitted_documents) > 0 ? `<span class="summary-scope-warn">另有 ${Number(coverage.omitted_documents)} 份未检查</span>` : ""}
    <span class="summary-scope-warn">非全文检查</span>
    ${sampled.length ? `<span class="summary-scope-warn">取样 ${sampled.length} 份</span>` : ""}
    ${unsampled.length ? `<span class="summary-scope-warn">检索未命中 ${unsampled.length} 份</span>` : ""}
  </div>`;
}

function summaryScopeHtml(summary) {
  if (!summary || !summary.document) return "";
  const total = Number(summary.total_chunks || 0);
  const read = Number(summary.read_chunks || 0);
  const pages = Array.isArray(summary.pages) ? `第 ${summary.pages[0]}–${summary.pages[1]} 页` : "页码未知";
  const segments = Number(summary.segments || 0);
  const summarized = Number(summary.summarized_segments || 0);
  const partial = String(summary.coverage || "").includes("partial") || read < total;
  const mode = summary.coverage === "parsed_text_prepared_not_generated" ? "已装载，未确认模型读取" : summary.mode === "segmented_map_reduce"
    ? `分段汇总 ${summarized}/${segments} 段`
    : summary.coverage === "not_read" ? "未读取" : "全文直读";
  return `<div class="summary-scope${partial ? " is-partial" : ""}" aria-label="文档总结阅读范围">
    <span class="summary-scope-title">文档总结</span>
    <span class="summary-scope-name" title="${escapeHtml(summary.document.filename)}">${escapeHtml(summary.document.filename)}</span>
    <span class="rank-chip">已读 ${read}/${total} 切片</span>
    <span class="rank-chip">${escapeHtml(pages)}</span>
    <span class="rank-chip">${escapeHtml(mode)}</span>
    ${partial ? `<span class="summary-scope-warn">${Math.max(total - read, 0)} 个切片未读或未形成可用笔记，不代表全文结论</span>` : ""}
  </div>`;
}

function renderConversationHistory(preserveScroll = false) {
  const statuses = {completed:"", needs_clarification:"请补充信息", failed:"本轮未完成", interrupted:"中断 · 结果未知", pending:"正在检索并回答", sending:"正在发送并检索", connection_unknown:"连接状态待确认"};
  const host = $("#conversation-history");
  const oldHeight = host.scrollHeight, oldTop = host.scrollTop;
  const position = () => {host.scrollTop = preserveScroll ? oldTop + host.scrollHeight - oldHeight : host.scrollHeight;};
  if (!state.turns.length) {
    host.innerHTML = '<div class="chat-empty"><span>从一个问题开始</span><p>发送消息后，在这里继续对话。</p></div>';
    return;
  }
  host.innerHTML = state.turns.map(turn => `
    <article class="conversation-turn">
      <div class="chat-message chat-user"><div class="chat-avatar" aria-hidden="true">你</div><div class="chat-content"><span class="chat-speaker">你</span><div class="chat-bubble"><p class="turn-question">${escapeHtml(turn.question)}</p></div></div></div>
      <div class="chat-message chat-assistant"><div class="chat-avatar" aria-hidden="true">F</div><div class="chat-content"><span class="chat-speaker">FinCite</span><div class="chat-bubble ${turn.status === 'failed' || turn.status === 'interrupted' ? 'chat-bubble-error' : ''}">
        ${turn.response ? `${partialAnswerHtml(turn.response.partial_answer)}${referentOptionsHtml(turn.response.referent_candidates)}${summaryScopeHtml(turn.response.document_summary)}${coverageHtml(turn.response.query_plan?.coverage)}<div class="turn-preview chat-answer" data-chat-answer="${escapeHtml(turn.id)}"></div>
          ${clarificationHintsHtml(turn.response.clarification)}
          <div class="chat-footer"><span>${escapeHtml(turn.status === "needs_clarification" ? "需要澄清" : turn.response.refused ? "当前依据不足" : `${turn.response.citations?.length || 0} 处引用`)}</span><button type="button" class="text-button" data-save-task="${escapeHtml(turn.id)}" aria-label="保存任务 · 第 ${turn.sequence} 轮">保存任务</button><button type="button" class="chat-details-button" data-turn-id="${escapeHtml(turn.id)}" aria-label="查看详情 · 第 ${turn.sequence} 轮">查看详情 <span aria-hidden="true">↗</span></button></div>`
          : `<p class="turn-status">${escapeHtml(statuses[turn.status] || turn.status)}</p><p class="turn-preview">${escapeHtml(turn.error?.message || "正在为你查找文档依据…")}</p>${turn.resolution?.round_budget ? roundBudgetLines(turn.resolution.round_budget).slice(0,2).map(text=>`<p class="retrieval-note">${escapeHtml(text)}</p>`).join("") : ""}${['pending','sending'].includes(turn.status) ? '<span class="chat-typing" aria-label="等待模型回答"><i></i><i></i><i></i></span>' : ''}`}
      </div></div></div>
    </article>`).join("");
  position();
  const epoch = state.conversationEpoch;
  Promise.all([...host.querySelectorAll("[data-chat-answer]")].map(node => {
    const turn = state.turns.find(t => t.id === node.dataset.chatAnswer);
    return renderRichAnswer(node, turn.response.answer);
  })).then(() => { if (epoch === state.conversationEpoch) position(); })
    .catch(() => { /* Rich renderer retains plain text on unsupported content. */ });
}

async function showTurn(turn) {
  const epoch = state.conversationEpoch;
  state.selectedTurnId = turn.id;
  $("#answer-detail-title").textContent = `回答详情 · 第 ${turn.sequence} 轮`;
  $("#answer-detail-question").textContent = turn.question;
  const intent = turn.resolution?.intent_context;
  const inputs = Array.isArray(intent?.referenced_user_inputs) ? intent.referenced_user_inputs : [];
  const rawRows=inputs.filter(r=>!String(r.turn_id).startsWith("memory:"));
  const memoryRows=inputs.filter(r=>String(r.turn_id).startsWith("memory:"));
  const rawInputs = intent ? `<p><strong>本轮原话：</strong>${escapeHtml(intent.current_user_input)}</p>
    ${intent.active_task ? `<p><strong>正在恢复的任务目标：</strong>${escapeHtml(intent.active_task.goal?.user_input)}</p><p>本轮明确更正优先，其次本任务按顺序保存的更正、任务目标；更早历史和记忆别名不能覆盖任务范围。</p>` : ""}
    <strong>引用的历史用户条件：</strong><ol>${rawRows.map(row => `<li>${escapeHtml(row.question)}${Array.isArray(row.supplements) ? row.supplements.map(text => `<p>补充：${escapeHtml(text)}</p>`).join("") : ""}</li>`).join("")}</ol>
    ${memoryRows.length ? `<strong>保存的记忆线索（仅帮助识别，不覆盖任务条件）：</strong><ul>${memoryRows.map(row=>`<li>${escapeHtml(row.question)}</li>`).join("")}</ul>` : ""}` : "";
  const resolution=turn.resolution || {};
  const traced=["conversation-reference-v4-history-aware","conversation-reference-v5-history-aware-constraints","conversation-reference-v6-topic-boundary","conversation-reference-v7-source-handles","conversation-reference-v8-cold-intent","conversation-reference-v9-cold-demonstrative","conversation-reference-v10-clarification-causes","conversation-reference-v11-partial-lookup","conversation-reference-v12-partial-assessment","conversation-reference-v13-lossless-partial","conversation-reference-v14-referent-candidates","conversation-reference-v15-adaptive-discovery","conversation-reference-v16-named-prior-binding"].includes(resolution.version);
  const statuses={resolved:"承接历史意图",standalone:"独立问题",clarify:"需要澄清"};
  const routing={history_present:"已有会话历史",pending_clarification:"待澄清事项",cold_reference:"首问缺少指代对象",cold_independent:"独立首问"};
  const onlySavedClues=resolution.reference_turn_ids?.length && resolution.reference_turn_ids.every(id=>id.startsWith("memory:"));
  const resolutionReasons={numeric_constraints_changed:"解析增加或遗漏了不允许变化的数字",resolution_timeout:"上下文解析超时",
    invalid_resolution_object:"响应结构不符",invalid_resolution_fields:"响应字段不符",invalid_reference_turn:"引用轮次无效",
    missing_reference_turn:"缺少引用轮次",pending_reference_omitted:"未引用待澄清事项",restatement_required:"需要完整重述",
    standalone_input_changed:"新话题未保持当前输入原文",nonfinal_model_output:"解析输出未正常完成",
    empty_resolution_content:"解析正文为空",authoritative_intent_budget_exceeded:"原始条件超出预算",
    invalid_resolution_json:"解析正文不是合法JSON",resolution_unavailable:"解析服务不可用，具体原因未分类",
    request_budget_exceeded:"完整请求超预算，未发起解析，需重述完整条件",
    round_budget_exhausted:"整轮额度不足，未派发解析请求",
    recent_intent_budget_exceeded:"最近意图超出预算",context_budget_exceeded:"待澄清内容超出预算",no_context:"缺少历史对象"};
  const number=value => Number.isFinite(value) ? String(value) : "未记录";
  const partialDecisions={omitted:"解析器未返回独立部分评估，未派发部分查询",none:"解析器未提议安全的独立部分；这不表示文档中没有答案",planned:"独立部分已进入查询流程，完整问题仍待补充",rejected:"部分方案未通过范围校验，未派发部分查询"};
  const traceHtml=traced ? `<p>上下文判断：${statuses[resolution.status] || "状态未知"}；入口：${onlySavedClues ? "用户保存的记忆线索" : routing[resolution.routing_reason] || "未知"}。</p>
    ${["resume_prior","follow_prior_named_object"].includes(resolution.pending_action) ? `<p>按本轮明确的对象承接旧问题：${escapeHtml(resolution.return_anchor)}；旧待澄清问题的条件未继承，本轮更正优先。</p>` : ""}
    ${resolution.partial_lookup_decision ? `<p>独立部分评估：${escapeHtml(partialDecisions[resolution.partial_lookup_decision] || "状态未知")}。</p>` : ""}
    <p>${resolution.request_attempted ? `上下文解析已发起一次调用 · ${Math.round(resolution.latency_ms || 0)} ms；实报输入 ${number(resolution.provider_usage?.prompt_tokens)}、输出 ${number(resolution.provider_usage?.completion_tokens)} token。缺少用量不记作零。` : "未发起上下文解析调用。"}</p>
    ${resolution.request_budget ? understandingBudgetLines(resolution.request_budget).map(text=>`<p>${escapeHtml(text)}</p>`).join("") : resolution.request_attempted ? "<p>未记录本阶段完整请求预算，保持未知。</p>" : ""}
    ${resolution.required_recent_intent_omitted ? "<p>最近一轮完整意图无法装入预算，未采用更早话题代替，请重述对象和条件。</p>" : ""}
    ${resolution.topic_scope ? `<p>当前承接最近一次已确认的独立话题及其后续；本次已读历史中未装入 ${number(resolution.topic_scope.older_loaded_turns_omitted)} 条更早意图，历史记录仍保留。需要返回旧话题时请明确对象和条件。</p>` : ""}
    ${resolution.status === "clarify" && resolution.reason ? `<p>未继续检索的原因：${resolutionReasons[resolution.reason] || "原因未知"}。</p>` : ""}
    ${resolution.status === "standalone" ? "<p>按当前输入原文重新检索，没有继承旧话题的条件。</p>" : ""}
    <small>这是模型判断与信息传递记录，不是语义正确率；会话与记忆线索不是文档证据，来源仍须重新检索。</small>` : "";
  const intentHtml=resolution.status === "resolved" ? `<p><strong>检索线索：</strong>${escapeHtml(turn.resolved_question)}</p>${rawInputs}<small>${intent ? "原始条件同时进入相关性、回答和依据复核；本轮更正优先。历史输入不是事实证据，模型仍可能理解错误。" : "仅借历史理解指代；答案重新检索。"}</small>` : "";
  const title=resolution.status === "resolved" ? (intent ? "检索问题与原始条件" : "本轮独立检索问题") : `上下文判断：${statuses[resolution.status] || "状态未知"}`;
  $("#answer-resolution").innerHTML = intentHtml || traceHtml ? `<details class="resolved-intent"><summary>${title}</summary>${traceHtml}${intentHtml}</details>` : "";
  if (resolution.memory_context || resolution.task_resume) {
    const box=document.createElement("details"), summary=document.createElement("summary");
    box.className="resolved-intent";summary.textContent="任务与记忆使用记录";box.append(summary);
    const memory=resolution.memory_context;
    const lines=memory ? [`已选择 ${memory.selected?.length || 0} 项记忆供考虑，模型引用 ${memory.referenced_ids?.length ?? "未知"} 项作为指代线索，预算 ${memory.selected_tokens ?? "未知"} / ${memory.budget_tokens} token。`,
      `方法：名称字面匹配和结构化表达偏好；${memory.conflicts?.length || 0} 组冲突，${memory.skipped?.length || 0} 项跳过。`,
      ...(memory.selected || []).map(m=>`记忆 ${m.id}，修订 ${m.revision}，范围 ${m.namespace}。`),
      "这些是用户线索，不是文档证据；历史展示保留当时修订，删除仅阻止未来使用。"] : [];
    if (resolution.response_style) lines.push(`表达偏好：${resolution.response_style}；本轮明确要求优先。`);
    if (resolution.task_resume) lines.push(`恢复任务 ${resolution.task_resume.task_id}；重新检索完整任务。来源变更/缺失 ${resolution.task_resume.source_versions_changed?.length || 0} 项；逐步覆盖尚未验证。`);
    for (const line of lines) {const p=document.createElement("p");p.textContent=line;box.append(p);}
    $("#answer-resolution").append(box);
  }
  renderTaskAlignment($("#answer-resolution"), resolution.task_alignment_checks, async()=>{
    const cid=turn.conversation_id || state.conversation?.id;
    if(!cid)return;
    try {
      const data=await api(`/conversations/${encodeURIComponent(cid)}/turns/${encodeURIComponent(turn.id)}/task-review`,{
        method:"POST",body:{request_id:crypto.randomUUID()}});
      if(epoch!==state.conversationEpoch)return;
      const current=state.turns.find(t=>t.id===turn.id);
      if(!current)return;
      const checks=[...(current.resolution?.task_alignment_checks || [])];
      const index=checks.findIndex(r=>r.request_id===data.check.request_id);
      if(index>=0)checks[index]=data.check;else checks.push(data.check);
      current.resolution={...current.resolution,task_alignment_checks:checks};
      if(state.selectedTurnId===turn.id && $("#answer-dialog").open){
        await showTurn(current);
        $(".task-alignment").open=true;
      }
      showToast(data.cached ? "已读取保存的核对，未重复调用模型" : "任务核对已保存；这是模型诊断");
    }catch(e){if(epoch===state.conversationEpoch)showToast(`${e.message} 可刷新对话查看核对记录。`,true);}
  });
  const exportBox=document.createElement("p"), exportButton=document.createElement("button");
  exportBox.className="retrieval-note";exportButton.type="button";exportButton.className="text-button";
  exportButton.dataset.exportReview=turn.id;
  exportButton.textContent="导出截至本轮的复核包";
  exportButton.title="本地下载此前对话、任务原话、答案、引用和空白复核表；不调用模型";
  exportButton.addEventListener("click",async()=>{
    const cid=turn.conversation_id || state.conversation?.id;if(!cid)return;
    exportButton.disabled=true;
    try{
      const bundle=await api(`/conversations/${encodeURIComponent(cid)}/turns/${encodeURIComponent(turn.id)}/review-export`);
      if(epoch!==state.conversationEpoch || state.selectedTurnId!==turn.id || !$("#answer-dialog").open)return;
      const url=URL.createObjectURL(new Blob([JSON.stringify(bundle,null,2)],{type:"application/json;charset=utf-8"}));
      const link=document.createElement("a");link.href=url;link.download=`conversation-review-${cid}-turn-${turn.sequence}.json`;
      document.body.append(link);link.click();link.remove();window.setTimeout(()=>URL.revokeObjectURL(url),1000);
      showToast("已发起复核包下载，请查看浏览器下载记录；没有调用模型");
    }catch(e){if(epoch===state.conversationEpoch)showToast(e.message,true);}
    finally{exportButton.disabled=false;}
  });
  exportBox.append(exportButton);$("#answer-resolution").append(exportBox);
  if (!$("#answer-dialog").open) $("#answer-dialog").showModal();
  await renderAnswer(turn.response);
  if (epoch === state.conversationEpoch && turn.status === "needs_clarification") $("#answer-panel h2").textContent = turn.response.partial_answer ? "部分处理 · 请补充问题" : "请补充问题";
}

$("#conversation-history").addEventListener("click", event => {
  const taskId=event.target.closest("[data-save-task]")?.dataset.saveTask;
  if (taskId) {saveTurnTask(taskId);return;}
  const id = event.target.closest("[data-turn-id]")?.dataset.turnId;
  const turn = state.turns.find(t => t.id === id);
  if (turn?.response) showTurn(turn);
});
$("#close-answer").addEventListener("click", () => $("#answer-dialog").close());

// Managers keep the chat layout unchanged; all user content is escaped or textContent.
function memoryPath() {return `/knowledge-bases/${encodeURIComponent(state.currentKnowledgeBase.id)}/memories`;}
function resetMemoryEditor() {
  state.memoryEdit=null;$("#memory-form").reset();$("#memory-scope").disabled=false;
  $("#memory-save").textContent="保存记忆";setError($("#memory-error"));
  $("#memory-kind-help").textContent="别名和备注按名称字面匹配，不会自动推断同义词。";
}
async function loadMemories(more=false) {
  const epoch=state.conversationEpoch, path=memoryPath();
  const data=await api(`${path}?offset=${more ? state.memories.length : 0}`);
  if(epoch!==state.conversationEpoch)return;
  state.memories=more ? [...state.memories,...data.items] : data.items;
  const names={alias:"对象别名",note:"用户备注",preference:"表达偏好",candidate:"候选",active:"已启用",disabled:"已停用"};
  $("#memory-list").innerHTML=state.memories.map(m=>`<article class="memory-row" data-memory-row="${escapeHtml(m.id)}">
    <header><strong>${escapeHtml(m.key)}</strong><span class="memory-state" data-status="${escapeHtml(m.status)}">${names[m.status] || "未知"}${m.expires_at && Date.parse(m.expires_at)<=Date.now() ? " · 已过期" : ""}</span></header>
    <p>${escapeHtml(m.content)}</p><small>${names[m.kind] || "未知"} · ${m.namespace==="user" ? "我的所有知识库" : "当前知识库"} · 修订 ${m.revision}${m.expires_at ? ` · 有效至 ${escapeHtml(new Date(m.expires_at).toLocaleString())}` : ""}</small>
    <details><summary>来源</summary><p>${m.source?.type==="explicit_user_entry" ? "用户手动填写，未核实为文档事实。" : `用户轮次 ${escapeHtml(m.source?.turn_id || "未记录")}：${escapeHtml(m.source?.quote || "未记录原话")}`}</p></details>
    <div class="memory-actions"><button type="button" class="text-button" data-edit-memory="${escapeHtml(m.id)}">${m.status==="candidate" ? "查看并启用" : "更正 / 停用"}</button><button type="button" class="text-button danger-link" data-delete-memory="${escapeHtml(m.id)}">删除记忆</button></div></article>`).join("") || '<p class="retrieval-note">还没有记忆。保存别名，或发送“记住：…”形成待启用候选。</p>';
  $("#memory-more").classList.toggle("is-hidden",!data.has_more);
}
$("#open-memories").addEventListener("click",async()=>{
  resetMemoryEditor();$("#memory-dialog").showModal();
  try {await loadMemories();}catch(e){setError($("#memory-error"),e.message);}
});
$("#close-memories").addEventListener("click",()=>$("#memory-dialog").close());
$("#memory-reset").addEventListener("click",resetMemoryEditor);
$("#memory-kind").addEventListener("change",()=>{
  const pref=$("#memory-kind").value==="preference";
  if(pref){$("#memory-key").value="response_style";$("#memory-content").value="concise";}
  $("#memory-kind-help").textContent=pref ? "表达偏好只支持 concise（简洁）、standard（标准）、detailed（详细）。本轮明确要求优先。" : "别名和备注按名称字面匹配，不会自动推断同义词。";
});
$("#memory-form").addEventListener("submit",async event=>{
  event.preventDefault();const epoch=state.conversationEpoch, edit=state.memoryEdit;
  const expiry=$("#memory-expiry").value;
  const body={scope:$("#memory-scope").value,kind:$("#memory-kind").value,key:$("#memory-key").value.trim(),content:$("#memory-content").value.trim(),status:$("#memory-status").value,expires_at:expiry ? new Date(expiry).toISOString() : null};
  if(edit){body.expected_revision=edit.revision;if(edit.source?.turn_id){body.source_turn_id=edit.source.turn_id;body.source_quote=edit.source.quote;}}
  $("#memory-save").disabled=true;setError($("#memory-error"));
  try {await api(memoryPath()+(edit ? `/${encodeURIComponent(edit.id)}` : ""),{method:edit ? "PUT" : "POST",body});
    if(epoch!==state.conversationEpoch)return;resetMemoryEditor();await loadMemories();showToast("记忆已保存");
  }catch(e){if(epoch===state.conversationEpoch)setError($("#memory-error"),e.message);}
  finally{$("#memory-save").disabled=false;}
});
$("#memory-list").addEventListener("click",async event=>{
  const edit=event.target.closest("[data-edit-memory]")?.dataset.editMemory;
  const remove=event.target.closest("[data-delete-memory]")?.dataset.deleteMemory;
  const row=state.memories.find(m=>m.id===(edit||remove));if(!row)return;
  if(edit){state.memoryEdit=row;$("#memory-scope").value=row.namespace==="user" ? "user" : "knowledge_base";$("#memory-scope").disabled=true;
    for(const key of ["kind","key","content","status"])$(`#memory-${key}`).value=row[key];
    $("#memory-expiry").value=row.expires_at ? new Date(Date.parse(row.expires_at)-new Date(row.expires_at).getTimezoneOffset()*60000).toISOString().slice(0,16) : "";
    $("#memory-save").textContent="保存更正";$("#memory-key").focus();$("#memory-form").scrollIntoView({block:"nearest"});return;
  }
  try {await api(`${memoryPath()}/${encodeURIComponent(row.id)}?expected_revision=${row.revision}`,{method:"DELETE"});resetMemoryEditor();await loadMemories();showToast("记忆已删除，后续请求不再使用");}
  catch(e){setError($("#memory-error"),e.message);}
});
$("#memory-more").addEventListener("click",()=>loadMemories(true).catch(e=>setError($("#memory-error"),e.message)));

async function saveTurnTask(turnId) {
  const epoch=state.conversationEpoch;
  try {await api(`/knowledge-bases/${encodeURIComponent(state.currentKnowledgeBase.id)}/tasks`,{method:"POST",body:{source_turn_id:turnId}});
    if(epoch===state.conversationEpoch)showToast("任务检查点已保存，可在右侧恢复");
  }catch(e){if(epoch===state.conversationEpoch)showToast(e.message,true);}
}
async function loadTasks(more=false) {
  const epoch=state.conversationEpoch;
  const data=await api(`/knowledge-bases/${encodeURIComponent(state.currentKnowledgeBase.id)}/tasks?offset=${more ? state.tasks.length : 0}`);
  if(epoch!==state.conversationEpoch)return;
  state.tasks=more ? [...state.tasks,...data.items] : data.items;
  const names={ready:"可恢复",awaiting_clarification:"待补充条件",running:"执行中",completed:"已结束执行",failed:"执行失败，可恢复",cancelled:"已取消"};
  $("#tasks-list").innerHTML=state.tasks.map(t=>`<article class="memory-row" data-task-row="${escapeHtml(t.id)}"><header><strong>${escapeHtml(t.title)}</strong><span class="memory-state">${names[t.status] || "未知"}</span></header>
    <p>修订 ${t.revision} · ${escapeHtml(new Date(t.updated_at).toLocaleString())}</p>
    <div class="memory-actions"><button type="button" class="text-button" data-task-detail="${escapeHtml(t.id)}">目标 / 步骤 / 尝试</button>
    ${t.status!=="cancelled" ? `<button type="button" class="text-button" data-resume-task="${escapeHtml(t.id)}">在新对话恢复</button><button type="button" class="text-button danger-link" data-cancel-task="${escapeHtml(t.id)}">取消任务</button>` : ""}</div><div class="task-detail"></div></article>`).join("") || '<p class="retrieval-note">还没有检查点，请在一轮回答下点击“保存任务”。</p>';
  $("#tasks-more").classList.toggle("is-hidden",!data.has_more);
}
$("#open-tasks").addEventListener("click",async()=>{$("#tasks-dialog").showModal();setError($("#tasks-error"));try{await loadTasks();}catch(e){setError($("#tasks-error"),e.message);}});
$("#close-tasks").addEventListener("click",()=>$("#tasks-dialog").close());
$("#tasks-more").addEventListener("click",()=>loadTasks(true).catch(e=>setError($("#tasks-error"),e.message)));
$("#tasks-list").addEventListener("click",async event=>{
  const detail=event.target.closest("[data-task-detail]")?.dataset.taskDetail;
  const resume=event.target.closest("[data-resume-task]")?.dataset.resumeTask;
  const cancel=event.target.closest("[data-cancel-task]")?.dataset.cancelTask;
  const row=state.tasks.find(t=>t.id===(detail||resume||cancel));if(!row)return;
  const epoch=state.conversationEpoch;
  try {
    if(cancel){await api(`/tasks/${encodeURIComponent(row.id)}/cancel`,{method:"POST",body:{expected_revision:row.revision}});if(epoch===state.conversationEpoch)await loadTasks();return;}
    const task=await api(`/tasks/${encodeURIComponent(row.id)}`);if(epoch!==state.conversationEpoch)return;
    if(resume){$("#tasks-dialog").close();startNewConversation();state.resumeTask=task;
      const bar=document.createElement("div");bar.id="resume-task-bar";bar.className="resume-task-bar";
      const label=document.createElement("span");label.textContent=`恢复任务：${task.title}（可在输入中补充更正）`;
      const clear=document.createElement("button");clear.type="button";clear.className="text-button";clear.textContent="取消本次恢复";clear.onclick=()=>{state.resumeTask=null;bar.remove();};bar.append(label,clear);$("#query-form").before(bar);
      $("#question").value="继续此任务，请重新检索并核对来源。";resizeQuestion();$("#question").focus();return;
    }
    const box=event.target.closest("[data-task-row]").querySelector(".task-detail");box.replaceChildren();
    const lines=[`用户目标：${task.state.goal}`,task.state.note,
      ...((task.state.user_inputs || []).map(r=>`${String(r.turn_id).startsWith("memory:") ? "记忆线索（仅助识别，不覆盖任务）" : "保存的用户条件（不是文档事实）"}：${r.question}${r.supplements?.length ? `；补充：${r.supplements.join("；")}` : ""}`)),
      ...(task.state.pending_intent ? [`待补充原始问题：${task.state.pending_intent.question || "需要完整重述"}`] : []),
      `原检查点来源版本 ${Object.keys(task.state.source_versions || {}).length} 项；恢复会重新核对，不直接复用旧证据。`,
      ...((task.state.plan?.constraints || []).map(c=>`条件 ${c.id}：${c.quote}`)),
      ...((task.state.plan?.steps || []).map(s=>`${s.id} ${s.kind}：${s.question}；依赖 ${s.depends_on?.join("、") || "无"}；逐步完成情况未验证。`)),
      ...(task.attempts || []).map(a=>`尝试 ${a.id}：${a.status}；前次 ${a.retry_of_attempt_id || "无"}；结果轮次 ${a.turn_id}；质量 ${a.result?.semantic_quality || "unknown"}；${a.result?.task_fence_accepted===false ? "取消或失去执行权后到达，未覆盖任务" : ""}`)];
    for(const text of lines){const p=document.createElement("p");p.textContent=text;box.append(p);}
  }catch(e){if(epoch===state.conversationEpoch)setError($("#tasks-error"),e.message);}
});
function startNewConversation() {
  if (!state.currentKnowledgeBase) return;
  const conversations = state.conversations;
  resetConversation(); state.conversations = conversations; renderConversationOptions();
  state.expandedKnowledgeBases.add(state.currentKnowledgeBase.id); renderKnowledgeBases();
  sessionStorage.removeItem(`knowflow_conversation_${state.currentKnowledgeBase.id}`);
  state.lastAnswer = null; $("#answer-panel").replaceChildren(); $("#answer-panel").classList.add("is-hidden");
  $("#question").focus();
}
async function selectConversation(id) {
  if (state.conversation?.id === id && !state.conversationBusy) return;
  const list = state.conversations;
  resetConversation(); state.conversations = list;
  state.conversationLoadTarget = id; renderConversationOptions();
  state.lastAnswer = null; $("#answer-panel").replaceChildren(); $("#answer-panel").classList.add("is-hidden");
  sessionStorage.setItem(`knowflow_conversation_${state.currentKnowledgeBase.id}`, id);
  const epoch = state.conversationEpoch;
  try { await loadConversation(id, state.conversationEpoch); }
  catch (error) { if (epoch === state.conversationEpoch) showRecordError(error.message); }
}
function showRecordError(message) {
  setError($("#query-error"), `${message} 可重新加载服务器记录，不会重新生成回答。`);
  $("#reload-records").classList.remove("is-hidden");
}
async function reloadConversationRecords() {
  const epoch = state.conversationEpoch;
  if (!state.token || state.conversationBusy || !state.currentKnowledgeBase) return;
  try {
    const id = state.conversationLoadTarget || state.conversation?.id;
    if (id) await loadConversation(id, epoch);
    else { await loadConversations(); if (epoch === state.conversationEpoch) {setError($("#query-error")); $("#reload-records").classList.add("is-hidden");} }
  } catch (error) { if (epoch === state.conversationEpoch) showRecordError(error.message); }
}
$("#reload-records").addEventListener("click", reloadConversationRecords);
window.addEventListener("focus", () => { if (state.conversation) reloadConversationRecords(); });
$("#older-turns").addEventListener("click", async () => {
  try { await loadConversation(state.conversation.id, state.conversationEpoch, true); }
  catch (error) { showRecordError(error.message); }
});

async function renderAnswer(response) {
  state.lastAnswer = response;
  const panel = $("#answer-panel");
  const citations = response.citations || [];
  const citationHtml = citations.map((citation, index) => {
    const assessment = response.retrieval_evidence?.find((row) => row.chunk_id === citation.chunk_id);
    const location = [citation.section, citation.page_number ? `第 ${citation.page_number} 页` : null, ...tabularLocations(citation.source_spans)]
      .filter(Boolean).join(" · ");
    const ranks = [
      assessment ? RELEVANCE_NAMES[assessment.assessment.verdict] || "未核验" : null,
      Number.isFinite(assessment?.cosine_similarity) ? `余弦相似度 ${assessment.cosine_similarity.toFixed(3)}` : null,
      citation.vector_rank ? `向量 ${citation.vector_rank}` : null,
      citation.keyword_rank ? `关键词 ${citation.keyword_rank}` : null,
      citation.graph_rank ? `图谱 ${citation.graph_rank}` : null,
      citation.table_rank ? `表格行 ${citation.table_rank}` : null,
      citation.reference_paths?.length ? `文内引用展开 · ${citation.reference_paths.map((path) => path.label).join("、")}` : null,
      citation.rerank_score !== null && citation.rerank_score !== undefined
        ? `重排 ${Number(citation.rerank_score).toFixed(3)}` : null,
    ].filter(Boolean);
    return `<article class="citation">
      <div class="citation-top"><div class="citation-source"><span class="citation-index">${index + 1}</span><span class="citation-name" title="${escapeHtml(citation.filename)}">${escapeHtml(citation.filename)}</span></div><span class="citation-location">${escapeHtml(location || "原文片段")}</span></div>
      ${ranks.length ? `<div class="citation-meta">${ranks.map((rank) => `<span class="rank-chip">${escapeHtml(rank)}</span>`).join("")}</div>` : ""}
      <blockquote class="citation-quote">${escapeHtml(citation.quote)}</blockquote>
      ${citation.context_excerpt ? `<details class="citation-context"><summary>更多原文上下文</summary><p>同一切片的补充节选，供核对；不表示这些内容均已通过逐句审核。</p><blockquote class="citation-quote">${escapeHtml(citation.context_excerpt)}</blockquote></details>` : ""}
      ${tabularLocations(citation.source_spans).length ? `<button type="button" class="text-button" data-table-source-chunk="${escapeHtml(citation.chunk_id)}">查看表格来源与原值</button>` : ""}
      ${citation.filename.toLowerCase().endsWith(".pdf") ? `<div class="citation-meta">${[...new Set([citation.page_number, ...(citation.source_spans || []).map((s) => s.page_number)].filter(Boolean))].map((page) => `<button type="button" class="text-button" data-source-chunk="${escapeHtml(citation.chunk_id)}" data-source-page="${page}">查看原页 · ${page}</button>`).join("")}</div>` : ""}
    </article>`;
  }).join("");
  panel.innerHTML = `
    <div class="answer-header"><h2>${response.refused ? "没有足够的文档依据" : "回答"}</h2><span class="answer-mode">${escapeHtml(STRATEGY_NAMES[response.strategy] || response.strategy)}${response.generation_route ? ` · ${escapeHtml(response.generation_route.model)} · ${response.generation_route.thinking === "enabled" ? "思考开启" : response.generation_route.thinking === "disabled" ? "思考关闭" : "服务默认"}` : ""}</span></div>
    ${partialAnswerHtml(response.partial_answer)}
    ${referentOptionsHtml(response.referent_candidates)}
    ${summaryScopeHtml(response.document_summary)}
    ${coverageHtml(response.query_plan?.coverage)}
    <div class="${response.refused ? "refusal-note" : "answer-copy"}"></div>
    ${clarificationHintsHtml(response.clarification)}
    <div class="retrieval-diagnostics"></div>
    ${citations.length ? `<div class="citations-heading"><h3>引用来源</h3><span>${citations.length} 个原文片段</span></div><div class="citation-list">${citationHtml}</div>` : ""}`;
  panel.classList.remove("is-hidden");
  renderRetrievalEvidence(panel.querySelector(".retrieval-diagnostics"), response);
  await renderRichAnswer(panel.querySelector(response.refused ? ".refusal-note" : ".answer-copy"), response.answer);
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

let sourceUrl = null;
let sourceRequest = 0;
let reportDocumentId = null;
const sourceDialog = $("#source-dialog");
function clearSourceUrl() {
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = null;
}
sourceDialog.addEventListener("close", () => { sourceRequest++; clearSourceUrl(); });
$("#close-source").addEventListener("click", () => sourceDialog.close());
$("#source-back").addEventListener("click", () => showParsingReport(reportDocumentId).catch((error) => showToast(error.message, true)));

async function showParsingReport(docId) {
  const request = ++sourceRequest;
  clearSourceUrl();
  reportDocumentId = docId;
  $("#source-title").textContent = "解析报告";
  $("#source-back").classList.add("is-hidden");
  $("#source-content").textContent = "正在加载解析报告…";
  if (!sourceDialog.open) sourceDialog.showModal();
  const doc = await api(`/documents/${encodeURIComponent(docId)}`);
  if (request !== sourceRequest) return;
  $("#source-title").textContent = `${doc.filename} · 解析报告`;
  renderParsingReport($("#source-content"), doc, (page) => showSourcePage(docId, page, []).catch((error) => { $("#source-content").textContent = error.message; }));
}

async function showDiagnostics(docId) {
  const request = ++sourceRequest;
  clearSourceUrl(); reportDocumentId = null;
  $("#source-title").textContent = "自动诊断";
  $("#source-back").classList.add("is-hidden");
  $("#source-content").textContent = "正在检查已保存的解析块与切片…";
  if (!sourceDialog.open) sourceDialog.showModal();
  try {
    const data = await api(`/documents/${encodeURIComponent(docId)}/diagnostics`);
    if (request !== sourceRequest) return;
    $("#source-title").textContent = `${data.filename} · 自动诊断`;
    renderDiagnostics($("#source-content"), data, (example) => {
      sourceDialog.close();
      showSegments(docId, example.chunk_id ? "chunks" : "blocks").catch((error) => showToast(error.message, true));
    });
  } catch (error) {
    if (request === sourceRequest) $("#source-content").textContent = error.message;
  }
}

async function showBatchDiagnostics(batchId) {
  batchDialog.close();
  const request = ++sourceRequest;
  clearSourceUrl(); reportDocumentId = null;
  $("#source-title").textContent = "批量来源与切片诊断";
  $("#source-back").classList.add("is-hidden");
  $("#source-content").textContent = "正在按预算检查本批文档，不调用模型…";
  if (!sourceDialog.open) sourceDialog.showModal();
  try {
    const data = await api(`/ingestion-batches/${encodeURIComponent(batchId)}/diagnostics`);
    if (request !== sourceRequest) return;
    renderBatchDiagnostics($("#source-content"), data, (docId, example) => {
      sourceDialog.close();
      showSegments(docId, example.chunk_id ? "chunks" : "blocks").catch((error) => showToast(error.message, true));
    });
  } catch (error) {
    if (request === sourceRequest) $("#source-content").textContent = error.message;
  }
}

async function showSourcePage(docId, page, spans) {
  const request = ++sourceRequest;
  clearSourceUrl();
  $("#source-title").textContent = `原文 · 第 ${page} 页`;
  $("#source-back").classList.toggle("is-hidden", !reportDocumentId);
  $("#source-content").textContent = "正在加载原页…";
  if (!sourceDialog.open) sourceDialog.showModal();
  const result = await fetch(`${API}/documents/${encodeURIComponent(docId)}/pages/${Number(page)}`, { headers: { Authorization: `Bearer ${state.token}` } });
  if (!result.ok) throw new Error("原页加载失败，请检查登录状态或文件是否仍存在。");
  const blob = await result.blob();
  if (request !== sourceRequest) return;
  sourceUrl = URL.createObjectURL(blob);
  renderSourcePage($("#source-content"), sourceUrl, Number(page), spans);
}

$("#answer-panel").addEventListener("click", async (event) => {
  const tableButton = event.target.closest("[data-table-source-chunk]");
  if (tableButton) {
    const citation = state.lastAnswer?.citations.find((item) => item.chunk_id === tableButton.dataset.tableSourceChunk);
    if (citation) await showSegments(citation.document_id, "blocks", citation.section || "");
    return;
  }
  const button = event.target.closest("[data-source-chunk]");
  if (!button) return;
  const citation = state.lastAnswer?.citations.find((item) => item.chunk_id === button.dataset.sourceChunk);
  if (!citation) return;
  reportDocumentId = null;
  try { await showSourcePage(citation.document_id, Number(button.dataset.sourcePage), citation.source_spans); }
  catch (error) { $("#source-content").textContent = error.message; }
});

const segmentState = { docId: null, offset: 0, limit: 20, total: 0, request: 0 };
const segmentsDialog = $("#segments-dialog");
$("#close-segments").addEventListener("click", () => segmentsDialog.close());
segmentsDialog.addEventListener("close", () => { segmentState.request++; });
async function showSegments(docId, view = "chunks", query = "") {
  segmentState.docId = docId; segmentState.offset = 0;
  $("#segments-view").value = view; $("#segments-search").value = query.slice(0, 200); $("#segments-page").value = "";
  if (!segmentsDialog.open) segmentsDialog.showModal();
  await loadSegments();
}
async function loadSegments() {
  const request = ++segmentState.request;
  $("#segments-content").textContent = "正在加载实际入库内容…";
  $("#segments-range").textContent = "";
  $("#segments-prev").disabled = true; $("#segments-next").disabled = true;
  const params = new URLSearchParams({ view: $("#segments-view").value, offset: segmentState.offset, limit: segmentState.limit, q: $("#segments-search").value });
  if ($("#segments-page").value) params.set("page_number", $("#segments-page").value);
  try {
    const response = await api(`/documents/${encodeURIComponent(segmentState.docId)}/segments?${params}`);
    if (request !== segmentState.request) return;
    segmentState.total = response.total;
    $("#segments-title").textContent = `${response.filename} · 切片检查`;
    renderSegments($("#segments-content"), response, (page, spans) => {
      reportDocumentId = null;
      showSourcePage(segmentState.docId, page, spans).catch((error) => { $("#source-content").textContent = error.message; });
    }, (blockKey, offset) => api(`/documents/${encodeURIComponent(segmentState.docId)}/table-cells?${new URLSearchParams({ block_key: blockKey, offset, limit: 40 })}`));
    $("#segments-range").textContent = response.total ? `${response.offset + 1}–${Math.min(response.offset + response.limit, response.total)} / ${response.total}` : "0 / 0";
    $("#segments-prev").disabled = response.offset === 0;
    $("#segments-next").disabled = response.offset + response.limit >= response.total;
  } catch (error) { if (request === segmentState.request) $("#segments-content").textContent = error.message; }
}
$("#segments-form").addEventListener("submit", (event) => { event.preventDefault(); segmentState.offset = 0; loadSegments(); });
$("#segments-view").addEventListener("change", () => { segmentState.offset = 0; loadSegments(); });
$("#segments-prev").addEventListener("click", () => { segmentState.offset = Math.max(0, segmentState.offset - segmentState.limit); loadSegments(); });
$("#segments-next").addEventListener("click", () => { segmentState.offset += segmentState.limit; loadSegments(); });

const ocrDialog = $("#ocr-dialog");
const ocrState = { docId: null, page: 1, request: 0, timer: null };
function stopOcrPolling() { clearTimeout(ocrState.timer); ocrState.timer = null; }
$("#close-ocr").addEventListener("click", () => ocrDialog.close());
ocrDialog.addEventListener("close", () => { ++ocrState.request; stopOcrPolling(); });
async function showOcrReview(docId) {
  const doc = state.documents.find((item) => item.id === docId);
  ocrState.docId = docId; ocrState.page = 1;
  $("#ocr-page").value = "1";
  const count = doc?.parsing_report?.source_page_count || doc?.parsing_report?.page_count;
  if (count) $("#ocr-page").max = String(count); else $("#ocr-page").removeAttribute("max");
  $("#ocr-title").textContent = `${doc?.filename || "PDF"} · 单页 OCR 对照`;
  if (!ocrDialog.open) ocrDialog.showModal();
  await loadOcrReview();
}
async function loadOcrReview(start = false, polling = false) {
  stopOcrPolling();
  const request = ++ocrState.request;
  const docId = ocrState.docId; const page = ocrState.page;
  $("#ocr-start").disabled = true;
  if (!polling) $("#ocr-content").textContent = start ? "正在提交单页 OCR 对照…" : "正在加载已保存的对照…";
  try {
    const review = await api(`/documents/${encodeURIComponent(docId)}/ocr-review?page_number=${page}`, { method: start ? "POST" : "GET" });
    if (request !== ocrState.request || !ocrDialog.open) return;
    renderOcrReview($("#ocr-content"), review, (physicalPage) => {
      reportDocumentId = null;
      showSourcePage(docId, physicalPage, []).catch((error) => { $("#source-content").textContent = error.message; });
    });
    $("#ocr-start").disabled = !review.can_start || review.status === "running";
    $("#ocr-start").textContent = review.status === "running" ? "OCR 运行中…" : review.status === "not_started" ? "运行单页 OCR" : "重新运行对照";
    if (review.status === "running") ocrState.timer = setTimeout(() => loadOcrReview(false, true), 2400);
  } catch (error) {
    if (request !== ocrState.request || !ocrDialog.open) return;
    $("#ocr-content").textContent = error.message;
    $("#ocr-start").disabled = false;
    $("#ocr-start").textContent = "重试单页 OCR";
  }
}
$("#ocr-form").addEventListener("submit", (event) => {
  event.preventDefault();
  ocrState.page = Number($("#ocr-page").value);
  loadOcrReview();
});
$("#ocr-start").addEventListener("click", () => {
  if (!$("#ocr-page").reportValidity()) return;
  ocrState.page = Number($("#ocr-page").value);
  loadOcrReview(true);
});

const referencesDialog = $("#references-dialog");
const referencesState = { docId: null, offset: 0, limit: 20, request: 0, building: false };
$("#close-references").addEventListener("click", () => referencesDialog.close());
referencesDialog.addEventListener("close", () => ++referencesState.request);
async function showReferences(docId) {
  referencesState.docId = docId; referencesState.offset = 0;
  $("#references-status").value = "all";
  if (!referencesDialog.open) referencesDialog.showModal();
  await loadReferences();
}
async function loadReferences() {
  const request = ++referencesState.request;
  const docId = referencesState.docId;
  $("#references-content").textContent = "正在加载文内引用索引…";
  $("#references-prev").disabled = true; $("#references-next").disabled = true;
  const params = new URLSearchParams({ offset: referencesState.offset, limit: referencesState.limit, status_filter: $("#references-status").value });
  try {
    const result = await api(`/documents/${encodeURIComponent(docId)}/references?${params}`);
    if (request !== referencesState.request) return;
    $("#references-title").textContent = `${result.filename} · 文内引用`;
    $("#rebuild-references").disabled = referencesState.building || result.document_status !== "processed";
    renderDocumentReferences($("#references-content"), result, (page, spans) => {
      reportDocumentId = null;
      showSourcePage(docId, page, spans).catch((error) => { $("#source-content").textContent = error.message; });
    });
    $("#references-range").textContent = result.total ? `${result.offset + 1}–${Math.min(result.offset + result.limit, result.total)} / ${result.total}` : "0 / 0";
    $("#references-prev").disabled = result.offset === 0;
    $("#references-next").disabled = result.offset + result.limit >= result.total;
  } catch (error) { if (request === referencesState.request) $("#references-content").textContent = error.message; }
}
$("#references-form").addEventListener("submit", (event) => { event.preventDefault(); referencesState.offset = 0; loadReferences(); });
$("#references-prev").addEventListener("click", () => { referencesState.offset = Math.max(0, referencesState.offset - referencesState.limit); loadReferences(); });
$("#references-next").addEventListener("click", () => { referencesState.offset += referencesState.limit; loadReferences(); });
$("#rebuild-references").addEventListener("click", async () => {
  const docId = referencesState.docId;
  referencesState.building = true; $("#rebuild-references").disabled = true;
  try {
    await api(`/documents/${encodeURIComponent(docId)}/reference-index`, { method: "POST" });
    if (docId === referencesState.docId && referencesDialog.open) { referencesState.offset = 0; await loadReferences(); }
    showToast("文内引用索引已更新，正文、向量和图谱未重建");
  } catch (error) { showToast(error.message, true); }
  finally { referencesState.building = false; $("#rebuild-references").disabled = false; }
});

function renderEvaluations() {
  const host = $("#evaluation-list");
  $("#empty-evaluations").classList.toggle("is-hidden", state.evaluations.length > 0);
  host.innerHTML = state.evaluations.map((run) => {
    const quality = run.kind === "rag_quality";
    const tableVariant = ["hybrid", "hybrid_rerank", "graph_hybrid"].includes(run.strategy)
      && typeof run.table_row_hints_enabled === "boolean"
      ? run.table_row_hints_enabled ? " · 含表格行" : " · 基础" : "";
    const referenceLabel = run.reference_expansion_enabled ? " · 文内引用" : "";
    const mrr = run.metrics?.mrr;
    const recallKey = Object.keys(run.metrics || {}).find((key) => key.startsWith("mean_recall@") || key.startsWith("recall@"));
    const qualityLabel = quality ? `质量验证 · ${run.status === "running" ? `运行中 ${run.completed_cases}/${run.total_cases}` : ({completed:"运行完成", incomplete:"未完整完成", failed:"失败", interrupted:"中断"})[run.status] || run.status}` : (STRATEGY_NAMES[run.strategy] || run.strategy) + tableVariant + referenceLabel;
    return `<button class="evaluation-run" type="button" data-run-id="${escapeHtml(run.run_id)}" data-run-kind="${escapeHtml(run.kind || "retrieval")}">
      <span class="run-top"><span class="run-strategy">${escapeHtml(qualityLabel)}</span><span class="run-date">${escapeHtml(formatDate(run.created_at))}</span></span>
      <span class="run-metrics"><span>MRR ${formatMetric(mrr)}</span><span>Recall ${formatMetric(recallKey ? run.metrics[recallKey] : null)}</span></span>
      <span class="run-detail is-hidden"></span>
    </button>`;
  }).join("");
}

function formatMetric(value) {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

$("#evaluation-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-run-id]");
  if (!button) return;
  if (button.dataset.runKind === "rag_quality") {
    qualityRunId = button.dataset.runId; qualityRevision = "";
    $("#quality-form").classList.add("is-hidden");
    $("#quality-title").textContent = "验证记录";
    $("#quality-dialog").showModal(); await loadQualityReport(); return;
  }
  const detail = button.querySelector(".run-detail");
  if (!detail.classList.contains("is-hidden")) {
    detail.classList.add("is-hidden");
    return;
  }
  if (!detail.dataset.loaded) {
    try {
      const report = await api(`/knowledge-bases/${encodeURIComponent(state.currentKnowledgeBase.id)}/evaluations/${encodeURIComponent(button.dataset.runId)}`);
      const rows = (report.cases || []).map((item) => {
        const metrics = item.metrics || {};
        const coverage = typeof metrics.candidate_evidence_recall === "number" ? ` · 展开候选证据覆盖 ${formatMetric(metrics.candidate_evidence_recall)}` : "";
        return `<div><strong>${escapeHtml(item.id)}</strong><br />Recall ${formatMetric(metrics[Object.keys(metrics).find((key) => key.startsWith("recall@"))])} · ${item.retrieved?.length || 0} 条基础召回${coverage} · ${Number(item.retrieval_latency_ms || 0).toFixed(0)} ms</div>`;
      }).join("");
      detail.innerHTML = `<div>数据集 ${escapeHtml(report.dataset_version)} · top-${escapeHtml(report.top_k)}</div>${rows}`;
      detail.dataset.loaded = "true";
    } catch (error) {
      detail.textContent = error.message;
    }
  }
  detail.classList.remove("is-hidden");
});

let graphOffset = 0;
let qualityRunId = null;
let qualityRevision = "";
async function loadGraph() {
  const response = await api(`/knowledge-bases/${encodeURIComponent(state.currentKnowledgeBase.id)}/graph?offset=${graphOffset}&limit=20&q=${encodeURIComponent($("#graph-search").value)}&status=${encodeURIComponent($("#graph-status").value)}`);
  renderGraphEvidence($("#graph-content"), response, async (row) => {
    try { await showSourcePage(row.document_id, row.page_number, row.source_spans); } catch (error) { showToast(error.message, true); }
  });
  $("#graph-range").textContent = `${response.filtered_edge_count ? graphOffset + 1 : 0}–${graphOffset + response.edges.length} / ${response.filtered_edge_count}`;
  $("#graph-prev").disabled = graphOffset === 0;
  $("#graph-next").disabled = graphOffset + response.edges.length >= response.filtered_edge_count;
}
$("#open-graph").addEventListener("click", async () => { graphOffset = 0; $("#graph-search").value = ""; $("#graph-status").value = "all"; $("#graph-dialog").showModal(); try { await loadGraph(); } catch (error) { $("#graph-content").textContent = error.message; } });
$("#close-graph").addEventListener("click", () => $("#graph-dialog").close());
$("#graph-form").addEventListener("submit", async (event) => { event.preventDefault(); graphOffset = 0; try { await loadGraph(); } catch (error) { showToast(error.message, true); } });
$("#graph-prev").addEventListener("click", async () => { graphOffset = Math.max(0, graphOffset - 20); try { await loadGraph(); } catch (error) { showToast(error.message, true); } });
$("#graph-next").addEventListener("click", async () => { graphOffset += 20; try { await loadGraph(); } catch (error) { showToast(error.message, true); } });

async function loadQualityReport() {
  const requestedRun = qualityRunId;
  const requestedKnowledgeBase = state.currentKnowledgeBase?.id;
  if (!requestedRun || !requestedKnowledgeBase) return;
  try {
    const report = await api(`/knowledge-bases/${encodeURIComponent(requestedKnowledgeBase)}/evaluations/${encodeURIComponent(requestedRun)}`);
    if (qualityRunId !== requestedRun || state.currentKnowledgeBase?.id !== requestedKnowledgeBase) return;
    const revision = `${requestedRun}:${report.status}:${report.completed_cases || 0}:${report.phase || ""}`;
    if (revision !== qualityRevision) { renderQualityReport($("#quality-content"), report); qualityRevision = revision; }
  } catch (error) { if (qualityRunId === requestedRun && state.currentKnowledgeBase?.id === requestedKnowledgeBase) $("#quality-content").textContent = error.message; }
}
$("#open-quality").addEventListener("click", () => {
  qualityRunId = null; qualityRevision = "";
  $("#quality-form").classList.remove("is-hidden");
  $("#quality-title").textContent = "新建质量验证";
  $("#quality-dataset").value = state.documents.some((d) => d.filename === "2206.01062.pdf") ? "tables" : state.documents.some((d) => d.filename === "2203.01017v2.pdf") ? "crosspage" : "engineering";
  $("#quality-graph").checked = state.documents.some((d) => d.graph_status === "processed" && d.parsing_report?.graph_index?.semantic_review?.eligible_relations > 0);
  const baseline = $("#quality-baseline"); baseline.replaceChildren(new Option("不对比历史", ""));
  for (const run of state.evaluations.filter((r) => r.kind === "rag_quality" && r.status !== "running")) baseline.append(new Option(`${formatDate(run.created_at)} · ${run.dataset_version} · ${run.status}`, run.run_id));
  $("#quality-content").textContent = "验证会逐题展示标准证据、召回排名、进入上下文与最终引用。图谱对照需先构建关系索引。";
  $("#quality-dialog").showModal();
});
$("#close-quality").addEventListener("click", () => $("#quality-dialog").close());
$("#quality-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const button = event.submitter; button.disabled = true;
  try {
    let custom = null;
    if ($("#quality-dataset").value === "custom") {
      const file = $("#quality-file").files[0];
      if (!file || file.size > 1024 * 1024) throw new Error("请选择不超过 1 MB 的题集 JSON 文件。");
      custom = JSON.parse(await file.text());
    }
    const report = await api(`/knowledge-bases/${encodeURIComponent(state.currentKnowledgeBase.id)}/quality-evaluations`, {method:"POST", body:{dataset:$("#quality-dataset").value, custom_dataset:custom, strategies:$("#quality-graph").checked ? ["hybrid","graph_hybrid"] : ["hybrid"], top_k:Number($("#quality-k").value), judge:$("#quality-judge").checked, baseline_run_id:$("#quality-baseline").value || null}});
    qualityRunId = report.run_id; qualityRevision = "";
    $("#quality-form").classList.add("is-hidden"); $("#quality-title").textContent = "验证记录";
    renderQualityReport($("#quality-content"), report); await refreshWorkspace();
  } catch (error) { $("#quality-content").textContent = error.message; }
  finally { button.disabled = false; }
});

if (state.token) {
  showAuthenticatedApp();
  refreshWorkspace();
} else {
  showAuth();
}
