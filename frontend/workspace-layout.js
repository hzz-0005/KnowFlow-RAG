// Local display preferences only. No questions, document text or credentials are stored.
const KEY = "knowflow_pane_widths_v1";
const MIN = {left:208, right:224};
const MAX = {left:520, right:480};
const DEFAULT = {left:248, right:288};
const clamp = (n, low, high) => Math.max(low, Math.min(high, n));

export function fitPanes(preferred, viewport, hasRight = true) {
  let left=clamp(Number.isFinite(preferred?.left) ? preferred.left : DEFAULT.left, MIN.left, MAX.left);
  let right=clamp(Number.isFinite(preferred?.right) ? preferred.right : DEFAULT.right, MIN.right, MAX.right);
  const reservedRight=viewport>980 && hasRight ? right : 0;
  if(viewport>680) left=Math.min(left,Math.max(MIN.left,viewport-360-reservedRight));
  if(viewport>980 && hasRight) right=Math.min(right,Math.max(MIN.right,viewport-360-left));
  return {left:Math.round(left),right:Math.round(right)};
}

export function installWorkspaceLayout(root) {
  if(!root || root.dataset.paneLayoutReady) return;
  root.dataset.paneLayoutReady="true";
  const doc=root.ownerDocument, win=doc.defaultView;
  const rail=doc.querySelector("#right-rail"), nav=doc.querySelector("#kb-list");
  const handles={left:doc.querySelector("#left-pane-divider"),right:doc.querySelector("#right-pane-divider")};
  let preferred={...DEFAULT}, widths, drag=null;
  try {preferred={...preferred,...JSON.parse(win.localStorage.getItem(KEY)||"{}")};} catch {}
  const hasRight=()=>!!rail && !rail.classList.contains("is-hidden");
  const enabled=side=>win.innerWidth>680 && (side==="left" || win.innerWidth>980 && hasRight());
  const maximum=side=>Math.max(MIN[side],Math.min(MAX[side],win.innerWidth-360-(
    side==="left" ? win.innerWidth>980 && hasRight() ? widths.right : 0 : widths.left
  )));
  function refresh() {
    widths=fitPanes(preferred,win.innerWidth,hasRight());
    root.style.setProperty("--left-rail-width",`${widths.left}px`);
    root.style.setProperty("--right-rail-width",`${widths.right}px`);
    root.dataset.sidebarCompact=String(widths.left<=260);
    for(const side of ["left","right"]) {
      const handle=handles[side];if(!handle) continue;
      handle.tabIndex=enabled(side)?0:-1;
      handle.setAttribute("aria-valuemin",String(MIN[side]));
      handle.setAttribute("aria-valuemax",String(maximum(side)));
      handle.setAttribute("aria-valuenow",String(widths[side]));
      handle.setAttribute("aria-valuetext",`${widths[side]} 像素`);
    }
  }
  function persist(){try{win.localStorage.setItem(KEY,JSON.stringify(preferred));}catch{}}
  function setWidth(side,width){preferred[side]=clamp(width,MIN[side],maximum(side));refresh();}
  function finish(cancel=false) {
    if(!drag) return;
    const current=drag;drag=null;
    if(cancel) preferred=current.before;
    root.classList.remove("is-resizing");
    if(current.handle.hasPointerCapture?.(current.id)) current.handle.releasePointerCapture(current.id);
    refresh();if(!cancel) persist();
  }
  for(const side of ["left","right"]) {
    const handle=handles[side];if(!handle) continue;
    handle.addEventListener("pointerdown",event=>{
      if(event.button!==0 || !enabled(side)) return;
      event.preventDefault();handle.focus({preventScroll:true});
      drag={side,id:event.pointerId,x:event.clientX,width:widths[side],before:{...preferred},handle};
      handle.setPointerCapture?.(event.pointerId);root.classList.add("is-resizing");
    });
    handle.addEventListener("pointermove",event=>{
      if(!drag || drag.id!==event.pointerId || drag.side!==side) return;
      setWidth(side,drag.width+(event.clientX-drag.x)*(side==="left"?1:-1));
    });
    handle.addEventListener("pointerup",()=>finish());
    handle.addEventListener("pointercancel",()=>finish(true));
    handle.addEventListener("lostpointercapture",()=>finish());
    handle.addEventListener("keydown",event=>{
      if(!enabled(side)) return;
      if(event.key==="Escape" && drag){event.preventDefault();finish(true);return;}
      const step=event.shiftKey?40:16;
      let target=event.key==="Home"?MIN[side]:event.key==="End"?maximum(side):
        event.key==="ArrowRight"?widths[side]+step*(side==="left"?1:-1):
        event.key==="ArrowLeft"?widths[side]+step*(side==="left"?-1:1):null;
      if(target===null) return;
      event.preventDefault();setWidth(side,target);persist();
    });
  }
  const stopName=row=>{if(row) delete row.dataset.nameScrolling;};
  function scrollName(row){
    const text=row?.querySelector(".nav-label-text");if(!text)return;
    stopName(row);
    const overflow=text.scrollWidth-text.clientWidth;
    if(overflow>2 && !win.matchMedia?.("(prefers-reduced-motion: reduce)").matches){
      text.style.setProperty("--name-travel",`${-overflow}px`);
      text.style.setProperty("--name-duration",`${Math.max(2,overflow/28+1.5)}s`);
      row.dataset.nameScrolling="true";
    }
  }
  const rowOf=target=>target?.closest?.(".kb-row,.chat-nav-row");
  nav?.addEventListener("pointerover",e=>{const row=rowOf(e.target);if(row&&!row.contains(e.relatedTarget))scrollName(row);});
  nav?.addEventListener("pointerout",e=>{const row=rowOf(e.target);if(row&&!row.contains(e.relatedTarget))stopName(row);});
  nav?.addEventListener("focusin",e=>scrollName(rowOf(e.target)));
  nav?.addEventListener("focusout",e=>{const row=rowOf(e.target);if(row&&!row.contains(e.relatedTarget))stopName(row);});
  win.addEventListener("resize",()=>{finish();refresh();});
  win.addEventListener("blur",()=>finish());
  const observer=new win.MutationObserver(refresh);
  observer.observe(root,{attributes:true,attributeFilter:["class"]});
  if(rail) observer.observe(rail,{attributes:true,attributeFilter:["class"]});
  refresh();
  return {refresh,getWidths:()=>({...widths})};
}
