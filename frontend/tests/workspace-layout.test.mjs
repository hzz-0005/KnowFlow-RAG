import assert from "node:assert/strict";
import {test} from "node:test";
import {JSDOM} from "jsdom";
import {fitPanes,installWorkspaceLayout} from "../workspace-layout.js";

function setup(width=1280,stored=null) {
  const dom=new JSDOM(`<div id="app-view"><div id="left-pane-divider"></div><div id="right-pane-divider"></div><aside id="right-rail"></aside><nav id="kb-list"><div class="kb-row"><button><span class="nav-label-text">A title</span></button></div></nav><textarea id="question">keep this draft</textarea></div>`,{url:"http://localhost/"});
  Object.defineProperty(dom.window,"innerWidth",{value:width,writable:true});
  if(stored!==null)dom.window.localStorage.setItem("knowflow_pane_widths_v1",stored);
  return {dom,win:dom.window,root:dom.window.document.querySelector('#app-view')};
}
const key=(win,target,name)=>target.dispatchEvent(new win.KeyboardEvent('keydown',{key:name,bubbles:true,cancelable:true}));
const pointer=(win,target,type,x,id=1)=>{
  const event=new win.MouseEvent(type,{clientX:x,button:0,bubbles:true,cancelable:true});
  Object.defineProperty(event,'pointerId',{value:id});target.dispatchEvent(event);
};

test("pane fitting preserves the desktop chat width and ignores invalid stored widths",()=>{
  for(const viewport of [681,800,981,1024,1440]) {
    const fitted=fitPanes({left:99999,right:99999},viewport);
    assert.ok(fitted.left>=208 && fitted.left<=520);
    assert.ok(fitted.right>=224 && fitted.right<=480);
    assert.ok(viewport-fitted.left-(viewport>980?fitted.right:0)>=360);
  }
  assert.deepEqual(fitPanes({left:"bad",right:null},1440),{left:248,right:288});
});

test("drag, cancel, keyboard clamps and refresh restoration preserve the draft",()=>{
  const h=setup();
  try{
    const layout=installWorkspaceLayout(h.root), doc=h.win.document;
    const left=doc.querySelector('#left-pane-divider'),right=doc.querySelector('#right-pane-divider');
    pointer(h.win,left,'pointerdown',248);pointer(h.win,left,'pointermove',350);pointer(h.win,left,'pointerup',350);
    assert.equal(layout.getWidths().left,350);
    pointer(h.win,right,'pointerdown',1000);pointer(h.win,right,'pointermove',950);pointer(h.win,right,'pointerup',950);
    assert.equal(layout.getWidths().right,338);
    const saved=h.win.localStorage.getItem('knowflow_pane_widths_v1');
    pointer(h.win,left,'pointerdown',350);pointer(h.win,left,'pointermove',500);pointer(h.win,left,'pointercancel',500);
    assert.equal(layout.getWidths().left,350);assert.equal(h.root.classList.contains('is-resizing'),false);
    assert.equal(h.win.localStorage.getItem('knowflow_pane_widths_v1'),saved);
    key(h.win,left,'Home');assert.equal(layout.getWidths().left,208);assert.equal(h.root.dataset.sidebarCompact,'true');
    key(h.win,right,'End');assert.ok(1280-layout.getWidths().left-layout.getWidths().right>=360);
    assert.equal(doc.querySelector('#question').value,'keep this draft');
    const next=setup(1280,h.win.localStorage.getItem('knowflow_pane_widths_v1'));
    try{assert.deepEqual(installWorkspaceLayout(next.root).getWidths(),layout.getWidths());}finally{next.dom.window.close();}
  }finally{h.dom.window.close();}
});

test("narrow viewport and corrupt local preferences do not break the workspace",()=>{
  const h=setup(390,'{"left":');
  try{
    const layout=installWorkspaceLayout(h.root),left=h.win.document.querySelector('#left-pane-divider');
    assert.equal(left.tabIndex,-1);key(h.win,left,'End');assert.equal(layout.getWidths().left,248);
    h.win.innerWidth=1024;h.win.dispatchEvent(new h.win.Event('resize'));
    assert.equal(left.tabIndex,0);assert.ok(1024-layout.getWidths().left-layout.getWidths().right>=360);
  }finally{h.dom.window.close();}
});

test("only overflowing names animate on deliberate hover and reset on exit",()=>{
  const h=setup();
  try{
    installWorkspaceLayout(h.root);
    const row=h.win.document.querySelector('.kb-row'),text=row.querySelector('.nav-label-text');
    Object.defineProperty(text,'clientWidth',{value:100});
    Object.defineProperty(text,'scrollWidth',{value:240,configurable:true});
    row.dispatchEvent(new h.win.MouseEvent('pointerover',{bubbles:true}));
    assert.equal(row.dataset.nameScrolling,'true');assert.equal(text.style.getPropertyValue('--name-travel'),'-140px');
    row.dispatchEvent(new h.win.MouseEvent('pointerout',{bubbles:true}));assert.equal(row.dataset.nameScrolling,undefined);
    Object.defineProperty(text,'scrollWidth',{value:100});
    row.dispatchEvent(new h.win.MouseEvent('pointerover',{bubbles:true}));assert.equal(row.dataset.nameScrolling,undefined);
  }finally{h.dom.window.close();}
});
