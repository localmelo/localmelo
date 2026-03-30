"""Smoke playground web UI and API routes."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from localmelo.support.gateway.playground import SmokePlayground

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>localmelo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#faf8f4;--sidebar:#f2efe9;--surface:#fff;
  --ink:#111;--muted:#888;--border:#e0dbd3;
  --accent:#1a7a7f;--accent-light:#e5f3f4;
  --user-bg:#1a7a7f;--user-fg:#fff;
  --code-bg:#1e1e1e;--code-fg:#d4d4d4;
  --r:10px;--rl:16px;
}
html,body{height:100%}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--ink);font-size:14px;line-height:1.5;
}

/* ── Layout ── */
.app{display:grid;grid-template-columns:220px 1fr;height:100dvh}

/* ── Sidebar ── */
.sidebar{
  display:flex;flex-direction:column;
  background:var(--sidebar);border-right:1px solid var(--border);
}
.sidebar-top{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 14px 12px;border-bottom:1px solid var(--border);
}
.sidebar-top h1{font-size:15px;font-weight:700;letter-spacing:-0.01em}
.new-btn{
  width:28px;height:28px;border:1px solid var(--border);border-radius:8px;
  background:var(--surface);cursor:pointer;font-size:16px;color:var(--ink);
  display:flex;align-items:center;justify-content:center;
}
.new-btn:hover{background:var(--accent-light)}
.session-list{flex:1;overflow-y:auto;padding:6px}
.session-item{
  display:flex;align-items:center;gap:6px;
  padding:9px 10px;border-radius:var(--r);cursor:pointer;margin-bottom:1px;
}
.session-item:hover{background:rgba(0,0,0,0.04)}
.session-item.active{background:var(--accent-light)}
.session-info{flex:1;min-width:0}
.session-model{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.session-memory{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.session-close{
  background:none;border:none;cursor:pointer;color:var(--muted);
  font-size:15px;padding:0 2px;opacity:0;flex-shrink:0;
}
.session-item:hover .session-close{opacity:0.6}
.session-close:hover{opacity:1!important;color:var(--ink)}
.settings-btn{
  display:flex;align-items:center;gap:6px;
  padding:12px 14px;border:none;border-top:1px solid var(--border);
  background:none;cursor:pointer;font:inherit;color:var(--muted);font-size:12px;
}
.settings-btn:hover{color:var(--ink)}

/* ── Chat ── */
.chat{display:flex;flex-direction:column;height:100dvh;min-width:0}
.chat-header{
  padding:12px 20px;border-bottom:1px solid var(--border);
  font-size:12px;color:var(--muted);flex-shrink:0;
}
.scroll{flex:1;overflow-y:auto;padding:20px 20px 8px}
.welcome{text-align:center;padding:80px 20px 40px;color:var(--muted)}
.welcome h2{font-size:20px;color:var(--ink);margin-bottom:6px;font-weight:600}
.welcome p{font-size:14px}
#messages{max-width:780px;margin:0 auto}

/* Messages */
.msg{margin-bottom:14px;display:flex}
.msg.user{justify-content:flex-end}
.msg.assistant{justify-content:flex-start}
.msg.note{justify-content:center}
.bubble{max-width:78%;padding:11px 15px;border-radius:var(--rl);font-size:14px;line-height:1.65}
.msg.user .bubble{background:var(--user-bg);color:var(--user-fg);border-bottom-right-radius:4px}
.msg.assistant .bubble{background:var(--surface);border:1px solid var(--border);border-bottom-left-radius:4px}
.msg.note .bubble{
  background:var(--accent-light);font-size:12px;color:var(--accent);
  padding:6px 14px;border-radius:999px;max-width:none;
}

/* Markdown inside bubbles */
.md p{margin-bottom:8px}.md p:last-child{margin-bottom:0}
.md code{background:var(--code-bg);color:var(--code-fg);padding:1px 5px;border-radius:4px;font-family:"SF Mono","Fira Code",monospace;font-size:12px}
.md pre{margin:8px 0;padding:12px;border-radius:var(--r);background:var(--code-bg);color:var(--code-fg);overflow-x:auto}
.md pre code{background:transparent;padding:0}
.md ul,.md ol{padding-left:18px;margin:8px 0}
.md strong{font-weight:700}

/* Thinking */
.think-toggle{margin-bottom:8px;background:rgba(26,122,127,0.06);border:1px solid rgba(26,122,127,0.12);border-radius:var(--r);overflow:hidden}
.think-toggle summary{cursor:pointer;padding:8px 11px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--accent);list-style:none}
.think-toggle summary::-webkit-details-marker{display:none}
.think-body{padding:0 11px 10px;font-size:12px;line-height:1.6;color:var(--muted);white-space:pre-wrap}

/* Typing indicator */
.typing{display:inline-flex;gap:5px;padding:4px 0}
.typing span{width:6px;height:6px;border-radius:50%;background:var(--accent);opacity:.3;animation:bounce 1s infinite ease-in-out}
.typing span:nth-child(2){animation-delay:.15s}
.typing span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:translateY(0);opacity:.2}40%{transform:translateY(-3px);opacity:1}}

/* Composer — always fixed at bottom */
.composer{flex-shrink:0;padding:10px 20px 14px;border-top:1px solid var(--border);background:var(--bg)}
.composer-inner{max-width:780px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
.composer-inner textarea{
  flex:1;border:1px solid var(--border);border-radius:var(--r);
  padding:10px 12px;font:inherit;resize:none;outline:none;
  min-height:42px;max-height:200px;background:var(--surface);
}
.composer-inner textarea:focus{border-color:var(--accent)}
.composer-inner button{
  padding:10px 16px;border:none;border-radius:var(--r);
  font:inherit;font-weight:600;font-size:13px;cursor:pointer;white-space:nowrap;
  flex-shrink:0;
}
#send-btn{background:var(--accent);color:#fff}
#send-btn:disabled{opacity:.35;cursor:default}
#stop-btn{background:#c62828;color:#fff}

/* ── Settings overlay ── */
.overlay{
  position:fixed;inset:0;background:var(--bg);z-index:100;
  overflow-y:auto;opacity:0;pointer-events:none;transition:opacity .15s ease;
}
.overlay.open{opacity:1;pointer-events:auto}
.settings{max-width:480px;margin:0 auto;padding:36px 20px 60px}
.settings-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px}
.settings-head h2{font-size:20px;font-weight:700}
.close-btn{
  width:32px;height:32px;border:1px solid var(--border);border-radius:var(--r);
  background:var(--surface);cursor:pointer;font-size:16px;
  display:flex;align-items:center;justify-content:center;color:var(--muted);
}
.close-btn:hover{color:var(--ink)}
.section{margin-bottom:24px}
.section h3{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:12px}
.ff{margin-bottom:10px}
.ff label{display:block;font-size:12px;font-weight:600;margin-bottom:3px}
.ff input,.ff select{
  width:100%;padding:9px 11px;border:1px solid var(--border);
  border-radius:var(--r);font:inherit;background:var(--surface);outline:none;
}
.ff input:focus,.ff select:focus{border-color:var(--accent)}
.btn-row{display:flex;gap:8px;margin:10px 0 6px}
.btn{padding:9px 16px;border:none;border-radius:var(--r);font:inherit;font-weight:600;font-size:13px;cursor:pointer}
.btn-outline{background:none;border:1px solid var(--border);color:var(--ink)}
.btn-outline:hover{background:rgba(0,0,0,.03)}
.status-text{font-size:11px;color:var(--muted);margin-bottom:10px;min-height:16px}
.memory-count{font-size:11px;color:var(--muted);margin-top:4px}
.create-btn{
  width:100%;padding:13px;background:var(--accent);color:#fff;border:none;
  border-radius:var(--r);font:inherit;font-size:14px;font-weight:700;cursor:pointer;margin-top:8px;
}
.create-btn:hover{opacity:.92}
.create-btn:disabled{opacity:.5;cursor:wait}
.create-error{
  font-size:13px;color:#c62828;background:#fce4ec;
  border:1px solid #ef9a9a;border-radius:var(--r);
  padding:10px 12px;margin-bottom:10px;display:none;
}
.create-error.show{display:block}
.btn-outline:disabled{opacity:.5;cursor:wait}

/* Responsive */
@media(max-width:700px){
  .app{grid-template-columns:1fr}
  .sidebar{display:none}
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sidebar-top">
      <h1>localmelo</h1>
      <button class="new-btn" id="new-btn" title="New session">+</button>
    </div>
    <div class="session-list" id="session-list"></div>
    <button class="settings-btn" id="settings-btn">Settings</button>
  </aside>

  <main class="chat">
    <div class="chat-header" id="chat-header"><span id="header-text">No active session</span></div>
    <div class="scroll" id="scroll">
      <div class="welcome" id="welcome">
        <h2>localmelo playground</h2>
        <p>Click <strong>+</strong> or <strong>Settings</strong> to connect a backend and start chatting.</p>
      </div>
      <div id="messages"></div>
    </div>
    <div class="composer">
      <div class="composer-inner">
        <textarea id="input" placeholder="Type a message..." rows="1"></textarea>
        <button id="send-btn" disabled>Send</button>
        <button id="stop-btn" style="display:none">Stop</button>
      </div>
    </div>
  </main>
</div>

<div class="overlay" id="overlay">
  <div class="settings">
    <div class="settings-head">
      <h2>Settings</h2>
      <button class="close-btn" id="close-settings">&times;</button>
    </div>

    <div class="section">
      <h3>Backend</h3>
      <div class="ff"><label>Adapter</label>
        <select id="s-adapter">
          <option value="auto">Auto-detect</option>
          <option value="mlc">MLC / OpenAI-compatible</option>
          <option value="ollama">Ollama native</option>
          <option value="vllm">vLLM</option>
          <option value="sglang">SGLang</option>
        </select>
      </div>
      <div class="ff"><label>Chat URL</label><input id="s-chat-url" placeholder="http://127.0.0.1:8400/v1"></div>
      <div class="ff"><label>Embedding URL</label><input id="s-embed-url" placeholder="same as chat if empty"></div>
      <div class="btn-row">
        <button class="btn btn-outline" id="discover-btn">Discover Models</button>
      </div>
      <div class="status-text" id="discover-status"></div>
      <div class="ff"><label>Chat Model</label><select id="s-chat-model"></select></div>
      <div class="ff"><label>Embedding Model</label><select id="s-embed-model"></select></div>
    </div>

    <div class="section">
      <h3>Memory</h3>
      <div class="ff"><label>Memory Set</label>
        <select id="s-memory">
          <option value="__all__">All</option>
          <option value="">None</option>
        </select>
      </div>
      <div class="memory-count" id="memory-count"></div>
    </div>

    <div class="create-error" id="create-error"></div>
    <button class="create-btn" id="create-btn">Create New Session</button>
  </div>
</div>

<script>
/* ── State ── */
const S={sessions:[],activeId:null,sending:false,abort:null,scenarios:[],scMap:{},discovery:null,lastCfg:null};
const KEY='localmelo-pg';
const $=id=>document.getElementById(id);

/* DOM refs */
const sessionListEl=$('session-list'),headerEl=$('header-text'),scrollEl=$('scroll');
const welcomeEl=$('welcome'),msgsEl=$('messages'),inputEl=$('input');
const sendBtn=$('send-btn'),stopBtn=$('stop-btn'),overlay=$('overlay');
const adapterEl=$('s-adapter'),chatUrlEl=$('s-chat-url'),embedUrlEl=$('s-embed-url');
const chatModelEl=$('s-chat-model'),embedModelEl=$('s-embed-model');
const memoryEl=$('s-memory'),memoryCountEl=$('memory-count'),discoverStatusEl=$('discover-status');

/* ── Helpers ── */
function esc(t){return(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

function md(text){
  let h=esc(text);
  h=h.replace(/```(\w*)\n([\s\S]*?)```/g,(_,__,c)=>'<pre><code>'+c.trimEnd()+'</code></pre>');
  h=h.replace(/`([^`]+)`/g,'<code>$1</code>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  h=h.replace(/(^|\n)([-*] .+(?:\n[-*] .+)*)/g,(_,p,b)=>{
    const li=b.split('\n').map(l=>'<li>'+l.replace(/^[-*] /,'')+'</li>').join('');
    return p+'<ul>'+li+'</ul>';
  });
  h=h.replace(/(^|\n)(\d+\. .+(?:\n\d+\. .+)*)/g,(_,p,b)=>{
    const li=b.split('\n').map(l=>'<li>'+l.replace(/^\d+\. /,'')+'</li>').join('');
    return p+'<ol>'+li+'</ol>';
  });
  h=h.replace(/\n{2,}/g,'</p><p>');
  h='<p>'+h+'</p>';
  h=h.replace(/<p>\s*<(pre|ul|ol)/g,'<$1');
  h=h.replace(/<\/(pre|ul|ol)>\s*<\/p>/g,'</$1>');
  h=h.replace(/<p><\/p>/g,'');
  return h;
}

function scrollBottom(){scrollEl.scrollTop=scrollEl.scrollHeight}
function active(){return S.sessions.find(s=>s.id===S.activeId)||null}

/* ── API ── */
async function api(path,opts,signal){
  const res=await fetch(path,{...opts,signal});
  const d=await res.json().catch(()=>({}));
  if(!res.ok)throw new Error(d.error||res.statusText||'Request failed');
  return d;
}

/* ── Render sessions ── */
function renderSessions(){
  sessionListEl.innerHTML='';
  S.sessions.forEach(s=>{
    const div=document.createElement('div');
    div.className='session-item'+(s.id===S.activeId?' active':'');
    const info=document.createElement('div');
    info.className='session-info';
    info.innerHTML='<div class="session-model">'+esc(s.chatModel)+'</div><div class="session-memory">'+esc(s.memoryName||'No memory')+'</div>';
    info.addEventListener('click',()=>switchSession(s.id));
    const del=document.createElement('button');
    del.className='session-close';del.textContent='\u00d7';
    del.addEventListener('click',e=>{e.stopPropagation();deleteSession(s.id)});
    div.appendChild(info);div.appendChild(del);
    sessionListEl.appendChild(div);
  });
}

/* ── Render messages ── */
function renderMessages(){
  msgsEl.innerHTML='';
  const s=active();
  if(!s){welcomeEl.style.display='';return}
  welcomeEl.style.display='none';
  s.messages.forEach(m=>{
    if(m.role==='user')addUser(m.content);
    else if(m.role==='assistant')addAssistant(m);
    else addNote(m.content);
  });
  scrollBottom();
}

function addUser(text){
  const r=document.createElement('div');r.className='msg user';
  r.innerHTML='<div class="bubble">'+esc(text)+'</div>';
  msgsEl.appendChild(r);scrollBottom();
}

function addAssistant(m){
  const r=document.createElement('div');r.className='msg assistant';
  const b=document.createElement('div');b.className='bubble';
  let html='';
  if(m.thinking){
    html+='<details class="think-toggle"><summary>Thinking</summary><div class="think-body">'+esc(m.thinking)+'</div></details>';
  }
  html+='<div class="md">'+md(m.answer||m.content||'')+'</div>';
  b.innerHTML=html;r.appendChild(b);msgsEl.appendChild(r);scrollBottom();
}

function addNote(text){
  const r=document.createElement('div');r.className='msg note';
  r.innerHTML='<div class="bubble">'+esc(text)+'</div>';
  msgsEl.appendChild(r);scrollBottom();
}

function showTyping(){
  const r=document.createElement('div');r.className='msg assistant';r.id='typing';
  r.innerHTML='<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  msgsEl.appendChild(r);scrollBottom();
}
function hideTyping(){const e=$('typing');if(e)e.remove()}

/* ── Sync UI ── */
function syncUI(){
  const s=active();
  headerEl.textContent=s?s.chatModel+(s.memoryName?' \u00b7 '+s.memoryName:''):'No active session';
  sendBtn.style.display=S.sending?'none':'';
  stopBtn.style.display=S.sending?'':'none';
  sendBtn.disabled=S.sending||!inputEl.value.trim()||!S.activeId;
}

/* ── Session ops ── */
function switchSession(id){S.activeId=id;renderSessions();renderMessages();syncUI()}

async function deleteSession(id){
  fetch('/v1/playground/session/'+id,{method:'DELETE'}).catch(()=>{});
  S.sessions=S.sessions.filter(s=>s.id!==id);
  if(S.activeId===id)S.activeId=S.sessions.length?S.sessions[S.sessions.length-1].id:null;
  renderSessions();renderMessages();syncUI();
}

/* ── Settings ── */
function openSettings(){hideCreateError();overlay.classList.add('open')}
function closeSettings(){overlay.classList.remove('open')}

function setOpts(sel,vals,pref){
  sel.innerHTML='';
  if(!vals||!vals.length){sel.innerHTML='<option value="">No models</option>';return}
  vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;if(pref===v)o.selected=true;sel.appendChild(o)});
}

function renderMemoryPreview(){
  const v=memoryEl.value;
  if(v==='__all__'){
    const t=S.scenarios.reduce((n,sc)=>n+(sc.seed_count||0),0);
    memoryCountEl.textContent=t+' seed memories from all sets';
  }else if(!v){
    memoryCountEl.textContent='No memories will be loaded';
  }else{
    const sc=S.scMap[v];
    memoryCountEl.textContent=sc?(sc.seed_count||0)+' seed memories':'';
  }
}

const discoverBtn=$('discover-btn'),createBtn=$('create-btn'),createErrorEl=$('create-error');

function showCreateError(msg){createErrorEl.textContent=msg;createErrorEl.classList.add('show')}
function hideCreateError(){createErrorEl.classList.remove('show')}

async function discover(){
  discoverBtn.disabled=true;discoverBtn.textContent='Discovering...';
  discoverStatusEl.textContent='';
  try{
    const d=await api('/v1/playground/discover',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({adapter:adapterEl.value,chat_url:chatUrlEl.value,embedding_url:embedUrlEl.value})});
    S.discovery=d;adapterEl.value=d.adapter;
    if(!embedUrlEl.value.trim())embedUrlEl.value=d.embedding_url;
    const saved=loadCfg();
    const pc=saved.chat_model||d.chat_models.find(m=>!/embed/i.test(m))||d.chat_models[0]||'';
    const pe=saved.embed_model||d.embedding_models.find(m=>/embed/i.test(m))||d.embedding_models[0]||'';
    setOpts(chatModelEl,d.chat_models,pc);setOpts(embedModelEl,d.embedding_models,pe);
    discoverStatusEl.textContent=d.chat_models.length+' model(s) found';saveCfg();
  }catch(err){discoverStatusEl.textContent='Failed: '+err.message}
  finally{discoverBtn.disabled=false;discoverBtn.textContent='Discover Models'}
}

async function createSession(){
  hideCreateError();

  /* Validate inputs */
  if(!chatUrlEl.value.trim()){showCreateError('Chat URL is required. Fill it in and click Discover Models first.');return}
  if(!S.discovery){
    /* Auto-discover if URL is set but discovery hasn't run */
    await discover();
    if(!S.discovery){showCreateError('Could not connect to backend. Check the URL and try Discover Models.');return}
  }
  if(!chatModelEl.value){showCreateError('No chat model selected. Run Discover Models first.');return}

  const mid=memoryEl.value;
  let mname='No memory';
  if(mid==='__all__')mname='All Memories';
  else if(mid&&S.scMap[mid])mname=S.scMap[mid].name;

  createBtn.disabled=true;createBtn.textContent='Creating...';
  try{
    const d=await api('/v1/playground/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({adapter:adapterEl.value,chat_url:chatUrlEl.value,embedding_url:embedUrlEl.value,chat_model:chatModelEl.value,embedding_model:embedModelEl.value,scenario_id:mid})});
    const sess={id:d.session_id,adapter:d.adapter,chatModel:d.chat_model,embedModel:d.embedding_model,memoryName:mname,messages:[]};
    if(d.scenario&&d.scenario.name){
      sess.messages.push({role:'note',content:d.scenario.name+' loaded ('+((d.scenario.seed_count)||0)+' memories)'});
    }
    S.lastCfg={adapter:adapterEl.value,chat_url:chatUrlEl.value,embed_url:embedUrlEl.value,chat_model:chatModelEl.value,embed_model:embedModelEl.value,scenario_id:mid,memoryName:mname};
    S.sessions.push(sess);S.activeId=sess.id;
    closeSettings();renderSessions();renderMessages();syncUI();saveCfg();inputEl.focus();
  }catch(err){showCreateError(err.message)}
  finally{createBtn.disabled=false;createBtn.textContent='Create New Session'}
}

/* Quick-create a new session reusing the last config (called by + button) */
async function quickSession(){
  if(!S.lastCfg){openSettings();return}
  const c=S.lastCfg;const newBtn=$('new-btn');
  newBtn.disabled=true;newBtn.textContent='\u00b7';
  try{
    const d=await api('/v1/playground/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({adapter:c.adapter,chat_url:c.chat_url,embedding_url:c.embed_url,chat_model:c.chat_model,embedding_model:c.embed_model,scenario_id:c.scenario_id})});
    const sess={id:d.session_id,adapter:d.adapter,chatModel:d.chat_model,embedModel:d.embed_model||c.embed_model||'',memoryName:c.memoryName,messages:[]};
    if(d.scenario&&d.scenario.name){
      sess.messages.push({role:'note',content:d.scenario.name+' loaded ('+((d.scenario.seed_count)||0)+' memories)'});
    }
    S.sessions.push(sess);S.activeId=sess.id;
    renderSessions();renderMessages();syncUI();inputEl.focus();
  }catch(err){
    /* Fall back to settings on error */
    openSettings();showCreateError(err.message);
  }finally{newBtn.disabled=false;newBtn.textContent='+'}
}

/* ── Chat ── */
async function send(){
  const q=inputEl.value.trim();const s=active();
  if(!q||S.sending||!s)return;
  S.sending=true;S.abort=new AbortController();syncUI();
  s.messages.push({role:'user',content:q});addUser(q);
  inputEl.value='';autoResize();showTyping();
  try{
    const d=await api('/v1/playground/session/'+s.id+'/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})},S.abort.signal);
    hideTyping();
    const m={role:'assistant',content:d.result,thinking:d.thinking,answer:d.answer};
    s.messages.push(m);addAssistant(m);
  }catch(err){
    hideTyping();
    const note=err.name==='AbortError'?'Interrupted':'Error: '+err.message;
    s.messages.push({role:'note',content:note});addNote(note);
  }finally{S.sending=false;S.abort=null;syncUI();inputEl.focus()}
}

function stopSending(){if(S.abort)S.abort.abort()}

function autoResize(){inputEl.style.height='auto';inputEl.style.height=Math.min(inputEl.scrollHeight,200)+'px'}

/* ── Persistence ── */
function saveCfg(){
  localStorage.setItem(KEY,JSON.stringify({adapter:adapterEl.value,chat_url:chatUrlEl.value,embed_url:embedUrlEl.value,chat_model:chatModelEl.value,embed_model:embedModelEl.value,memory:memoryEl.value}));
}
function loadCfg(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(_){return{}}}
function restoreCfg(){
  const c=loadCfg();
  if(c.adapter)adapterEl.value=c.adapter;
  if(c.chat_url)chatUrlEl.value=c.chat_url;
  if(c.embed_url)embedUrlEl.value=c.embed_url;
  if(c.memory!==undefined&&c.memory!==null)memoryEl.value=c.memory;
}

/* ── Init ── */
async function init(){
  restoreCfg();
  try{
    const d=await api('/v1/playground/scenarios');
    S.scenarios=d.scenarios;S.scMap={};
    d.scenarios.forEach(sc=>{S.scMap[sc.id]=sc});
    memoryEl.innerHTML='<option value="__all__">All</option><option value="">None</option>';
    d.scenarios.forEach(sc=>{const o=document.createElement('option');o.value=sc.id;o.textContent=sc.name;memoryEl.appendChild(o)});
    const saved=loadCfg();
    if(saved.memory!==undefined&&saved.memory!==null){
      for(let i=0;i<memoryEl.options.length;i++){if(memoryEl.options[i].value===saved.memory){memoryEl.value=saved.memory;break}}
    }
    renderMemoryPreview();
  }catch(err){console.error('scenarios load failed',err)}
  syncUI();
}

/* ── Events ── */
$('new-btn').addEventListener('click',quickSession);
$('settings-btn').addEventListener('click',openSettings);
$('close-settings').addEventListener('click',closeSettings);
discoverBtn.addEventListener('click',discover);
createBtn.addEventListener('click',createSession);
sendBtn.addEventListener('click',send);
stopBtn.addEventListener('click',stopSending);
inputEl.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
inputEl.addEventListener('input',()=>{autoResize();syncUI()});
memoryEl.addEventListener('change',()=>{renderMemoryPreview();saveCfg()});
[adapterEl,chatUrlEl,embedUrlEl,chatModelEl,embedModelEl].forEach(el=>el.addEventListener('change',saveCfg));

init();
</script>
</body>
</html>"""


def _manager(app: FastAPI) -> SmokePlayground:
    manager = getattr(app.state, "smoke_playground", None)
    if manager is None:
        manager = SmokePlayground()
        app.state.smoke_playground = manager
    return manager


def mount(app: FastAPI) -> None:
    """Register the smoke playground routes on the app."""
    if getattr(app.state, "_localmelo_webapp_mounted", False):
        return
    app.state._localmelo_webapp_mounted = True

    _manager(app)

    @app.on_event("shutdown")
    async def _close_smoke_playground() -> None:
        await _manager(app).close_all()

    @app.get("/", response_class=HTMLResponse)
    async def chat_ui() -> str:
        return _HTML

    @app.get("/v1/playground/scenarios")
    async def playground_scenarios() -> JSONResponse:
        return JSONResponse(content={"scenarios": _manager(app).scenarios_payload()})

    @app.post("/v1/playground/discover")
    async def playground_discover(body: dict[str, Any]) -> JSONResponse:
        try:
            payload = await _manager(app).discover(
                chat_url=str(body.get("chat_url", "")),
                embedding_url=str(body.get("embedding_url", "")),
                adapter=str(body.get("adapter", "auto")),
            )
            return JSONResponse(content=payload)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    @app.post("/v1/playground/session")
    async def playground_session(body: dict[str, Any]) -> JSONResponse:
        try:
            payload = await _manager(app).create_session(
                chat_url=str(body.get("chat_url", "")),
                chat_model=str(body.get("chat_model", "")),
                embedding_url=str(body.get("embedding_url", "")),
                embedding_model=str(body.get("embedding_model", "")),
                adapter=str(body.get("adapter", "auto")),
                scenario_id=str(body.get("scenario_id", "")),
            )
            return JSONResponse(content=payload)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    @app.post("/v1/playground/session/{session_id}/run")
    async def playground_run(session_id: str, body: dict[str, Any]) -> JSONResponse:
        try:
            payload = await _manager(app).run(session_id, str(body.get("query", "")))
            return JSONResponse(content=payload)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except TimeoutError:
            return JSONResponse(
                status_code=504, content={"error": "Agent timed out (600s limit)"}
            )
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.delete("/v1/playground/session/{session_id}")
    async def playground_close(session_id: str) -> JSONResponse:
        closed = await _manager(app).close(session_id)
        if not closed:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return JSONResponse(content={"closed": session_id})
