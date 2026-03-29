#!/usr/bin/env python3
"""Web-based log viewer for openpilot.

Serves a browser UI that shows historical + live logs with scrollback,
level filtering, and text search.

Usage:
  python tools/log_viewer.py                    # default: listen on 0.0.0.0:8086
  python tools/log_viewer.py --port 9000        # custom port
  python tools/log_viewer.py --addr 127.0.0.1   # listen on loopback only
  python tools/log_viewer.py --history 500       # load last 500 historical lines (default 200)
"""
import argparse
import asyncio
import json
import os
import glob
import time
from collections import deque
from pathlib import Path

from aiohttp import web
from aiohttp.web import middleware

import cereal.messaging as messaging
from openpilot.system.hardware.hw import Paths

LEVELS = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
MAX_BUFFER = 5000


def _parse_swaglog_line(raw: str) -> dict | None:
  """Parse a single JSON line from a swaglog file into a normalized log entry."""
  try:
    obj = json.loads(raw)
  except (json.JSONDecodeError, ValueError):
    return None

  msg = obj.get("msg$s") or obj.get("msg") or ""
  if isinstance(msg, dict):
    msg = json.dumps(msg)

  levelnum = obj.get("levelnum") or obj.get("levelnum$i") or 20
  if isinstance(levelnum, str):
    levelnum = int(levelnum)

  return {
    "t": obj.get("created") or obj.get("created$f") or 0,
    "level": LEVELS.get(levelnum, "INFO"),
    "levelnum": levelnum,
    "msg": str(msg),
    "filename": obj.get("filename") or obj.get("filename$s", ""),
    "lineno": obj.get("lineno") or obj.get("lineno$i", ""),
    "func": obj.get("funcName") or obj.get("funcName$s", ""),
    "process": obj.get("ctx", {}).get("daemon") or obj.get("ctx", {}).get("daemon$s", ""),
  }


def _parse_logmessage(raw_json: str) -> dict | None:
  """Parse a cereal logMessage (same JSON as swaglog, but without $-suffixed keys)."""
  try:
    obj = json.loads(raw_json)
  except (json.JSONDecodeError, ValueError):
    return None

  msg = obj.get("msg", "")
  if isinstance(msg, dict):
    msg = json.dumps(msg)

  levelnum = obj.get("levelnum", 20)

  return {
    "t": obj.get("created", time.time()),
    "level": LEVELS.get(levelnum, "INFO"),
    "levelnum": levelnum,
    "msg": str(msg),
    "filename": obj.get("filename", ""),
    "lineno": obj.get("lineno", ""),
    "func": obj.get("funcName", ""),
    "process": obj.get("ctx", {}).get("daemon", ""),
  }


def load_history(max_lines: int) -> list[dict]:
  """Read the most recent swaglog files and return up to max_lines parsed entries."""
  root = Paths.swaglog_root()
  if not os.path.isdir(root):
    return []

  files = sorted(glob.glob(os.path.join(root, "swaglog.*")), reverse=True)
  entries: list[dict] = []

  for fpath in files:
    if len(entries) >= max_lines:
      break
    try:
      with open(fpath) as f:
        lines = f.readlines()
    except OSError:
      continue
    for line in reversed(lines):
      line = line.strip()
      if not line:
        continue
      entry = _parse_swaglog_line(line)
      if entry:
        entries.append(entry)
        if len(entries) >= max_lines:
          break

  entries.reverse()
  return entries


class LogBroadcaster:
  """Subscribes to cereal logMessage and fans out to SSE clients."""

  def __init__(self, history_lines: int):
    self.clients: list[asyncio.Queue] = []
    self.buffer: deque[dict] = deque(maxlen=MAX_BUFFER)
    self._task: asyncio.Task | None = None
    self._history_lines = history_lines

  def start(self):
    for entry in load_history(self._history_lines):
      self.buffer.append(entry)
    self._task = asyncio.create_task(self._poll_loop())

  async def stop(self):
    if self._task:
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass

  def subscribe(self) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=512)
    self.clients.append(q)
    return q

  def unsubscribe(self, q: asyncio.Queue):
    try:
      self.clients.remove(q)
    except ValueError:
      pass

  async def _poll_loop(self):
    sm = messaging.SubMaster(['logMessage'])
    while True:
      sm.update(100)
      if sm.updated['logMessage']:
        entry = _parse_logmessage(sm['logMessage'])
        if entry:
          self.buffer.append(entry)
          data = json.dumps(entry)
          dead = []
          for q in self.clients:
            try:
              q.put_nowait(data)
            except asyncio.QueueFull:
              dead.append(q)
          for q in dead:
            self.clients.remove(q)
      await asyncio.sleep(0)


async def handle_history(request: web.Request) -> web.Response:
  broadcaster: LogBroadcaster = request.app["broadcaster"]
  entries = list(broadcaster.buffer)
  return web.json_response(entries)


async def handle_stream(request: web.Request) -> web.StreamResponse:
  broadcaster: LogBroadcaster = request.app["broadcaster"]
  q = broadcaster.subscribe()

  resp = web.StreamResponse(
    status=200,
    reason="OK",
    headers={
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "Access-Control-Allow-Origin": "*",
    },
  )
  await resp.prepare(request)

  try:
    while True:
      data = await q.get()
      await resp.write(f"data: {data}\n\n".encode())
  except (asyncio.CancelledError, ConnectionResetError):
    pass
  finally:
    broadcaster.unsubscribe(q)
  return resp


async def handle_index(request: web.Request) -> web.Response:
  return web.Response(content_type="text/html", text=INDEX_HTML)


@middleware
async def cors_mw(request, handler):
  resp = await handler(request)
  resp.headers["Access-Control-Allow-Origin"] = "*"
  return resp


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>openpilot logs</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --border: #30363d;
    --fg: #c9d1d9; --fg-dim: #8b949e;
    --debug: #8b949e; --info: #58a6ff; --warning: #d29922; --error: #f85149; --critical: #ff7b72;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; background: var(--bg); color: var(--fg); font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 13px; }
  body { display: flex; flex-direction: column; }

  .toolbar {
    display: flex; gap: 8px; align-items: center; padding: 8px 12px;
    background: var(--bg2); border-bottom: 1px solid var(--border); flex-shrink: 0; flex-wrap: wrap;
  }
  .toolbar label { color: var(--fg-dim); font-size: 12px; cursor: pointer; user-select: none; }
  .toolbar label input { margin-right: 3px; }
  .toolbar input[type=text] {
    background: var(--bg); border: 1px solid var(--border); color: var(--fg);
    padding: 4px 8px; border-radius: 4px; font-size: 12px; width: 200px; font-family: inherit;
  }
  .toolbar input[type=text]:focus { outline: none; border-color: var(--info); }
  .toolbar .spacer { flex: 1; }
  .toolbar button {
    background: var(--bg); border: 1px solid var(--border); color: var(--fg); padding: 4px 10px;
    border-radius: 4px; cursor: pointer; font-size: 12px; font-family: inherit;
  }
  .toolbar button:hover { border-color: var(--info); }
  .toolbar button.active { background: var(--info); color: #000; border-color: var(--info); }
  .toolbar .count { color: var(--fg-dim); font-size: 11px; }

  #log-container {
    flex: 1; overflow-y: auto; padding: 0; scroll-behavior: auto;
  }
  .log-line {
    padding: 1px 12px; white-space: pre-wrap; word-break: break-all;
    border-bottom: 1px solid transparent; line-height: 1.5; display: flex; gap: 8px;
  }
  .log-line:hover { background: rgba(255,255,255,0.03); }
  .log-line .ts { color: var(--fg-dim); flex-shrink: 0; min-width: 80px; }
  .log-line .lv { flex-shrink: 0; min-width: 55px; font-weight: 600; }
  .log-line .src { color: var(--fg-dim); flex-shrink: 0; min-width: 100px; max-width: 250px; overflow: hidden; text-overflow: ellipsis; }
  .log-line .msg { flex: 1; }

  .log-line.DEBUG .lv { color: var(--debug); }
  .log-line.INFO .lv { color: var(--info); }
  .log-line.WARNING .lv { color: var(--warning); }
  .log-line.WARNING { background: rgba(210,153,34,0.05); }
  .log-line.ERROR .lv { color: var(--error); }
  .log-line.ERROR { background: rgba(248,81,73,0.06); }
  .log-line.CRITICAL .lv { color: var(--critical); font-weight: 800; }
  .log-line.CRITICAL { background: rgba(255,123,114,0.1); }

  .paused-banner {
    position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: var(--warning); color: #000; padding: 6px 16px; border-radius: 6px;
    font-size: 12px; font-weight: 600; cursor: pointer; z-index: 10; display: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }

  mark { background: rgba(88,166,255,0.3); color: inherit; border-radius: 2px; }
</style>
</head>
<body>

<div class="toolbar">
  <label><input type="checkbox" data-level="DEBUG"> DEBUG</label>
  <label><input type="checkbox" data-level="INFO" checked> INFO</label>
  <label><input type="checkbox" data-level="WARNING" checked> WARNING</label>
  <label><input type="checkbox" data-level="ERROR" checked> ERROR</label>
  <label><input type="checkbox" data-level="CRITICAL" checked> CRITICAL</label>
  <input type="text" id="search" placeholder="Filter text...">
  <input type="text" id="proc-filter" placeholder="Process..." style="width:120px">
  <span class="spacer"></span>
  <span class="count" id="count"></span>
  <button id="pause-btn">Pause</button>
  <button id="clear-btn">Clear</button>
</div>

<div id="log-container"></div>
<div class="paused-banner" id="paused-banner">Scroll paused — click or press End to resume</div>

<script>
(function() {
  const container = document.getElementById('log-container');
  const searchInput = document.getElementById('search');
  const procInput = document.getElementById('proc-filter');
  const pauseBtn = document.getElementById('pause-btn');
  const clearBtn = document.getElementById('clear-btn');
  const pausedBanner = document.getElementById('paused-banner');
  const countEl = document.getElementById('count');
  const levelBoxes = document.querySelectorAll('[data-level]');

  let allEntries = [];
  let autoScroll = true;
  let searchTerm = '';
  let procTerm = '';
  const MAX_DOM = 8000;
  const MAX_ENTRIES = 20000;

  function getEnabledLevels() {
    const s = new Set();
    levelBoxes.forEach(cb => { if (cb.checked) s.add(cb.dataset.level); });
    return s;
  }

  function matchesFilter(entry) {
    const levels = getEnabledLevels();
    if (!levels.has(entry.level)) return false;
    if (searchTerm && !entry._lower.includes(searchTerm)) return false;
    if (procTerm && !(entry.process || '').toLowerCase().includes(procTerm)) return false;
    return true;
  }

  function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
      + '.' + String(d.getMilliseconds()).padStart(3, '0');
  }

  function highlightSearch(text) {
    if (!searchTerm) return escapeHtml(text);
    const escaped = escapeHtml(text);
    const re = new RegExp('(' + escapeRegex(escapeHtml(searchTerm)) + ')', 'gi');
    return escaped.replace(re, '<mark>$1</mark>');
  }

  function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function makeLine(entry) {
    const div = document.createElement('div');
    div.className = 'log-line ' + entry.level;
    const src = entry.process || entry.filename || '';
    const loc = entry.func ? entry.func + ':' + entry.lineno : (entry.filename ? entry.filename + ':' + entry.lineno : '');
    div.innerHTML =
      '<span class="ts">' + formatTime(entry.t) + '</span>' +
      '<span class="lv">' + entry.level + '</span>' +
      '<span class="src" title="' + escapeHtml(loc) + '">' + escapeHtml(src || entry.filename) + '</span>' +
      '<span class="msg">' + highlightSearch(entry.msg) + '</span>';
    return div;
  }

  function rebuildDOM() {
    const frag = document.createDocumentFragment();
    let visible = 0;
    for (let i = allEntries.length - 1; i >= 0 && visible < MAX_DOM; i--) {
      const e = allEntries[i];
      if (matchesFilter(e)) {
        frag.prepend(makeLine(e));
        visible++;
      }
    }
    container.innerHTML = '';
    container.appendChild(frag);
    countEl.textContent = visible + ' shown / ' + allEntries.length + ' total';
    if (autoScroll) scrollToBottom();
  }

  function appendEntry(entry) {
    entry._lower = (entry.msg + ' ' + (entry.process || '') + ' ' + (entry.filename || '')).toLowerCase();
    allEntries.push(entry);
    if (allEntries.length > MAX_ENTRIES) allEntries.splice(0, allEntries.length - MAX_ENTRIES);

    if (matchesFilter(entry)) {
      container.appendChild(makeLine(entry));
      while (container.children.length > MAX_DOM) container.removeChild(container.firstChild);
      countEl.textContent = container.children.length + ' shown / ' + allEntries.length + ' total';
      if (autoScroll) scrollToBottom();
    }
  }

  function scrollToBottom() {
    container.scrollTop = container.scrollHeight;
  }

  function setAutoScroll(v) {
    autoScroll = v;
    pauseBtn.classList.toggle('active', !v);
    pauseBtn.textContent = v ? 'Pause' : 'Resume';
    pausedBanner.style.display = v ? 'none' : 'block';
    if (v) scrollToBottom();
  }

  container.addEventListener('scroll', () => {
    const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 40;
    if (!atBottom && autoScroll) setAutoScroll(false);
    if (atBottom && !autoScroll) setAutoScroll(true);
  });

  pauseBtn.addEventListener('click', () => setAutoScroll(!autoScroll));
  pausedBanner.addEventListener('click', () => setAutoScroll(true));
  clearBtn.addEventListener('click', () => { allEntries = []; container.innerHTML = ''; countEl.textContent = '0'; });

  let filterTimeout;
  function debouncedRebuild() {
    clearTimeout(filterTimeout);
    filterTimeout = setTimeout(rebuildDOM, 150);
  }
  searchInput.addEventListener('input', () => { searchTerm = searchInput.value.toLowerCase(); debouncedRebuild(); });
  procInput.addEventListener('input', () => { procTerm = procInput.value.toLowerCase(); debouncedRebuild(); });
  levelBoxes.forEach(cb => cb.addEventListener('change', rebuildDOM));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'End') setAutoScroll(true);
    if (e.key === '/' && document.activeElement !== searchInput && document.activeElement !== procInput) {
      e.preventDefault(); searchInput.focus();
    }
  });

  async function init() {
    try {
      const resp = await fetch('/api/history');
      const history = await resp.json();
      for (const entry of history) appendEntry(entry);
    } catch(e) { console.warn('Failed to load history:', e); }

    const evtSource = new EventSource('/api/stream');
    evtSource.onmessage = (event) => {
      try {
        appendEntry(JSON.parse(event.data));
      } catch(e) {}
    };
    evtSource.onerror = () => {
      setTimeout(() => { evtSource.close(); init(); }, 3000);
    };
  }

  init();
})();
</script>
</body>
</html>
"""


async def on_startup(app: web.Application):
  app["broadcaster"].start()


async def on_cleanup(app: web.Application):
  await app["broadcaster"].stop()


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--addr", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
  parser.add_argument("--port", type=int, default=8086, help="Port (default 8086)")
  parser.add_argument("--history", type=int, default=200, help="Number of historical log lines to load (default 200)")
  args = parser.parse_args()

  broadcaster = LogBroadcaster(history_lines=args.history)

  app = web.Application(middlewares=[cors_mw])
  app["broadcaster"] = broadcaster
  app.on_startup.append(on_startup)
  app.on_cleanup.append(on_cleanup)

  app.router.add_get("/", handle_index)
  app.router.add_get("/api/history", handle_history)
  app.router.add_get("/api/stream", handle_stream)

  print(f"Log viewer: http://{args.addr}:{args.port}")
  web.run_app(app, host=args.addr, port=args.port, print=None)


if __name__ == "__main__":
  main()
