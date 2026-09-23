/* ═══════════════════════════════════════════════════════════════════
   app.js — Nikhil AI Chatbot Frontend Logic
   Features: chat, typing indicator, metrics modal, sidebar stats,
             copy button, regenerate, auto-resize textarea, timestamps
   ═══════════════════════════════════════════════════════════════════ */

const API = '';   // empty = same origin

// ─── DOM Refs ──────────────────────────────────────────────────────────────
const $input      = document.getElementById('userInput');
const $send       = document.getElementById('btnSend');
const $regen      = document.getElementById('btnRegen');
const $clear      = document.getElementById('btnClear');
const $metrics    = document.getElementById('btnMetrics');
const $messages   = document.getElementById('messagesContainer');
const $typing     = document.getElementById('typingWrapper');
const $welcome    = document.getElementById('welcomeScreen');
const $charCount  = document.getElementById('charCount');
const $status     = document.getElementById('statusBadge');
const $statusTxt  = $status.querySelector('.status-text');
const $modal      = document.getElementById('metricsModal');
const $modalClose = document.getElementById('btnModalClose');
const $modalBody  = document.getElementById('metricsBody');
const $statTotal  = document.getElementById('statTotal');
const $statConf   = document.getElementById('statConfident');
const $statFall   = document.getElementById('statFallbacks');

// ─── State ─────────────────────────────────────────────────────────────────
let lastBotResponse = null;
let sessionStats    = { total: 0, confident: 0, fallbacks: 0 };
let isWaiting       = false;
let modelInfo       = null;

// ─── Init ──────────────────────────────────────────────────────────────────
(async function init() {
  await checkHealth();
  await loadSidebarData();
  $input.focus();
})();

// ─── Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    if (r.ok) {
      const d = await r.json();
      if (d.model_loaded) {
        setStatus('online', '✅ Model Ready');
      } else {
        setStatus('error', '❌ Model not loaded');
      }
    } else {
      setStatus('error', '⚠ Server error');
    }
  } catch {
    setStatus('error', '🔴 Offline');
  }
}

function setStatus(cls, text) {
  $status.className = `status-badge ${cls}`;
  $statusTxt.textContent = text;
}

// ─── Sidebar data ──────────────────────────────────────────────────────────
async function loadSidebarData() {
  try {
    const r = await fetch(`${API}/api/metrics`);
    if (!r.ok) { showSidebarPlaceholder(); return; }
    const d = await r.json();
    modelInfo = d;
    renderModelCard(d);
    renderDatasetCard(d);
    renderPerfCard(d);
  } catch {
    showSidebarPlaceholder();
  }
}

function renderModelCard(d) {
  const mi = d.best_model || 'Unknown';
  document.getElementById('modelInfoBody').innerHTML = `
    <div class="info-row"><span class="info-label">Algorithm</span><span class="info-val highlight">${shortModelName(mi)}</span></div>
    <div class="info-row"><span class="info-label">Framework</span><span class="info-val">scikit-learn</span></div>
    <div class="info-row"><span class="info-label">Classes</span><span class="info-val">${d.dataset_info?.num_intents || 151}</span></div>
    <div class="info-row"><span class="info-label">Seed</span><span class="info-val">42</span></div>
  `;
}

function renderDatasetCard(d) {
  const di = d.dataset_info || {};
  document.getElementById('datasetBody').innerHTML = `
    <div class="info-row"><span class="info-label">Dataset</span><span class="info-val highlight">CLINC150</span></div>
    <div class="info-row"><span class="info-label">Train</span><span class="info-val">${(di.train_size||15100).toLocaleString()}</span></div>
    <div class="info-row"><span class="info-label">Val</span><span class="info-val">${(di.val_size||3100).toLocaleString()}</span></div>
    <div class="info-row"><span class="info-label">Test</span><span class="info-val">${(di.test_size||5500).toLocaleString()}</span></div>
  `;
}

function renderPerfCard(d) {
  const tm = d.test_metrics || {};
  const acc = tm.accuracy || 0;
  const f1  = tm.f1_macro || 0;
  const f1w = tm.f1_weighted || 0;
  document.getElementById('perfBody').innerHTML = `
    ${metricBar('Accuracy',   acc)}
    ${metricBar('F1 Macro',   f1)}
    ${metricBar('F1 Weighted',f1w)}
  `;
}

function metricBar(name, val) {
  const pct = (val * 100).toFixed(1);
  return `
    <div class="metric-item">
      <div class="metric-header">
        <span class="metric-name">${name}</span>
        <span class="metric-score">${pct}%</span>
      </div>
      <div class="metric-bar-bg"><div class="metric-bar" style="width:${pct}%"></div></div>
    </div>`;
}

function showSidebarPlaceholder() {
  ['modelInfoBody','datasetBody','perfBody'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = `<p style="font-size:0.75rem;color:var(--text-muted)">Run <code>python train.py</code> first.</p>`;
  });
}

function shortModelName(full) {
  if (full.includes('SVM'))    return 'Linear SVM';
  if (full.includes('Logistic'))return 'Logistic Reg.';
  if (full.includes('Forest')) return 'Random Forest';
  return full.split('(')[0].trim();
}

// ─── Send Message ──────────────────────────────────────────────────────────
async function sendMessage(text) {
  text = (text || $input.value).trim();
  if (!text || isWaiting) return;

  // Hide welcome screen
  $welcome.style.display = 'none';

  appendMessage('user', text);
  $input.value = '';
  autoResize();
  updateCharCount();
  setWaiting(true);

  try {
    const r = await fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      appendBotMessage('⚠️ Error: ' + (err.detail || r.statusText), null, 0, false, []);
    } else {
      const d = await r.json();
      lastBotResponse = d;
      appendBotMessage(d.response, d.intent, d.confidence, d.is_confident, d.top_k || [], d.timestamp);
      updateSessionStats(d.is_confident);
    }
  } catch (e) {
    appendBotMessage('⚠️ Could not reach the server. Is `python app.py` running?', null, 0, false, []);
  }

  setWaiting(false);
  $regen.disabled = false;
}

function appendMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `message-group ${role}`;

  const avatar = document.createElement('div');
  avatar.className = role === 'bot' ? 'bot-avatar' : 'user-avatar';
  avatar.innerHTML = role === 'bot' ? '🤖' : '👤';

  const bubbleWrap = document.createElement('div');
  bubbleWrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = `bubble bubble-${role}`;
  bubble.innerHTML = formatText(text);

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.innerHTML = `<span class="msg-time">${timeStr()}</span>`;

  if (role === 'bot') {
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.textContent = '📋 Copy';
    copyBtn.onclick = () => copyText(text, copyBtn);
    meta.appendChild(copyBtn);
  }

  bubbleWrap.appendChild(bubble);
  bubbleWrap.appendChild(meta);
  wrap.appendChild(avatar);
  wrap.appendChild(bubbleWrap);
  $messages.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function appendBotMessage(text, intent, confidence, isConfident, topK, timestamp) {
  const wrap = appendMessage('bot', text);
  const meta = wrap.querySelector('.msg-meta');

  if (intent && intent !== 'greeting_thanks') {
    const pill = document.createElement('span');
    pill.className = 'intent-pill';
    pill.textContent = intent;
    meta.insertBefore(pill, meta.firstChild);

    const confPill = document.createElement('span');
    confPill.className = `conf-pill ${isConfident ? 'high' : 'low'}`;
    confPill.textContent = (confidence * 100).toFixed(0) + '%';
    meta.insertBefore(confPill, meta.querySelector('.copy-btn') || null);
  }
}

// ─── Typing Indicator ──────────────────────────────────────────────────────
function setWaiting(v) {
  isWaiting = v;
  $typing.style.display = v ? 'flex' : 'none';
  $send.disabled = v;
  if (v) scrollBottom();
}

// ─── Regenerate ────────────────────────────────────────────────────────────
$regen.addEventListener('click', async () => {
  if (!lastBotResponse || isWaiting) return;
  // Remove last bot message
  const groups = $messages.querySelectorAll('.message-group.bot');
  if (groups.length) groups[groups.length - 1].remove();
  // Re-request with same last user message
  const userGroups = $messages.querySelectorAll('.message-group.user');
  if (!userGroups.length) return;
  const lastUserBubble = userGroups[userGroups.length - 1].querySelector('.bubble');
  const lastUserText   = lastUserBubble?.textContent?.trim();
  if (lastUserText) {
    setWaiting(true);
    try {
      const r = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: lastUserText }),
      });
      if (r.ok) {
        const d = await r.json();
        lastBotResponse = d;
        appendBotMessage(d.response, d.intent, d.confidence, d.is_confident, d.top_k || [], d.timestamp);
      }
    } catch {}
    setWaiting(false);
  }
});

// ─── Clear Chat ────────────────────────────────────────────────────────────
$clear.addEventListener('click', async () => {
  if (!confirm('Clear all messages?')) return;
  $messages.innerHTML = '';
  $welcome.style.display = 'flex';
  $regen.disabled = true;
  lastBotResponse = null;
  sessionStats = { total: 0, confident: 0, fallbacks: 0 };
  updateStatDisplay();
  await fetch(`${API}/api/history`, { method: 'DELETE' }).catch(() => {});
});

// ─── Metrics Modal ─────────────────────────────────────────────────────────
$metrics.addEventListener('click', openMetrics);
$modalClose.addEventListener('click', () => $modal.classList.remove('open'));
$modal.addEventListener('click', e => { if (e.target === $modal) $modal.classList.remove('open'); });

async function openMetrics() {
  $modal.classList.add('open');
  $modalBody.innerHTML = '<div class="metrics-loading">Loading metrics…</div>';
  try {
    const r = await fetch(`${API}/api/metrics`);
    if (!r.ok) {
      $modalBody.innerHTML = '<p>Run <code>python train.py</code> to generate metrics.</p>';
      return;
    }
    const d = await r.json();
    renderMetricsModal(d);
  } catch {
    $modalBody.innerHTML = '<p style="color:var(--rose-500)">Could not load metrics.</p>';
  }
}

function renderMetricsModal(d) {
  const tm = d.test_metrics || {};
  const ds = d.dataset_info || {};
  const cmp= d.comparison || {};
  const ea = d.error_analysis || {};

  let html = `
    <div>
      <p class="metrics-section-title">📊 Test Set Performance</p>
      <div class="metrics-grid">
        ${mCard('Accuracy',    ((tm.accuracy||0)*100).toFixed(2)+'%')}
        ${mCard('Precision',   ((tm.precision_macro||0)*100).toFixed(2)+'%')}
        ${mCard('Recall',      ((tm.recall_macro||0)*100).toFixed(2)+'%')}
        ${mCard('F1 Macro',    ((tm.f1_macro||0)*100).toFixed(2)+'%')}
        ${mCard('F1 Weighted', ((tm.f1_weighted||0)*100).toFixed(2)+'%')}
        ${mCard('Samples',     (ds.test_size||0).toLocaleString())}
      </div>
    </div>

    <div>
      <p class="metrics-section-title">🏋 Dataset Info</p>
      <div class="metrics-grid">
        ${mCard('Train',   (ds.train_size||0).toLocaleString())}
        ${mCard('Val',     (ds.val_size||0).toLocaleString())}
        ${mCard('Test',    (ds.test_size||0).toLocaleString())}
        ${mCard('Intents', ds.num_intents||151)}
      </div>
    </div>

    <div>
      <p class="metrics-section-title">⚖ Model Comparison (Validation Set)</p>
      <table class="comparison-table">
        <thead>
          <tr>
            <th>Model</th><th>Accuracy</th><th>F1 Macro</th><th>F1 Weighted</th><th>Time</th>
          </tr>
        </thead>
        <tbody>
          ${Object.entries(cmp).map(([name, v]) => {
            const isBest = name === d.best_model;
            return `<tr class="${isBest ? 'best' : ''}">
              <td>${name}${isBest ? '<span class="best-badge">BEST</span>' : ''}</td>
              <td>${((v.accuracy||0)*100).toFixed(2)}%</td>
              <td>${((v.f1_macro||0)*100).toFixed(2)}%</td>
              <td>${((v.f1_weighted||0)*100).toFixed(2)}%</td>
              <td>${v.train_time_sec||'–'}s</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>

    <div>
      <p class="metrics-section-title">🔍 Error Analysis</p>
      <div class="metrics-grid">
        ${mCard('Correct',   (ea.correct||0).toLocaleString())}
        ${mCard('Errors',    (ea.incorrect||0).toLocaleString())}
        ${mCard('Error Rate',((ea.error_rate||0)*100).toFixed(2)+'%')}
      </div>
    </div>
  `;
  $modalBody.innerHTML = html;
}

function mCard(label, val) {
  return `<div class="metric-card">
    <div class="metric-card-val">${val}</div>
    <div class="metric-card-label">${label}</div>
  </div>`;
}

// ─── Session Stats ─────────────────────────────────────────────────────────
function updateSessionStats(isConfident) {
  sessionStats.total++;
  if (isConfident) sessionStats.confident++;
  else             sessionStats.fallbacks++;
  updateStatDisplay();
}
function updateStatDisplay() {
  $statTotal.textContent = sessionStats.total;
  $statConf.textContent  = sessionStats.confident;
  $statFall.textContent  = sessionStats.fallbacks;
}

// ─── Suggestion Chips ──────────────────────────────────────────────────────
function sendSuggestion(btn) {
  sendMessage(btn.textContent);
}
window.sendSuggestion = sendSuggestion;

// ─── Input Handling ────────────────────────────────────────────────────────
$input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
$input.addEventListener('input', () => { autoResize(); updateCharCount(); });
$send.addEventListener('click', () => sendMessage());

function autoResize() {
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 150) + 'px';
}

function updateCharCount() {
  const len = $input.value.length;
  $charCount.textContent = `${len} / 1000`;
  $charCount.className = 'char-count' + (len > 900 ? ' danger' : len > 750 ? ' warn' : '');
}

// ─── Helpers ───────────────────────────────────────────────────────────────
function scrollBottom() {
  requestAnimationFrame(() => {
    $messages.scrollTop = $messages.scrollHeight;
    $typing.scrollIntoView?.({ behavior: 'smooth' });
  });
}

function timeStr() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatText(text) {
  // Convert simple markdown-like to HTML
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code style="background:var(--indigo-50);padding:1px 5px;border-radius:4px;font-family:\'JetBrains Mono\',monospace;font-size:0.82em">$1</code>')
    .replace(/\n/g, '<br>');
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text.replace(/<[^>]+>/g, ''));
    const orig = btn.textContent;
    btn.textContent = '✅ Copied';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1800);
  } catch {}
}

// ─── Keyboard shortcut: Esc closes modal ───────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') $modal.classList.remove('open');
});
