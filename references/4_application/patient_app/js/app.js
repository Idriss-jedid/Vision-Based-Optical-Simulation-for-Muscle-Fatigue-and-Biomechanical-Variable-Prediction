/* VisionBiomech — app logic. Upload-only, fully dynamic (no hardcoded sample paths). */
'use strict';

const BACKEND = '';                          // same origin (served by the backend)

/* ---- PWA: register service worker ---- */
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  });
}

/* ---- Element refs ---- */
const $ = (id) => document.getElementById(id);
const els = { bstat: $('bstat') };

/* ---- State ---- */
/* ---- Backend health ---- */
fetch(BACKEND + '/health')
  .then(r => r.json())
  .then(() => { els.bstat.textContent = 'online'; els.bstat.className = 'status'; })
  .catch(() => { els.bstat.textContent = 'offline'; els.bstat.className = 'status off'; });

/* ---- Camera file ---- */
/* make a <video> actually play + loop (play on loadeddata, not before) */
function ensurePlay(v) {
  v.muted = true; v.defaultMuted = true; v.setAttribute('muted', '');
  v.loop = true; v.playsInline = true; v.setAttribute('playsinline', ''); v.autoplay = true;
  const go = () => { const p = v.play(); if (p && p.catch) p.catch(() => {}); };
  v.onloadedmetadata = go; v.onloadeddata = go; v.oncanplay = go; go();
}
/* autoplay unlock: first user interaction resumes every loaded clip */
function resumeAllVideos() {
  document.querySelectorAll('video').forEach(v => { if (v.src && v.paused) { const p = v.play(); if (p && p.catch) p.catch(() => {}); } });
}
['click', 'touchstart', 'pointerdown'].forEach(ev => addEventListener(ev, resumeAllVideos, { passive: true }));
/* keep-alive: if any loaded clip drifts to paused, resume it (both pages) */
setInterval(() => {
  if (!$('p1').classList.contains('on')) return;
  const v = $('sv');
  if (S && S.file_v && v.paused && v.readyState >= 2 && !v.srcObject) v.play().catch(() => {});
}, 1500);

/* ---- Results render ---- */
const GMETA = {
  torque: ['T', '#0a84ff', 'Joint torque'],
  activation: ['A', '#30d158', 'Muscle activation'],
  force: ['F', '#bf5af2', 'Muscle force'],
  fatigue: ['Fatigue', '#ff9f0a', 'Muscle fatigue'],
};
/* shared results renderer — muscle diagram + dashboard + 13 grouped charts (both pages) */
function renderFit(data, E) {
  const V = data.vars;
  const FL = [['BIClong', 'Biceps (long)'], ['BICshort', 'Biceps (short)'], ['BRA', 'Brachialis'], ['BRD_hand', 'Brachioradialis']];
  const peakAbs = (tg) => Math.max(...V[tg].pred.map(Math.abs));
  const meanOf = (tg) => V[tg].pred.reduce((a, b) => a + b, 0) / V[tg].pred.length;
  const endOf = (tg) => V[tg].pred[V[tg].pred.length - 1];
  const set = (id, val) => { const e = $(id); if (e) e.textContent = val; };
  set(E.torque, peakAbs('elbow_moment').toFixed(1));
  set(E.force, Math.round(Math.max(...FL.map(m => peakAbs('frc_' + m[0])))));
  set(E.act, (FL.reduce((a, m) => a + meanOf('act_' + m[0]), 0) / 4).toFixed(2));
  set(E.fat, (FL.reduce((a, m) => a + endOf('MF_' + m[0]), 0) / 4).toFixed(1));
  if (E.r2) set(E.r2, 'R² ' + Math.round((data.model_R2_validated || 0) * 1000) / 10 + '%');
  const pa = {}; FL.forEach(m => pa[m[0]] = peakAbs('act_' + m[0]));
  const fill = (id, a) => { const e = $(id); if (e) e.setAttribute('fill', muscleColor(a)); };
  fill(E.mbic, (pa.BIClong + pa.BICshort) / 2); fill(E.mbra, pa.BRA); fill(E.mbrd, pa.BRD_hand);
  const amax = Math.max(0.15, ...Object.values(pa));
  const ml = $(E.mlist);
  if (ml) ml.innerHTML = FL.map(m => { const a = pa[m[0]], f = Math.round(peakAbs('frc_' + m[0]));
    return `<div class="fit-mrow"><span class="mn">${m[1]}</span><div class="fit-mbar"><i style="width:${Math.round(a / amax * 100)}%"></i></div><span class="mv">${f} N</span></div>`; }).join('');
  const CH = $(E.charts); if (!CH) return; CH.innerHTML = '';
  if (window.Chart) { Chart.defaults.color = '#6b7791'; Chart.defaults.font.family = 'system-ui'; }
  const GC = { torque: '#3b5bdb', activation: '#1fbf75', force: '#8a5cf6', fatigue: '#f5a524' };
  const GL = { torque: 'Joint torque', activation: 'Muscle activation', force: 'Muscle force', fatigue: 'Muscle fatigue' };
  const groups = { torque: [], activation: [], force: [], fatigue: [] };
  Object.values(V).forEach(v => groups[v.group].push(v));
  ['torque', 'activation', 'force', 'fatigue'].forEach(g => {
    if (!groups[g].length) return;
    const h = document.createElement('div'); h.className = 'grp';
    h.innerHTML = `<span class="gi" style="background:${GC[g]}22;color:${GC[g]}">${GL[g].charAt(0)}</span>${GL[g]}`;
    CH.appendChild(h);
    groups[g].forEach((v, idx) => {
      const cc = document.createElement('div'); cc.className = 'cc'; const cid = E.prefix + '_' + g + '_' + idx;
      cc.innerHTML = `<div class="t"><h5>${v.label}</h5><span class="u">${v.unit}</span></div><canvas id="${cid}"></canvas>`;
      CH.appendChild(cc);
      requestAnimationFrame(() => aiChart(cid, data.time, v.pred, GC[g], v.unit));
    });
  });
}
function aiChart(id, time, vals, color, unit) {
  const cv = document.getElementById(id); if (!cv) return;
  const ctx = cv.getContext('2d');
  const g = ctx.createLinearGradient(0, 0, 0, 96);
  g.addColorStop(0, color + '66'); g.addColorStop(1, color + '05');
  const ylab = (unit && unit.trim()) ? unit : 'value';
  new Chart(cv, {
    type: 'line',
    data: { labels: time, datasets: [{ label: ylab, data: vals, borderColor: color, backgroundColor: g, fill: true, borderWidth: 2.2, tension: 0.4, pointRadius: 0 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: true, callbacks: {
          title: (it) => 't = ' + time[it[0].dataIndex] + ' s',
          label: (it) => it.parsed.y.toFixed(2) + ' ' + (unit || '') } }
      },
      scales: {
        x: { title: { display: true, text: 'time (s)', color: '#8a93ab', font: { size: 9 } },
             ticks: { maxTicksLimit: 5, callback: (v) => time[v] != null ? (+time[v]).toFixed(0) : '', font: { size: 8 }, color: '#8a93ab' },
             grid: { display: false } },
        y: { title: { display: true, text: ylab, color: '#8a93ab', font: { size: 9 } },
             ticks: { maxTicksLimit: 4, font: { size: 8 }, color: '#8a93ab' },
             grid: { color: 'rgba(120,128,160,.18)' } }
      }
    }
  });
}

/* ============================================================
   SINGLE-CAMERA · LEVEL 3  (monocular, 2D -> BiGRU)
   ============================================================ */
const S = {
  slot: $('s_slot'), v: $('sv'), k: $('sk'), ph: $('s_ph'), rm: $('s_rm'),
  file: $('sfile'), run: $('srun'), hint: $('s_hint'), proc: $('sproc'), err: $('serr'),
  results: $('sresults'), charts: $('scharts'), mlist: $('m_list'), again: $('sagain'),
  file_v: null, kp: null, fps: 50, stride: 3, running: false,
};
const FLEX = [['BIClong', 'Biceps (long)'], ['BICshort', 'Biceps (short)'], ['BRA', 'Brachialis'], ['BRD_hand', 'Brachioradialis']];

/* pick video */
S.slot.addEventListener('click', (e) => { if (e.target === S.rm) return; S.file.click(); });
S.file.addEventListener('change', (e) => { const f = e.target.files[0]; if (f) setSingle(f); S.file.value = ''; });
S.rm.addEventListener('click', (e) => { e.stopPropagation(); clearSingle(); });
function setSingle(f) {
  S.file_v = f; S.slot.classList.add('has'); S.ph.style.display = 'none'; S.rm.hidden = false;
  S.v.srcObject = null; S.v.src = URL.createObjectURL(f); S.v.load(); ensurePlay(S.v);
  S.run.disabled = false; S.hint.textContent = 'Ready · one video ✓';
}
function clearSingle() {
  S.file_v = null; S.kp = null; S.slot.classList.remove('has'); S.ph.style.display = 'flex'; S.rm.hidden = true;
  S.v.removeAttribute('src'); S.v.srcObject = null; S.v.load(); S.run.disabled = true;
  S.hint.textContent = 'Upload a video, or record one with your camera';
}

/* ---- record with the device camera ---- */
let mediaRec = null, recChunks = [], recStream = null;
$('srecord').addEventListener('click', async () => {
  const btn = $('srecord');
  if (mediaRec && mediaRec.state === 'recording') { mediaRec.stop(); return; }
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 900, height: 900 }, audio: false });
  } catch (e) { S.err.style.display = 'block'; S.err.innerHTML = '📷 Camera not available (' + e.message + '). On http it may be blocked — use localhost or upload a file.'; return; }
  S.file_v = null; S.slot.classList.add('has'); S.ph.style.display = 'none'; S.rm.hidden = false;
  S.v.srcObject = recStream; S.v.muted = true; S.v.play().catch(() => {});
  recChunks = [];
  const mime = MediaRecorder.isTypeSupported('video/mp4') ? 'video/mp4' : 'video/webm';
  mediaRec = new MediaRecorder(recStream, { mimeType: mime });
  mediaRec.ondataavailable = e => { if (e.data && e.data.size) recChunks.push(e.data); };
  mediaRec.onstop = () => {
    recStream.getTracks().forEach(t => t.stop()); S.v.srcObject = null;
    const blob = new Blob(recChunks, { type: mime });
    const ext = mime.includes('mp4') ? 'mp4' : 'webm';
    setSingle(new File([blob], 'recorded.' + ext, { type: mime }));
    btn.textContent = '📹 Record with camera'; btn.classList.remove('rec');
  };
  mediaRec.start();
  btn.textContent = '⏹ Stop recording'; btn.classList.add('rec');
  S.hint.textContent = 'Recording… film the side view of your curl, then tap Stop';
});

/* overlay on the single video (right arm highlighted) */
function drawSingle() {
  // Single-camera view shows ONLY the original video (no ViTPose overlay). Keep the canvas cleared.
  const v = S.v, c = S.k;
  if (v && c) {
    if (c.width !== v.clientWidth) { c.width = v.clientWidth; c.height = v.clientHeight; }
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }
  requestAnimationFrame(drawSingle);
}
requestAnimationFrame(drawSingle);

/* colour a muscle by activation (light -> deep blue) */
function muscleColor(a) {
  const t = Math.max(0, Math.min(a / 0.4, 1));
  const c1 = [207, 215, 238], c2 = [41, 66, 184];
  return `rgb(${c1.map((x, i) => Math.round(x + (c2[i] - x) * t)).join(',')})`;
}

/* run single-camera analysis */
async function runSingle() {
  if (S.running || !S.file_v) return;
  S.running = true; S.err.style.display = 'none'; S.run.disabled = true; S.run.textContent = 'Processing…';
  S.proc.classList.add('on'); S.results.classList.remove('on');
  const st = [...S.proc.querySelectorAll('.fit-stage')];
  st.forEach(s => s.classList.remove('act', 'done'));
  st[0].classList.add('act'); st[0].querySelector('.fsi').innerHTML = '<span class="spin"></span>';

  const fd = new FormData(); fd.append('cam1', S.file_v); fd.append('stride', '3');
  try {
    const res = await fetch(BACKEND + '/analyze_single', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    st[0].classList.replace('act', 'done'); st[0].querySelector('.fsi').textContent = '✓';
    st[1].classList.add('done'); st[1].querySelector('.fsi').textContent = '✓';
    S.kp = data.keypoints; S.fps = data.fps; S.stride = data.stride;
    renderSingle(data);
    window.__vbLast = { data: data, mode: 'single', videos: [S.file_v] };
    if (window.onAnalyzed) window.onAnalyzed(window.__vbLast);
    S.results.classList.add('on'); S.results.scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    S.err.style.display = 'block';
    S.err.innerHTML = '⚠️ Backend not reachable. Start it:<br><code>uvicorn analysis.app:app --port 8000</code><br>(' + e.message + ')';
  }
  S.run.textContent = 'Analyze movement'; S.run.disabled = false; S.running = false;
}
$('srun').addEventListener('click', runSingle);
S.again.addEventListener('click', () => { S.results.classList.remove('on'); S.proc.classList.remove('on'); document.getElementById('p1').scrollIntoView({ behavior: 'smooth' }); });

/* render muscle diagram + dashboard + charts */
const E_SINGLE = { torque: 'd_torque', force: 'd_force', act: 'd_act', fat: 'd_fat', r2: 's_r2', mbic: 'm_bic', mbra: 'm_bra', mbrd: 'm_brd', mlist: 'm_list', charts: 'scharts', prefix: 's' };
function renderSingle(data) { renderFit(data, E_SINGLE); }