/* Rehab layer for the PWA: patient login + assigned exercises + save session + history detail.
   Reuses the app's renderFit()/muscleColor() so results & history look EXACTLY like the analyzer. */
'use strict';
(function () {
  const REHAB = (location.port === '8100') ? '' : 'http://localhost:8100';  // rehab API base
  let PTOKEN = localStorage.getItem('vb_ptoken') || '';
  let CURRENT_RX = null;                 // the prescription the patient is doing now
  let LAST = null;                        // { data, mode, videos:[File] }
  const $ = (id) => document.getElementById(id);

  async function rapi(path, opts = {}) {
    const r = await fetch(REHAB + path, { ...opts,
      headers: { 'Content-Type': 'application/json', ...(PTOKEN ? { Authorization: 'Bearer ' + PTOKEN } : {}), ...(opts.headers || {}) } });
    if (r.status === 401) { plogout(); throw new Error('unauthorized'); }
    if (!r.ok) throw new Error((await r.json().catch(() => ({ detail: r.status }))).detail);
    return r.json();
  }

  // ---------------- injected UI ----------------
  function inject() {
    const wrap = document.createElement('div');
    wrap.innerHTML = `
    <div id="rehab_login">
      <div class="vb-auth">
        <!-- faux status bar, for the native-app feel -->
        <div class="vb-statusbar">
          <span>9:41</span>
          <span class="vb-sb"><i class="sig"></i><i class="wifi"></i><i class="batt"></i></span>
        </div>

        <div class="vb-hero">
          <!-- on-brand decoration: the three arm keypoints the app actually tracks -->
          <svg class="vb-art" viewBox="0 0 200 200" fill="none" aria-hidden="true">
            <path d="M40 168 L120 120 L150 60" stroke="#fff" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" opacity=".12"/>
            <path d="M40 168 L120 120 L150 60" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="40" cy="168" r="7" fill="#fff"/><circle cx="120" cy="120" r="7" fill="#fff"/><circle cx="150" cy="60" r="7" fill="#fff"/>
          </svg>

          <div class="vb-brand">
            <span class="mark">
              <svg viewBox="0 0 24 24" fill="none"><path d="M5 18 L14 14 L9 6" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="5" cy="18" r="2" fill="#fff"/><circle cx="14" cy="14" r="2" fill="#fff"/><circle cx="9" cy="6" r="2" fill="#fff"/></svg>
            </span>
            VisionBiomech
          </div>
          <div class="vb-eyebrow">Home rehabilitation</div>
          <h1 class="vb-title">Your rehab,<br>at home.</h1>
        </div>

        <div class="vb-sheet">
          <div class="vb-grip"></div>
          <h2 class="vb-h">Welcome back</h2>

          <label class="vb-lab">Email</label>
          <div class="vb-field"><span class="fi">
            <svg viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" stroke-width="1.7"/><path d="M4 7l8 6 8-6" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
          </span><input id="rl_email" value="patient@demo.com" autocomplete="username" placeholder="you@example.com"></div>

          <div class="vb-labrow"><label class="vb-lab">Password</label><span class="vb-forgot">Forgot?</span></div>
          <div class="vb-field"><span class="fi">
            <svg viewBox="0 0 24 24" fill="none"><rect x="5" y="10" width="14" height="10" rx="2.5" stroke="currentColor" stroke-width="1.7"/><path d="M8 10V7a4 4 0 018 0v3" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
          </span><input id="rl_pass" type="password" value="demo1234" autocomplete="current-password" placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;">
          <span class="vb-eye" id="rl_eye" role="button" aria-label="show password">
            <svg viewBox="0 0 24 24" fill="none"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" stroke="currentColor" stroke-width="1.7"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.7"/></svg>
          </span></div>

          <div id="rl_err" class="vb-err"></div>
          <button class="vb-btn" id="rl_btn">Sign in <span class="arr">&rarr;</span></button>

          <div class="vb-or"><span>or</span></div>
          <button class="vb-faceid" type="button" id="rl_face">
            <svg viewBox="0 0 24 24" fill="none"><path d="M4 8V6a2 2 0 012-2h2M16 4h2a2 2 0 012 2v2M20 16v2a2 2 0 01-2 2h-2M8 20H6a2 2 0 01-2-2v-2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M9 10v1M15 10v1M12 9v4l-1 1M9.5 15c1.5 1 3.5 1 5 0" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
            Sign in with Face ID
          </button>

          <div class="vb-trust">&#128274; Your videos stay on your device.</div>
          <div class="vb-home"></div>
        </div>
      </div>
    </div>

    <main class="page fit" id="p_home">
      <div class="fit-hero"><div class="fit-kick">HOME REHABILITATION</div>
        <h2 class="fit-title" id="rh_hi">Your exercises</h2>
        <p class="fit-sub">Do your prescribed exercises and track your progress.</p></div>
      <div class="fit-sec">Assigned by your doctor</div>
      <div id="rx_home" style="padding:0 18px;display:flex;flex-direction:column;gap:12px"></div>
    </main>

    <main class="page fit" id="p_hist">
      <div class="fit-hero"><div class="fit-kick">HISTORY</div><h2 class="fit-title">My sessions</h2>
        <p class="fit-sub">Tap a session to see the full analysis.</p></div>
      <div id="hist_list" style="padding:16px 18px;display:flex;flex-direction:column;gap:11px"></div>
      <div id="hist_detail" class="fit-results" style="display:none">
        <button class="fit-btn ghost" id="hd_back" style="margin:14px 18px 0">‹ Back to history</button>
        <div id="hd_head" style="margin:14px 18px 0"></div>
        <div id="hd_videos" style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 18px 0"></div>
        <div class="fit-card fit-muscle" style="margin:16px 18px 0">
          <div class="fit-card-h">Muscles involved <span class="fit-badge">4 elbow flexors</span></div>
          <div class="fit-arm"><svg viewBox="0 0 220 150" id="hd_armsvg" xmlns="http://www.w3.org/2000/svg">
            <path class="seg" d="M40 40 L110 70"/><path class="seg" d="M110 70 L170 40"/>
            <ellipse id="hd_bic" class="musc" cx="72" cy="53" rx="26" ry="12" transform="rotate(24 72 53)"/>
            <ellipse id="hd_bra" class="musc" cx="95" cy="66" rx="18" ry="8" transform="rotate(24 95 66)"/>
            <ellipse id="hd_brd" class="musc" cx="140" cy="55" rx="22" ry="9" transform="rotate(-27 140 55)"/>
            <circle class="joint" cx="40" cy="40" r="5"/><circle class="joint" cx="110" cy="70" r="6"/><circle class="joint" cx="170" cy="40" r="5"/>
            <text x="34" y="30" class="jlab">shoulder</text><text x="150" y="30" class="jlab">wrist</text><text x="98" y="90" class="jlab">elbow</text>
          </svg></div>
          <div class="fit-mlist" id="hd_mlist"></div>
        </div>
        <div class="fit-dash" style="margin:16px 18px 0">
          <div class="fit-stat"><div class="fsv" id="hd_torque">–</div><div class="fsl">Peak torque</div><div class="fsu">N·m</div></div>
          <div class="fit-stat"><div class="fsv" id="hd_force">–</div><div class="fsl">Peak force</div><div class="fsu">N</div></div>
          <div class="fit-stat"><div class="fsv" id="hd_act">–</div><div class="fsl">Mean activation</div><div class="fsu">0–1</div></div>
          <div class="fit-stat"><div class="fsv" id="hd_fat">–</div><div class="fsl">End fatigue</div><div class="fsu">%</div></div>
        </div>
        <div class="fit-card" style="margin:16px 18px 0"><div class="fit-card-h">Signals over time <span class="fit-badge alt" id="hd_r2">R²</span></div><div id="hd_charts"></div></div>
      </div>
    </main>

    <div id="rehab_banner" style="display:none;margin:12px 18px 0;background:#eef2fe;border:1px solid #c7d0ea;border-radius:12px;padding:10px 14px;font-size:.85rem;color:#2942b8">
      Exercise: <b id="rb_name"></b> — record it with your camera below.</div>

    <div id="rehab_savebar" style="display:none;position:fixed;left:0;right:0;bottom:70px;max-width:440px;margin:0 auto;padding:0 18px;z-index:60">
      <button class="fit-btn" id="rehab_save" style="margin:0;box-shadow:0 8px 22px rgba(59,91,219,.4)">✓ Save this session to my history</button>
    </div>`;
    const st = document.createElement('style');
    st.textContent = `
      /* ---- login screen -------------------------------------------------
         Every rule is scoped under #rehab_login so it cannot collide with
         app.css / fit.css, which already define .btn, .brand, .field, … */
      #rehab_login{position:fixed;inset:0;z-index:9999;overflow-y:auto;background:#1e2c73}
      #rehab_login .vb-auth{min-height:100vh;min-height:100dvh;max-width:480px;margin:0 auto;
        display:flex;flex-direction:column;
        background:radial-gradient(120% 80% at 80% -10%, #5c7cff 0%, transparent 55%),
                   linear-gradient(165deg,#3b5bdb 0%,#2a3fb0 52%,#1e2c73 100%)}
      #rehab_login .vb-statusbar{display:flex;justify-content:space-between;align-items:center;
        padding:calc(8px + env(safe-area-inset-top)) 22px 2px;color:#fff;font-size:.82rem;font-weight:700}
      #rehab_login .vb-sb{display:flex;align-items:center;gap:6px}
      #rehab_login .vb-sb i{display:block;background:#fff;border-radius:1px}
      #rehab_login .vb-sb .sig{width:16px;height:10px;clip-path:polygon(0 100%,20% 100%,20% 55%,40% 55%,40% 100%,60% 100%,60% 30%,80% 30%,80% 100%,100% 100%,100% 0,0 0)}
      #rehab_login .vb-sb .wifi{width:15px;height:11px;clip-path:polygon(50% 100%,0 40%,15% 26%,50% 58%,85% 26%,100% 40%)}
      #rehab_login .vb-sb .batt{width:22px;height:11px;border-radius:3px;position:relative}
      #rehab_login .vb-sb .batt::after{content:"";position:absolute;right:-3px;top:3px;width:2px;height:5px;background:#fff;border-radius:1px}
      #rehab_login .vb-hero{flex:1;display:flex;flex-direction:column;justify-content:center;
        padding:20px 28px 26px;color:#fff;position:relative;overflow:hidden}
      #rehab_login .vb-art{position:absolute;top:14px;right:-30px;width:210px;height:210px;opacity:.9;transform:rotate(4deg)}
      #rehab_login .vb-brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.02rem;position:relative}
      #rehab_login .vb-brand .mark{width:34px;height:34px;border-radius:11px;
        background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25);
        display:flex;align-items:center;justify-content:center;box-shadow:0 6px 16px rgba(0,0,0,.15)}
      #rehab_login .vb-brand .mark svg{width:20px;height:20px}
      #rehab_login .vb-eyebrow{margin-top:26px;font-size:.72rem;font-weight:800;letter-spacing:.18em;
        text-transform:uppercase;opacity:.72;position:relative}
      #rehab_login .vb-title{font-size:2.35rem;font-weight:900;line-height:1.06;margin:8px 0 0;
        letter-spacing:-.01em;position:relative;text-shadow:0 2px 20px rgba(0,0,0,.12)}
      #rehab_login .vb-sheet{background:#fff;border-radius:30px 30px 0 0;
        padding:10px 26px calc(14px + env(safe-area-inset-bottom));box-shadow:0 -14px 40px rgba(21,27,46,.22)}
      #rehab_login .vb-grip{width:40px;height:5px;border-radius:5px;background:#e6eaf3;margin:8px auto 12px}
      #rehab_login .vb-h{font-size:1.4rem;font-weight:900;letter-spacing:-.01em;margin:0 0 12px}
      #rehab_login .vb-lab{display:block;font-size:.76rem;font-weight:600;color:#7580a0;margin:12px 0 5px}
      #rehab_login .vb-labrow{display:flex;justify-content:space-between;align-items:baseline}
      #rehab_login .vb-forgot{color:#3b5bdb;font-size:.76rem;font-weight:700;margin-top:12px}
      #rehab_login .vb-field{position:relative;display:flex;align-items:center}
      #rehab_login .vb-field .fi{position:absolute;left:14px;color:#7580a0;display:flex;pointer-events:none}
      #rehab_login .vb-field .fi svg{width:19px;height:19px}
      #rehab_login .vb-field input{width:100%;padding:0 44px;height:54px;border-radius:15px;background:#f5f7fc;
        border:1.5px solid transparent;font:inherit;font-size:1rem;color:#151b2e}
      #rehab_login .vb-field input::placeholder{color:#aab2cc}
      #rehab_login .vb-field input:focus{outline:none;border-color:#5c7cff;background:#fff;
        box-shadow:0 0 0 4px rgba(92,124,255,.14)}
      #rehab_login .vb-eye{position:absolute;right:12px;color:#7580a0;display:flex;padding:6px;cursor:pointer}
      #rehab_login .vb-eye svg{width:20px;height:20px}
      #rehab_login .vb-eye.on{color:#3b5bdb}
      #rehab_login .vb-err{color:#e5484d;font-size:.82rem;margin-top:8px;min-height:16px}
      #rehab_login .vb-btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;
        height:54px;margin-top:20px;border:0;border-radius:16px;color:#fff;font:inherit;font-weight:700;
        font-size:1.02rem;cursor:pointer;background:linear-gradient(120deg,#3b5bdb,#5c7cff);
        box-shadow:0 10px 24px rgba(59,91,219,.34)}
      #rehab_login .vb-btn:disabled{background:#c3cbe4;box-shadow:none}
      #rehab_login .vb-btn .arr{transition:transform .2s}
      #rehab_login .vb-btn:active .arr{transform:translateX(4px)}
      #rehab_login .vb-or{display:flex;align-items:center;gap:12px;color:#aab2cc;font-size:.78rem;font-weight:600;margin:16px 0}
      #rehab_login .vb-or::before,#rehab_login .vb-or::after{content:"";flex:1;height:1px;background:#e2e7f2}
      #rehab_login .vb-faceid{display:flex;align-items:center;justify-content:center;gap:9px;width:100%;height:52px;
        border-radius:16px;background:#fff;border:1.5px solid #e2e7f2;color:#151b2e;font:inherit;font-weight:700;
        font-size:.95rem;cursor:pointer}
      #rehab_login .vb-faceid svg{width:22px;height:22px;color:#3b5bdb}
      #rehab_login .vb-faceid:active{background:#f5f7fc}
      #rehab_login .vb-trust{text-align:center;color:#7580a0;font-size:.76rem;margin-top:16px}
      #rehab_login .vb-home{width:130px;height:5px;border-radius:5px;background:#0b1020;opacity:.18;margin:12px auto 2px}
      /* ---- rest of the injected rehab UI ---- */
      .lab{display:block;font-size:.76rem;font-weight:600;color:#7580a0;margin:12px 0 5px}
      .inp{width:100%;padding:12px;border:1px solid #e2e7f2;border-radius:11px;font:inherit;background:#fff}
      .rx-card{background:#fff;border-radius:16px;padding:16px;box-shadow:0 4px 14px rgba(21,27,46,.06);display:flex;align-items:center;justify-content:space-between;gap:12px}
      .rx-card .t{font-weight:700;font-size:1.05rem}.rx-card .m{color:#7580a0;font-size:.8rem;margin-top:3px}
      .rx-card .go{background:#eaeefc;color:#3b5bdb;border:none;border-radius:11px;padding:11px 15px;font-weight:700}
      .pill{display:inline-block;background:#eaeefc;color:#3b5bdb;font-size:.68rem;font-weight:700;padding:3px 9px;border-radius:100px;margin-top:6px}
      .hist-item{background:#fff;border-radius:15px;padding:14px;box-shadow:0 4px 14px rgba(21,27,46,.06);cursor:pointer}
      .hist-item .h{display:flex;justify-content:space-between;font-weight:700}.hist-item .d{color:#7580a0;font-weight:400;font-size:.76rem}
      .hist-item .s{display:flex;gap:14px;margin-top:6px;font-size:.78rem;color:#7580a0}.hist-item .s b{color:#151b2e}
      #hd_videos video{width:130px;height:96px;object-fit:cover;border-radius:12px;background:#000}`;
    document.head.appendChild(st);
    document.body.appendChild(wrap);

    // rebuild bottom nav with 4 tabs
    const nav = document.querySelector('.nav');
    nav.innerHTML = `
      <button data-tab="home"><span class="ic">🏠</span>Home</button>
      <button data-tab="single"><span class="ic">📱</span>Record</button>
      <button data-tab="history"><span class="ic">📊</span>History</button>`;
    nav.querySelectorAll('button').forEach((b) => b.onclick = () => setPage(b.dataset.tab));

    $('rl_btn').onclick = plogin;
    $('rl_pass').addEventListener('keydown', (e) => { if (e.key === 'Enter') plogin(); });
    // reveal the password, and let Face ID stand in for Sign in on the demo
    $('rl_eye').onclick = () => {
      const p = $('rl_pass'), show = p.type === 'password';
      p.type = show ? 'text' : 'password';
      $('rl_eye').classList.toggle('on', show);
    };
    $('rl_face').onclick = () => $('rl_btn').click();
    $('rehab_save').onclick = saveSession;
    $('hd_back').onclick = () => { $('hist_detail').style.display = 'none'; $('hist_list').style.display = 'flex'; };
  }

  // ---------------- navigation ----------------
  function setPage(p) {
    const PAGE_OF = { home: 'p_home', single: 'p1', history: 'p_hist' };
    Object.values(PAGE_OF).forEach((id) => $(id) && $(id).classList.remove('on'));
    if (PAGE_OF[p]) $(PAGE_OF[p]).classList.add('on');
    document.querySelectorAll('.nav button').forEach((b) => b.classList.toggle('on', b.dataset.tab === p));
    if (p === 'home' || p === 'history') { const bn = $('rehab_banner'); if (bn) bn.style.display = 'none'; }
    if (p === 'home') loadExercises();
    if (p === 'history') { $('hist_detail').style.display = 'none'; $('hist_list').style.display = 'flex'; loadHistory(); }
  }

  // ---------------- auth ----------------
  async function plogin() {
    $('rl_err').textContent = '';
    try {
      const d = await rapi('/auth/login', { method: 'POST', body: JSON.stringify({ email: $('rl_email').value, password: $('rl_pass').value }) });
      if (d.user.role !== 'patient') { $('rl_err').textContent = 'Use a patient account.'; return; }
      PTOKEN = d.token; localStorage.setItem('vb_ptoken', PTOKEN); showRehab(d.user);
    } catch (e) { $('rl_err').textContent = 'Wrong email or password.'; }
  }
  function plogout() {
    PTOKEN = ''; localStorage.removeItem('vb_ptoken');
    $('rehab_login').style.display = 'block';
    const out = $('rh_out'); if (out) out.hidden = true;
    const h = $('rh_hi'); if (h) h.textContent = 'Your exercises';
    const l = $('rx_home'); if (l) l.innerHTML = '';
    const hl = $('hist_list'); if (hl) hl.innerHTML = '';
    setPage('home');
  }
  async function pboot() { if (!PTOKEN) return; try { const me = await rapi('/me'); if (me.role === 'patient') showRehab(me); } catch (e) {} }
  function showRehab(user) {
    $('rehab_login').style.display = 'none';
    $('rh_hi').textContent = 'Hi, ' + user.name.split(' ')[0];
    const st = document.querySelector('.status'); if (st) { st.textContent = user.name.split(' ')[0]; }
    const out = $('rh_out');
    if (out) {
      out.hidden = false;
      if (!out.dataset.bound) { out.dataset.bound = '1'; out.onclick = plogout; }
    }
    setPage('home');
  }

  // ---------------- home: assigned exercises ----------------
  async function loadExercises() {
    const rx = await rapi('/my/prescriptions');
    $('rx_home').innerHTML = rx.length ? rx.map((r, i) => `
      <div class="rx-card"><div><div class="t">${r.exercise ? r.exercise.name : 'Exercise'}</div>
        <div class="m">${r.reps} reps × ${r.sets} sets</div><span class="pill">${r.schedule || 'anytime'}</span></div>
        <button class="go" data-i="${i}">Start ›</button></div>`).join('')
      : '<div style="color:#7580a0;text-align:center;padding:24px">No exercises assigned yet.</div>';
    $('rx_home').querySelectorAll('[data-i]').forEach((b) => b.onclick = () => startExercise(rx[+b.dataset.i]));
  }
  function startExercise(rx) {
    CURRENT_RX = rx; LAST = null; $('rehab_savebar').style.display = 'none';
    startRecording();                             // one camera: no mode to choose
  }
  function startRecording() {
    $('rb_name').textContent = CURRENT_RX && CURRENT_RX.exercise ? CURRENT_RX.exercise.name : 'Exercise';
    $('rehab_banner').style.display = 'block';
    setPage('single');
  }

  // ---------------- save after analysis ----------------
  function countReps(data) {
    const p = ((data.vars || {}).elbow_moment || {}).pred || [];
    if (p.length < 3) return 0;
    const mn = Math.min(...p), mx = Math.max(...p), thr = mn + (mx - mn) * 0.5;
    let n = 0, up = false; for (const v of p) { if (v > thr && !up) { n++; up = true; } else if (v < thr) up = false; } return n;
  }
  window.onAnalyzed = (payload) => {   // called by app.js after a real analysis
    LAST = payload;
    if (CURRENT_RX) $('rehab_savebar').style.display = 'block';
  };
  async function saveSession() {
    if (!LAST || !CURRENT_RX) return;
    const btn = $('rehab_save'); btn.disabled = true; btn.textContent = 'Saving…';
    try {
      const sess = await rapi('/my/sessions', { method: 'POST', body: JSON.stringify({
        exercise_id: CURRENT_RX.exercise_id, prescription_id: CURRENT_RX.id, mode: LAST.mode,
        reps_done: countReps(LAST.data), results: LAST.data }) });
      for (const f of (LAST.videos || [])) {
        if (!f) continue;
        const fd = new FormData(); fd.append('file', f);
        await fetch(REHAB + `/my/sessions/${sess.id}/video`, { method: 'POST', headers: { Authorization: 'Bearer ' + PTOKEN }, body: fd }).catch(() => {});
      }
      $('rehab_savebar').style.display = 'none'; CURRENT_RX = null; setPage('history');
    } catch (e) { alert('Save failed: ' + e.message); }
    btn.disabled = false; btn.textContent = '✓ Save this session to my history';
  }

  // ---------------- history + detail ----------------
  let EXMAP = {};
  async function loadHistory() {
    if (!Object.keys(EXMAP).length) (await rapi('/exercises')).forEach((e) => EXMAP[e.id] = e.name);
    const list = await rapi('/my/sessions');
    $('hist_list').innerHTML = list.length ? list.map((s, i) => {
      const V = s.results || {}; const tq = (V.vars || {}).elbow_moment, fat = (V.vars || {}).MF_BIClong;
      const pk = (a) => a && a.pred ? Math.max(...a.pred.map(Math.abs)).toFixed(1) : '–';
      const en = (a) => a && a.pred ? a.pred[a.pred.length - 1].toFixed(1) : '–';
      return `<div class="hist-item" data-i="${i}"><div class="h"><span>${EXMAP[s.exercise_id] || 'Exercise'}</span>
        <span class="d">${new Date(s.created_at).toLocaleString()}</span></div>
        <div class="s"><span><b>${s.reps_done}</b> reps</span><span>torque <b>${pk(tq)}</b></span><span>fatigue <b>${en(fat)}</b>%</span>
        <span style="color:#3b5bdb">details ›</span></div></div>`;
    }).join('') : '<div style="color:#7580a0;text-align:center;padding:24px">No sessions yet. Do an exercise from Home.</div>';
    $('hist_list').querySelectorAll('[data-i]').forEach((b) => b.onclick = () => showDetail(list[+b.dataset.i]));
  }
  const E_HD = { torque: 'hd_torque', force: 'hd_force', act: 'hd_act', fat: 'hd_fat', r2: 'hd_r2',
    mbic: 'hd_bic', mbra: 'hd_bra', mbrd: 'hd_brd', mlist: 'hd_mlist', charts: 'hd_charts', prefix: 'hd' };
  function showDetail(s) {
    $('hist_list').style.display = 'none'; $('hist_detail').style.display = 'block';
    $('hd_head').innerHTML = `<div style="font-weight:800;font-size:1.3rem">${EXMAP[s.exercise_id] || 'Exercise'}</div>
      <div style="color:#7580a0;font-size:.85rem">${new Date(s.created_at).toLocaleString()} · ${s.reps_done} reps · ${s.mode || 'single'} camera</div>`;
    $('hd_videos').innerHTML = (s.videos || []).map((u) => `<video src="${REHAB + u}" muted loop autoplay playsinline></video>`).join('')
      || '<div style="color:#9aa6c8;font-size:.8rem">no video saved</div>';
    if (s.results && s.results.vars) renderFit(s.results, E_HD);   // reuse the app's exact renderer
  }

  // ---------------- boot ----------------
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
  function start() { inject(); pboot(); }
})();
