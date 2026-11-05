/* ============================================================================
   VisionBiomech — the real pipeline, in the browser. No server.

     frame -> 900x900 -> YOLOX-tiny (person) -> RTMPose-s (SimCC) -> kp[6,8,10]
           -> 22 physics channels -> BiGRU (ONNX) -> 13 biomechanical signals

   This is a faithful port of analysis/single_camera.py. The Python version of
   this exact chain was checked against the service's PyTorch predictor and
   agreed to 0.0000 N.m, so the numbers here are the real ones.

   IMPORTANT — the 900x900 resize is not cosmetic. The L3 model consumes RAW
   PIXEL coordinates and was trained on 900x900 video, so feeding it a 1080p
   phone frame inflates velocities and accelerations and therefore the torque.
   Every frame is squared to 900x900 before pose, exactly as in training.
   ============================================================================ */
'use strict';

window.VBCV = (function () {

  let MODELS = './models';          // override with VBCV.setBase(path)
  const SIDE   = 900;               // training resolution (Fit3D is 900x900)
  const DET_SZ = 416;               // YOLOX input
  const PW = 192, PH = 256;         // RTMPose input (w,h)
  const SIMCC = 2.0;                // SimCC split ratio
  const RID = [6, 8, 10];           // COCO-17: right shoulder, elbow, wrist

  let det = null, pose = null, l3 = null, meta = null, ready = false;

  /* ---------------------------------------------------------------- loading */
  async function load(onProgress = () => {}) {
    if (ready) return;
    if (typeof ort === 'undefined') throw new Error('onnxruntime-web not loaded');
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    ort.env.wasm.simd = true;

    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    onProgress('detector', 0);
    det  = await ort.InferenceSession.create(`${MODELS}/yolox-tiny.onnx`, opt);
    onProgress('pose', 33);
    pose = await ort.InferenceSession.create(`${MODELS}/rtmpose-s.onnx`, opt);
    onProgress('predictor', 66);
    l3   = await ort.InferenceSession.create(`${MODELS}/l3.onnx`, opt);
    meta = await (await fetch(`${MODELS}/l3_meta.json`)).json();
    onProgress('ready', 100);
    ready = true;
  }

  /* ------------------------------------------------------- frame -> 900x900 */
  const sq = document.createElement('canvas');
  sq.width = sq.height = SIDE;
  const sqx = sq.getContext('2d', { willReadFrequently: true });

  function square(video) {
    sqx.drawImage(video, 0, 0, SIDE, SIDE);          // square resize, as in training
    return sqx.getImageData(0, 0, SIDE, SIDE);
  }

  /* Seek and WAIT FOR THE FRAME TO BE PRESENTED.
     'seeked' fires when the seek completes, not when the new frame is painted: drawing on
     'seeked' alone silently captures the PREVIOUS frame, which makes every sample identical
     and flattens the motion (and therefore the torque). requestVideoFrameCallback is the
     only reliable signal that a new frame is actually available to draw. */
  function seek(video, t) {
    return new Promise(resolve => {
      const done = () => {
        if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(() => resolve());
        else requestAnimationFrame(() => requestAnimationFrame(resolve));
      };
      if (Math.abs(video.currentTime - t) < 1e-4) return done();
      video.addEventListener('seeked', done, { once: true });
      video.currentTime = t;
    });
  }

  /* -------------------------------------------------------- YOLOX (person) */
  const dc = document.createElement('canvas');
  dc.width = dc.height = DET_SZ;
  const dcx = dc.getContext('2d', { willReadFrequently: true });

  async function detect() {
    // 900x900 -> 416x416. The frame is already square, so this is a plain scale (r = 416/900).
    dcx.drawImage(sq, 0, 0, SIDE, SIDE, 0, 0, DET_SZ, DET_SZ);
    const d = dcx.getImageData(0, 0, DET_SZ, DET_SZ).data;

    const n = DET_SZ * DET_SZ;
    const t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {                                 // BGR, no normalisation (mmdet YOLOX)
      t[i]         = d[i * 4 + 2];
      t[n + i]     = d[i * 4 + 1];
      t[2 * n + i] = d[i * 4];
    }
    const out = await det.run({ input: new ort.Tensor('float32', t, [1, 3, DET_SZ, DET_SZ]) });
    const dets = out.dets.data, labels = out.labels.data;
    const m = out.dets.dims[1];
    const r = DET_SZ / SIDE;
    let best = null, bs = 0.3;
    for (let i = 0; i < m; i++) {
      if (Number(labels[i]) !== 0) continue;                      // class 0 = person
      const s = dets[i * 5 + 4];
      if (s <= bs) continue;
      bs = s;
      best = [dets[i * 5] / r, dets[i * 5 + 1] / r, dets[i * 5 + 2] / r, dets[i * 5 + 3] / r];
    }
    return best;
  }

  /* ---------------------------------------------------- RTMPose (top-down) */
  const pc = document.createElement('canvas');
  pc.width = PW; pc.height = PH;
  const pcx = pc.getContext('2d', { willReadFrequently: true });

  function affine(box) {
    // expand the box to the model aspect, then pad 1.25x  (mirrors top_down_affine)
    let [x1, y1, x2, y2] = box;
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    let bw = x2 - x1, bh = y2 - y1;
    const ar = PW / PH;
    if (bw > bh * ar) bh = bw / ar; else bw = bh * ar;
    bw *= 1.25; bh *= 1.25;
    return { cx, cy, bw, bh };
  }

  async function keypoints(box) {
    const { cx, cy, bw, bh } = affine(box);
    pcx.clearRect(0, 0, PW, PH);
    pcx.drawImage(sq, cx - bw / 2, cy - bh / 2, bw, bh, 0, 0, PW, PH);   // the warp
    const d = pcx.getImageData(0, 0, PW, PH).data;

    const n = PW * PH;
    const t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {                                  // BGR, raw (rtmlib does no mean/std)
      t[i]         = d[i * 4 + 2];
      t[n + i]     = d[i * 4 + 1];
      t[2 * n + i] = d[i * 4];
    }
    // rtmlib normalises the crop: (img - mean) / std, applied to the BGR image as-is
    // (rtmlib/tools/base.py does NOT convert BGR->RGB). Skipping this yields garbage keypoints.
    const MEAN = [123.675, 116.28, 103.53], STD = [58.395, 57.12, 57.375];
    for (let c = 0; c < 3; c++)
      for (let i = 0; i < n; i++) t[c * n + i] = (t[c * n + i] - MEAN[c]) / STD[c];

    const out = await pose.run({ input: new ort.Tensor('float32', t, [1, 3, PH, PW]) });
    const sx = out.simcc_x, sy = out.simcc_y;
    const K = sx.dims[1], NX = sx.dims[2], NY = sy.dims[2];

    const kp = [];
    for (let k = 0; k < K; k++) {
      let bx = 0, vx = -1e9, by = 0, vy = -1e9;
      for (let i = 0; i < NX; i++) { const v = sx.data[k * NX + i]; if (v > vx) { vx = v; bx = i; } }
      for (let i = 0; i < NY; i++) { const v = sy.data[k * NY + i]; if (v > vy) { vy = v; by = i; } }
      // SimCC bin -> model input px -> back to the 900x900 frame
      const mx = bx / SIMCC, my = by / SIMCC;
      kp.push([
        (mx / PW) * bw + (cx - bw / 2),
        (my / PH) * bh + (cy - bh / 2),
        Math.min(vx, vy)
      ]);
    }
    return kp;
  }

  /* ------------------------------------------- 22 channels (== build_l3_input) */
  function gradient(X, T, C) {                    // numpy.gradient along axis 0
    const g = new Float32Array(T * C);
    for (let c = 0; c < C; c++) {
      for (let t = 0; t < T; t++) {
        let v;
        if (t === 0)          v = X[(1) * C + c] - X[0 * C + c];
        else if (t === T - 1) v = X[(T - 1) * C + c] - X[(T - 2) * C + c];
        else                  v = (X[(t + 1) * C + c] - X[(t - 1) * C + c]) / 2;
        g[t * C + c] = v;
      }
    }
    return g;
  }

  function features(arm, fps) {                  // arm: [T][3][2] -> Float32Array(T*22)
    const T = arm.length, dt = 1 / fps;
    const X = new Float32Array(T * 6);
    for (let t = 0; t < T; t++)
      for (let j = 0; j < 3; j++) { X[t * 6 + j * 2] = arm[t][j][0]; X[t * 6 + j * 2 + 1] = arm[t][j][1]; }

    const vel = gradient(X, T, 6);
    const acc = gradient(vel, T, 6);

    const F = new Float32Array(T * 22);
    let cum = 0;
    for (let t = 0; t < T; t++) {
      const shx = X[t*6], shy = X[t*6+1], elx = X[t*6+2], ely = X[t*6+3], wrx = X[t*6+4], wry = X[t*6+5];
      const uax = elx - shx, uay = ely - shy, fax = wrx - elx, fay = wry - ely;
      const nu = Math.hypot(uax, uay) + 1e-9, nf = Math.hypot(fax, fay) + 1e-9;
      const cu = Math.max(-1, Math.min(1, (uax * fax + uay * fay) / (nu * nf)));
      const q_el = Math.acos(cu) * 180 / Math.PI;
      const q_sh = Math.acos(Math.max(-1, Math.min(1, -uay / nu))) * 180 / Math.PI;
      const g = Math.abs(Math.sin((q_sh + q_el) * Math.PI / 180));
      cum += g * dt;

      const o = t * 22;
      for (let c = 0; c < 6; c++) { F[o + c] = X[t*6+c]; F[o + 6 + c] = vel[t*6+c]; F[o + 12 + c] = acc[t*6+c]; }
      F[o + 18] = g;
      F[o + 19] = cum;
      F[o + 20] = t * dt;
      F[o + 21] = t / Math.max(T - 1, 1);
    }
    return F;
  }

  /* ---------------------------------------------------------------- predict */
  async function predict(arm, fps) {
    const T = arm.length;
    const F = features(arm, fps);
    const { xm, xs, ym, ys, targets } = meta;
    for (let t = 0; t < T; t++)
      for (let c = 0; c < 22; c++) F[t * 22 + c] = (F[t * 22 + c] - xm[c]) / xs[c];

    const out = await l3.run({ x: new ort.Tensor('float32', F, [1, T, 22]) });
    const Y = out.y.data;

    const sig = {};
    targets.forEach((name, k) => {
      const a = new Float32Array(T);
      for (let t = 0; t < T; t++) {
        let v = Y[t * 13 + k] * ys[k] + ym[k];
        if (k >= 1 && k <= 4) v = Math.max(0, Math.min(1, v));     // activations in [0,1]
        else if (k >= 5)      v = Math.max(0, v);                  // forces / fatigue >= 0
        a[t] = v;
      }
      sig[name] = a;
    });
    return sig;
  }

  /* ------------------------------------------------------------------- run */
  /** Analyse a <video> element end-to-end. onFrame(i, n, kp) is called per frame. */
  async function analyse(video, { stride = 3, onFrame = () => {}, onStage = () => {} } = {}) {
    await load(onStage);
    const fps = 50;                                   // the model's assumed rate (as in the app)
    const dur = video.duration;
    const step = stride / fps;
    const n = Math.max(2, Math.floor(dur / step));

    onStage('pose', 0);
    const arm = [];
    for (let i = 0; i < n; i++) {
      await seek(video, i * step);
      square(video);
      const box = await detect();
      let k = null;
      if (box) {
        const kp = await keypoints(box);
        k = RID.map(j => [kp[j][0], kp[j][1]]);
        onFrame(i, n, kp, box);
      } else {
        onFrame(i, n, null, null);
      }
      arm.push(k);
      if (i % 6 === 0) onStage('pose', Math.round(i / n * 100));
    }

    // fill gaps where no person was found (linear, as the backend does)
    for (let j = 0; j < 3; j++) for (let d = 0; d < 2; d++) {
      const idx = [], val = [];
      arm.forEach((a, i) => { if (a) { idx.push(i); val.push(a[j][d]); } });
      if (!idx.length) throw new Error('No person detected in the video.');
      arm.forEach((a, i) => {
        if (a) return;
        let p = 0; while (p < idx.length - 1 && idx[p + 1] < i) p++;
        const i0 = idx[Math.max(0, p)], i1 = idx[Math.min(idx.length - 1, p + 1)];
        const t = i1 === i0 ? 0 : (i - i0) / (i1 - i0);
        const v = val[Math.max(0, p)] * (1 - t) + val[Math.min(val.length - 1, p + 1)] * t;
        if (!arm[i]) arm[i] = [[0,0],[0,0],[0,0]];
        arm[i][j][d] = v;
      });
    }

    onStage('predict', 0);
    const sig = await predict(arm, fps / stride);
    onStage('done', 100);
    return { signals: sig, frames: arm.length, fps: fps / stride };
  }

  function setBase(p) { MODELS = p.replace(/\/$/, ''); }

  /* debug hook: run detect+pose on the video's CURRENT frame and return the raw numbers,
     so the JS port can be checked against the Python reference frame by frame. */
  async function _probe(video) {
    square(video);
    const box = await detect(null);
    if (!box) return null;
    const kp = await keypoints(box);
    return { box, arm: RID.map(j => [kp[j][0], kp[j][1], kp[j][2]]), kp };
  }

  return { load, analyse, predict, features, setBase, _probe, seek,
           get ready() { return ready; }, SIDE };
})();
