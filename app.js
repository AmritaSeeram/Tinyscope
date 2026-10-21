(() => {
  const D = 512, LR = 0.5, LAM = 0.001;
  const COL = { a: [15, 157, 138], b: [194, 50, 125] };
  const STARTER = [
    ["warm blanket and a mug of tea", 0], ["rainy afternoon with a good book", 0],
    ["candles glowing by the window", 0], ["soft socks and slow music", 0],
    ["fireplace crackling at dusk", 0], ["hot cocoa under a quilt", 0],
    ["alarms blaring in every room", 1], ["ten tabs open and none make sense", 1],
    ["spilled coffee on the keyboard", 1], ["traffic horns and shouting everywhere", 1],
    ["deadline in an hour and nothing works", 1], ["dogs barking at the vacuum", 1],
  ];
  const $ = (id) => document.getElementById(id);
  const state = { ex: STARTER.map(([text, y]) => ({ text, y })), w: new Float32Array(D), b: 0 };
  const cache = new Map();

  // ---- features: hashed words + letter trios -------------------------------
  const hash = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };
  const tokenize = (t) => t.toLowerCase().match(/[a-z0-9']+/g) || [];
  function vec(text) {
    if (cache.has(text)) return cache.get(text);
    const toks = tokenize(text), entries = [], x = new Float32Array(D);
    const add = (k, v, ti) => { const h = hash(k); entries.push({ i: h % D, v: (hash(k + "#") & 1 ? 1 : -1) * v, ti }); };
    toks.forEach((tok, ti) => {
      add("w:" + tok, 1, ti);
      const p = "<" + tok + ">";
      for (let i = 0; i + 3 <= p.length; i++) add("c:" + p.slice(i, i + 3), 0.35, ti);
    });
    entries.forEach((e) => { x[e.i] += e.v; });
    const norm = Math.hypot(...x) || 1;
    for (let i = 0; i < D; i++) x[i] /= norm;
    const out = { x, entries, norm, toks };
    cache.set(text, out);
    return out;
  }

  // ---- model ---------------------------------------------------------------
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const dot = (x) => { let s = state.b; for (let i = 0; i < D; i++) s += state.w[i] * x[i]; return s; };
  const probB = (text) => sigmoid(dot(vec(text).x));
  function epoch() {
    const items = state.ex.map((e) => ({ x: vec(e.text).x, y: e.y }));
    for (let i = items.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [items[i], items[j]] = [items[j], items[i]]; }
    for (const it of items) {
      const g = sigmoid(dot(it.x)) - it.y;
      for (let i = 0; i < D; i++) state.w[i] -= LR * (g * it.x[i] + LAM * state.w[i]);
      state.b -= LR * g;
    }
  }

  // ---- 2D map: top two principal directions of your examples ---------------
  function plane() {
    const X = state.ex.map((e) => Float32Array.from(vec(e.text).x)), n = X.length;
    const m = new Float32Array(D);
    X.forEach((r) => r.forEach((v, i) => { m[i] += v / n; }));
    X.forEach((r) => { for (let i = 0; i < D; i++) r[i] -= m[i]; });
    const pcs = [];
    for (let c = 0; c < 2; c++) {
      let v = Float32Array.from({ length: D }, (_, i) => Math.sin(i * 12.9898 + c * 78.233));
      for (let it = 0; it < 30; it++) {
        const u = X.map((r) => r.reduce((s, val, i) => s + val * v[i], 0));
        const nv = new Float32Array(D);
        X.forEach((r, k) => { for (let i = 0; i < D; i++) nv[i] += r[i] * u[k]; });
        const nn = Math.hypot(...nv) || 1;
        for (let i = 0; i < D; i++) nv[i] /= nn;
        v = nv;
      }
      pcs.push(v);
      X.forEach((r) => { const p = r.reduce((s, val, i) => s + val * v[i], 0); for (let i = 0; i < D; i++) r[i] -= p * v[i]; });
    }
    return { m, pcs };
  }
  const project = (x, pl) => pl.pcs.map((pc) => x.reduce((s, v, i) => s + (v - pl.m[i]) * pc[i], 0));

  // ---- drawing -------------------------------------------------------------
  const canvas = $("map"), ctx = canvas.getContext("2d");
  let pts = [];
  const mix = (p) => COL.a.map((a, i) => Math.round(a + (COL.b[i] - a) * p));
  function draw() {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (state.ex.length < 3) { ctx.fillStyle = "#5d6b7c"; ctx.font = "16px sans-serif"; ctx.fillText("Add at least 3 examples to draw the map.", 20, 40); pts = []; return; }
    const pl = plane(), q = $("query").value;
    const coords = state.ex.map((e) => project(vec(e.text).x, pl));
    const qc = q.trim() ? project(vec(q).x, pl) : null;
    const all = qc ? coords.concat([qc]) : coords;
    const xs = all.map((c) => c[0]), ys = all.map((c) => c[1]);
    let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const px = Math.max(x1 - x0, 0.3) * 0.25, py = Math.max(y1 - y0, 0.3) * 0.25;
    x0 -= px; x1 += px; y0 -= py; y1 += py;
    const sx = (v) => ((v - x0) / (x1 - x0)) * W, sy = (v) => H - ((v - y0) / (y1 - y0)) * H;
    const cell = 8, x = new Float32Array(D);
    for (let cy = 0; cy < H; cy += cell) for (let cx = 0; cx < W; cx += cell) {
      const a = x0 + ((cx + cell / 2) / W) * (x1 - x0), b = y0 + (1 - (cy + cell / 2) / H) * (y1 - y0);
      for (let i = 0; i < D; i++) x[i] = pl.m[i] + a * pl.pcs[0][i] + b * pl.pcs[1][i];
      const p = sigmoid(dot(x)), c = mix(p);
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.08 + Math.abs(p - 0.5) * 0.5})`;
      ctx.fillRect(cx, cy, cell, cell);
    }
    pts = state.ex.map((e, i) => ({ px: sx(coords[i][0]), py: sy(coords[i][1]), text: e.text }));
    state.ex.forEach((e, i) => {
      const c = e.y ? COL.b : COL.a;
      ctx.beginPath(); ctx.arc(pts[i].px, pts[i].py, 6, 0, 7);
      ctx.fillStyle = `rgb(${c})`; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
    });
    if (qc) { ctx.beginPath(); ctx.arc(sx(qc[0]), sy(qc[1]), 9, 0, 7); ctx.strokeStyle = "#14213d"; ctx.lineWidth = 3.5; ctx.stroke(); }
  }
  canvas.addEventListener("mousemove", (ev) => {
    const r = canvas.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const hit = pts.find((p) => Math.hypot(p.px - mx, p.py - my) < 10), tip = $("tip");
    tip.hidden = !hit;
    if (hit) { tip.textContent = hit.text; tip.style.left = Math.min(mx + 12, r.width - 270) + "px"; tip.style.top = my + 12 + "px"; }
  });
  canvas.addEventListener("mouseleave", () => { $("tip").hidden = true; });

  // ---- explanations --------------------------------------------------------
  const names = () => [$("nameA").value || "A", $("nameB").value || "B"];
  const chip = (text, extra, style) => { const s = document.createElement("span"); s.className = "chip"; s.textContent = text; if (style) s.style.background = style; if (extra) { const sm = document.createElement("small"); sm.textContent = extra; s.appendChild(sm); } return s; };
  function renderVerdict() {
    const q = $("query").value, [A, B] = names();
    if (!tokenize(q).length) { $("vtext").textContent = "Type a sentence to test the model."; return; }
    const p = probB(q);
    $("barfill").style.left = p * 100 + "%";
    $("vtext").textContent = `${p < 0.5 ? A : B}, ${Math.round((p < 0.5 ? 1 - p : p) * 100)}% sure`;
  }
  function renderAsk() {
    renderVerdict();
    const q = $("query").value, v = vec(q), [A, B] = names();
    const toks = $("tokens"), why = $("why"), drop = $("drop");
    toks.replaceChildren(); why.replaceChildren(); drop.replaceChildren();
    if (!v.toks.length) return;
    v.toks.forEach((t) => toks.appendChild(chip(t, "#" + (hash("w:" + t) % D))));
    const contrib = v.toks.map((_, ti) => v.entries.filter((e) => e.ti === ti).reduce((s, e) => s + (state.w[e.i] * e.v) / v.norm, 0));
    const mx = Math.max(...contrib.map(Math.abs), 1e-6);
    v.toks.forEach((t, ti) => {
      const c = contrib[ti], col = c > 0 ? COL.b : COL.a, a = 0.12 + (Math.abs(c) / mx) * 0.6;
      why.appendChild(chip(t, (c > 0 ? "toward " + B : "toward " + A), `rgba(${col},${a})`));
    });
    const p0 = probB(q);
    v.toks.slice(0, 12).forEach((t, ti) => {
      const rest = v.toks.filter((_, k) => k !== ti).join(" "), li = document.createElement("li");
      if (!rest) return;
      const p = probB(rest), flips = (p < 0.5) !== (p0 < 0.5);
      li.textContent = `Without “${t}”: ${p < 0.5 ? A : B}, ${Math.round((p < 0.5 ? 1 - p : p) * 100)}% sure` + (flips ? " (the answer flips)" : "");
      if (flips) li.className = "flip";
      drop.appendChild(li);
    });
  }

  // ---- training loop & list ------------------------------------------------
  let frames = 0, running = false;
  function loop() {
    if (state.ex.length) for (let k = 0; k < 4; k++) epoch();
    draw(); renderVerdict();
    if (--frames > 0) requestAnimationFrame(loop); else { running = false; renderAsk(); }
  }
  function kick() { frames = 60; if (!running) { running = true; requestAnimationFrame(loop); } }
  function renderList() {
    const ul = $("exList"), [A, B] = names();
    ul.replaceChildren();
    state.ex.forEach((e, i) => {
      const li = document.createElement("li"), d = document.createElement("span"), t = document.createElement("span"), x = document.createElement("button");
      d.className = "dot"; d.style.background = `rgb(${e.y ? COL.b : COL.a})`; d.title = e.y ? B : A;
      t.className = "t"; t.textContent = e.text;
      x.className = "x"; x.type = "button"; x.textContent = "×"; x.setAttribute("aria-label", "Remove example: " + e.text);
      x.onclick = () => { state.ex.splice(i, 1); renderList(); kick(); };
      li.append(d, t, x); ul.appendChild(li);
    });
  }
  function syncNames() {
    const [A, B] = names();
    $("legA").textContent = A; $("legB").textContent = B;
    const sel = $("newLabel"), cur = sel.value;
    sel.replaceChildren(new Option(A, 0), new Option(B, 1)); sel.value = cur || 0;
    renderList(); renderAsk();
  }
  $("addForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = $("newText").value.trim();
    if (!tokenize(text).length) return;
    state.ex.push({ text, y: Number($("newLabel").value) });
    $("newText").value = ""; renderList(); kick();
  });
  $("reset").onclick = () => { state.ex = STARTER.map(([text, y]) => ({ text, y })); state.w.fill(0); state.b = 0; renderList(); kick(); };
  $("nameA").addEventListener("input", syncNames);
  $("nameB").addEventListener("input", syncNames);
  $("query").addEventListener("input", () => { draw(); renderAsk(); });
  window.addEventListener("resize", draw);
  syncNames(); kick();
})();
