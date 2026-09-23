/* Step 1C-1 localization review UI. No framework, no build step.
 *
 * Two views of the SAME source-resolution verification frame: a fit-to-screen view and
 * a zoomed view centred on the region of interest.  Bounding boxes are drawn in
 * container coordinates computed from source-frame pixels, so nothing is ever shown or
 * saved in CSS/crop/tile coordinates.
 *
 * Candidates are presented only as A/B/C with a method name — no score, no model
 * identity, no confidence, because the proposal helper must not look like evidence.
 */
(() => {
  "use strict";

  const state = {
    meta: null,
    queue: [],
    current: null,
    detail: null,
    picked: null,       // "A" | "B" | "C" | null
    note: "",
    message: "",
    error: "",
  };

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const PROPOSAL_CLASS = { A: "a", B: "b", C: "c" };

  async function api(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(payload.error || `HTTP ${response.status}`);
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  const post = (path, body) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  // ---------------------------------------------------------------- loading

  async function loadMeta() {
    state.meta = await api("/api/meta");
    state.queue = state.meta.queue;
    const cameras = [...new Set(state.queue.map((r) => r.camera_id))].sort();
    $("f-camera").innerHTML = '<option value="all">全部</option>' +
      cameras.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    $("meta").textContent =
      `${state.meta.required_episode_count} REQUIRED episodes · Gold ${state.meta.gold_sha256.slice(0, 12)}`;
    renderProgress();
    renderQueue();
    const first = visibleRows()[0];
    if (first) selectEpisode(first.episode_id);
  }

  function renderProgress() {
    const p = state.meta.progress;
    $("progress").textContent = `${p.reviewed} / ${p.total}`;
    $("status-line").textContent = `已审核 ${p.reviewed} · 待审核 ${p.pending}`;
  }

  function visibleRows() {
    const status = $("f-status").value;
    const origin = $("f-origin").value;
    const risk = $("f-risk").value;
    const camera = $("f-camera").value;
    const search = $("f-search").value.trim().toLowerCase();
    return state.queue.filter((row) => {
      const reviewed = !!row.reviewed;
      if (status === "pending" && reviewed) return false;
      if (status !== "all" && status !== "pending" && row.localization_status !== status) return false;
      if (origin !== "all" && row.origin !== origin) return false;
      if (risk === "bad" && !row.likely_box_bad) return false;
      if (risk === "ok" && row.likely_box_bad) return false;
      if (camera !== "all" && row.camera_id !== camera) return false;
      if (search && !row.episode_id.toLowerCase().includes(search)) return false;
      return true;
    });
  }

  function renderQueue() {
    const rows = visibleRows();
    $("queue").innerHTML = rows.map((row) => {
      const badges = [];
      if (row.likely_box_bad) badges.push(`<span class="badge bad">CHECK</span>`);
      if (row.origin === "manual_missing_target") badges.push(`<span class="badge manual">POINT</span>`);
      if (row.origin === "box_wrong") badges.push(`<span class="badge wrong">BOX_WRONG</span>`);
      if (row.origin === "split_derived") badges.push(`<span class="badge">SPLIT</span>`);
      if (row.localization_status === "VERIFIED_BBOX") badges.push(`<span class="badge done">VERIFIED</span>`);
      if (row.localization_status === "LOCALIZATION_UNRESOLVED") badges.push(`<span class="badge unres">UNRESOLVED</span>`);
      if (row.localization_status === "TRUTH_REVIEW_REQUIRED") badges.push(`<span class="badge unres">TRUTH?</span>`);
      const active = state.current === row.episode_id ? " active" : "";
      return `<button class="item${active}" data-eid="${esc(row.episode_id)}">
        <span>${esc(row.camera_id)} · ${esc(row.source_timestamp || "")}</span>
        <small>${esc(row.episode_id)}</small>
        <span class="badges">${badges.join("")}</span>
      </button>`;
    }).join("") || '<p class="muted" style="padding:10px">没有符合条件的 episode</p>';
    document.querySelectorAll(".item").forEach((node) => {
      node.onclick = () => selectEpisode(node.dataset.eid);
    });
  }

  async function selectEpisode(episodeId) {
    state.current = episodeId;
    state.picked = null;
    state.note = "";
    state.message = "";
    state.error = "";
    try {
      state.detail = await api(`/api/episode?id=${encodeURIComponent(episodeId)}`);
      state.note = state.detail.note || "";
      if (state.detail.localization_decision === "PROPOSAL_SELECTED") {
        state.picked = state.detail.selected_proposal_id || null;
      }
    } catch (error) {
      state.detail = null;
      state.error = String(error.message || error);
    }
    renderQueue();
    renderDetail();
  }

  function move(delta) {
    const rows = visibleRows();
    if (!rows.length) return;
    const at = rows.findIndex((r) => r.episode_id === state.current);
    const next = at < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, at + delta));
    selectEpisode(rows[next].episode_id);
  }

  // ---------------------------------------------------------------- views

  function collectBoxes(detail) {
    const boxes = [];
    if (detail.original_bbox) {
      boxes.push({ cls: "orig", bbox: detail.original_bbox, label: "original" });
    }
    (detail.proposals || []).forEach((p) => {
      boxes.push({ cls: (PROPOSAL_CLASS[p.proposal_id] || "orig"), bbox: p.bbox,
                   label: p.proposal_id, proposal: p.proposal_id });
    });
    if (detail.verified_bbox && detail.verified_bbox_norm) {
      const [x1, y1, x2, y2] = detail.verified_bbox_norm;
      const w = detail.source_width, h = detail.source_height;
      boxes.push({ cls: "verif", bbox: [x1 * w, y1 * h, x2 * w, y2 * h], label: "verified" });
    }
    return boxes;
  }

  function roiFor(boxes, detail, mode) {
    const w = detail.source_width, h = detail.source_height;
    if (mode === "full") return { x: 0, y: 0, w: w, h: h };
    const pts = [];
    boxes.forEach((b) => { pts.push([b.bbox[0], b.bbox[1]], [b.bbox[2], b.bbox[3]]); });
    if (detail.original_point_source && detail.original_point_source.ok) {
      pts.push(detail.original_point_source.point_source);
    }
    if (!pts.length) return { x: 0, y: 0, w: w, h: h };
    let x1 = Math.min(...pts.map((p) => p[0])), y1 = Math.min(...pts.map((p) => p[1]));
    let x2 = Math.max(...pts.map((p) => p[0])), y2 = Math.max(...pts.map((p) => p[1]));
    const pad = Math.max(48, 0.35 * Math.max(x2 - x1, y2 - y1));
    x1 = Math.max(0, x1 - pad); y1 = Math.max(0, y1 - pad);
    x2 = Math.min(w, x2 + pad); y2 = Math.min(h, y2 + pad);
    return { x: x1, y: y1, w: Math.max(16, x2 - x1), h: Math.max(16, y2 - y1) };
  }

  function paintView(rootId, detail, mode) {
    const root = $(rootId);
    if (!root) return;
    const boxes = collectBoxes(detail);
    const roi = roiFor(boxes, detail, mode);
    const cw = root.clientWidth || 520;
    const ch = mode === "full" ? Math.round(cw * detail.source_height / detail.source_width) : 460;
    root.style.height = ch + "px";
    const k = Math.min(cw / roi.w, ch / roi.h);
    const padX = (cw - roi.w * k) / 2;
    const padY = (ch - roi.h * k) / 2;

    const img = root.querySelector("img");
    img.style.width = detail.source_width + "px";
    img.style.height = detail.source_height + "px";
    img.style.transform =
      `translate(${padX - k * roi.x}px, ${padY - k * roi.y}px) scale(${k})`;

    const layer = root.querySelector(".overlay");
    layer.innerHTML = boxes.map((b) => {
      const [x1, y1, x2, y2] = b.bbox;
      const left = padX + (x1 - roi.x) * k, top = padY + (y1 - roi.y) * k;
      const width = Math.max(2, (x2 - x1) * k), height = Math.max(2, (y2 - y1) * k);
      const chosen = state.picked && b.proposal === state.picked ? " sel" : "";
      const border = b.proposal ? Math.max(2, Math.min(6, k * 2)) : 2;
      return `<div class="box ${b.cls}${chosen}" style="left:${left}px;top:${top}px;
        width:${width}px;height:${height}px;border-width:${border}px"></div>`;
    }).join("") + (detail.original_point_source && detail.original_point_source.ok
      ? (() => {
          const [px, py] = detail.original_point_source.point_source;
          return `<div class="pt" style="left:${padX + (px - roi.x) * k}px;
            top:${padY + (py - roi.y) * k}px"></div>`;
        })() : "");
  }

  function paintAllViews() {
    if (!state.detail) return;
    paintView("view-full", state.detail, "full");
    paintView("view-zoom", state.detail, "zoom");
  }

  // ---------------------------------------------------------------- detail

  function renderDetail() {
    const detail = state.detail;
    if (!detail) {
      $("detail").innerHTML = `<p class="err">${esc(state.error || "加载失败")}</p>`;
      return;
    }
    const reasons = (detail.screen && detail.screen.screen_reasons) || [];
    const reasonHtml = reasons.length
      ? `<div class="reasons"><b>自动初筛（仅排序提示，不是判定）：</b> ${esc(reasons.join(", "))}</div>`
      : '<div class="reasons ok">自动初筛：无风险提示</div>';
    const proposalRows = detail.proposals || [];

    $("detail").innerHTML = `
      <div class="card">
        <div class="kv">
          <dt>episode_id</dt><dd><b>${esc(detail.episode_id)}</b></dd>
          <dt>camera_id</dt><dd>${esc(detail.camera_id)}</dd>
          <dt>timestamp</dt><dd>${esc(detail.source_timestamp)}</dd>
          <dt>origin</dt><dd>${esc(detail.origin)} ${detail.origin_flags.length > 1
            ? esc("(" + detail.origin_flags.join(", ") + ")") : ""}</dd>
          <dt>source</dt><dd>${detail.source_width}×${detail.source_height}</dd>
          <dt>历史定位</dt><dd>${esc(detail.original_location_type)} ·
            ${detail.original_bbox ? esc(JSON.stringify(detail.original_bbox.map(Math.round)))
              : '<span class="muted">无 bbox</span>'}</dd>
          <dt>人工 point</dt><dd>${detail.original_point_source && detail.original_point_source.ok
            ? `${detail.original_point_source.point_source.map((v) => v.toFixed(1)).join(", ")}
               <span class="muted">(crop→source 映射已校验)</span>`
            : '<span class="muted">无</span>'}</dd>
          <dt>当前 localization</dt><dd>${esc(detail.localization_status)}</dd>
        </div>
        ${reasonHtml}
      </div>

      <div class="card">
        <div class="views">
          <div>
            <div class="view" id="view-full"><img src="${esc(detail.frame_url)}" alt="source frame"><div class="overlay"></div></div>
            <div class="viewcap">source-native 全图（fit）· 蓝=原 bbox · 绿=verified · 绿十字=人工 point</div>
          </div>
          <div>
            <div class="view" id="view-zoom"><img src="${esc(detail.frame_url)}" alt="zoom"><div class="overlay"></div></div>
            <div class="viewcap">局部放大（ROI）</div>
          </div>
        </div>
      </div>

      ${proposalRows.length ? `<div class="card">
        <b>Machine proposal（仅定位辅助，不是真值来源）</b>
        <div class="cands">
          ${proposalRows.map((p) => `
            <button class="cand ${PROPOSAL_CLASS[p.proposal_id] || ""}${state.picked === p.proposal_id ? " sel" : ""}"
              data-pick="${esc(p.proposal_id)}">
              ${esc(p.proposal_id)}
              <small>${esc(p.method)}<br>${p.bbox.map((v) => Math.round(v)).join(", ")}</small>
            </button>`).join("")}
        </div>
        <p class="hint">按 A / B / C 选择，或点击候选。选出后按 <kbd>Enter</kbd> 或下方按钮保存。</p>
      </div>` : ""}

      <div class="card">
        <textarea id="note" rows="2" placeholder="可选备注 / TRUTH_REVIEW 原因">${esc(state.note)}</textarea>
        <p class="hint">
          <kbd>1</kbd> BOX_OK · <kbd>2</kbd> BOX_BAD → 生成 A/B/C · <kbd>A</kbd><kbd>B</kbd><kbd>C</kbd> 选择 ·
          <kbd>Enter</kbd> 保存选择 · <kbd>X</kbd> 都不对 · <kbd>T</kbd> TRUTH_REVIEW_REQUIRED ·
          <kbd>S</kbd> 跳过 · <kbd>R</kbd> 清除 · <kbd>←</kbd><kbd>→</kbd> 上下一条
        </p>
        <div class="${state.error ? "err" : "hint"}">${esc(state.error || state.message)}</div>
      </div>

      <div class="actions">
        <button class="act primary" data-act="BOX_OK"><kbd>1</kbd> BOX_OK 可训练</button>
        <button class="act warn" data-act="BOX_BAD"><kbd>2</kbd> BOX_BAD 需修复</button>
        ${state.picked ? '<button class="act primary" data-act="SAVE_PICK"><kbd>Enter</kbd> 保存所选</button>' : ""}
        <button class="act danger" data-act="UNRESOLVED"><kbd>X</kbd> 都不对 / 无法判断</button>
        <button class="act" data-act="TRUTH_REVIEW_REQUIRED"><kbd>T</kbd> TRUTH_REVIEW_REQUIRED</button>
        <button class="act" data-act="SKIP"><kbd>S</kbd> 跳过</button>
        <button class="act" data-act="RESET"><kbd>R</kbd> 清除本条</button>
      </div>`;

    bindDetail();
    const img = $("view-full") ? $("view-full").querySelector("img") : null;
    if (img) {
      if (img.complete) paintAllViews();
      img.onload = paintAllViews;
    }
    paintAllViews();
  }

  function bindDetail() {
    document.querySelectorAll("[data-pick]").forEach((node) => {
      node.onclick = () => { state.picked = node.dataset.pick; renderDetail(); };
    });
    document.querySelectorAll("[data-act]").forEach((node) => {
      node.onclick = () => action(node.dataset.act);
    });
    const note = $("note");
    if (note) note.oninput = () => { state.note = note.value; };
  }

  // ---------------------------------------------------------------- actions

  function currentNote() {
    const node = $("note");
    return node ? node.value.trim() : (state.note || "");
  }

  async function action(name) {
    const detail = state.detail;
    state.error = "";
    state.message = "";
    if (!detail) return;
    const episodeId = detail.episode_id;
    try {
      if (name === "BOX_BAD") {
        await post("/api/proposals", { episode_id: episodeId });
        await selectEpisode(episodeId);
        state.message = "已生成 A/B/C 候选，请选择（不会自动成为真值）";
        renderDetail();
        return;
      }
      if (name === "SAVE_PICK") {
        if (!state.picked) throw new Error("请先选择 A / B / C");
        await post("/api/decision", { episode_id: episodeId, decision: "PROPOSAL_SELECTED",
                                      proposal_id: state.picked, note: currentNote() });
        await advance(episodeId);
        return;
      }
      if (name === "BOX_OK") {
        await post("/api/decision", { episode_id: episodeId, decision: "BOX_OK",
                                      note: currentNote() });
        await advance(episodeId);
        return;
      }
      if (name === "UNRESOLVED") {
        await post("/api/decision", { episode_id: episodeId, decision: "UNRESOLVED",
                                      note: currentNote() });
        await advance(episodeId);
        return;
      }
      if (name === "TRUTH_REVIEW_REQUIRED") {
        const reason = window.prompt("TRUTH_REVIEW_REQUIRED 原因（必填）：", currentNote());
        if (!reason || !reason.trim()) return;
        await post("/api/decision", { episode_id: episodeId,
                                      decision: "TRUTH_REVIEW_REQUIRED",
                                      reason: reason.trim(), note: currentNote() });
        await advance(episodeId);
        return;
      }
      if (name === "SKIP") { move(1); return; }
      if (name === "RESET") {
        await post("/api/reset", { episode_id: episodeId });
        state.picked = null;
        await selectEpisode(episodeId);
        return;
      }
    } catch (error) {
      state.error = String(error.message || error);
      renderDetail();
    }
  }

  async function advance(fromEpisodeId) {
    try {
      const meta = await api("/api/meta");
      state.meta = meta;
      state.queue = meta.queue;
      renderProgress();
    } catch (error) {
      state.error = String(error.message || error);
    }
    const rows = visibleRows();
    const at = rows.findIndex((r) => r.episode_id === fromEpisodeId);
    const next = rows[at + 1] || rows[0];
    if (next) await selectEpisode(next.episode_id);
    else renderQueue();
  }

  // ---------------------------------------------------------------- bindings

  function bind() {
    ["f-status", "f-origin", "f-risk", "f-camera"].forEach((id) => {
      $(id).onchange = renderQueue;
    });
    $("f-search").oninput = renderQueue;
    window.addEventListener("resize", paintAllViews);
    document.addEventListener("keydown", (event) => {
      const tag = event.target.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") {
        if (event.key === "Enter" && tag === "TEXTAREA" && state.picked) {
          event.preventDefault();
          action("SAVE_PICK");
        }
        return;
      }
      if (event.key === "1") { event.preventDefault(); action("BOX_OK"); return; }
      if (event.key === "2") { event.preventDefault(); action("BOX_BAD"); return; }
      if (event.key === "Enter") { event.preventDefault(); action("SAVE_PICK"); return; }
      const pick = { a: "A", b: "B", c: "C", A: "A", B: "B", C: "C" }[event.key];
      if (pick) { event.preventDefault(); state.picked = pick; renderDetail(); return; }
      if (event.key === "x" || event.key === "X") { event.preventDefault(); action("UNRESOLVED"); return; }
      if (event.key === "t" || event.key === "T") { event.preventDefault(); action("TRUTH_REVIEW_REQUIRED"); return; }
      if (event.key === "s" || event.key === "S") { event.preventDefault(); action("SKIP"); return; }
      if (event.key === "r" || event.key === "R") { event.preventDefault(); action("RESET"); return; }
      if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); return; }
      if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
    });
  }

  bind();
  loadMeta().catch((error) => {
    document.body.innerHTML = `<pre class="err">${esc(error.message || error)}</pre>`;
  });
})();
