/* Step 1C-1R truth reconciliation UI. No framework, no build step.
 *
 * This page decides TRUTH only.  It intentionally offers no localization controls:
 * no box verdict buttons, no candidate selection, no bbox pick, no point click.
 * Step 1C-1 localization is frozen and is shown as read-only context.
 *
 * The Step 1C-1 free-text reason is displayed prominently but is NEVER used to
 * pre-select or auto-classify anything — the reviewer clicks.
 */
(() => {
  "use strict";

  const state = {
    meta: null, queue: [], current: null, detail: null,
    reason: "", message: "", error: "",
  };

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const KEYS = ["1", "2", "3", "4", "5"];
  const ORDER = ["KEEP_REQUIRED", "IGNORE_SMALL", "NON_LITTER", "UNCERTAIN",
                 "IDENTITY_AMBIGUOUS"];
  const BADGE = { KEEP_REQUIRED: "keep", IGNORE_SMALL: "unc", NON_LITTER: "drop",
                  UNCERTAIN: "unc", IDENTITY_AMBIGUOUS: "amb" };

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
    const frozen = state.meta.frozen || {};
    $("meta").textContent =
      `TRUTH_REVIEW_REQUIRED ${state.meta.in_scope_count} 条 · ` +
      `Step 1C-1 已冻结不重审：${frozen.verified_bbox} VERIFIED_BBOX / ` +
      `${frozen.localization_unresolved} LOCALIZATION_UNRESOLVED`;
    renderProgress();
    renderQueue();
    const first = visibleRows()[0];
    if (first) selectEpisode(first.episode_id);
  }

  function renderProgress() {
    const p = state.meta.progress;
    $("progress").textContent = `${p.reviewed} / ${p.total}`;
    $("status-line").textContent = `已复核 ${p.reviewed} · 待复核 ${p.pending}`;
  }

  function visibleRows() {
    const status = $("f-status").value;
    const camera = $("f-camera").value;
    const origin = $("f-origin").value;
    const search = $("f-search").value.trim().toLowerCase();
    return state.queue.filter((row) => {
      if (status === "pending" && row.reviewed) return false;
      if (status === "decided" && !row.reviewed) return false;
      if (ORDER.includes(status) && row.reconciliation_decision !== status) return false;
      if (camera !== "all" && row.camera_id !== camera) return false;
      if (origin !== "all" && row.origin !== origin) return false;
      if (search && !row.episode_id.toLowerCase().includes(search)) return false;
      return true;
    });
  }

  function renderQueue() {
    const rows = visibleRows();
    $("queue").innerHTML = rows.map((row) => {
      const badges = [];
      if (row.reconciliation_decision) {
        badges.push(`<span class="badge ${BADGE[row.reconciliation_decision] || ""}">${esc(row.reconciliation_decision)}</span>`);
      }
      if (row.origin === "box_wrong") badges.push('<span class="badge drop">BOX_WRONG</span>');
      if (row.origin === "manual_missing_target") badges.push('<span class="badge amb">POINT</span>');
      if (row.origin === "split_derived") badges.push('<span class="badge">SPLIT</span>');
      const active = state.current === row.episode_id ? " active" : "";
      return `<button class="item${active}" data-eid="${esc(row.episode_id)}">
        <span>${esc(row.camera_id)} · ${esc(row.source_timestamp || "")}</span>
        <small>${esc(row.prior_truth_review_reason || "NO_PRIOR_REASON")}</small>
        <span class="badges">${badges.join("")}</span>
      </button>`;
    }).join("") || '<p class="muted" style="padding:10px">没有符合条件的 episode</p>';
    document.querySelectorAll(".item").forEach((node) => {
      node.onclick = () => selectEpisode(node.dataset.eid);
    });
  }

  async function selectEpisode(episodeId) {
    state.current = episodeId;
    state.message = "";
    state.error = "";
    try {
      state.detail = await api(`/api/episode?id=${encodeURIComponent(episodeId)}`);
      state.reason = state.detail.review_reason_optional || "";
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

  function roiFor(detail) {
    const w = detail.source_width, h = detail.source_height;
    const pts = [];
    if (detail.original_bbox) {
      pts.push([detail.original_bbox[0], detail.original_bbox[1]],
               [detail.original_bbox[2], detail.original_bbox[3]]);
    }
    if (detail.original_point_frame) {
      pts.push(detail.original_point_frame);
    } else if (detail.original_crop_box) {
      // the Step 1C-1 crop the click was made in is the only known geometry
      const b = detail.original_crop_box;
      pts.push([b[0], b[1]], [b[2], b[3]]);
    }
    if (!pts.length) return { x: 0, y: 0, w: w, h: h };
    let x1 = Math.min(...pts.map((p) => p[0])), y1 = Math.min(...pts.map((p) => p[1]));
    let x2 = Math.max(...pts.map((p) => p[0])), y2 = Math.max(...pts.map((p) => p[1]));
    const pad = Math.max(64, 0.6 * Math.max(x2 - x1, y2 - y1));
    x1 = Math.max(0, x1 - pad); y1 = Math.max(0, y1 - pad);
    x2 = Math.min(w, x2 + pad); y2 = Math.min(h, y2 + pad);
    return { x: x1, y: y1, w: Math.max(24, x2 - x1), h: Math.max(24, y2 - y1) };
  }

  function paintView(rootId, detail, mode) {
    const root = $(rootId);
    if (!root) return;
    const roi = mode === "full"
      ? { x: 0, y: 0, w: detail.source_width, h: detail.source_height } : roiFor(detail);
    const cw = root.clientWidth || 520;
    const ch = mode === "full"
      ? Math.round(cw * detail.source_height / detail.source_width) : 440;
    root.style.height = ch + "px";
    const k = Math.min(cw / roi.w, ch / roi.h);
    const padX = (cw - roi.w * k) / 2, padY = (ch - roi.h * k) / 2;
    const img = root.querySelector("img");
    img.style.width = detail.source_width + "px";
    img.style.height = detail.source_height + "px";
    img.style.transform = `translate(${padX - k * roi.x}px, ${padY - k * roi.y}px) scale(${k})`;
    const layer = root.querySelector(".overlay");
    let html = "";
    if (detail.original_bbox) {
      const [x1, y1, x2, y2] = detail.original_bbox;
      html += `<div class="box" style="left:${padX + (x1 - roi.x) * k}px;
        top:${padY + (y1 - roi.y) * k}px;width:${Math.max(2, (x2 - x1) * k)}px;
        height:${Math.max(2, (y2 - y1) * k)}px"></div>`;
    }
    if (detail.original_point_frame) {
      const p = detail.original_point_frame;
      html += `<div class="pt" title="历史点击位置（已映射到 source 像素）"
        style="left:${padX + (p[0] - roi.x) * k}px;
        top:${padY + (p[1] - roi.y) * k}px"></div>`;
    }
    layer.innerHTML = html;
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
    const chosen = detail.reconciliation_decision;
    $("detail").innerHTML = `
      <div class="prior">
        <div class="label">Step 1C-1 上一轮 TRUTH_REVIEW 理由（仅作参考，不参与自动判定）</div>
        <div class="value">${esc(detail.prior_truth_review_reason || "NO_PRIOR_REASON")}</div>
        <div class="hint">Step 1C-1 判断：${esc(detail.localization_status)}
          ${detail.localization_decision ? " / " + esc(detail.localization_decision) : ""}
          —— localization 已冻结，本步骤不修 bbox</div>
      </div>

      <div class="card">
        <div class="kv">
          <dt>episode_id</dt><dd><b>${esc(detail.episode_id)}</b></dd>
          <dt>camera_id</dt><dd>${esc(detail.camera_id)}</dd>
          <dt>timestamp</dt><dd>${esc(detail.source_timestamp)}</dd>
          <dt>origin</dt><dd>${esc(detail.origin)}</dd>
          <dt>原历史定位</dt><dd>${detail.original_bbox
            ? esc(JSON.stringify(detail.original_bbox.map(Math.round)))
            : (detail.original_point_frame
                ? "POINT → source " + esc(detail.original_point_frame.map(Math.round).join(","))
                : (detail.original_point
                    ? '<span class="err">POINT 无法映射到 source 帧（无 crop 几何），不绘制标记</span>'
                    : "无"))}</dd>
          <dt>当前 reconciliation</dt><dd>${chosen
            ? esc(chosen) + " → " + esc(detail.reconciled_truth_class === null
                ? "null（不伪造 truth class）" : detail.reconciled_truth_class)
            : '<span class="muted">未复核</span>'}</dd>
        </div>
      </div>

      <div class="card">
        <div class="views">
          <div>
            <div class="view" id="view-full"><img src="${esc(detail.frame_url)}" alt="source frame"><div class="overlay"></div></div>
            <div class="viewcap">source-native 全图 · 蓝=原历史 bbox · 绿十字=历史 point（已映射）</div>
          </div>
          <div>
            <div class="view" id="view-zoom"><img src="${esc(detail.frame_url)}" alt="zoom"><div class="overlay"></div></div>
            <div class="viewcap">局部放大</div>
          </div>
        </div>
      </div>

      <div class="card">
        <textarea id="reason" rows="2" placeholder="可选备注；选 IDENTITY_AMBIGUOUS 时必须填写简短理由">${esc(state.reason)}</textarea>
        <p class="hint">
          <kbd>1</kbd> KEEP_REQUIRED · <kbd>2</kbd> IGNORE_SMALL · <kbd>3</kbd> NON_LITTER ·
          <kbd>4</kbd> UNCERTAIN · <kbd>5</kbd> IDENTITY_AMBIGUOUS ·
          <kbd>S</kbd> 跳过 · <kbd>R</kbd> 清除 · <kbd>←</kbd><kbd>→</kbd> 上下一条
        </p>
        <div class="${state.error ? "err" : "hint"}">${esc(state.error || state.message)}</div>
      </div>

      <div class="actions">
        <button class="act keep${chosen === "KEEP_REQUIRED" ? " sel" : ""}" data-act="KEEP_REQUIRED"><kbd>1</kbd> KEEP_REQUIRED</button>
        <button class="act small${chosen === "IGNORE_SMALL" ? " sel" : ""}" data-act="IGNORE_SMALL"><kbd>2</kbd> IGNORE_SMALL</button>
        <button class="act non${chosen === "NON_LITTER" ? " sel" : ""}" data-act="NON_LITTER"><kbd>3</kbd> NON_LITTER</button>
        <button class="act unc${chosen === "UNCERTAIN" ? " sel" : ""}" data-act="UNCERTAIN"><kbd>4</kbd> UNCERTAIN</button>
        <button class="act amb${chosen === "IDENTITY_AMBIGUOUS" ? " sel" : ""}" data-act="IDENTITY_AMBIGUOUS"><kbd>5</kbd> IDENTITY_AMBIGUOUS</button>
        <button class="act" data-act="SKIP"><kbd>S</kbd> 跳过</button>
        <button class="act" data-act="RESET"><kbd>R</kbd> 清除本条</button>
      </div>`;

    document.querySelectorAll("[data-act]").forEach((node) => {
      node.onclick = () => action(node.dataset.act);
    });
    const reason = $("reason");
    if (reason) reason.oninput = () => { state.reason = reason.value; };
    const img = $("view-full") ? $("view-full").querySelector("img") : null;
    if (img) { if (img.complete) paintAllViews(); img.onload = paintAllViews; }
    paintAllViews();
  }

  // ---------------------------------------------------------------- actions

  function currentReason() {
    const node = $("reason");
    return node ? node.value.trim() : (state.reason || "");
  }

  async function action(name) {
    const detail = state.detail;
    state.error = "";
    state.message = "";
    if (!detail) return;
    const episodeId = detail.episode_id;
    try {
      if (name === "SKIP") { move(1); return; }
      if (name === "RESET") {
        const atIndex = visibleRows().findIndex((r) => r.episode_id === episodeId);
        await post("/api/reset", { episode_id: episodeId });
        await advance(atIndex, "已清除本条 reconciliation");
        return;
      }
      if (!ORDER.includes(name)) throw new Error(`unknown action ${name}`);
      // Remember the position, not the id: under the "未复核" filter the decided
      // episode disappears from the list, so keeping the index lands on the next one.
      const atIndex = visibleRows().findIndex((r) => r.episode_id === episodeId);
      let reason = currentReason();
      if (name === "IDENTITY_AMBIGUOUS" && !reason) {
        reason = window.prompt("IDENTITY_AMBIGUOUS 的简短理由（必填）：", "") || "";
        if (!reason.trim()) return;
        state.reason = reason.trim();
      }
      await post("/api/decision", { episode_id: episodeId, decision: name, reason });
      await advance(atIndex);
    } catch (error) {
      state.error = String(error.message || error);
      renderDetail();
    }
  }

  async function advance(atIndex, message) {
    try {
      const meta = await api("/api/meta");
      state.meta = meta;
      state.queue = meta.queue;
      renderProgress();
    } catch (error) {
      state.error = String(error.message || error);
    }
    const rows = visibleRows();
    if (!rows.length) { renderQueue(); return; }
    const at = atIndex < 0 ? 0 : Math.min(atIndex, rows.length - 1);
    await selectEpisode(rows[at].episode_id);
    if (message) { state.message = message; renderDetail(); }
  }

  function bind() {
    ["f-status", "f-camera", "f-origin"].forEach((id) => { $(id).onchange = renderQueue; });
    $("f-search").oninput = renderQueue;
    window.addEventListener("resize", paintAllViews);
    document.addEventListener("keydown", (event) => {
      const tag = event.target.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
      const index = KEYS.indexOf(event.key);
      if (index >= 0) { event.preventDefault(); action(ORDER[index]); return; }
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
