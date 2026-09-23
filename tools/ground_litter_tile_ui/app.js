/* Step 1C-2 annotation-completeness review.  No framework, no build step.
 *
 * The single question per tile is whether every Required Litter in this 640x640 tile
 * already has a usable box.  The page therefore offers no way to add, move, resize or
 * re-propose a box: a problem is reported, never fixed here.
 *
 * The tile PNG is served untouched; every overlay below is CSS drawn on top of it and
 * never written back to the image bytes.
 */
(() => {
  "use strict";

  const state = {
    meta: null, queue: [], current: null, tile: null,
    note: "", zoom: 1, focus: 0, message: "", error: "",
  };

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const KEYS = ["1", "2", "3", "4"];
  const ORDER = ["ANNOTATION_COMPLETE", "MISSING_REQUIRED", "BOX_PROBLEM",
                 "UNCERTAIN_COMPLETENESS"];
  const BADGE = { ANNOTATION_COMPLETE: "ok", MISSING_REQUIRED: "warn",
                  BOX_PROBLEM: "bad", UNCERTAIN_COMPLETENESS: "amb" };
  const SHORT = { ANNOTATION_COMPLETE: "COMPLETE", MISSING_REQUIRED: "MISSING",
                  BOX_PROBLEM: "BOX_PROBLEM", UNCERTAIN_COMPLETENESS: "UNCERTAIN" };

  async function api(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }
  const post = (path, body) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const pct = (v) => (100 * v / 640).toFixed(4) + "%";

  // ---------------------------------------------------------------- loading

  async function loadMeta(keep) {
    state.meta = await api("/api/meta");
    state.queue = state.meta.queue;
    const cameras = [...new Set(state.queue.map((r) => r.camera_id))].sort();
    $("f-camera").innerHTML = '<option value="all">全部</option>' +
      cameras.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    $("meta").textContent =
      `${state.meta.reviewable_count} 个可复核 tile · 640×640 source-native · ` +
      `候选 ${state.meta.candidate_count} · 生成阶段已排除 ` +
      `${state.meta.excluded_by_generation_count} · class 0 = ${state.meta.class_name}`;
    renderProgress();
    renderQueue();
    if (keep && state.current) return;
    const first = visibleRows()[0];
    if (first) selectTile(first.tile_id);
  }

  function renderProgress() {
    const p = state.meta.progress;
    $("progress").textContent = `${p.reviewed} / ${p.reviewable}`;
    $("status-line").textContent = `待复核 ${p.pending} · skip ${p.skipped}`;
  }

  function visibleRows() {
    const status = $("f-status").value;
    const camera = $("f-camera").value;
    const labels = $("f-labels").value;
    const search = $("f-search").value.trim().toLowerCase();
    return state.queue.filter((row) => {
      const reviewed = row.review_status && row.review_status !== "PENDING";
      if (status === "pending" && reviewed) return false;
      if (ORDER.includes(status) && row.review_status !== status) return false;
      if (camera !== "all" && row.camera_id !== camera) return false;
      if (labels === "1" && row.label_count !== 1) return false;
      if (labels === "2plus" && !(row.label_count >= 2)) return false;
      if (search && !(row.tile_id.toLowerCase().includes(search)
                      || row.primary_episode_id.toLowerCase().includes(search))) return false;
      return true;
    });
  }

  function renderQueue() {
    const rows = visibleRows();
    $("queue").innerHTML = rows.map((row) => {
      const badges = [];
      if (row.review_status && row.review_status !== "PENDING") {
        badges.push(`<span class="badge ${BADGE[row.review_status] || ""}">` +
                    `${esc(SHORT[row.review_status] || row.review_status)}</span>`);
      }
      const n = row.label_count || 0;
      badges.push(`<span class="badge${n >= 2 ? " amb" : ""}">${n} box</span>`);
      if (row.known_unlocalized_required_present) {
        badges.push('<span class="badge warn">UNLOCALIZED</span>');
      }
      const active = state.current === row.tile_id ? " active" : "";
      return `<button class="item${active}" data-tid="${esc(row.tile_id)}">
        <span>${esc(row.camera_id)} · ${esc(row.source_timestamp || "")}</span>
        <small>${esc(row.primary_episode_id)}</small>
        <span class="badges">${badges.join("")}</span>
      </button>`;
    }).join("") || '<p class="muted" style="padding:10px">没有符合条件的 tile</p>';
    document.querySelectorAll(".item").forEach((node) => {
      node.onclick = () => selectTile(node.dataset.tid);
    });
  }

  async function selectTile(tileId) {
    state.current = tileId;
    state.message = "";
    state.error = "";
    state.focus = 0;
    try {
      state.tile = await api(`/api/tile?id=${encodeURIComponent(tileId)}`);
      state.note = state.tile.annotation_review_note || "";
    } catch (error) {
      state.tile = null;
      state.error = String(error.message || error);
    }
    renderQueue();
    renderDetail();
  }

  function move(delta) {
    const rows = visibleRows();
    if (!rows.length) return;
    const at = rows.findIndex((r) => r.tile_id === state.current);
    const next = at < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, at + delta));
    selectTile(rows[next].tile_id);
  }

  // ---------------------------------------------------------------- overlays

  function overlayMarkup(tile) {
    return tile.labels.map((label, index) => {
      const [x1, y1, x2, y2] = label.tile_xyxy;
      const primary = label.episode_ids.includes(tile.primary_episode_id);
      return `<div class="box${primary ? " primary" : ""}"
        style="left:${pct(x1)};top:${pct(y1)};
               width:${pct(x2 - x1)};height:${pct(y2 - y1)}">
        <span class="tag">${esc(label.letter)}</span></div>`;
    }).join("");
  }

  function renderZoom() {
    const wrap = $("zoomwrap");
    if (!wrap || !state.tile) return;
    const img = wrap.querySelector("img");
    const overlay = wrap.querySelector(".overlay");
    const box = 640 / state.zoom;
    const labels = state.tile.labels;
    const focusLabel = labels[Math.min(state.focus, labels.length - 1)] || null;
    let cx = 320, cy = 320;
    if (focusLabel) {
      const [x1, y1, x2, y2] = focusLabel.tile_xyxy;
      cx = (x1 + x2) / 2; cy = (y1 + y2) / 2;
    }
    let ox = Math.max(0, Math.min(640 - box, cx - box / 2));
    let oy = Math.max(0, Math.min(640 - box, cy - box / 2));
    const size = wrap.clientWidth || 420;
    const scale = size / box;
    // The image is scaled to 640*k screen pixels and translated so that the view window
    // (ox, oy, box, box) fills the square container.  This is display-only scaling: the
    // served PNG is never resampled.
    img.style.width = (640 * scale) + "px";
    img.style.height = (640 * scale) + "px";
    img.style.transformOrigin = "0 0";
    img.style.transform = `translate(${-ox * scale}px, ${-oy * scale}px)`;
    overlay.style.transform = `translate(${-ox * scale}px, ${-oy * scale}px)`;
    overlay.innerHTML = labels.map((label) => {
      const [x1, y1, x2, y2] = label.tile_xyxy;
      return `<div class="box" style="left:${x1 * scale}px;top:${y1 * scale}px;
        width:${(x2 - x1) * scale}px;height:${(y2 - y1) * scale}px">
        <span class="tag">${esc(label.letter)}</span></div>`;
    }).join("");
    if (focusLabel) {
      const [x1, y1, x2, y2] = focusLabel.tile_xyxy;
      overlay.innerHTML += `<div class="box primary" style="left:${x1 * scale}px;
        top:${y1 * scale}px;width:${(x2 - x1) * scale}px;
        height:${(y2 - y1) * scale}px"></div>`;
    }
  }

  // ---------------------------------------------------------------- detail

  function renderDetail() {
    if (!state.tile) {
      $("detail").innerHTML = `<p class="err">${esc(state.error || "加载失败")}</p>`;
      return;
    }
    const tile = state.tile;
    const chosen = tile.annotation_review_status;
    const warnings = [];
    if (tile.known_unlocalized_required_present) {
      warnings.push(`同帧存在 KNOWN_UNLOCALIZED_REQUIRED：` +
        `${(tile.known_unlocalized_required_same_frame_ids || []).length} 个 Required ` +
        `没有可靠 bbox，且其位置与本 crop 相交（生成阶段已标记不可训练）`);
    } else if ((tile.known_unlocalized_required_same_frame_ids || []).length) {
      warnings.push(`同帧有 ${tile.known_unlocalized_required_same_frame_ids.length} 个 ` +
        `Required 没有可靠 bbox，但其位置可证明不在本 crop 内`);
    }
    if ((tile.ignore_small_in_crop_ids || []).length) {
      warnings.push(`crop 内有 IGNORE_SMALL ${tile.ignore_small_in_crop_ids.length} 个：` +
        `按 §15 不要求给它 bbox`);
    }
    if ((tile.non_litter_same_frame_ids || []).length) {
      warnings.push(`同帧 NON_LITTER ${tile.non_litter_same_frame_ids.length} 个（不标注）`);
    }
    if ((tile.uncertain_truth_in_crop_ids || []).length) {
      warnings.push(`crop 内有 UNCERTAIN/IDENTITY_AMBIGUOUS ` +
        `${tile.uncertain_truth_in_crop_ids.length} 个（Step 0B §6.4：不可训练）`);
    }
    const rows = tile.labels.map((label) => `<tr>
      <td><b>${esc(label.letter)}</b></td>
      <td>${esc(label.episode_ids.join(", "))}</td>
      <td>${esc(JSON.stringify(label.source_xyxy.map(Math.round)))}</td>
      <td>${esc(JSON.stringify(label.tile_xyxy.map(Math.round)))}</td>
      <td>${esc(label.yolo_xywh_norm.map((v) => v.toFixed(4)).join(" "))}</td>
      <td>${esc(label.source_short_side_px)} px (${esc(label.size_bucket)})</td>
    </tr>`).join("");

    $("detail").innerHTML = `
      <div class="question">
        <div class="label">本步骤唯一问题</div>
        <div class="value">这个 640×640 tile 里，所有需要稳定检测的 Required Litter
          是不是都已经有正确框？</div>
        <div class="hint">crop 是 source 原始像素切片，未 resize；框只作 UI 标识，
          不会写进训练图片。本页不能新增/修改框。</div>
      </div>

      ${warnings.length ? `<div class="warnbox"><div class="label">提示</div>
        ${warnings.map((w) => `<div>${esc(w)}</div>`).join("")}</div>` : ""}

      <div class="card">
        <div class="kv">
          <dt>tile_id</dt><dd><b>${esc(tile.tile_id)}</b></dd>
          <dt>primary_episode_id</dt><dd>${esc(tile.primary_episode_id)}</dd>
          <dt>全部 label episode</dt><dd>${esc((tile.all_known_label_episode_ids || []).join(", "))}</dd>
          <dt>camera_id</dt><dd>${esc(tile.camera_id)}</dd>
          <dt>timestamp（1C-0 / 1C-2）</dt>
          <dd>${esc(tile.step1c0_decoded_timestamp)} → ${esc(tile.step1c2_decoded_timestamp)}
            （Δ ${esc(tile.timestamp_delta_ms)} ms / 帧间隔 ${esc(tile.frame_interval_ms)} ms）</dd>
          <dt>source_file_id</dt><dd>${esc(tile.source_file_id)}</dd>
          <dt>source 尺寸</dt><dd>${esc(tile.source_width)}×${esc(tile.source_height)}</dd>
          <dt>crop source xyxy</dt><dd>${esc(JSON.stringify(tile.source_crop_xyxy))}</dd>
          <dt>primary source bbox</dt><dd>${esc(JSON.stringify(tile.primary_source_bbox))}</dd>
          <dt>最小 label 边距</dt><dd>${esc(tile.min_label_margin_px)} px</dd>
          <dt>label_count</dt><dd>${esc(tile.label_count)}</dd>
          <dt>image_sha256</dt><dd>${esc(tile.image_sha256)}</dd>
          <dt>当前结论</dt><dd>${chosen && chosen !== "PENDING"
            ? esc(chosen) + (tile.positive_training_ready ? " → positive_training_ready"
                                                          : " → 排除训练")
            : '<span class="muted">未复核</span>'}</dd>
        </div>
      </div>

      <div class="card">
        <div class="views">
          <div>
            <div class="tilewrap" id="tilewrap">
              <img src="${esc(tile.image_url)}" alt="640x640 source-native tile">
              <div class="overlay">${overlayMarkup(tile)}</div>
            </div>
            <div class="viewcap">1:1 source-native tile（640×640）· 绿=A(primary) · 蓝=其它已确认 Required</div>
          </div>
          <div>
            <div class="zoomwrap" id="zoomwrap">
              <img src="${esc(tile.image_url)}" alt="zoom">
              <div class="overlay"></div>
            </div>
            <div class="zoomctl">
              <span class="hint">放大</span>
              ${[1, 2, 4, 8].map((z) => `<button data-zoom="${z}"
                 class="${state.zoom === z ? "sel" : ""}">${z}×</button>`).join("")}
              <span class="hint">聚焦</span>
              ${tile.labels.map((label, index) => `<button data-focus="${index}"
                 class="${state.focus === index ? "sel" : ""}">${esc(label.letter)}</button>`)
                 .join("")}
            </div>
            <div class="viewcap">局部放大（不改变像素，只做显示放大）</div>
          </div>
        </div>
      </div>

      <div class="card">
        <table class="labels">
          <tr><th></th><th>episode</th><th>source xyxy</th><th>tile xyxy</th>
              <th>YOLO cx cy w h</th><th>短边</th></tr>
          ${rows || '<tr><td colspan="6" class="err">没有 label（不应出现）</td></tr>'}
        </table>
      </div>

      <div class="card">
        <textarea id="note" rows="2" placeholder="可选备注（MISSING_REQUIRED / BOX_PROBLEM / UNCERTAIN_COMPLETENESS 建议写一句）">${esc(state.note)}</textarea>
        <p class="hint">
          <kbd>1</kbd> COMPLETE（全部 Required 都有框） ·
          <kbd>2</kbd> MISSING_REQUIRED（有漏框 → 排除） ·
          <kbd>3</kbd> BOX_PROBLEM（现有框不可用 → 排除） ·
          <kbd>4</kbd> UNCERTAIN_COMPLETENESS（无法确认 → 排除） ·
          <kbd>S</kbd> 跳过 · <kbd>R</kbd> 清除 · <kbd>←</kbd><kbd>→</kbd> 上下一条
        </p>
        <div class="${state.error ? "err" : "hint"}">${esc(state.error || state.message)}</div>
      </div>

      <div class="actions">
        <button class="act complete${chosen === "ANNOTATION_COMPLETE" ? " sel" : ""}" data-act="ANNOTATION_COMPLETE"><kbd>1</kbd> COMPLETE</button>
        <button class="act missing${chosen === "MISSING_REQUIRED" ? " sel" : ""}" data-act="MISSING_REQUIRED"><kbd>2</kbd> MISSING_REQUIRED</button>
        <button class="act bad${chosen === "BOX_PROBLEM" ? " sel" : ""}" data-act="BOX_PROBLEM"><kbd>3</kbd> BOX_PROBLEM</button>
        <button class="act unc${chosen === "UNCERTAIN_COMPLETENESS" ? " sel" : ""}" data-act="UNCERTAIN_COMPLETENESS"><kbd>4</kbd> UNCERTAIN_COMPLETENESS</button>
        <button class="act" data-act="SKIP"><kbd>S</kbd> 跳过</button>
        <button class="act" data-act="RESET"><kbd>R</kbd> 清除本条</button>
      </div>`;

    document.querySelectorAll("[data-act]").forEach((node) => {
      node.onclick = () => action(node.dataset.act);
    });
    document.querySelectorAll("[data-zoom]").forEach((node) => {
      node.onclick = () => { state.zoom = Number(node.dataset.zoom); renderDetail(); };
    });
    document.querySelectorAll("[data-focus]").forEach((node) => {
      node.onclick = () => { state.focus = Number(node.dataset.focus); renderDetail(); };
    });
    const note = $("note");
    if (note) note.oninput = () => { state.note = note.value; };
    const img = $("tilewrap") ? $("tilewrap").querySelector("img") : null;
    if (img) { if (img.complete) renderZoom(); img.onload = renderZoom; }
    renderZoom();
  }

  // ---------------------------------------------------------------- actions

  function currentNote() {
    const node = $("note");
    return node ? node.value.trim() : (state.note || "");
  }

  async function action(name) {
    const tile = state.tile;
    state.error = "";
    state.message = "";
    if (!tile) return;
    try {
      if (name === "SKIP") { move(1); return; }
      if (name === "RESET") {
        const at = visibleRows().findIndex((r) => r.tile_id === tile.tile_id);
        await post("/api/reset", { tile_id: tile.tile_id });
        await advance(at, "已清除本条结论");
        return;
      }
      if (!ORDER.includes(name)) throw new Error(`unknown action ${name}`);
      const at = visibleRows().findIndex((r) => r.tile_id === tile.tile_id);
      await post("/api/review", { tile_id: tile.tile_id, decision: name,
                                  note: currentNote() });
      await advance(at);
    } catch (error) {
      state.error = String(error.message || error);
      renderDetail();
    }
  }

  async function advance(atIndex, message) {
    try {
      await loadMeta(true);
    } catch (error) {
      state.error = String(error.message || error);
    }
    const rows = visibleRows();
    if (!rows.length) { renderQueue(); return; }
    const at = atIndex < 0 ? 0 : Math.min(atIndex, rows.length - 1);
    await selectTile(rows[at].tile_id);
    if (message) { state.message = message; renderDetail(); }
  }

  function bind() {
    ["f-status", "f-camera", "f-labels"].forEach((id) => { $(id).onchange = renderQueue; });
    $("f-search").oninput = renderQueue;
    window.addEventListener("resize", renderZoom);
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
  loadMeta(false).catch((error) => {
    document.body.innerHTML = `<pre class="err">${esc(error.message || error)}</pre>`;
  });
})();
