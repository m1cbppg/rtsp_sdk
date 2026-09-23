/* Step 1D hard-negative completeness review.  No framework, no build step.
 *
 * One question per tile: "does this 640x640 tile really contain no REQUIRED Litter?"
 * There is deliberately no way to create, adjust or promote a box: a tile that turns out
 * to contain litter is reported as REQUIRED_PRESENT and simply excluded.
 *
 * The anchor box shows where the historically mis-detected non-litter object is.  It is
 * context only -- never a training label.  The tile PNG is served untouched; every
 * overlay below is CSS drawn on top of it.
 */
(() => {
  "use strict";

  const state = {
    meta: null, queue: [], current: null, tile: null,
    note: "", message: "", error: "",
  };

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
  const pct = (v) => (100 * v / 640).toFixed(4) + "%";
  const round = (v) => Math.round(Number(v));

  const KEYS = ["1", "2", "3", "4"];
  const ORDER = ["NEGATIVE_OK", "REQUIRED_PRESENT", "UNCERTAIN", "BAD_CROP"];
  const BADGE = { NEGATIVE_OK: "ok", REQUIRED_PRESENT: "bad", UNCERTAIN: "warn",
                  BAD_CROP: "amb" };
  const SHORT = { NEGATIVE_OK: "NEG_OK", REQUIRED_PRESENT: "REQUIRED!",
                  UNCERTAIN: "UNCERTAIN", BAD_CROP: "BAD_CROP" };
  const RISK_LABEL = {
    KNOWN_REQUIRED_PRESENT: "已知 verified REQUIRED 与 crop 相交（生成阶段已排除，不应出现）",
    UNCERTAIN_TRUTH_IN_CROP: "tile 内有 UNCERTAIN / IDENTITY_AMBIGUOUS 目标：无法确认时请选 UNCERTAIN",
    NEARBY_VERIFIED_REQUIRED_OTHER_FRAME: "同一 PS 邻近帧有 verified REQUIRED 落在此 crop：请特别仔细确认",
    IGNORE_SMALL_IN_CROP: "tile 内有 IGNORE_SMALL（允许存在，不阻塞 negative）",
  };

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

  // ---------------------------------------------------------------- loading

  async function loadMeta(keep) {
    state.meta = await api("/api/meta");
    state.queue = state.meta.queue;
    const cameras = [...new Set(state.queue.map((r) => r.camera_id))].sort();
    $("f-camera").innerHTML = '<option value="all">全部</option>' +
      cameras.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    $("meta").textContent =
      `${state.meta.reviewable_count} 个可审核 negative candidate · 640×640 source-native · ` +
      `候选 ${state.meta.candidate_count} · 生成阶段已排除 ` +
      `${state.meta.excluded_by_generation_count}（已知 verified REQUIRED 与 crop 相交）`;
    renderProgress();
    renderQueue();
    if (keep && state.current) return;
    const first = visibleRows()[0];
    if (first) selectCandidate(first.negative_tile_id);
  }

  function renderProgress() {
    const p = state.meta.progress;
    $("progress").textContent = `${p.reviewed} / ${p.reviewable}`;
    $("status-line").textContent = `待审核 ${p.pending} · skip ${p.skipped}`;
  }

  function visibleRows() {
    const status = $("f-status").value;
    const camera = $("f-camera").value;
    const origin = $("f-origin").value;
    const search = $("f-search").value.trim().toLowerCase();
    return state.queue.filter((row) => {
      const reviewed = row.review_status && row.review_status !== "PENDING";
      if (status === "pending" && reviewed) return false;
      if (status !== "pending" && status !== "all"
          && row.review_status !== status) return false;
      if (camera !== "all" && row.camera_id !== camera) return false;
      if (origin !== "all" && row.origin !== origin) return false;
      if (search && !row.negative_tile_id.toLowerCase().includes(search)
          && !row.camera_id.includes(search)) return false;
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
      badges.push(`<span class="badge">${esc(row.hardness_source || "")}</span>`);
      if ((row.risk_flags || []).includes("UNCERTAIN_TRUTH_IN_CROP")) {
        badges.push('<span class="badge warn">UNCERTAIN</span>');
      }
      if ((row.risk_flags || []).includes("IGNORE_SMALL_IN_CROP")) {
        badges.push('<span class="badge">IGNORE_SMALL</span>');
      }
      const active = state.current === row.negative_tile_id ? " active" : "";
      return `<button class="item${active}" data-cid="${esc(row.negative_tile_id)}">
        <span>${esc(row.camera_id)} · ${esc(row.timestamp || "")}</span>
        <small>${esc(row.negative_tile_id)}</small>
        <span class="badges">${badges.join("")}</span>
      </button>`;
    }).join("") || '<p class="muted" style="padding:10px">没有符合条件的 candidate</p>';
    document.querySelectorAll(".item").forEach((node) => {
      node.onclick = () => selectCandidate(node.dataset.cid);
    });
  }

  async function selectCandidate(candidateId) {
    state.current = candidateId;
    state.message = "";
    state.error = "";
    try {
      state.tile = await api(`/api/candidate?id=${encodeURIComponent(candidateId)}`);
      state.note = state.tile.review_reason || "";
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
    const at = rows.findIndex((r) => r.negative_tile_id === state.current);
    const next = at < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, at + delta));
    selectCandidate(rows[next].negative_tile_id);
  }

  // ---------------------------------------------------------------- detail

  function renderDetail() {
    if (!state.tile) {
      $("detail").innerHTML = `<p class="err">${esc(state.error || "加载失败")}</p>`;
      return;
    }
    const tile = state.tile;
    const chosen = tile.review_status;
    const risk = tile.risk_flags || [];
    const anchor = tile.anchor_tile_xyxy;

    const warnings = [];
    risk.forEach((flag) => warnings.push(RISK_LABEL[flag] || flag));
    const errbox = state.error
      ? `<div class="errbox"><div class="label">错误</div>${esc(state.error)}</div>` : "";

    $("detail").innerHTML = `
      <div class="question">
        <div class="label">本步骤唯一问题</div>
        <div class="value">这个 640×640 tile 里，是否真的不存在需要稳定检测的 REQUIRED_LITTER？</div>
        <div class="hint">小碎屑 / IGNORE_SMALL <b>可以存在</b>；只要能看到任何需要检测的垃圾，
          就选 REQUIRED_PRESENT。本页不能画框、不能把它转成正例。</div>
      </div>

      ${warnings.length ? `<div class="warnbox"><div class="label">提示</div>
        ${warnings.map((w) => `<div>${esc(w)}</div>`).join("")}</div>` : ""}
      ${errbox}

      <div class="card">
        <div class="kv">
          <dt>negative_tile_id</dt><dd><b>${esc(tile.negative_tile_id)}</b></dd>
          <dt>camera / 时间</dt><dd>${esc(tile.camera_id)} · ${esc(tile.timestamp)}
            → decoded ${esc(tile.decoded_timestamp)}（Δ ${esc(tile.timestamp_delta_ms)} ms）</dd>
          <dt>source_file_id</dt><dd>${esc(tile.source_file_id)}</dd>
          <dt>origin / 历史来源</dt><dd>${esc(tile.origin)} · ${esc(tile.historical_label)}
            · ${esc(tile.historical_source)}</dd>
          <dt>hardness_source</dt><dd>${esc(tile.hardness_source)}</dd>
          <dt>source card</dt><dd>${esc(tile.source_card_id)}（${esc(tile.source_review_batch)}）</dd>
          <dt>历史审核备注</dt><dd>${esc(tile.note || "—")}</dd>
          <dt>crop source xyxy</dt><dd>${esc(JSON.stringify(tile.source_crop_xyxy))}</dd>
          <dt>anchor source xyxy</dt><dd>${esc(JSON.stringify(tile.anchor_source_xyxy))}
            · tile ${esc(JSON.stringify(anchor))} · 边距 ${esc(tile.anchor_min_margin_px)} px</dd>
          <dt>source 尺寸</dt><dd>${esc(tile.source_width)}×${esc(tile.source_height)}</dd>
          <dt>当前结论</dt><dd>${chosen && chosen !== "PENDING"
            ? esc(chosen) + (tile.hard_negative_ready ? " → hard_negative_ready" : " → 排除")
            : '<span class="muted">未审核</span>'}</dd>
          <dt>image sha256</dt><dd>${esc(tile.image_sha256)}</dd>
        </div>
      </div>

      <div class="card">
        <div class="tilewrap">
          <img src="${esc(tile.image_url)}" alt="640x640 source-native negative tile">
          <div class="overlay">
            ${anchor ? `<div class="anchor" style="left:${pct(anchor[0])};top:${pct(anchor[1])};
              width:${pct(anchor[2] - anchor[0])};height:${pct(anchor[3] - anchor[1])}">
              <span class="tag">历史上误识别的是这里（非训练标签）</span></div>` : ""}
          </div>
        </div>
        <div class="viewcap">640×640 source-native tile · 橙色虚线 = 历史 NON_LITTER anchor（仅提示，不是 label）</div>
      </div>

      <div class="card">
        <textarea id="note" rows="2" placeholder="REQUIRED_PRESENT / UNCERTAIN / BAD_CROP 需要一句简短原因；NEGATIVE_OK 可留空">${esc(state.note)}</textarea>
        <p class="hint">
          <kbd>1</kbd> NEGATIVE_OK（整 tile 无 REQUIRED_LITTER） ·
          <kbd>2</kbd> REQUIRED_PRESENT（有垃圾 → 排除） ·
          <kbd>3</kbd> UNCERTAIN（无法可靠确认 → 排除） ·
          <kbd>4</kbd> BAD_CROP（anchor 没保留 / crop 或映射有问题 → 排除） ·
          <kbd>S</kbd> 跳过 · <kbd>R</kbd> 清除 · <kbd>←</kbd><kbd>→</kbd> 上下一条
        </p>
        <div class="${state.error ? "err" : "hint"}">${esc(state.message)}</div>
      </div>

      <div class="actions">
        <button class="act ok${chosen === "NEGATIVE_OK" ? " sel" : ""}" data-act="NEGATIVE_OK"><kbd>1</kbd> NEGATIVE_OK</button>
        <button class="act bad${chosen === "REQUIRED_PRESENT" ? " sel" : ""}" data-act="REQUIRED_PRESENT"><kbd>2</kbd> REQUIRED_PRESENT</button>
        <button class="act unc${chosen === "UNCERTAIN" ? " sel" : ""}" data-act="UNCERTAIN"><kbd>3</kbd> UNCERTAIN</button>
        <button class="act crop${chosen === "BAD_CROP" ? " sel" : ""}" data-act="BAD_CROP"><kbd>4</kbd> BAD_CROP</button>
        <button class="act" data-act="SKIP"><kbd>S</kbd> 跳过</button>
        <button class="act" data-act="RESET"><kbd>R</kbd> 清除本条</button>
      </div>`;

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
    const tile = state.tile;
    state.error = "";
    state.message = "";
    if (!tile) return;
    try {
      if (name === "SKIP") { move(1); return; }
      if (name === "RESET") {
        const at = visibleRows().findIndex((r) => r.negative_tile_id === tile.negative_tile_id);
        await post("/api/reset", { candidate_id: tile.negative_tile_id });
        await advance(at, "已清除本条结论");
        return;
      }
      if (!ORDER.includes(name)) throw new Error(`unknown action ${name}`);
      const reason = currentNote();
      if (name !== "NEGATIVE_OK" && !reason) {
        state.error = `${name} 需要一句简短原因`;
        renderDetail();
        return;
      }
      const at = visibleRows().findIndex((r) => r.negative_tile_id === tile.negative_tile_id);
      await post("/api/review", { candidate_id: tile.negative_tile_id, decision: name,
                                  reason });
      await advance(at, name === "NEGATIVE_OK" ? "已确认 negative" : `已排除（${name}）`);
    } catch (error) {
      state.error = String(error.message || error);
      renderDetail();
    }
  }

  async function advance(atIndex, message) {
    try { await loadMeta(true); } catch (error) { state.error = String(error.message || error); }
    const rows = visibleRows();
    if (!rows.length) { renderQueue(); return; }
    const at = atIndex < 0 ? 0 : Math.min(atIndex, rows.length - 1);
    await selectCandidate(rows[at].negative_tile_id);
    if (message) { state.message = message; renderDetail(); }
  }

  function bind() {
    ["f-status", "f-camera", "f-origin"].forEach((id) => { $(id).onchange = renderQueue; });
    $("f-search").oninput = renderQueue;
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
