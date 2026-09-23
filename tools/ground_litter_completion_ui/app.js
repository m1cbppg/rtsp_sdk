/* Step 1C-2M supplemental-target completion.  No framework, no build step.
 *
 * The reviewer clicks the centre of a missed Required Litter, the machine offers at
 * most three boxes (A/B/C), and the reviewer picks one.  There is deliberately no way
 * to create, adjust, resize or modify a box, and the original verified boxes are
 * read-only.
 *
 * The tile PNG is served untouched; every overlay below is CSS drawn on top of it.
 */
(() => {
  "use strict";

  const state = {
    meta: null, queue: [], current: null, tile: null,
    note: "", message: "", error: "", warning: null,
    armed: false, pending: false, activeTarget: null, confirm: null,
    elapsed: null,
  };

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
  const pct = (v) => (100 * v / 640).toFixed(4) + "%";
  const round = (v) => Math.round(Number(v));

  const KEYS = { "a": "A", "b": "B", "c": "C" };

  async function api(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) {
      const error = new Error(payload.message || payload.error || `HTTP ${response.status}`);
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

  async function loadMeta(keep) {
    state.meta = await api("/api/meta");
    state.queue = state.meta.queue;
    const cameras = [...new Set(state.queue.map((r) => r.camera_id))].sort();
    $("f-camera").innerHTML = '<option value="all">全部</option>' +
      cameras.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    $("meta").textContent =
      `${state.meta.queue_count} 个 MISSING_REQUIRED tile · 只补 REQUIRED_LITTER · ` +
      `proposal 尺寸先验 ${round(state.meta.size_prior.width)}×` +
      `${round(state.meta.size_prior.height)} px` +
      `（${state.meta.size_prior.sample_count} 个已 verified label）`;
    renderProgress();
    renderQueue();
    if (keep && state.current) return;
    const first = visibleRows()[0];
    if (first) selectTile(first.tile_id);
  }

  function renderProgress() {
    const p = state.meta.progress;
    $("progress").textContent = `${p.reviewed} / ${p.queue}`;
    $("status-line").textContent = `待处理 ${p.pending} · skip ${p.skipped}`;
  }

  function visibleRows() {
    const status = $("f-status").value;
    const camera = $("f-camera").value;
    const labels = $("f-labels").value;
    const search = $("f-search").value.trim().toLowerCase();
    return state.queue.filter((row) => {
      const done = row.completion_status !== "NEEDS_COMPLETION";
      if (status === "pending" && done) return false;
      if (status !== "pending" && status !== "all"
          && row.completion_status !== status) return false;
      if (camera !== "all" && row.camera_id !== camera) return false;
      if (labels === "1" && row.existing_label_count !== 1) return false;
      if (labels === "2" && row.existing_label_count !== 2) return false;
      if (labels === "3plus" && !(row.existing_label_count >= 3)) return false;
      if (search && !(row.tile_id.toLowerCase().includes(search)
                      || row.primary_episode_id.toLowerCase().includes(search))) return false;
      return true;
    });
  }

  function renderQueue() {
    const rows = visibleRows();
    $("queue").innerHTML = rows.map((row) => {
      const badges = [`<span class="badge">${row.existing_label_count} label</span>`];
      if (row.supplemental_count) {
        badges.push(`<span class="badge new">+${row.supplemental_verified}/${row.supplemental_count}</span>`);
      }
      if (row.completion_status && row.completion_status !== "NEEDS_COMPLETION") {
        const cls = row.completion_status === "ANNOTATION_COMPLETE_AFTER_SUPPLEMENT"
          ? "ok" : (row.completion_status === "STILL_MISSING_REQUIRED" ? "warn" : "bad");
        badges.push(`<span class="badge ${cls}">${esc(row.completion_status.replace(/_/g, " "))}</span>`);
      }
      if (row.known_unlocalized_required_present) {
        badges.push('<span class="badge warn">§17 风险</span>');
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
    state.warning = null;
    state.armed = false;
    state.activeTarget = null;
    state.confirm = null;
    state.elapsed = null;
    try {
      state.tile = await api(`/api/tile?id=${encodeURIComponent(tileId)}`);
      state.note = state.tile.completion_note || "";
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

  function boxMarkup(box, cls, tag) {
    return `<div class="box ${cls}" style="left:${pct(box[0])};top:${pct(box[1])};
      width:${pct(box[2] - box[0])};height:${pct(box[3] - box[1])}">
      ${tag ? `<span class="tag">${esc(tag)}</span>` : ""}</div>`;
  }

  function overlayMarkup(tile) {
    let html = "";
    (tile.existing_labels || []).forEach((label) => {
      html += boxMarkup(label.tile_xyxy, "", label.letter);
    });
    (tile.supplemental_targets || []).forEach((target, index) => {
      if (target.verified_tile_xyxy) {
        const trunc = target.localization_status === "TARGET_TRUNCATED_BY_TILE";
        html += boxMarkup(target.verified_tile_xyxy, trunc ? "trunc" : "new",
                          `S${index + 1}`);
      }
      const click = target.current_click || target.original_click;
      if (click && state.activeTarget === target.supplemental_target_id) {
        html += `<div class="point" style="left:${pct(click.tile_x)};top:${pct(click.tile_y)}"></div>`;
      }
    });
    const active = (tile.supplemental_targets || [])
      .find((t) => t.supplemental_target_id === state.activeTarget);
    if (active && !active.selected_proposal) {
      (active.proposal_candidates || []).forEach((candidate) => {
        html += boxMarkup(candidate.bbox_tile_xyxy, "cand", candidate.letter);
      });
    }
    return html;
  }

  function tilePointFromEvent(event) {
    const wrap = $("tilewrap");
    const img = wrap.querySelector("img");
    const rect = img.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width * 640;
    const y = (event.clientY - rect.top) / rect.height * 640;
    return [Math.max(0, Math.min(639, x)), Math.max(0, Math.min(639, y))];
  }

  // ---------------------------------------------------------------- detail

  function renderDetail() {
    if (!state.tile) {
      $("detail").innerHTML = `<p class="err">${esc(state.error || "加载失败")}</p>`;
      return;
    }
    const tile = state.tile;
    const chosen = tile.completion_status;
    const targets = tile.supplemental_targets || [];
    const verified = targets.filter((t) => t.localization_status === "VERIFIED_BBOX");
    const unresolved = targets.filter((t) => t.localization_status ===
      "SUPPLEMENTAL_LOCALIZATION_UNRESOLVED");
    const truncated = targets.filter((t) => t.localization_status === "TARGET_TRUNCATED_BY_TILE");

    const labelRows = (tile.merged_labels || []).map((label) => {
      const supp = label.origin === "step1c2m_supplemental";
      return `<tr class="${supp ? "supp" : ""}">
        <td>${supp ? "S" : "A/B/C"} <b>${esc(label.label_id)}</b></td>
        <td>${esc((label.episode_ids || []).join(", ") || (label.supplemental_target_ids || []).join(", "))}</td>
        <td>${esc(JSON.stringify(label.tile_xyxy.map(round)))}</td>
        <td>${esc(JSON.stringify(label.source_xyxy.map(round)))}</td>
        <td>${esc(label.yolo_xywh_norm.map((v) => v.toFixed(4)).join(" "))}</td>
        <td>${esc(label.source_short_side_px)} px (${esc(label.size_bucket)})</td>
      </tr>`;
    }).join("");

    const activeTarget = targets.find((t) => t.supplemental_target_id === state.activeTarget);
    const candidateRows = activeTarget ? (activeTarget.proposal_candidates || []).map((c) => `
      <div class="candrow ${activeTarget.selected_proposal === c.letter ? "sel" : ""}">
        <button data-pick="${esc(c.letter)}" data-target="${esc(activeTarget.supplemental_target_id)}">
          ${esc(c.letter)}</button>
        <span class="meta">${esc(c.method)}<br>
          ${esc(JSON.stringify(c.bbox_tile_xyxy.map(round)))} px
          ${c.contains_point ? "" : "<b>（不含点击点）</b>"}
          ${c.touches_border ? `<b>（接触 tile 边缘：${esc(c.touches_border)}）</b>` : ""}</span>
      </div>`).join("") : "";

    const targetRows = targets.map((target, index) => `
      <tr class="${target.localization_status === "VERIFIED_BBOX" ? "supp" : ""}">
        <td>S${index + 1} <b>${esc(target.supplemental_target_id)}</b></td>
        <td>click tile (${round(target.original_click.tile_x)}, ${round(target.original_click.tile_y)})
          → source (${round(target.original_click.source_x)}, ${round(target.original_click.source_y)})</td>
        <td>rev ${esc(target.proposal_revision)} · ${esc(target.selected_proposal || "未选择")}</td>
        <td>${esc(target.localization_status || "未定位")}${target.truncated_sides && target.truncated_sides.length ? " · " + esc(target.truncated_sides.join("/")) : ""}</td>
        <td>${target.verified_source_xyxy ? esc(JSON.stringify(target.verified_source_xyxy.map(round))) : "—"}</td>
        <td>
          <button data-repoint="${esc(target.supplemental_target_id)}">重新点中心</button>
          <button data-delete="${esc(target.supplemental_target_id)}">删除</button>
        </td>
      </tr>`).join("");

    const warnings = [];
    if (tile.known_unlocalized_required_present) {
      warnings.push("本 tile 带 §17 风险标记：同帧存在未定位的 REQUIRED_LITTER（历史 bbox / manual point），" +
        "可能落在本 crop；补标完成后仍需据实选择 COMPLETE / UNCERTAIN。");
    }
    if (state.warning) {
      warnings.push(`${state.warning.warning}：${state.warning.reason || ""} ` +
        `${(state.warning.matches || []).map((m) => `${m.label_id}(IoU ${m.iou})`).join("、")}` +
        `${(state.warning.sides || []).length ? " 接触边界：" + state.warning.sides.join("/") : ""}`);
    }
    const errbox = state.error
      ? `<div class="errbox"><div class="label">错误</div>${esc(state.error)}</div>` : "";

    $("detail").innerHTML = `
      <div class="question">
        <div class="label">本步骤唯一任务</div>
        <div class="value">在这个 640×640 tile 里，把“希望模型稳定检测”的遗漏 REQUIRED_LITTER 补上框。</div>
        <div class="hint">只补 REQUIRED_LITTER；小碎屑 / IGNORE_SMALL 不需要框。你不画框：点目标中心 →
          机器给 A/B/C → 你选一个。crop 不移动，图片不修改，原 verified 框只读。</div>
      </div>

      ${warnings.length ? `<div class="warnbox"><div class="label">提示</div>
        ${warnings.map((w) => `<div>${esc(w)}</div>`).join("")}</div>` : ""}
      ${errbox}

      <div class="card">
        <div class="kv">
          <dt>tile_id</dt><dd><b>${esc(tile.tile_id)}</b></dd>
          <dt>primary_episode_id</dt><dd>${esc(tile.primary_episode_id)}</dd>
          <dt>camera / 时间</dt><dd>${esc(tile.camera_id)} · ${esc(tile.step1c0_decoded_timestamp)}</dd>
          <dt>crop source xyxy</dt><dd>${esc(JSON.stringify(tile.source_crop_xyxy))}</dd>
          <dt>已有 verified label</dt><dd>${esc(tile.existing_label_count)}</dd>
          <dt>supplemental</dt><dd>${targets.length}（已定位 ${verified.length} · 未定位 ${unresolved.length} · 截断 ${truncated.length}）</dd>
          <dt>合并后 label 数</dt><dd>${esc(tile.final_label_count)}</dd>
          <dt>Step 1C-2 复核备注</dt><dd>${esc(tile.review_note || "—")}</dd>
          <dt>当前完成状态</dt><dd>${chosen && chosen !== "NEEDS_COMPLETION"
            ? esc(chosen) + (tile.positive_training_ready ? " → 可进入 accepted_v2" : " → 排除")
            : '<span class="muted">未处理</span>'}</dd>
          <dt>图片 sha256</dt><dd>${esc(tile.image_sha256)}</dd>
        </div>
      </div>

      <div class="tilearea">
        <div>
          <div class="tilewrap${state.armed ? " armed" : ""}" id="tilewrap">
            <img src="${esc(tile.image_url)}" alt="640x640 source-native tile">
            <div class="overlay">${overlayMarkup(tile)}</div>
          </div>
          <div class="viewcap hint">640×640 source-native tile · 绿=A/B/C 原 verified 框（只读）·
            蓝=S 新增框 · 紫虚线=proposal 候选 · 黄十字=点击点</div>
          ${state.pending ? '<div class="loading">正在生成候选…</div>' : ""}
          ${state.elapsed !== null ? `<div class="hint">proposal 用时 ${esc(state.elapsed)} ms</div>` : ""}
        </div>
        <div class="side">
          <div class="card" style="margin:0">
            <div class="hint">当前操作</div>
            <div><b>${state.armed ? "请点击遗漏垃圾的中心" :
              (activeTarget ? "选择 A/B/C，或重新点中心" : "点击『新增遗漏 REQUIRED』开始")}</b></div>
            ${candidateRows ? `<div class="candlist" style="margin-top:6px">${candidateRows}</div>` : ""}
            ${activeTarget ? `<div class="hint" style="margin-top:6px">
              target ${esc(activeTarget.supplemental_target_id)} · revision
              ${esc(activeTarget.proposal_revision)}/${esc(tile.max_proposal_revisions)}</div>` : ""}
            ${activeTarget && !activeTarget.selected_proposal ? `
              <div style="margin-top:6px">
                <button class="act bad" data-reject="${esc(activeTarget.supplemental_target_id)}">X 三个都不对</button>
              </div>` : ""}
          </div>
          <div class="card" style="margin:0">
            <div class="hint">supplemental targets</div>
            <table class="labels">
              <tr><th>id</th><th>点击</th><th>proposal</th><th>状态</th><th>source bbox</th><th></th></tr>
              ${targetRows || '<tr><td colspan="6" class="muted">还没有新增</td></tr>'}
            </table>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="hint">最终 labels（原 verified ∪ 人工选择的 supplemental，已按 IoU≥0.95 去重）</div>
        <table class="labels">
          <tr><th>来源</th><th>provenance</th><th>tile xyxy</th><th>source xyxy</th>
              <th>YOLO cx cy w h</th><th>短边</th></tr>
          ${labelRows}
        </table>
      </div>

      <div class="card">
        <textarea id="note" rows="2" placeholder="可选备注">${esc(state.note)}</textarea>
        ${state.confirm ? `
          <div class="confirmbox">
            <div class="label">确认</div>
            <div>${esc(state.confirm.text)}</div>
            <div style="margin-top:6px">
              <button class="act complete" id="confirm-yes">确认</button>
              <button class="act" id="confirm-no">取消</button>
            </div>
          </div>` : ""}
        <p class="hint">
          <kbd>N</kbd> 新增遗漏 REQUIRED · <kbd>A</kbd><kbd>B</kbd><kbd>C</kbd> 选择 proposal ·
          <kbd>X</kbd> 三个都不对 · <kbd>D</kbd> 删除当前 · <kbd>P</kbd> 重新点中心 ·
          <kbd>←</kbd><kbd>→</kbd> 上下一条
        </p>
        <div class="${state.error ? "err" : "hint"}">${esc(state.message)}</div>
      </div>

      <div class="actions">
        <button class="act${state.armed ? " armed" : ""}" id="btn-add" ${targets.length >= tile.max_supplemental_per_tile ? "disabled" : ""}>
          <kbd>N</kbd> 新增遗漏 REQUIRED</button>
        <button class="act" id="btn-clear" ${targets.length ? "" : "disabled"}><kbd>R</kbd> 清除本 tile 全部新增</button>
        <button class="act complete" id="btn-complete" ${verified.length && !unresolved.length && !truncated.length ? "" : "disabled"}>
          <kbd>1</kbd> RECHECK COMPLETE</button>
        <button class="act missing" id="btn-stillmissing"><kbd>2</kbd> STILL_MISSING</button>
        <button class="act bad" id="btn-unresolved"><kbd>3</kbd> UNRESOLVED</button>
        <button class="act unc" id="btn-uncertain"><kbd>4</kbd> UNCERTAIN</button>
        <button class="act" id="btn-skip"><kbd>S</kbd> 跳过</button>
      </div>`;

    bindDetail();
    const img = $("tilewrap") ? $("tilewrap").querySelector("img") : null;
    if (img) img.onload = () => { state.overlayReady = true; };
  }

  // ---------------------------------------------------------------- actions

  function currentNote() {
    const node = $("note");
    return node ? node.value.trim() : (state.note || "");
  }

  function setBusy(flag) {
    state.pending = flag;
    const line = document.querySelector(".loading");
    if (flag && !line) {
      const wrap = $("tilewrap");
      if (wrap) wrap.insertAdjacentHTML("afterend",
        '<div class="loading">正在生成候选…</div>');
    } else if (!flag) {
      document.querySelectorAll(".loading").forEach((n) => n.remove());
    }
  }

  async function requestPoint(tileX, tileY, targetId) {
    const tile = state.tile;
    if (!tile) return;
    setBusy(true);
    state.error = "";
    state.warning = null;
    state.message = "";
    try {
      const body = { tile_id: tile.tile_id, tile_x: tileX, tile_y: tileY };
      if (targetId) body.target_id = targetId;
      const reply = await post("/api/point", body);
      state.elapsed = reply.elapsed_ms;
      state.activeTarget = reply.supplemental_target_id;
      state.armed = false;
      state.message = reply.candidates.length
        ? "已生成候选，请选择 A/B/C（不会自动保存）"
        : "未生成候选";
      await refreshTile();
    } catch (error) {
      state.armed = false;
      state.error = String(error.payload && error.payload.error_code
        ? `${error.payload.error_code}: ${error.payload.message}` : error.message || error);
      setBusy(false);
      renderDetail();
    }
  }

  async function refreshTile(message) {
    const tileId = state.tile ? state.tile.tile_id : null;
    if (!tileId) return;
    const active = state.activeTarget;
    try {
      state.tile = await api(`/api/tile?id=${encodeURIComponent(tileId)}`);
      state.note = state.tile.completion_note || state.note;
      state.activeTarget = active;
      await loadMeta(true);
    } catch (error) {
      state.error = String(error.message || error);
    }
    if (message) state.message = message;
    renderDetail();
  }

  async function pick(letter, targetId) {
    state.error = "";
    state.warning = null;
    try {
      const reply = await post("/api/select", {
        tile_id: state.tile.tile_id, target_id: targetId, letter,
      });
      if (reply.warning === "POSSIBLE_DUPLICATE_TARGET") {
        state.warning = reply;
        state.confirm = {
          kind: "duplicate", letter, targetId,
          text: "该 bbox 与已有目标高度重叠（或点击点落在已有框内）。请确认这是另一个独立垃圾；" +
                "否则取消并改点其它位置。",
        };
        renderDetail();
        return;
      }
      if (reply.warning === "TARGET_TRUNCATED_BY_TILE") {
        state.warning = reply;
        state.message = "该目标接触 tile 边缘，按 §16 不能作为训练标注（不能 COMPLETE）";
        await refreshTile(state.message);
        return;
      }
      state.activeTarget = null;
      await refreshTile(`已保存 supplemental bbox（${letter}）`);
    } catch (error) {
      state.error = String(error.payload && error.payload.error_code
        ? `${error.payload.error_code}: ${error.payload.message}` : error.message || error);
      renderDetail();
    }
  }

  function confirmDuplicate() {
    const pending = state.confirm;
    state.confirm = null;
    if (!pending) return;
    state.warning = null;
    post("/api/select", { tile_id: state.tile.tile_id, target_id: pending.targetId,
                          letter: pending.letter, confirm_independent: true })
      .then(async (reply) => {
        if (reply.warning === "TARGET_TRUNCATED_BY_TILE") {
          state.warning = reply;
          state.message = "该目标接触 tile 边缘，不能作为训练标注";
          setBusy(false);
          await refreshTile(state.message);
          return;
        }
        state.activeTarget = null;
        await refreshTile(`已确认独立目标并保存（${pending.letter}）`);
      })
      .catch((error) => { state.error = String(error.message || error); renderDetail(); });
  }

  async function recheck(decision, note) {
    state.confirm = null;
    if (decision === "ANNOTATION_COMPLETE_AFTER_SUPPLEMENT" && !note) {
      state.confirm = {
        kind: "complete", decision,
        text: "确认：当前 640×640 tile 中所有可见 REQUIRED_LITTER 都已有合理 bbox。",
      };
      renderDetail();
      return;
    }
    try {
      await post("/api/recheck", { tile_id: state.tile.tile_id, decision,
                                  note: note || currentNote() });
      const rows = visibleRows();
      const at = rows.findIndex((r) => r.tile_id === state.tile.tile_id);
      await refreshTile();
      const next = visibleRows()[Math.min(Math.max(at, 0), Math.max(0, visibleRows().length - 1))];
      if (next) await selectTile(next.tile_id);
      else { renderQueue(); renderDetail(); }
    } catch (error) {
      state.error = String(error.payload && error.payload.message
        ? error.payload.message : error.message || error);
      renderDetail();
    }
  }

  function bindDetail() {
    const add = $("btn-add");
    if (add) add.onclick = () => { state.armed = true; state.activeTarget = null;
      state.message = "请点击遗漏垃圾的中心"; renderDetail(); };
    const clear = $("btn-clear");
    if (clear) clear.onclick = () => post("/api/clear", { tile_id: state.tile.tile_id })
      .then(() => { state.activeTarget = null; return refreshTile("已清除本 tile 的新增"); })
      .catch((e) => { state.error = String(e.message || e); renderDetail(); });
    const complete = $("btn-complete");
    if (complete) complete.onclick = () => recheck("ANNOTATION_COMPLETE_AFTER_SUPPLEMENT",
                                                  currentNote());
    const still = $("btn-stillmissing");
    if (still) still.onclick = () => recheck("STILL_MISSING_REQUIRED", currentNote());
    const unresolved = $("btn-unresolved");
    if (unresolved) unresolved.onclick = () => recheck("SUPPLEMENTAL_LOCALIZATION_UNRESOLVED",
                                                      currentNote());
    const uncertain = $("btn-uncertain");
    if (uncertain) uncertain.onclick = () => recheck("UNCERTAIN_COMPLETENESS", currentNote());
    const skip = $("btn-skip");
    if (skip) skip.onclick = () => post("/api/skip", { tile_id: state.tile.tile_id })
      .then(() => { move(1); }).catch((e) => { state.error = String(e.message || e); renderDetail(); });

    document.querySelectorAll("[data-pick]").forEach((node) => {
      node.onclick = () => pick(node.dataset.pick, node.dataset.target);
    });
    document.querySelectorAll("[data-repoint]").forEach((node) => {
      node.onclick = () => { state.activeTarget = node.dataset.repoint;
        state.armed = true; state.message = "请重新点击该目标的中心（最多两轮）"; renderDetail(); };
    });
    document.querySelectorAll("[data-delete]").forEach((node) => {
      node.onclick = () => post("/api/delete", { tile_id: state.tile.tile_id,
                                                 target_id: node.dataset.delete })
        .then(() => { state.activeTarget = null; return refreshTile("已删除"); })
        .catch((e) => { state.error = String(e.message || e); renderDetail(); });
    });
    document.querySelectorAll("[data-reject]").forEach((node) => {
      node.onclick = () => post("/api/reject", { tile_id: state.tile.tile_id,
                                                 target_id: node.dataset.reject,
                                                 reason: currentNote() })
        .then(() => { state.activeTarget = null;
                      return refreshTile("已标记：两轮 proposal 均不正确"); })
        .catch((e) => { state.error = String(e.message || e); renderDetail(); });
    });
    const yes = $("confirm-yes");
    if (yes) yes.onclick = () => {
      if (state.confirm && state.confirm.kind === "duplicate") confirmDuplicate();
      else recheck(state.confirm.decision, "confirmed by reviewer");
    };
    const no = $("confirm-no");
    if (no) no.onclick = () => { state.confirm = null; state.warning = null;
      state.message = "已取消"; renderDetail(); };

    const wrap = $("tilewrap");
    if (wrap) wrap.onclick = (event) => {
      if (!state.armed) return;
      const [x, y] = tilePointFromEvent(event);
      requestPoint(x, y, state.activeTarget);
    };
    const note = $("note");
    if (note) note.oninput = () => { state.note = note.value; };
  }

  function bind() {
    ["f-status", "f-camera", "f-labels"].forEach((id) => { $(id).onchange = renderQueue; });
    $("f-search").oninput = renderQueue;
    document.addEventListener("keydown", (event) => {
      const tag = event.target.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
      const key = event.key.toLowerCase();
      const target = state.tile && (state.tile.supplemental_targets || [])
        .find((t) => t.supplemental_target_id === state.activeTarget);
      if (KEYS[key] && target && !state.confirm) {
        const candidate = (target.proposal_candidates || [])
          .find((c) => c.letter === KEYS[key]);
        if (candidate) { event.preventDefault(); pick(KEYS[key], target.supplemental_target_id); }
        return;
      }
      if (key === "n") { event.preventDefault();
        state.armed = true; state.activeTarget = null;
        state.message = "请点击遗漏垃圾的中心"; renderDetail(); return; }
      if (key === "x" && target) { event.preventDefault();
        document.querySelector("[data-reject]").click(); return; }
      if (key === "d" && target) { event.preventDefault();
        post("/api/delete", { tile_id: state.tile.tile_id,
                             target_id: target.supplemental_target_id })
          .then(() => { state.activeTarget = null; return refreshTile("已删除"); })
          .catch((e) => { state.error = String(e.message || e); renderDetail(); });
        return; }
      if (key === "p" && target) { event.preventDefault();
        state.armed = true; state.message = "请重新点击该目标的中心";
        renderDetail(); return; }
      if (key === "s") { event.preventDefault();
        post("/api/skip", { tile_id: state.tile.tile_id })
          .then(() => move(1)).catch((e) => { state.error = String(e.message || e);
                                              renderDetail(); });
        return; }
      if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); return; }
      if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
    });
  }

  bind();
  loadMeta(false).catch((error) => {
    document.body.innerHTML = `<pre class="err">${esc(error.message || error)}</pre>`;
  });
})();
