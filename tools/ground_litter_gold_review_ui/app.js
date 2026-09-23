/* Step 1B gold-episode review UI. No framework, no build step.
 *
 * Deliberately never renders detector source / score / model name: Step 1B is a
 * human episode-identity decision, and the frozen Step 0B protocol requires Blind
 * review.  Only Step 1A grouping evidence is shown, and only as auxiliary info.
 */
(() => {
  "use strict";

  const state = {
    meta: null,
    queue: [],
    decision: {},
    current: null,
    mode: "normal",        // normal | split | merge
    split: null,           // {groups:[{members:[], truth_class}]}
    mergeSel: new Set(),
    message: "",
    error: "",
  };

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const DECISION_LABEL = {
    CONFIRM: "REQUIRED_LITTER / 单一 episode",
    SPLIT: "拆分为多个 episode",
    MERGE: "与其它 candidate 合并",
    NON_LITTER: "NON_LITTER",
    IGNORE_SMALL: "IGNORE_SMALL",
    UNCERTAIN: "UNCERTAIN",
  };

  async function api(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  async function post(path, body) {
    return api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  // ---------------------------------------------------------------- loading

  async function loadMeta() {
    state.meta = await api("/api/meta");
    state.queue = state.meta.queue;
    state.decision = {};
    state.queue.forEach((row) => { state.decision[row.episode_candidate_id] = row; });
    const cameras = [...new Set(state.queue.map((r) => r.camera_id))].sort();
    const select = $("f-camera");
    select.innerHTML = '<option value="all">全部</option>' +
      cameras.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    $("meta").textContent =
      `${state.meta.candidate_count} candidates · ${state.meta.member_card_count} member cards · Step1A ${state.meta.artifact_sha256.slice(0, 12)} · commit ${String(state.meta.step1a_code_commit).slice(0, 8)}`;
    renderProgress();
    renderQueue();
    const first = visibleRows()[0];
    if (first) selectCandidate(first.episode_candidate_id);
  }

  function renderProgress() {
    const p = state.meta.progress;
    $("progress").textContent = `${p.reviewed} / ${p.total}`;
    $("status-line").textContent =
      `已审核 ${p.reviewed} · 已跳过 ${p.skipped} · 待审核 ${p.pending}`;
  }

  function visibleRows() {
    const status = $("f-status").value;
    const risk = $("f-risk").value;
    const camera = $("f-camera").value;
    const members = $("f-members").value;
    const search = $("f-search").value.trim().toLowerCase();
    return state.queue.filter((row) => {
      const record = state.decision[row.episode_candidate_id] || {};
      const rowStatus = record.status || "pending";
      if (status !== "all" && rowStatus !== status) return false;
      if (risk === "grouping" && !row.grouping_risk) return false;
      if (risk === "lineage" && !row.lineage_only) return false;
      if (risk === "none" && (row.grouping_risk || row.lineage_only)) return false;
      if (camera !== "all" && row.camera_id !== camera) return false;
      if (members === "multi" && row.member_count < 2) return false;
      if (members === "single" && row.member_count !== 1) return false;
      if (search && !row.episode_candidate_id.toLowerCase().includes(search)) return false;
      return true;
    });
  }

  function renderQueue() {
    const rows = visibleRows();
    $("queue").innerHTML = rows.map((row) => {
      const record = state.decision[row.episode_candidate_id] || {};
      const badges = [];
      if (row.grouping_risk) badges.push(`<span class="badge p1">GROUPING_RISK</span>`);
      if (row.member_count > 1) badges.push(`<span class="badge p2">${row.member_count} cards</span>`);
      if ((row.original_labels_summary || {}).BOX_WRONG) badges.push(`<span class="badge p3">BOX_WRONG</span>`);
      if (row.lineage_only) badges.push(`<span class="badge p5">LINEAGE_ONLY</span>`);
      if (record.decision) badges.push(`<span class="badge done">${esc(record.decision)}</span>`);
      const active = state.current === row.episode_candidate_id ? " active" : "";
      return `<button class="item${active}" data-cid="${esc(row.episode_candidate_id)}">
        <span>${esc(row.camera_id)} · ${esc(row.start_timestamp || "")}</span>
        <small>${esc(row.episode_candidate_id)}</small>
        <span class="badges">${badges.join("")}</span>
      </button>`;
    }).join("") || '<p class="muted" style="padding:10px">没有符合条件的 candidate</p>';
    document.querySelectorAll(".item").forEach((node) => {
      node.onclick = () => selectCandidate(node.dataset.cid);
    });
  }

  async function selectCandidate(candidateId) {
    state.current = candidateId;
    state.mode = "normal";
    state.split = null;
    state.mergeSel = new Set();
    state.message = "";
    state.error = "";
    try {
      state.detail = await api(`/api/candidate?id=${encodeURIComponent(candidateId)}`);
    } catch (error) {
      state.error = String(error.message || error);
      state.detail = null;
    }
    renderQueue();
    renderDetail();
  }

  function move(delta) {
    const rows = visibleRows();
    if (!rows.length) return;
    const at = rows.findIndex((r) => r.episode_candidate_id === state.current);
    const next = at < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, at + delta));
    selectCandidate(rows[next].episode_candidate_id);
  }

  // ---------------------------------------------------------------- detail

  function tile(slot, data) {
    if (!data || data.missing || !data.url) {
      return `<figure class="tile"><div class="missing">MISSING</div>
        <figcaption>${esc(slot)}</figcaption></figure>`;
    }
    const note = data.derived_from ? `<br>(${esc(data.derived_from)})` : "";
    return `<figure class="tile"><img src="${esc(data.url)}" alt="${esc(slot)}" loading="lazy">
      <figcaption>${esc(slot)}${note}</figcaption></figure>`;
  }

  function renderMembers(detail) {
    return detail.members.map((member, index) => {
      const tiles = ["context", "crop", "before", "current", "after"]
        .map((slot) => tile(slot, member.tiles[slot])).join("");
      const bbox = member.bbox ? `[${member.bbox.map((v) => Math.round(v)).join(", ")}]` : "MISSING";
      const chips = state.mode === "split"
        ? `<div>归属：${state.split.groups.map((group, gi) =>
            `<span class="chip ${group.members.includes(member.card_id) ? "on" : ""}"
              data-assign="${esc(member.card_id)}" data-group="${gi}">组 ${gi + 1}</span>`).join("")}</div>`
        : "";
      return `<div class="member" data-index="${index}">
        <div class="member-head">
          <span><b>${esc(member.card_id)}</b></span>
          <span class="muted">${esc(member.timestamp || "MISSING timestamp")} ·
            原始标签 ${esc(member.original_label || "MISSING")} ·
            frame ${esc(member.frame_id || "MISSING")} · bbox ${esc(bbox)}</span>
        </div>
        <div class="tiles">${tiles}</div>
        ${chips}
      </div>`;
    }).join("");
  }

  function renderSuggestions(detail) {
    if (state.mode !== "merge") return "";
    const rows = detail.merge_suggestions || [];
    if (!rows.length) return '<p class="muted">没有可建议的邻近 candidate。</p>';
    return `<div class="suggestions">${rows.map((s) => `
      <div class="sugg">
        <div>${s.representative_url
          ? `<img src="${esc(s.representative_url)}" alt="suggestion">`
          : '<div class="missing">MISSING</div>'}</div>
        <div>
          <div><label><input type="checkbox" data-merge="${esc(s.candidate_id)}"
            ${state.mergeSel.has(s.candidate_id) ? "checked" : ""}> 合并 <b>${esc(s.candidate_id)}</b></label></div>
          <div class="hint">${esc(s.start_timestamp)} ~ ${esc(s.end_timestamp)} · ${s.member_count} cards</div>
          <div class="hint">time gap ${s.time_gap_seconds}s · center dist ${s.min_center_distance_px}px · ${esc(s.risk_reason)}</div>
        </div>
      </div>`).join("")}</div>`;
  }

  function renderSplitEditor(detail) {
    if (state.mode !== "split") return "";
    const groups = state.split.groups;
    const unassigned = detail.members
      .map((m) => m.card_id)
      .filter((card) => !groups.some((g) => g.members.includes(card)));
    return `<div class="card">
      <b>SPLIT 分组</b>
      <p class="hint">点击每个 member 上方的“组 N”把它分配到该组。所有 member 必须且只能归属一个组。</p>
      ${groups.map((group, gi) => `
        <div class="splitgroup">
          <h4>组 ${gi + 1} · ${group.members.length} cards · 分类
            <select data-truth="${gi}">
              ${["REQUIRED_LITTER", "IGNORE_SMALL", "NON_LITTER", "UNCERTAIN"].map((t) =>
                `<option value="${t}" ${group.truth_class === t ? "selected" : ""}>${t}</option>`).join("")}
            </select>
          </h4>
          <div class="hint">${group.members.map(esc).join(" , ") || "（空）"}</div>
        </div>`).join("")}
      <button class="act" id="add-group">+ 添加组</button>
      <span class="hint">未分配 ${unassigned.length} 个：${esc(unassigned.join(", ")) || "无"}</span>
    </div>`;
  }

  function renderDetail() {
    const detail = state.detail;
    if (!detail) {
      $("detail").innerHTML = `<p class="err">${esc(state.error || "加载失败")}</p>`;
      return;
    }
    const risk = detail.risk || {};
    const evidence = detail.grouping_evidence || {};
    const record = state.decision[detail.episode_candidate_id] || {};
    const riskHtml = (risk.grouping_risk_reasons || []).length || (risk.lineage_reasons || []).length
      ? `<div class="riskbox">
           ${(risk.grouping_risk_reasons || []).length
             ? `<div><b>GROUPING_RISK</b>: ${esc(risk.grouping_risk_reasons.join(", "))}</div>` : ""}
           ${(risk.lineage_reasons || []).length
             ? `<div>LINEAGE_ONLY: ${esc(risk.lineage_reasons.join(", "))}</div>` : ""}
           ${(risk.same_frame_neighbour_candidate_ids || []).length
             ? `<div>同帧邻近 candidate: ${esc(risk.same_frame_neighbour_candidate_ids.join(", "))}</div>` : ""}
         </div>`
      : '<div class="riskbox" style="border-color:var(--ok)">无分组风险标记</div>';

    $("detail").innerHTML = `
      <div class="card">
        <div class="kv">
          <dt>episode_candidate_id</dt><dd><b>${esc(detail.episode_candidate_id)}</b></dd>
          <dt>camera_id</dt><dd>${esc(detail.camera_id)}</dd>
          <dt>scene_version</dt><dd>${esc(detail.scene_version)}</dd>
          <dt>时间范围</dt><dd>${esc(detail.start_timestamp)} ~ ${esc(detail.end_timestamp)}</dd>
          <dt>member_count</dt><dd>${detail.member_count}</dd>
          <dt>原始标签汇总</dt><dd>${esc(JSON.stringify(detail.original_labels_summary || {}))}</dd>
          <dt>当前决定</dt><dd>${record.decision ? esc(DECISION_LABEL[record.decision] || record.decision)
              : '<span class="muted">未审核</span>'}</dd>
        </div>
        ${riskHtml}
        <details><summary>Step 1A grouping evidence（辅助信息，不是 truth）</summary>
          <div class="evidence">${esc(JSON.stringify(evidence, null, 1))}</div>
        </details>
      </div>

      <div class="card">
        <b>全部 member（${detail.members.length}）</b>
        ${renderMembers(detail)}
      </div>

      ${renderSplitEditor(detail)}
      ${renderSuggestions(detail)}

      <div class="card">
        <textarea id="note" rows="2" style="width:100%"
          placeholder="可选备注 / KEEP_SEPARATE 说明">${esc(record.note || "")}</textarea>
        <p class="hint">1 CONFIRM · 2 SPLIT · 3 MERGE · 4 NON_LITTER · 5 IGNORE_SMALL · 6 UNCERTAIN ·
          S 跳过 · ← → 上/下一个</p>
        <div id="msg" class="${state.error ? "err" : "hint"}">${esc(state.error || state.message)}</div>
      </div>

      <div class="actions">
        <button class="act primary" data-act="CONFIRM"><kbd>1</kbd> CONFIRM</button>
        <button class="act ${state.mode === "split" ? "mode" : ""}" data-act="SPLIT"><kbd>2</kbd> SPLIT</button>
        <button class="act ${state.mode === "merge" ? "mode" : ""}" data-act="MERGE"><kbd>3</kbd> MERGE</button>
        <button class="act" data-act="NON_LITTER"><kbd>4</kbd> NON_LITTER</button>
        <button class="act warn" data-act="IGNORE_SMALL"><kbd>5</kbd> IGNORE_SMALL</button>
        <button class="act" data-act="UNCERTAIN"><kbd>6</kbd> UNCERTAIN</button>
        ${state.mode === "split" ? '<button class="act primary" data-act="SPLIT_APPLY">应用 SPLIT</button>' : ""}
        ${state.mode === "merge" ? '<button class="act warn" data-act="MERGE_APPLY">应用 MERGE</button>' : ""}
        ${state.mode === "merge" ? '<button class="act" data-act="KEEP_SEPARATE">KEEP_SEPARATE</button>' : ""}
        <button class="act" data-act="SKIP"><kbd>S</kbd> 跳过</button>
        <button class="act danger" data-act="RESET">清除本条决定</button>
      </div>`;

    bindDetail();
  }

  function bindDetail() {
    document.querySelectorAll("[data-assign]").forEach((chip) => {
      chip.onclick = () => {
        const card = chip.dataset.assign;
        const groupIndex = Number(chip.dataset.group);
        state.split.groups.forEach((group, gi) => {
          group.members = group.members.filter((m) => m !== card);
          if (gi === groupIndex) group.members.push(card);
        });
        renderDetail();
      };
    });
    document.querySelectorAll("[data-truth]").forEach((select) => {
      select.onchange = () => {
        state.split.groups[Number(select.dataset.truth)].truth_class = select.value;
      };
    });
    const addGroup = $("add-group");
    if (addGroup) {
      addGroup.onclick = () => {
        state.split.groups.push({ members: [], truth_class: "REQUIRED_LITTER" });
        renderDetail();
      };
    }
    document.querySelectorAll("[data-merge]").forEach((box) => {
      box.onchange = () => {
        if (box.checked) state.mergeSel.add(box.dataset.merge);
        else state.mergeSel.delete(box.dataset.merge);
      };
    });
    document.querySelectorAll("[data-act]").forEach((button) => {
      button.onclick = () => action(button.dataset.act);
    });
  }

  // ---------------------------------------------------------------- actions

  function noteValue() {
    const node = $("note");
    return node ? node.value.trim() : "";
  }

  async function action(name) {
    const detail = state.detail;
    state.error = "";
    state.message = "";
    if (!detail) return;
    const candidateId = detail.episode_candidate_id;
    try {
      if (name === "SPLIT") {
        state.mode = "split";
        state.split = {
          groups: [
            { members: [], truth_class: "REQUIRED_LITTER" },
            { members: [], truth_class: "REQUIRED_LITTER" },
          ],
        };
        renderDetail();
        return;
      }
      if (name === "MERGE") {
        state.mode = "merge";
        state.mergeSel = new Set();
        renderDetail();
        return;
      }
      if (name === "SPLIT_APPLY") {
        const groups = state.split.groups.filter((g) => g.members.length);
        if (groups.length < 2) throw new Error("SPLIT 至少需要两个非空分组");
        const assigned = groups.flatMap((g) => g.members);
        if (new Set(assigned).size !== assigned.length) throw new Error("同一 member 被分配到多个组");
        if (assigned.length !== detail.members.length) {
          throw new Error(`还有 ${detail.members.length - assigned.length} 个 member 未分配`);
        }
        await submit(candidateId, "SPLIT", {
          episodes: groups.map((group, index) => ({
            draft_id: `e${index + 1}`,
            member_card_ids: group.members,
            truth_class: group.truth_class,
            localization_status: "OK",
          })),
        });
        return;
      }
      if (name === "MERGE_APPLY") {
        if (!state.mergeSel.size) throw new Error("请至少勾选一个要合并的 candidate");
        await submit(candidateId, "MERGE", { merge_targets: [...state.mergeSel] });
        return;
      }
      if (name === "KEEP_SEPARATE") {
        const targets = (detail.merge_suggestions || []).map((s) => s.candidate_id);
        state.mode = "normal";
        state.message = `已记录：与 ${targets.join(", ") || "(无建议)"} 保持分离（未改变 episode 决定）`;
        renderDetail();
        return;
      }
      if (name === "SKIP") {
        await submit(candidateId, "SKIP", {});
        return;
      }
      if (name === "RESET") {
        await post("/api/reset", { candidate_id: candidateId });
        await refreshAfter(candidateId, true);
        return;
      }
      await submit(candidateId, name, {});
    } catch (error) {
      state.error = String(error.message || error);
      renderDetail();
    }
  }

  async function submit(candidateId, decision, extra) {
    const body = Object.assign({ candidate_id: candidateId, note: noteValue() }, extra || {});
    if (decision === "SKIP") {
      await post("/api/skip", body);
      markLocal(candidateId, { status: "skipped", decision: null });
    } else {
      body.decision = decision;
      const result = await post("/api/decision", body);
      markLocal(candidateId, result.decision || { status: "reviewed", decision });
    }
    await refreshAfter(candidateId, false);
  }

  function markLocal(candidateId, record) {
    const row = state.decision[candidateId] || {};
    state.decision[candidateId] = Object.assign({}, row, record);
  }

  async function refreshAfter(candidateId, staying) {
    const previous = state.current;
    try {
      const meta = await api("/api/meta");
      state.meta.progress = meta.progress;
      meta.queue.forEach((row) => {
        const existing = state.decision[row.episode_candidate_id] || {};
        state.decision[row.episode_candidate_id] =
          Object.assign({}, row, { status: row.status, decision: row.decision, note: existing.note });
      });
      renderProgress();
    } catch (error) {
      state.error = String(error.message || error);
    }
    state.message = state.message || "已保存";
    if (staying) {
      await selectCandidate(previous);
    } else {
      const rows = visibleRows();
      const at = rows.findIndex((r) => r.episode_candidate_id === candidateId);
      const next = rows[at + 1] || rows[at] || rows[0];
      if (next) await selectCandidate(next.episode_candidate_id);
      else renderQueue();
    }
  }

  // ---------------------------------------------------------------- bindings

  function bind() {
    ["f-status", "f-risk", "f-camera", "f-members"].forEach((id) => {
      $(id).onchange = renderQueue;
    });
    $("f-search").oninput = renderQueue;
    document.addEventListener("keydown", (event) => {
      if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT"
          || event.target.tagName === "SELECT") return;
      const map = { "1": "CONFIRM", "2": "SPLIT", "3": "MERGE", "4": "NON_LITTER",
                    "5": "IGNORE_SMALL", "6": "UNCERTAIN" };
      if (map[event.key]) { event.preventDefault(); action(map[event.key]); return; }
      if (event.key === "s" || event.key === "S") { event.preventDefault(); action("SKIP"); return; }
      if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); return; }
      if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
    });
  }

  bind();
  loadMeta().catch((error) => {
    document.body.innerHTML = `<pre class="err">${esc(error.message || error)}</pre>`;
  });
})();
