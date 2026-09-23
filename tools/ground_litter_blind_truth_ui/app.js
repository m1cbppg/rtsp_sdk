/* Blind Truth review UI.
 *
 * Hard rule: this file never requests, renders or stores any detector output.  The only
 * server endpoints it touches are raw-frame decode, the human truth store, the classic
 * CV localization proposals (used only after a human confirmed the target), sampling and
 * freeze.  No per-box probability, no network output and no model box is shown.
 */
'use strict';

const state = {
  inventory: null,
  files: [],
  cameras: [],
  truth: [],
  episodes: [],
  coverage: null,
  frozen: null,
  camera: null,
  fileId: null,
  decodedTs: 0,
  sourceWidth: 2560,
  sourceHeight: 1440,
  roi: [],
  pendingPoint: null,       // [x, y] in source pixels
  selectedTruth: null,      // truth_id
  selectedEpisode: null,    // episode_id
  zoom: null,               // {x, y, half} in source pixels
  playing: false,
  playTimer: null,
  proposals: [],
};

const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return response.headers.get('content-type')?.includes('json')
    ? response.json() : response.text();
}

function setStatus(text) { $('status').textContent = text || ''; }

async function boot() {
  state.inventory = await api('/api/inventory');
  state.files = state.inventory.files;
  state.cameras = state.inventory.cameras;
  const cameraSelect = $('camera');
  cameraSelect.innerHTML = '';
  for (const camera of state.cameras) {
    const option = document.createElement('option');
    option.value = camera;
    option.textContent = `${camera} (${state.inventory.per_camera_count[camera]} PS)`;
    cameraSelect.appendChild(option);
  }
  const resume = state.inventory.resume || {};
  state.camera = resume.camera_id && state.cameras.includes(resume.camera_id)
    ? resume.camera_id : state.cameras[0];
  cameraSelect.value = state.camera;
  fillFileSelect(resume.file_id, resume.decoded_timestamp);
  await refreshState();
  await loadFrame();
}

function fillFileSelect(preferFileId, preferTs) {
  const select = $('file');
  select.innerHTML = '';
  const files = state.files.filter((f) => f.camera_id === state.camera);
  for (const file of files) {
    const option = document.createElement('option');
    option.value = file.file_id;
    const status = (state.coverage?.per_file?.[file.file_id]) || 'PENDING';
    option.textContent = `${file.record_start.slice(11)}–${file.record_end.slice(11)} [${status}]`;
    select.appendChild(option);
  }
  const match = files.find((f) => f.file_id === preferFileId);
  state.fileId = match ? match.file_id : (files[0] && files[0].file_id);
  if (state.fileId) select.value = state.fileId;
  state.decodedTs = match && preferTs ? Number(preferTs) : 0;
}

function currentFile() { return state.files.find((f) => f.file_id === state.fileId); }

async function refreshState() {
  const payload = await api('/api/state');
  state.truth = payload.truth_objects;
  state.episodes = payload.episodes;
  state.coverage = payload.coverage;
  state.frozen = payload.frozen;
  renderLists();
  renderCoverage();
}

function renderLists() {
  const truthList = $('truth-list');
  truthList.innerHTML = '';
  for (const row of state.truth) {
    const li = document.createElement('li');
    li.className = `${row.truth_class}${state.selectedTruth === row.truth_id ? ' sel' : ''}`;
    const geometry = row.source_bbox_xyxy
      ? `box ${row.source_bbox_xyxy.map((v) => Math.round(v)).join(',')}`
      : (row.source_point ? `pt ${row.source_point.map((v) => Math.round(v)).join(',')}` : 'no geom');
    li.innerHTML = `<b>${row.truth_id}</b> ${row.truth_class}<span class="badge">${row.localization_status}</span>`
      + `<br><span class="meta">${row.camera_id} ${row.timestamp.slice(11)} ${geometry}`
      + `${row.episode_id ? ' → ' + row.episode_id : ' (未分组)'}</span>`;
    li.onclick = () => {
      state.selectedTruth = row.truth_id;
      state.selectedEpisode = row.episode_id || state.selectedEpisode;
      if (row.camera_id !== state.camera) {
        state.camera = row.camera_id;
        $('camera').value = row.camera_id;
        fillFileSelect();
      }
      state.fileId = row.source_file_id;
      $('file').value = row.source_file_id;
      state.decodedTs = row.decoded_timestamp ?? 0;
      state.pendingPoint = row.source_point || state.pendingPoint;
      state.zoom = null;
      loadFrame();
      renderLists();
    };
    truthList.appendChild(li);
  }
  $('truth-count').textContent = `${state.truth.length}`;

  const episodeList = $('episode-list');
  episodeList.innerHTML = '';
  for (const row of state.episodes) {
    const li = document.createElement('li');
    li.className = `${row.truth_class}${state.selectedEpisode === row.episode_id ? ' sel' : ''}`;
    li.innerHTML = `<b>${row.episode_id}</b> ${row.truth_class}<span class="badge">${row.review_status}</span>`
      + `<br><span class="meta">${row.camera_id} ${row.first_confirmable_timestamp.slice(11)}`
      + ` → ${row.last_confirmable_timestamp.slice(11)} · ${row.observation_count ?? 0} obs`
      + ` · files ${(row.source_file_ids || []).length}</span>`;
    li.onclick = () => { state.selectedEpisode = row.episode_id; renderLists(); };
    episodeList.appendChild(li);
  }
  $('episode-count').textContent = `${state.episodes.length}`;
}

function renderCoverage() {
  const coverage = state.coverage;
  if (!coverage) return;
  $('coverage').textContent = `${coverage.reviewed}/${coverage.total_files} PS`;
  const parts = Object.entries(coverage.per_camera)
    .map(([cam, slot]) => `${cam}: ${slot.reviewed}/${slot.total}`);
  $('coverage-detail').textContent = parts.join(' · ');
  $('freeze-info').textContent = state.frozen
    ? `已冻结 ${state.frozen.frozen_at} truth_sha256=${String(state.frozen.truth_sha256).slice(0, 16)}…`
    : (coverage.complete ? 'Review 已完成，可以 freeze' : 'Review 未完成，不能 freeze');
}

async function loadFrame() {
  const file = currentFile();
  if (!file) return;
  state.roi = file.roi || [];
  const url = `/api/frame?file_id=${encodeURIComponent(state.fileId)}`
    + `&t=${state.decodedTs}&w=1280`;
  const meta = await api(`/api/frame-meta?file_id=${encodeURIComponent(state.fileId)}&t=${state.decodedTs}`);
  state.sourceWidth = meta.source_width;
  state.sourceHeight = meta.source_height;
  state.decodedTs = meta.decoded_timestamp;
  $('time-input').value = meta.decoded_timestamp.toFixed(2);
  $('file-meta').textContent = `${file.camera_id} ${file.record_start} → ${file.record_end}`
    + ` · ${meta.source_width}×${meta.source_height} · ${file.record_start.slice(11)}+${meta.decoded_timestamp.toFixed(1)}s`;
  $('frame').src = `${url}&_=${Date.now()}`;
  $('frame').onload = () => { drawOverlay(); };
  if (state.zoom) await loadCrop(); else setStatus('');
}

function displayScale() {
  const image = $('frame');
  const width = image.clientWidth || 1;
  return width / state.sourceWidth;
}

function drawOverlay() {
  const image = $('frame');
  const canvas = $('overlay');
  canvas.width = image.clientWidth;
  canvas.height = image.clientHeight;
  const scale = displayScale();
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (state.zoom) return;   // the crop view draws its own overlay

  if (state.roi && state.roi.length) {
    ctx.strokeStyle = '#4fd98a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.roi.forEach(([nx, ny], index) => {
      const x = nx * image.clientWidth;
      const y = ny * image.clientHeight;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }

  for (const row of state.truth) {
    if (row.source_file_id !== state.fileId) continue;
    const point = row.source_point;
    const box = row.source_bbox_xyxy;
    ctx.lineWidth = 2;
    if (box) {
      ctx.strokeStyle = row.truth_class === 'REQUIRED_LITTER' ? '#d9534f' : '#d9a24f';
      ctx.strokeRect(box[0] * scale, box[1] * scale,
                     (box[2] - box[0]) * scale, (box[3] - box[1]) * scale);
    }
    if (point) {
      ctx.strokeStyle = state.selectedTruth === row.truth_id ? '#ffffff' : '#f2d24f';
      ctx.beginPath();
      ctx.arc(point[0] * scale, point[1] * scale, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(point[0] * scale - 11, point[1] * scale);
      ctx.lineTo(point[0] * scale + 11, point[1] * scale);
      ctx.moveTo(point[0] * scale, point[1] * scale - 11);
      ctx.lineTo(point[0] * scale, point[1] * scale + 11);
      ctx.stroke();
    }
  }

  if (state.pendingPoint) {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    const [x, y] = state.pendingPoint;
    ctx.beginPath();
    ctx.arc(x * scale, y * scale, 9, 0, Math.PI * 2);
    ctx.stroke();
  }
}

async function loadCrop() {
  const zoom = state.zoom;
  const url = `/api/crop?file_id=${encodeURIComponent(state.fileId)}&t=${state.decodedTs}`
    + `&cx=${zoom.x}&cy=${zoom.y}&half=${zoom.half}`;
  const meta = await api(`/api/crop-meta?file_id=${encodeURIComponent(state.fileId)}&t=${state.decodedTs}`
    + `&cx=${zoom.x}&cy=${zoom.y}&half=${zoom.half}`);
  $('frame').src = `${url}&_=${Date.now()}`;
  state.cropMeta = meta;
  $('frame').onload = () => {
    const image = $('frame');
    const canvas = $('overlay');
    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.pendingPoint) {
      const scaleX = image.clientWidth / (meta.x2 - meta.x1);
      const scaleY = image.clientHeight / (meta.y2 - meta.y1);
      const cx = (state.pendingPoint[0] - meta.x1) * scaleX;
      const cy = (state.pendingPoint[1] - meta.y1) * scaleY;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.stroke();
    }
    setStatus(`source-native crop x[${meta.x1},${meta.x2}] y[${meta.y1},${meta.y2}]`);
  };
}

function imagePointToSource(event) {
  const image = $('frame');
  const rect = image.getBoundingClientRect();
  const fx = (event.clientX - rect.left) / rect.width;
  const fy = (event.clientY - rect.top) / rect.height;
  if (state.zoom && state.cropMeta) {
    const meta = state.cropMeta;
    return [meta.x1 + fx * (meta.x2 - meta.x1), meta.y1 + fy * (meta.y2 - meta.y1)];
  }
  return [fx * state.sourceWidth, fy * state.sourceHeight];
}

function setPoint(point) {
  state.pendingPoint = [Math.round(point[0] * 100) / 100, Math.round(point[1] * 100) / 100];
  $('point-info').textContent = `source ${state.pendingPoint[0]}, ${state.pendingPoint[1]}`
    + ` (${state.sourceWidth}×${state.sourceHeight})`;
  drawOverlay();
}

async function addTruth(truthClass) {
  if (!state.pendingPoint) { setStatus('先在画面上点击目标中心'); return; }
  const body = {
    action: 'add', truth_class: truthClass,
    camera_id: state.camera, source_file_id: state.fileId,
    decoded_timestamp: state.decodedTs,
    source_point: state.pendingPoint,
  };
  try {
    const reply = await api('/api/truth', { method: 'POST',
      headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    state.selectedTruth = reply.truth_id;
    if (reply.episode_id) state.selectedEpisode = reply.episode_id;
    setStatus(`已记录 ${reply.truth_id} ${truthClass}${reply.in_roi === false ? '（注意：在 ROI 外）' : ''}`);
    await refreshState();
    loadFrame();
  } catch (error) { setStatus(`记录失败 ${error.message}`); }
}

async function truthAction(action, extra) {
  if (!state.selectedTruth) { setStatus('先选一个 truth 对象'); return; }
  try {
    await api('/api/truth', { method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ action, truth_id: state.selectedTruth, ...extra }) });
    await refreshState();
    loadFrame();
  } catch (error) { setStatus(`${action} 失败 ${error.message}`); }
}

async function episodeAction(action, extra) {
  try {
    const reply = await api('/api/episode', { method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ action, truth_id: state.selectedTruth,
                             episode_id: state.selectedEpisode, ...extra }) });
    state.selectedEpisode = reply.episode_id;
    await refreshState();
  } catch (error) { setStatus(`${action} 失败 ${error.message}`); }
}

async function proposalStep() {
  if (!state.pendingPoint) { setStatus('先点击目标中心，再做经典 CV 定位'); return; }
  const [x, y] = state.pendingPoint;
  const payload = await api(`/api/proposals?file_id=${encodeURIComponent(state.fileId)}`
    + `&t=${state.decodedTs}&x=${x}&y=${y}`);
  state.proposals = payload.proposals;
  const list = $('proposal-list');
  list.innerHTML = '';
  if (!payload.proposals.length) {
    list.innerHTML = '<span class="meta">经典 CV 无法给出可靠候选 → 保留 point truth，'
      + '标记 LOCALIZATION_UNRESOLVED，交给人工裁决</span>';
    $('proposals').classList.remove('hidden');
    return;
  }
  payload.proposals.forEach((proposal, index) => {
    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.src = proposal.preview_url;
    image.width = 180;
    image.title = `${proposal.label} area=${proposal.area}`;
    image.onclick = () => {
      truthAction('update', { source_bbox_xyxy: proposal.bbox_xyxy,
                              localization_status: 'PROPOSAL_SELECTED' });
      $('proposals').classList.add('hidden');
    };
    const caption = document.createElement('figcaption');
    caption.textContent = proposal.label;
    figure.appendChild(image);
    figure.appendChild(caption);
    list.appendChild(figure);
  });
  $('proposals').classList.remove('hidden');
}

async function markReviewed() {
  await api('/api/review', { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ file_id: state.fileId, status: 'REVIEWED' }) });
  await refreshState();
  const files = state.files.filter((f) => f.camera_id === state.camera);
  const index = files.findIndex((f) => f.file_id === state.fileId);
  const next = files[index + 1];
  if (next) { state.fileId = next.file_id; $('file').value = next.file_id; state.decodedTs = 0; }
  await loadFrame();
}

async function move(deltaFrames) {
  state.decodedTs = Math.max(0, state.decodedTs + deltaFrames / 25.0);
  await loadFrame();
}

function togglePlay() {
  state.playing = !state.playing;
  $('btn-play').classList.toggle('active', state.playing);
  if (state.playing) {
    state.playTimer = setInterval(() => { move(2).catch(() => {}); }, 220);
  } else {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
}

function bind() {
  $('camera').onchange = async (event) => {
    state.camera = event.target.value;
    state.zoom = null;
    fillFileSelect();
    await loadFrame();
  };
  $('file').onchange = async (event) => {
    state.fileId = event.target.value;
    state.decodedTs = 0;
    state.zoom = null;
    await loadFrame();
  };
  $('btn-prev').onclick = () => move(-1);
  $('btn-next').onclick = () => move(1);
  $('btn-prev-sec').onclick = () => move(-25);
  $('btn-next-sec').onclick = () => move(25);
  $('btn-play').onclick = togglePlay;
  $('btn-goto').onclick = async () => {
    state.decodedTs = Math.max(0, Number($('time-input').value) || 0);
    await loadFrame();
  };
  $('btn-zoom-in').onclick = async () => {
    const [x, y] = state.pendingPoint || [state.sourceWidth / 2, state.sourceHeight / 2];
    const half = state.zoom ? Math.max(60, state.zoom.half / 1.6) : 320;
    state.zoom = { x, y, half };
    await loadCrop();
  };
  $('btn-zoom-out').onclick = async () => {
    if (!state.zoom) return;
    const half = state.zoom.half * 1.6;
    if (half > Math.max(state.sourceWidth, state.sourceHeight)) {
      state.zoom = null;
      await loadFrame();
    } else {
      state.zoom = { ...state.zoom, half };
      await loadCrop();
    }
  };
  $('btn-zoom-reset').onclick = async () => { state.zoom = null; await loadFrame(); };
  $('btn-proposals').onclick = proposalStep;
  $('btn-proposal-cancel').onclick = () => $('proposals').classList.add('hidden');
  $('btn-delete-truth').onclick = () => truthAction('delete', { reason: 'human correction' });
  $('btn-review').onclick = markReviewed;
  $('btn-new-episode').onclick = () => episodeAction('new');
  $('btn-assign-episode').onclick = () => episodeAction('assign');
  $('btn-confirm-episode').onclick = () => episodeAction('confirm');
  $('btn-set-start').onclick = () => truthAction('set_interval', { which: 'start' });
  $('btn-set-end').onclick = () => truthAction('set_interval', { which: 'end' });
  $('btn-sample').onclick = async () => {
    const reply = await api('/api/sample', { method: 'POST' });
    setStatus(`采样完成：visible ${reply.visible_frames} 帧 / global ${reply.global_frames} 帧`);
    await refreshState();
  };
  $('btn-freeze').onclick = async () => {
    try {
      const reply = await api('/api/freeze', { method: 'POST' });
      setStatus(`已冻结 truth_sha256=${reply.truth_sha256}`);
      await refreshState();
    } catch (error) { setStatus(`freeze 被拒绝：${error.message}`); }
  };
  $('frame').onclick = (event) => setPoint(imagePointToSource(event));
  for (const button of document.querySelectorAll('[data-class]')) {
    button.onclick = () => addTruth(button.dataset.class);
  }
  document.addEventListener('keydown', (event) => {
    if (event.target.tagName === 'INPUT') return;
    const map = { r: 'REQUIRED_LITTER', i: 'IGNORE_SMALL', u: 'UNCERTAIN', d: 'NON_LITTER' };
    const key = event.key.toLowerCase();
    if (map[key]) { addTruth(map[key]); event.preventDefault(); return; }
    if (key === 'n') { episodeAction('new'); event.preventDefault(); return; }
    if (key === 'e') { episodeAction('assign'); event.preventDefault(); return; }
    if (key === 'c') { episodeAction('confirm'); event.preventDefault(); return; }
    if (key === 'm') { markReviewed(); event.preventDefault(); return; }
    if (key === 'x') { truthAction('delete', { reason: 'human correction' }); event.preventDefault(); return; }
    if (key === 'arrowleft') { move(event.shiftKey ? -25 : -1); event.preventDefault(); return; }
    if (key === 'arrowright') { move(event.shiftKey ? 25 : 1); event.preventDefault(); return; }
    if (key === ' ') { togglePlay(); event.preventDefault(); }
  });
}

bind();
boot().catch((error) => setStatus(`初始化失败：${error.message}`));
