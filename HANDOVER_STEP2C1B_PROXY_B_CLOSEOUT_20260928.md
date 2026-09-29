# Step 2C-1B Review Proxy B 收口 — 交接文档

> ## ✅ 本任务已于 2026-09-29 04:10 CST 全部完成，**不再需要交接**。
>
> 本文档写于 2026-09-28 23:00（当时批处理还在跑），保留作为**过程记录与运维参考**。
> **请勿重跑 §1 的步骤** —— 尤其**绝对不要重跑 reset**：`reset-official-review` 会销毁用户
> 新一轮的人工审核数据。最终结果见文末 **§11 完成记录**。
>
> 原始写作时的状态（已过期）：批处理 38/65、ETA 约 3.7 h。§1 列出的「剩余步骤」现已**全部执行完毕**。

---

## 0. 一句话现状

**B proxy 批处理正在服务器上跑**：当时 `38 / 65 VERIFIED`，**全部通过、0 失败**，平均 **500 s/条**，**ETA 约 3.7 小时（预计 02:40 CST 前后完成）**。

配方冻结（A）、备份（D-前置）、official reset（D）**都已完成并验证过**。**只剩四步**：

```
等 65/65  →  独立校验 + 生成 manifest  →  切换 8801 到 B  →  终验 + 汇报
```

---

## 1. 你的任务（按顺序，可直接复制）

### 步骤 0 — 确认批处理完成

```bash
ssh -p 21002 sf01@14.21.88.97 'D=/home/sf01/step2c1-blind-truth/review-proxy-b-v1; \
  echo "finished=$(ls $D/*-h264.mp4 2>/dev/null | wc -l)/65"; \
  tail -3 /tmp/gl_ab/gen_b_full.log; \
  pgrep -cf "gen_proxy_b.py --shard"'
```

判据：`finished=65/65`，且日志最后一行以 `shard 0 done` 结尾，generator 进程数为 0。
`*.partial.mp4` 必须为 0。

如果 `finished` 长时间不动而你确认 generator 进程已死（被 kill / 掉线），**直接重跑即可，是安全的、可续的**：

```bash
ssh -p 21002 sf01@14.21.88.97 'nohup setsid bash /tmp/gl_ab/run_gen_b.sh \
  > /tmp/gl_ab/gen_b_full.log 2>&1 < /dev/null & sleep 5; tail -2 /tmp/gl_ab/gen_b_full.log'
```

已 `VERIFIED` 的文件会按 `proxy-b-status/shard_0.jsonl` 跳过。

### 步骤 1 — 独立校验并生成 `review_proxy_b_manifest.json`

```bash
ssh -p 21002 sf01@14.21.88.97 \
  'cd /tmp/gl_ab && PYTHONPATH=/home/sf01/step2c1-blind-truth/code-step2c1b \
   /home/sf01/ground_litter_train/.venv/bin/python -B verify_proxy_b.py \
   --out /home/sf01/step2c1-blind-truth/review-proxy-b-v1/review_proxy_b_manifest.json'
```

它**不信任生成时的结果**，会从 frozen Development manifest 出发、对磁盘上真实存在的每个文件重跑全套验收：
codec=h264 / 2560×1440 / 25 fps / duration 与对应 A proxy 差 ≤0.5 s / start_time 与 A 差 ≤0.05 s /
faststart（moov 在 mdat 前）/ 全片 decode 无 error / sha256。

**退出码非 0 表示不是 65/65 或还有 partial —— 此时绝对不要切换 UI。**

### 步骤 2 — 把 official 8801 切到 B

```bash
ssh -p 21002 sf01@14.21.88.97 'bash /tmp/gl_ab/switch_official_to_b.sh'
```

它只改两个启动参数，其余一律不动：

```
--proxy-root  /home/sf01/step2c1-blind-truth/review-proxy-b-v1   (原来是 .../review-proxy)
--proxy-recipe b-v1
```

含义：**只换媒体来源**。坐标系统、truth 点、时间戳语义、coverage 语义、episode 语义、audit 语义、
localization 语义全部不变。用户点击仍然落在 **2560×1440 source coordinate**；source frame / QA /
localization / evaluation 仍然回**原始 PS**。proxy B 只用于连续观看。

### 步骤 3 — 终验

```bash
ssh -p 21002 sf01@14.21.88.97 'bash /tmp/gl_ab/final_accept.sh'
```

依次输出：manifest 摘要 → frozen 双 hash → 切换结果 → 切换后 official 只读状态 →
**隔离 8807 上的媒体探测（不在 8801 上探测，原因见 §5）** → artifact blank-slate 复查。

### 步骤 4 — 按 §8 模板汇报

---

## 2. 环境速查

| 项 | 值 |
|---|---|
| SSH | `ssh -p 21002 sf01@14.21.88.97` |
| 工作根 | `/home/sf01/step2c1-blind-truth` |
| 部署代码 | `/home/sf01/step2c1-blind-truth/code-step2c1b`（`CODE_COMMIT=45823bc`） |
| Python | `/home/sf01/ground_litter_train/.venv/bin/python` |
| ffmpeg | `/home/sf01/step2c1-blind-truth/private-deps/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`（**没有 ffprobe**，脚本用 `ffmpeg -i` 解析） |
| Development 根 | `/home/sf01/ground-litter-feasibility/20260923-r1/archive/ground-litter-detector-feasibility-20260923-r1/development` |
| ROI 目录 | `.../code-step2c1b/output/ground_litter_final_roi_20260922/config` |
| official artifact | `/home/sf01/step2c1-blind-truth/artifact` |
| proxy A（保留） | `/home/sf01/step2c1-blind-truth/review-proxy` |
| proxy B（新建） | `/home/sf01/step2c1-blind-truth/review-proxy-b-v1` |
| 批处理状态 | `/home/sf01/step2c1-blind-truth/proxy-b-status/shard_0.jsonl`（append-only） |
| 续作工具（**持久副本**） | `/home/sf01/step2c1-blind-truth/proxy-b-tools-20260928/` |
| 工具工作副本 + 全部日志 | `/tmp/gl_ab/`（易失；若被清，把持久副本拷回去） |
| official 服务 | `127.0.0.1:8801`（**用户专属**），当前 pid 见 `ss -ltnp \| grep 8801` |
| 目标 PS | `01021 2026-09-22 16:10:44–16:15:48`（manifest 第 4 条） |

### 冻结输入（**byte-identical，绝不能变**）

```
66f908f475f8e3972e371f09987b060481632b9361d5404e4dd00c5beb178496  development_manifest.json
2f8ea4b3a2c7552cadf1a586afa51631e16566dbb1e0dce5a06b0ee7f97c274b  audit_manifest.json
```

`audit_manifest.json` 的不可变性尤其重要：**audit windows 已经预注册，重开人工审核 ≠ 重新抽 audit。**

---

## 3. 已完成事项与证据

### (A) 冻结 Proxy B 配方 — 已完成

配方 `b-v1`（与 A/B 验收通过的那份逐参数一致）：

```
libx264 -preset medium -b:v 5500k -maxrate 7500k -bufsize 15000k
        -g 250 -keyint_min 250 -sc_threshold 0 -pix_fmt yuv420p
        -movflags +faststart -avoid_negative_ts make_zero
容器：2560×1440 / 25 fps / yuv420p / 源时间轴对齐 / 与 A 完全同构
```

落地方式：代码里新增单一配方表 `PROXY_RECIPES = {"a": [...CRF18...], "b-v1": [...]}`，
`ReviewProxy` 接受 `recipe=` 参数，`ensure()` 用该表；新增 CLI `--proxy-recipe {a,b-v1}`，
**默认仍是 `a`**（不改变既有 proxy 的生成方式）。

* 提交：`45823bc`（worktree `/private/tmp/ground-litter-blind-review`，分支 `feature/ground-litter-blind-review`）
* 已部署到 `code-step2c1b`，`CODE_COMMIT=45823bc`，文件 sha 前缀 `8e3b2a79b5e97322`
* 回归测试：`tests/test_ground_litter_blind_review_ux.py::ProxyRecipeFreezeTests`（5 项）把两个配方和生成的 ffmpeg 命令全部钉死；该文件 47 项全绿；全量 `Ran 1481 tests` 与基线**同样数量的既有失败**（未引入新失败）
* 部署前旧文件备份：`/tmp/gl_ab/serve_before_recipe_20260928-172738.py`

**A/B 依据（用户已在 2026-09-28 正式 PASS 人工视觉验收）**：同一 PS，A=CRF18/8.95 Mbps，
B=5.32 Mbps；8 Mbps 限速 1× 下 A 停 8 次 / 累计 9.0 s / 最坏 2.4 s（45 s 只推进 36.7 s 媒体），
**B 停 0 次、完整推进 45.7 s**；HTTP/1.0 与 HTTP/1.1 逐格无差别（所以没上 HTTP/1.1）。
视觉：B 中所有被标记物体仍可辨认，仅地砖/网格等高频纹理变柔。

### (B) 批量生成 — 进行中（见 §1 步骤 0）

* 单 worker + **x264 默认线程**，这是刻意的（原因见 §6.1、§6.2）
* 每条：`encode → <hash>-h264.<pid>.partial.mp4 → 全套校验 → os.replace()`；校验不过就删临时文件、**永不生成正式文件名**，所以正式播放器不可能看到半成品
* 逐条实时结果在 `proxy-b-status/shard_0.jsonl`；截至目前 **0 失败**
* 目标 PS（manifest 第 4 条）第一次就用新配方重生成并 VERIFIED（424–446 s）

### (C) UI 切换 — **未执行**（被 65/65 闸门挡住，按用户要求）

脚本已写好并做过语法检查：`/tmp/gl_ab/switch_official_to_b.sh`。

### (D) official reset — 已完成

* **备份**：`/home/sf01/step2c1-blind-truth/artifact-backup-before-full-review-reset-20260928-173452/`
  内含 `review_state.json`、`truth_objects.jsonl`、`episodes.jsonl`、`ui_cache/`、两个 frozen manifest 的副本、
  `_sidecar/`、`SHA256SUMS`、`README.md`（写明"用户明确授权 reset 前的最后 snapshot，禁止删除"）、
  以及 reset 工具自己写的 `RESET-official-review.json`。**禁止删除。**
* **reset 用代码自带的 builder**（不是手删字段）：
  `reset-official-review --confirm-reset-official-review --backup-manifest <backup>/SHA256SUMS`
* before → after：`reviewed 3→0`、`truth 6→0`、`episodes 5→0`、`active 3→0`、`pending 62→65`
* 删除：`truth_objects.jsonl`、`episodes.jsonl`、`review_state.json`、`ui_cache/`
* `preserved_byte_identical: true`；`sealed_accessed/detector_loaded/checkpoint_accessed` 全 `false`
* 当前 artifact **只有两个 frozen manifest**（这就是"初始态"：`review_state.json` 由代码自身默认值定义，首次使用才落盘）
* 已用隔离服务验证 reset 后的 DOM **干净**：`episodeNodes 0`、无 `ep-000x`、无 `E1–E5`、
  coverage `0.00%`、`resume` 不显示、`truthCount 0`、`episodeCount 0`

### (E) 派生产物清理与分类 — 已完成

* 已删/已重置：见 (D)
* **未删、仅分类报告**（符合"不确定就不要删"）：`/tmp` 无关产物之外，服务器上仍有旧 truth/episodes 副本的目录：
  `artifact-backup-before-reset-20260928-140328/`、`artifact-backup-before-restore-20260928-161609/`、
  `pilot-artifact/`、`pilot-step2c1b-artifact/`、`pilot-step2c1b-flow-20260928-{1,2}/`、`test-e2e-artifact/`
* **无泄漏路径**：全机只有 **一个** server 进程、**一个** listener（8801），且指向干净的 `artifact/`；
  `app.js` 不使用 `localStorage`/`sessionStorage`/`indexedDB`，hard refresh 不会恢复旧状态

---

## 4. 硬约束（违反即返工）

1. `development_manifest.json` / `audit_manifest.json` 必须 **byte-identical**（hash 见 §2）。
2. **proxy A 必须保留**（`review-proxy/`，35 个文件，9.4 GB）。
3. **proxy B 必须保留**（`review-proxy-b-v1/`）。
4. Development PS 原文件保留；**Sealed 绝对不访问**；不改 Step 1C-2M / 1D / 2A / 2B 训练产物。
5. 不改：分辨率、preset、码率、GOP 或其它编码参数（已冻结为 `b-v1`）。
6. 不改：坐标系统 / truth 点 / 时间戳语义 / coverage / episode / audit / localization 语义。

---

## 5. 安全红线

* **所有自动化测试只能 isolated artifact + isolated port。** 我用的隔离端口是 **8802–8807**，本地隧道 **18802–18819**。
  8801 **只允许**：部署、读静态状态/hash、健康检查。**不得**模拟用户审核行为、不得自动播放。
* **不要给 8801 建隧道。** 历史教训：一个 `while true` 的 `tunnel_supervisor.sh`（18801→8801）在后台反复重建隧道，
  即使删掉脚本文件也没用（循环还在跑），**两次把 official review_state 写脏**。现在这些循环已被全部 `kill -9`，
  相关脚本也被改成一次性失败退出。续作时**不要再引入任何自动重连的隧道**。
* **8801 的 `/api/proxy` 不是只读。** 它会把 `media_duration_seconds` 写进 `review_state.json`。
  我自己在"只读"探测时踩过一次（产生了 210 B 的 review_state.json），已删除复原，探针文件留在
  `/tmp/gl_ab/review_state.probe-created.json`。`final_accept.sh` 已改为**不在 8801 上探测**，改用隔离 8807。
* 本任务全程：`sealed_accessed=false`、`detector_loaded=false`、`checkpoint_accessed=false`、无 detector 推理。

---

## 6. 已经踩过的坑（务必避免重复）

1. **x264 不是逐位可复现的。** 同一条命令、同样 24 线程，重编同一 PS 得到 202,267,622 B，
   而 A/B 验收那份是 202,159,092 B（差 0.054%，均值 5324.2 vs 5321.4 kbps）。
   原因：默认 `mbtree=1` + 帧级多线程。**"冻结配方"= 参数冻结，不是字节可复现。**
   不要用"文件是否 byte-identical"当作验收判据。
2. **`-threads` 会改变输出。** 同一 1000 帧片段：默认 24,143,220 B，`-threads 4` 25,654,153 B（**+6%**），
   `-threads 8` 25,280,892 B。低线程会劣化 rate control，等于偷偷改掉已验证的码率。
   **所以批量生成必须单 worker + 默认线程，不要并行 shard。** 这也是 ETA ≈8 h 的原因。
3. **源 PS 无法按时间 seek。** `01021` 的 `.ps` 是 **HEVC 且带 `start_time 41353.035`**，
   `-ss <秒>` 只会得到**灰帧**（`Could not find ref with POC …`）。
   要取 source 原生帧必须**按帧号 `select=eq(n\,N)` 一次解码到底**（`/tmp/gl_ab/visual_frames.py` 就是这么做的）。
   已有 34 张原生 source 帧预渲染在 `/home/sf01/step2c1-blind-truth/proxy-b-review-only-20260928/source/`。
4. **`set -e` + `$(… | grep …)` 空匹配会中止脚本。** 我写 `sanity_serve.sh` 时因此让整个脚本静默退出、
   服务没起来。用 `|| true` 或去掉 `set -e`。
5. **monitor 的字段错位。** `f=$(grep -c X file || echo 0)` 在没有匹配时会输出**两行**（`0` 和 `0`），
   导致后续 `sed -n Np` 全部错位、把进度报成 `0/65`。用 `|| true`（`monitor_batch2.sh` 已修）。
6. **不要用 `timeout` 命令**（macOS 本地没有）。SSH 用 `-o ConnectTimeout=` / `ServerAliveInterval=`。
7. **本 session 的长 sleep / 长命令不稳定**：`sleep 200` 以上经常被 cap 掉且**不返回尾部输出**
   （但墙钟确实被消耗、批处理确实在前进）。建议：先做**快速状态检查**（可靠），再用一个长调用消耗本轮时间。
8. **`LAST_ARTIFACT_BACKUP` 曾被写进备份目录内部**（因为脚本先 `cd $B` 再 `echo > LAST_ARTIFACT_BACKUP`），
   导致 reset 读到旧 manifest 而**正确拒绝执行**。现在它的值应是
   `artifact-backup-before-full-review-reset-20260928-173452`。
9. **reset 的拒绝不是 bug 是保护**：它要求 `--backup-manifest` 里的 sha256 与当前待删文件**完全匹配**。
   若被拒绝，先检查 `LAST_ARTIFACT_BACKUP` 指向的 manifest 是否对应现在的文件。
10. **`cmd_prewarm_proxies` / `/api/prewarm` 是同步的**，缺 proxy 时会**在请求线程里跑 6 分钟编码**。
    做任何网络/播放测量前必须先把相邻 PS 的 proxy 备好，否则测量会被编码抢 CPU 污染。

---

## 7. 回滚方案

| 想回滚什么 | 怎么做 |
|---|---|
| UI 回到 proxy A | 重跑 8801，把 `--proxy-root` 指回 `.../review-proxy`，去掉 `--proxy-recipe`（或 `--proxy-recipe a`） |
| 代码回到加配方前 | 用 `/tmp/gl_ab/serve_before_recipe_20260928-172738.py` 覆盖 `code-step2c1b/scripts/serve_ground_litter_blind_truth.py`，`CODE_COMMIT` 改回 `12663bf` |
| 恢复被 reset 掉的人工审核数据 | 从 `artifact-backup-before-full-review-reset-20260928-173452/` 拷回 `review_state.json`、`truth_objects.jsonl`、`episodes.jsonl`、`ui_cache/`（拷回后 `chmod 600`）。**注意：这会撤销用户授权的 reset，只在用户明确要求时做。** |
| 取消 B 批处理 | `pkill -f 'gen_proxy_b.py --shard'` + 删掉 `review-proxy-b-v1/*.partial.mp4`。已 VERIFIED 的文件留着无害 |

---

## 8. 汇报模板（用户要求的格式）

```markdown
## Proxy B
generated / verified : X / 65
proxy root           : /home/sf01/step2c1-blind-truth/review-proxy-b-v1
encoding config      : libx264 -preset medium -b:v 5500k -maxrate 7500k -bufsize 15000k
                       -g 250 -keyint_min 250 -sc_threshold 0 -pix_fmt yuv420p +faststart
manifest sha256      : <review_proxy_b_manifest.json 的 sha256>
failed files         : <无 或 列表>

## Old A
确认 retained        : review-proxy/ 35 files 9.4G，问题 PS = 339,916,149 B

## Reset Backup
backup path          : artifact-backup-before-full-review-reset-20260928-173452
SHA256SUMS           : 已校验

## Frozen Inputs
development_manifest : 66f908f4…  before == after
audit_manifest       : 2f8ea4b3…  before == after

## New Official State
reviewed / truth / episodes / active : 0/65 · 0 · 0 · 0
current PS           : 01021 15:55:33–16:00:37（第一条 Development）
coverage             : empty        resume : none

## Derived Cleanup
<删除/重置了什么；哪些只分类未删>

## Safety
Sealed access = 0 · detector = false · checkpoint inference = false
automated official playback = false

## Official UI
确认使用 review-proxy-b-v1 (recipe b-v1)  → URL / hard refresh / 从哪里开始
```

---

## 9. 给用户的最后交代（切换完成后）

* 打开 **`http://127.0.0.1:8801/`**（或用户平时的入口）
* **必须 hard refresh**（macOS：`Cmd+Shift+R`）—— 清掉旧 episode 卡片 / truth marker / resume / coverage
* 正式重新审核**从第一条 Development PS 开始**：`01021 15:55:33–16:00:37`
* coverage 必须由**用户本人**播放产生，任何人都不要代替用户观看

---

## 10. 相关产物（本次 A/B 与验收的可读证据）

* `output/ground_litter_ab_20260928/README.md` —— A/B 配置全表（尺寸/码率/p50/p95/max）+ 图片索引
* `output/ground_litter_ab_20260928/PS4_f00*_ROI.png` —— 列序固定 **SOURCE | A | B** 的三列对照图
* `/home/sf01/step2c1-blind-truth/proxy-b-review-only-20260928/` —— 当时给用户做 B 目视验收的独立只读页面
  （`server.py` + `index.html` + 34 张原生 source 帧 + `B.mp4`）。**该页面没有 POST/PUT/PATCH/DELETE 处理函数**，
  `/api/truth`、`/api/eof`、`/api/playback` 实测全部 405，可复用。

---

## 11. 完成记录（2026-09-29 04:10 CST）

四个阶段全部执行并验证完毕，**无需任何后续动作**。

### 批处理结果

```
65 / 65 VERIFIED      0 failed      0 missing      0 partial
总编码耗时 8.9 h      平均 490 s/条
shard 0 done in 524.1 min
mean size 199,707,809 B  (≈5.26 Mbps)
manifest sha256 b6fc98f5f8f1e18290140b5707f552cf9fc785f2658ca5de086faf5778fb82c6
```

manifest 校验通过：`source_manifest.sha256` = `66f908f4…`（= frozen development_manifest），
`audit_manifest_sha256` = `2f8ea4b3…`（= frozen audit_manifest）。抽样条目：
`duration 303.92 / start 0.08 / codec h264 / 2560×1440 / 25 fps / faststart true / VERIFIED`。

### 切换结果

```
旧 pid 872502  --proxy-root .../review-proxy            (recipe a, CRF18)
新 pid 2285313 --proxy-root .../review-proxy-b-v1 --proxy-recipe b-v1
```

同一份隔离配置（8807）对第一条 Development PS 返回 **200,564,734 B**（= proxy B），
而不是 A 的 321,526,193 B，且 `/api/state` 仍为 0/65。

### 终验结果

| 检查 | 结果 |
|---|---|
| `development_manifest.json` | `66f908f4…` **未变** |
| `audit_manifest.json` | `2f8ea4b3…` **未变** |
| artifact 内容 | 只有两个 frozen manifest（blank slate） |
| reviewed | **0 / 65** |
| truth / episodes / active | **0 / 0 / 0** |
| resume | `None` |
| 第一条 PS | `01021 15:55:33–16:00:37`，未开始 |
| 隔离浏览器实测 | 1× 2.50 s、2× 5.00 s、4× 10.00 s（每 2.5 s 墙钟），readyState 4，2560×1440 |
| reset 后 DOM | 0 episode 卡片、无 `ep-000x`、无 `E1–E5`、coverage 0.00%、无 resume |
| 监听端口 | 只有 official `8801`（pid 2285313） |
| 保留资产 | proxy A 35 files / 9.4 G；proxy B 65 files / 13 G；备份目录保留；磁盘 262 G free |
| 安全 | Sealed 0 · detector false · checkpoint false · automated official playback false |

### 交接对象可以忽略本文档的 §1

§1 的四个步骤已全部执行。若日后仍需重做，请注意：
* 生成：`run_gen_b.sh`（可续、幂等）
* 校验：`verify_proxy_b.py`（约 45–60 min，全量 decode）
* 切换：`switch_official_to_b.sh`
* 终验：`final_accept.sh`
四者都在 `/home/sf01/step2c1-blind-truth/proxy-b-tools-20260928/`。

工具与本文档同时存在于服务器 `/home/sf01/step2c1-blind-truth/HANDOVER_STEP2C1B_PROXY_B_CLOSEOUT_20260928.md`。
