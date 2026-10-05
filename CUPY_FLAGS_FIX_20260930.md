# 2026-09-30 生产故障：cupy `flags.so` 镜像层损坏导致旁路流 SIGSEGV

## 1. 现象

- 2026-09-30 16:24–16:57（CST），`rtsp-yolo-api` 的流任务约一半启动即 `exit_code=-11`（SIGSEGV）。
- `kern.log` 共 13 次 segfault，其中 11 次故障指令在动态链接器：
  - 10 次：`... in ld-linux-x86-64.so.2[f90c,...] error 6/7`，fault 尾号 `f868`
  - 1 次（16:24）：`ld-linux-x86-64.so.2[fb7b,...] error 6`，fault 尾号 `f188`
- 正常流可跑到 `capture_fps≈25 / publish_fps≈25 / healthy`，说明主链、NVENC、MediaMTX 均正常。
- 同时期服务器刚由人工在 16:19 安装 `nvidia-driver-580-open` 并 16:21 重启（此前因内核 7.0.0-34
  缺少 NVIDIA 模块而停服）。**该驱动变更与本故障无关**，见 §3。

## 2. 根因

`cupy` wheel 的 Cython 扩展在镜像层里被损坏：

```
/usr/local/lib/python3.12/dist-packages/cupy/_core/flags.cpython-312-x86_64-linux-gnu.so
```

该文件有 ≥4 处“bit 被置 1”的字节改动，**包含代码段**，导致：

1. 两条重定位项越界：
   - `r_offset 0x20020f868`（正确 `0x20f868`，多 `2^33`），类型 `R_X86_64_RELATIVE`
   - `r_offset 0x20020f188`（正确 `0x20f188`），类型 `R_X86_64_JUMP_SLOT`（`PyErr_Clear`）
   → `ld.so` 按 `base + r_offset` 写到模块外 8.6 GB：RELATIVE 走加载期重定位
   （`ld.so+0xf90c`），JUMP_SLOT 走 lazy binding（`ld.so+0xfb7b`），与内核日志的两种签名完全对应。
2. `.dynstr` 中符号名损坏：`PyDict_Next` → `PyDict_Ngxt`。
3. 修掉以上 3 处后仍与 wheel 记录哈希不符，且 `import cupy` 变成 SIGILL（rc=132）→ 代码段也有坏字节。

`import cupy` 在该镜像中**100% 崩溃**（`python3 -c "import cupy"` → rc=139），且**所有历史镜像
（8 个 tag，含 8/21、9/16、9/18）该文件 sha256 完全相同**，属共享基础层缺陷。

触发链路（gdb 从 apport core 解出）：

```
#10 dlopen()  ←  #25 __Pyx_Import  ←  #26 __pyx_pymod_exec_core (cupy/_core/core.cpp)
#15 _PyEval_EvalFrameDefault  ...  rtsp_annotator.deepstream_worker
```

代码中唯一 `import cupy` 的位置是 `deepstream_worker._frame_to_small_numpy()` 的兜底分支：
仅当 ServiceMaker 回调拿到的是 CUDA tensor（`np.from_dlpack` 无法直接消费）时才会执行。
因此“时好时坏”= 是否走到该 GPU tensor 分支。16:24 那次还说明坏重定位**可能写进某个可写页而
“成功”**，随后才在 lazy binding 处崩溃，即存在静默内存破坏风险。

## 3. 排除项与完整性核验

- **与 NVIDIA 驱动/内核无关**：干净 `docker run` 即可复现 `import cupy` 段错误；所有历史镜像
  该文件哈希一致。
- cupy wheel 自带 `RECORD`（安装时记录的每文件 sha256）：
  - 期望 `sha256=LcusLOrJmaM13Kn3AheTLsvIkLLyLqzmpc4pVQkI_sQ`
    （hex `2dcbac2ceac999a335dca9f70217932ecbc890b2f22eace6a5ce29550908fec4`）
  - 实际 `9cac4961c285426d731eb569b21cd0e6fedb3555c9db99028882c45999ced82c`
- 全量完整性扫描：85 个 pip 包 / **24678 个文件**按 `RECORD` 校验，**仅此 1 个不匹配**；
  `dpkg -V` 无校验和不匹配（只有 man page 裁剪等正常缺失）。
- 全部 kern.log 轮转无 ext4 / I/O / 块设备错误；磁盘剩余 260 GB。
- 结论：**孤立单文件损坏，不是 wheel 上游缺陷，也不是存储大面积劣化**（但成因未知，见 §7）。

## 4. 修复

从 PyPI 取同一版本 wheel，抽出该文件并校验哈希：

```
cupy_cuda12x-13.4.1-cp312-cp312-manylinux2014_x86_64.whl
  成员 cupy/_core/flags.cpython-312-x86_64-linux-gnu.so
  size 410704
  sha256 2dcbac2ceac999a335dca9f70217932ecbc890b2f22eace6a5ce29550908fec4
  == 镜像 RECORD 期望值  ✅
```

增量镜像（不改任何业务代码）：

```dockerfile
# releases/cupy-flags-fix-20260930/Dockerfile
FROM rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918
COPY flags.cpython-312-x86_64-linux-gnu.so \
     /usr/local/lib/python3.12/dist-packages/cupy/_core/flags.cpython-312-x86_64-linux-gnu.so
```

## 5. 部署记录

| 项 | 值 |
| --- | --- |
| 新镜像 | `rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260930-cupyfix`（manifest `dbce808fd146`） |
| 回滚镜像 | `rtsp-yolo-annotator:deepstream8-before-cupyfix-20260930` → `7aa71d92df2e`（原生产镜像） |
| 新增 compose | `docker-compose.cupy-flags-fix-20260930.override.yml`（链上第 7 个，只改 `api.image`） |
| 变更范围 | 仅 `up -d --no-deps api`；`rtsp-mediamtx`、`camera-control`、`rtsp-web-gateway` 未重启 |
| 构建/部署证据 | 构建上下文 `releases/cupy-flags-fix-20260930/`；`docker compose config -q` 通过 |

**回滚**：去掉第 7 个 `-f docker-compose.cupy-flags-fix-20260930.override.yml` 再
`docker compose <原 6 个 -f> up -d --no-deps api`（不必重建镜像）。

## 6. 验证结果

- 运行中容器内该文件 sha256 = `2dcbac2c…`（正确）。
- 运行中容器内 `import cupy` → `cupy 13.4.1`、`devices 1`、`gpu_sum 499500.0`、`cupy._core.flags` 正常。
- 容器 `Status=running`、`Restarts=0`、`Error=` 空；启动日志无 traceback/error。
- `http://14.21.88.97:38080/health` = 200、`/docs` = 200、`/v1/streams`（无 key）= 401（可达），
  带 `X-API-Key`（服务器本机）= 200。
- **未做真实摄像机端到端验收**：API 重启清空了内存中的流任务，需要业务方用新的签名摄像头地址重建；
  建流后再观察 ground_litter/gas_cylinder 旁路是否稳定。

## 7. 遗留与风险

- **损坏成因未知**：单文件、多处 bit 置 1、跨所有共享该层的镜像一致。存储面证据干净（无 I/O 错误、
  其余 24678 个文件完好）。建议在维护窗口跑 `sudo smartctl -a /dev/nvme0n1` 看 SMART，并观察是否复发。
- 旧镜像层里的坏文件仍在（所有历史 tag 的 cupy 依旧不可用）；只有 cupyfix 之后的镜像可用。
- 兜底分支仍是 `import cupy`。若需更强健壮性，可改为 `torch.from_dlpack(...).cpu().numpy()`
  （镜像内已有 torch），本次未改代码。
- `/var/lib/apport/coredump` 累积约 11 GB、`/var/crash` 约 1 GB（每次崩溃写 ≈2 GB core），
  需 sudo 清理，并建议限制 apport/core dump。
- 事件/截图、`data/` 与 `engines/` 未改动。
