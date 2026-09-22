# 暂停的 v6 草稿存档（2026-09-21）

本目录只做**保存现场**，不重置、不清理、不提交到 main。

## 已跟踪文件（binary patch）

`v6_wip.patch` 由以下命令生成（相对 `682fad1` 的工作区改动）：

```bash
git diff --binary -- \
  rtsp_annotator/ground_litter_profile_bank.py \
  rtsp_annotator/ground_litter_profile_sampling.py \
  rtsp_annotator/ground_litter_profile_match.py \
  scripts/build_ground_litter_profile_bank.py \
  tests/profile_bank_fixtures.py \
  tests/test_ground_litter_profile_c1c4.py \
  > output/profile_factory_v6_paused_20260921/v6_wip.patch
```

恢复草稿（在**另一个** worktree 或临时分支上执行，不要污染 main）：

```bash
git apply --binary output/profile_factory_v6_paused_20260921/v6_wip.patch
```

## 未跟踪文档（原样保留）

- `docs/plans/2026-09-21-profile-factory-v6-boundary-and-feasibility.md`

## 说明

- 工作区**保持原样**（含未提交改动），未 `reset`、未 `stash`、未删除任何文件。
- v6 草稿**没有**提交到 main，也没有复制进实验 worktree。
