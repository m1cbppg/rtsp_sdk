"""B1：纯有状态 Selector（方案二 §2/§3.2/§4/§5）。

本模块**只有**时序逻辑：没有图像、没有 I/O、没有模型。离线工厂（A4/A5）与未来在线
旁路必须运行同一个 ``ProfileSelector``，否则「N 由完整 Selector 定稿」就没有意义。

实现的设计复核项：

* **F1 扩展检索**：Top-K 只是快速路径。找不到合格候选或候选全尺寸验证失败时，
  建立覆盖全库的待查队列，每 tick 有界推进，保留游标；冷却项不占名额，
  已经排队的候选仍有被检查的机会。
* **F4 动态定稿**：静态覆盖只是预选输入；本模块输出动态可用覆盖、暂停区间、
  切换次数与提交时延，供工厂剪枝时判断「过渡参考是否缩短暂停」。

状态机与 ``matcher.json`` 的 ``selection`` 参数一一对应，全部可从配置覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

STATUS_MATCHED = "MATCHED"
STATUS_SWITCH_PENDING = "SWITCH_PENDING"
STATUS_UNKNOWN = "UNKNOWN"

PHASE_STEADY = "STEADY"
PHASE_SEARCHING = "SEARCHING"
PHASE_PREPARING = "PREPARING"
PHASE_VALIDATING = "VALIDATING"

SEARCH_REASON_OK = "OK"
SEARCH_REASON_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"
SEARCH_REASON_NO_ELIGIBLE = "NO_ELIGIBLE_PROFILE"
SEARCH_REASON_WAITING_RECOVERY = "WAITING_RECOVERY_SPAN"
SEARCH_REASON_WAITING_SWITCH = "WAITING_SWITCH_EVIDENCE"
SEARCH_REASON_WAITING_DWELL = "WAITING_MIN_DWELL"

DEFAULT_SELECTION_CONFIG: dict[str, Any] = {
    "top_k": 3,
    "max_small_matches_per_tick": 4,
    "switch_improvement_ratio": 0.15,
    "switch_min_samples": 3,
    "switch_min_span_seconds": 4.0,
    "min_dwell_seconds": 10.0,
    "recovery_min_samples": 2,
    "recovery_min_span_seconds": 2.0,
    "failed_candidate_cooldown_seconds": 10.0,
    "max_result_age_seconds": 4.0,
    "max_observation_gap_seconds": 4.0,
    "score_floor": 0.25,
    "expanded_new_candidates_per_tick": 3,
    # 源录像之间的空隙（换文件/停机）不是「Selector 暂停」；超过该值的间隔
    # 会被当作时间线断开：清空连续证据，并且不计入覆盖率分母。
    "join_gap_seconds": 300.0,
    # 计划采样周期：一次观测最多只能证明到下一个计划观察点之前（R3）。
    "tick_interval_seconds": 2.0,
    # 结果有效期：旧证据不得延长有效覆盖。
    "result_validity_seconds": 4.0,
}


class SelectorError(ValueError):
    """Selector 输入非法。"""


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """本 tick 从共享 matcher 得到的候选结果（小图或全尺寸验证都走这里）。

    ``verified=True`` 表示它已经过全尺寸验证；未验证的候选只能用于请求预加载。
    """

    profile_id: str
    score: float
    enter_eligible: bool
    hold_eligible: bool
    rank: int = 0
    verified: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    bank_id: str
    bank_version: str
    view_id: str
    selected_profile_id: str | None
    candidate_profile_id: str | None
    status: str
    phase: str
    prior_allowed: bool
    score_current: float | None
    score_best: float | None
    reason: str
    visible_fraction: float
    valid_fraction: float
    tested_profile_ids: tuple[str, ...]
    search_cursor: int
    search_exhausted: bool
    budget_exhausted: bool
    input_timestamp: float
    profile_generation: int
    alignment_generation: int
    improvement: float | None
    commit_requested: bool
    commit_profile_id: str | None
    switch_count: int
    dwell_seconds: float
    # 跨模块契约（v4 复核后）：环境匹配能力与 prior 能力分开输出。
    #   active_profile_id  —— 当前最匹配环境的 Profile；
    #   prior_profile_id    —— 允许产生 prior 的 Profile；unsuitable 时为 None；
    #   profile_match_available —— 是否有可用参考（区分"无匹配"与"匹配但 prior 不可靠"）；
    #   prior_available     —— 本 tick 是否允许 prior-only 候选；
    #   prior_unavailable_reason —— NO_MATCHED_PROFILE / PROFILE_PRIOR_UNSUITABLE /
    #                               NOT_OBSERVABLE / None。
    active_profile_id: str | None = None
    prior_profile_id: str | None = None
    profile_match_available: bool = False
    prior_available: bool = False
    prior_unavailable_reason: str | None = None
    prior_generation: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "bank_version": self.bank_version,
            "view_id": self.view_id,
            "selected_profile_id": self.selected_profile_id,
            "candidate_profile_id": self.candidate_profile_id,
            "status": self.status,
            "phase": self.phase,
            "prior_allowed": self.prior_allowed,
            "active_profile_id": self.active_profile_id,
            "prior_profile_id": self.prior_profile_id,
            "profile_match_available": self.profile_match_available,
            "prior_available": self.prior_available,
            "prior_unavailable_reason": self.prior_unavailable_reason,
            "prior_generation": self.prior_generation,
            "score_current": None if self.score_current is None
            else round(self.score_current, 5),
            "score_best": None if self.score_best is None
            else round(self.score_best, 5),
            "reason": self.reason,
            "visible_fraction": round(self.visible_fraction, 4),
            "valid_fraction": round(self.valid_fraction, 4),
            "tested_profile_ids": list(self.tested_profile_ids),
            "search_cursor": self.search_cursor,
            "search_exhausted": self.search_exhausted,
            "budget_exhausted": self.budget_exhausted,
            "input_timestamp": self.input_timestamp,
            "profile_generation": self.profile_generation,
            "alignment_generation": self.alignment_generation,
            "improvement": None if self.improvement is None else round(self.improvement, 5),
            "commit_requested": self.commit_requested,
            "commit_profile_id": self.commit_profile_id,
            "switch_count": self.switch_count,
            "dwell_seconds": round(self.dwell_seconds, 3),
            "diagnostics": self.diagnostics,
        }


@dataclass(slots=True)
class _Evidence:
    profile_id: str
    timestamps: list[float] = field(default_factory=list)

    def observe(self, timestamp: float, *, max_gap: float) -> None:
        if self.timestamps and not 0 < timestamp - self.timestamps[-1] <= max_gap:
            self.timestamps.clear()
        self.timestamps.append(float(timestamp))

    def count(self) -> int:
        return len(self.timestamps)

    def span(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def clear(self) -> None:
        self.timestamps.clear()


@dataclass(slots=True)
class _SearchState:
    queue: list[str] = field(default_factory=list)
    cursor: int = 0
    visited: set[str] = field(default_factory=set)
    round_index: int = 0

    def exhausted(self) -> bool:
        return self.cursor >= len(self.queue)


PRIOR_UNAVAILABLE_NO_MATCH = "NO_MATCHED_PROFILE"
PRIOR_UNAVAILABLE_UNSUITABLE = "PROFILE_PRIOR_UNSUITABLE"
PRIOR_UNAVAILABLE_NOT_OBSERVABLE = "NOT_OBSERVABLE"


class ProfileSelector:
    """有状态、纯逻辑的 Profile 选择器。"""

    def __init__(
        self, *, bank_id: str, bank_version: str, view_id: str,
        profile_ids: Sequence[str], config: Mapping[str, Any] | None = None,
        algorithm_version: str = "selector_r3",
        prior_suitable_profile_ids: Sequence[str] | None = None,
    ) -> None:
        if not profile_ids:
            raise SelectorError("Selector 至少需要一个候选 Profile")
        if len(set(profile_ids)) != len(profile_ids):
            raise SelectorError("profile_ids 不能重复")
        self.bank_id = str(bank_id)
        self.bank_version = str(bank_version)
        self.view_id = str(view_id)
        self.profile_ids = tuple(str(item) for item in profile_ids)
        # prior 能力集合：缺省（老调用/离线诊断）视为"未知"，
        # 此时保持旧行为——不做 prior 门禁，避免静默改变历史对照。
        if prior_suitable_profile_ids is None:
            self.prior_suitable_profile_ids: frozenset[str] | None = None
        else:
            declared = {str(item) for item in prior_suitable_profile_ids}
            unknown = declared - set(self.profile_ids)
            if unknown:
                raise SelectorError(
                    f"prior_suitable_profile_ids 含未知 Profile: {sorted(unknown)}"
                )
            self.prior_suitable_profile_ids = frozenset(declared)
        self.config = {**DEFAULT_SELECTION_CONFIG, **dict(config or {})}
        self.algorithm_version = algorithm_version
        self.prior_generation = 0
        self._prior_suitability_last: bool | None = None

        self.selected_profile_id: str | None = None
        self.pending_profile_id: str | None = None
        self.profile_generation = 0
        self.alignment_generation = 0
        self.switch_count = 0
        self._dwell_start: float | None = None
        self._last_timestamp: float | None = None
        self._cooldown_until: dict[str, float] = {}
        self._search = _SearchState()
        self._challenger: _Evidence | None = None
        self._recovery: _Evidence | None = None
        self._last_matched_at: float | None = None
        self._paused_since: float | None = None
        self._timeline: list[dict[str, Any]] = []
        self._last_reason = SEARCH_REASON_NO_ELIGIBLE
        self._last_budget_exhausted = False
        self._tick_index = 0

    # -- 参数访问 ---------------------------------------------------------- #

    def _number(self, key: str) -> float:
        try:
            value = float(self.config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectorError(f"selection.{key} 非法") from exc
        if not math.isfinite(value) or value < 0:
            raise SelectorError(f"selection.{key} 必须是非负有限值")
        return value

    @property
    def score_floor(self) -> float:
        return max(self._number("score_floor"), 1e-6)

    @property
    def max_observation_gap(self) -> float:
        return max(self._number("max_observation_gap_seconds"), 1e-6)

    # -- 检索队列 ---------------------------------------------------------- #

    def reset_search(self, *, reason: str) -> None:
        """开启新的一轮全库检索（bank/view 改变或持续显著全局外观变化）。"""
        self._search.queue = list(self.profile_ids)
        self._search.cursor = 0
        self._search.visited = set()
        self._search.round_index += 1
        self._last_budget_exhausted = False
        self._timeline.append({
            "kind": "search_round", "round": self._search.round_index,
            "reason": reason, "at": self._last_timestamp,
        })

    def _ensure_search(self) -> None:
        if not self._search.queue:
            self.reset_search(reason="initial")

    def _eligible_pending(self, now: float) -> list[str]:
        return [
            profile_id for profile_id in self._search.queue[self._search.cursor:]
            if self._cooldown_until.get(profile_id, 0.0) <= now
        ]

    def _consume(self, count: int, now: float, prefer: Sequence[str] = ()) -> list[str]:
        """按游标推进，返回本 tick 要检查的 profile_id（冷却项跳过但计入访问）。

        ``prefer`` 里的候选优先占用名额：正在累计恢复/切换证据的候选必须能被
        反复观测，否则「两次且跨度至少两秒」永远无法满足。新候选仍会在剩余
        名额里推进，不会被偏好项永久饿死。
        """
        self._ensure_search()
        picked: list[str] = []
        for profile_id in prefer:
            if len(picked) >= count:
                break
            if profile_id in picked or profile_id not in self.profile_ids:
                continue
            if self._cooldown_until.get(profile_id, 0.0) > now:
                continue
            picked.append(profile_id)
        while len(picked) < count and self._search.cursor < len(self._search.queue):
            profile_id = self._search.queue[self._search.cursor]
            self._search.cursor += 1
            if profile_id in self._search.visited or profile_id in picked:
                continue
            self._search.visited.add(profile_id)
            if self._cooldown_until.get(profile_id, 0.0) > now:
                continue
            picked.append(profile_id)
        return picked

    def plan_tick(self, *, timestamp: float, current_hold_eligible: bool,
                  current_enter_eligible: bool = False) -> dict[str, Any]:
        """决定本 tick 要检查哪些候选。

        快速路径给 Top-K（由调用方按粗距离排序后传入 ``fast_rank``），扩展路径
        每 tick 至少推进一个新候选，保证第 K+1 个仍有机会（F1）。
        """
        self._ensure_search()
        if current_hold_eligible:
            plan = {
                "mode": "fast" if not current_enter_eligible else "hold",
                "limit": int(self.config["max_small_matches_per_tick"]),
                "reserved": list(self.profile_ids[: int(self.config["top_k"])]),
            }
        else:
            plan = {
                "mode": "expanded",
                "limit": int(self.config["max_small_matches_per_tick"]),
                "reserved": self._consume(
                    int(self.config.get("expanded_new_candidates_per_tick", 3)),
                    timestamp, prefer=self._pending_observations(),
                ),
            }
        return plan

    def _pending_observations(self) -> list[str]:
        """正在累计恢复证据的候选：优先重看，保证跨度条件可满足。"""
        if self._recovery is not None:
            return [self._recovery.profile_id]
        if self._challenger is not None:
            return [self._challenger.profile_id]
        return []

    # -- 主入口 ------------------------------------------------------------ #

    def observe(
        self, *,
        timestamp: float,
        current: CandidateMatch | None,
        candidates: Sequence[CandidateMatch],
        visible_fraction: float = 1.0,
        valid_fraction: float = 1.0,
        tested_profile_ids: Sequence[str] | None = None,
        budget_exhausted: bool = False,
        alignment_generation: int | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        observable: bool = True,
        non_observable_reason: str = "",
        tick_interval_seconds: float | None = None,
    ) -> SelectionDecision:
        """消费一个分析 tick 的匹配结果，返回使用/切换意图。

        本方法**只提出意图**：真正提交必须由调用方完成全尺寸验证后调用
        ``commit``/``fail``。小图评分通过绝不等于已经切换。
        """
        if not math.isfinite(timestamp):
            raise SelectorError("timestamp 必须是有限值")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            # 源时间倒退 = 新时间代际；清空连续证据而不是把它当负间隔。
            self._challenger = None
            self._recovery = None
            self.alignment_generation += 1
        elif (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp > self._number("join_gap_seconds")
        ):
            # 跨文件的长时间空隙不是「连续观测」：断开证据但不改代际。
            self._challenger = None
            self._recovery = None
        self._last_timestamp = float(timestamp)
        self._tick_index += 1
        # 本次观测最多能证明到 min(计划采样周期, 结果有效期) 之后（R3）。
        interval = (
            self._number("tick_interval_seconds")
            if tick_interval_seconds is None else float(tick_interval_seconds)
        )
        # 观测间隔上限必须随**实际**节拍自适应：默认 4s 只对 2s 级节拍成立。
        # 离线回放/低分析频率下若沿用 4s，每个 tick 都会清空连续证据，
        # 动态覆盖会恒为 0（这正是评审 R3/F4 陷阱的另一种表现）。
        if interval > 0:
            required_gap = max(self._number("max_observation_gap_seconds"),
                               2.0 * interval)
            if required_gap > self._number("max_observation_gap_seconds"):
                self.config["max_observation_gap_seconds"] = float(required_gap)
        self._observation_span = max(
            0.0, min(interval, self._number("result_validity_seconds")),
        )
        if alignment_generation is not None:
            self.alignment_generation = int(alignment_generation)
        self._last_budget_exhausted = bool(budget_exhausted)

        if current is not None and current.profile_id != self.selected_profile_id:
            # 调用方报告的 current 必须与内部选择一致，否则会串代。
            raise SelectorError("current 与 selected_profile_id 不一致")

        candidate_by_id = {item.profile_id: item for item in candidates}
        if current is not None:
            candidate_by_id.setdefault(current.profile_id, current)
        # 只有本 tick 真正被检查过的候选才能参与决策；否则会把「粗距离靠前」
        # 误当成「已评估」，扩展开销也就失去意义。
        if tested_profile_ids is not None:
            tested_set = set(tested_profile_ids)
            candidates = [
                item for item in candidates
                if item.profile_id in tested_set
                or (current is not None and item.profile_id == current.profile_id)
            ]

        self._expire_state(timestamp)
        commit_requested = False
        commit_profile_id: str | None = None
        improvement: float | None = None

        current_hold = bool(current and current.hold_eligible)
        current_enter = bool(current and current.enter_eligible)

        if current_hold:
            self._mark_matched(timestamp)
            challenger = self._pick_challenger(current, candidates, timestamp)
            if challenger is None:
                self._challenger = None
                decision_status = STATUS_MATCHED
                decision_phase = PHASE_STEADY
                self._last_reason = SEARCH_REASON_OK
            else:
                improvement = self._improvement(current.score, challenger.score)
                if self._challenger is None or self._challenger.profile_id != challenger.profile_id:
                    self._challenger = _Evidence(challenger.profile_id)
                self._challenger.observe(timestamp, max_gap=self.max_observation_gap)
                dwell = self.dwell_seconds
                enough = (
                    self._challenger.count() >= int(self.config["switch_min_samples"])
                    and self._challenger.span() >= self._number("switch_min_span_seconds")
                )
                if enough and dwell >= self._number("min_dwell_seconds"):
                    self.pending_profile_id = challenger.profile_id
                    commit_requested = True
                    commit_profile_id = challenger.profile_id
                    decision_status = STATUS_SWITCH_PENDING
                    decision_phase = PHASE_VALIDATING
                    self._last_reason = "SWITCH_READY"
                else:
                    decision_status = STATUS_MATCHED
                    decision_phase = PHASE_PREPARING
                    if not enough:
                        self._last_reason = SEARCH_REASON_WAITING_SWITCH
                    else:
                        self._last_reason = SEARCH_REASON_WAITING_DWELL
        else:
            # 旧参考失效：当帧停用 prior 证据，不受驻留限制。
            if self._paused_since is None:
                self._paused_since = timestamp
            self._challenger = None
            # 上一 tick 未提交的切换意图随旧参考失效作废；本 tick 若重新
            # 请求提交会在下面重新写入 pending。
            self.pending_profile_id = None
            recovery = self._pick_recovery(candidates, timestamp)
            if recovery is None:
                decision_status = STATUS_UNKNOWN
                decision_phase = PHASE_SEARCHING
                self._last_reason = (
                    SEARCH_REASON_BUDGET_EXHAUSTED if budget_exhausted
                    else SEARCH_REASON_NO_ELIGIBLE
                )
            else:
                if self._recovery is None or self._recovery.profile_id != recovery.profile_id:
                    self._recovery = _Evidence(recovery.profile_id)
                fresh = recovery.diagnostics.get("fresh", True)
                if fresh:
                    self._recovery.observe(timestamp, max_gap=self.max_observation_gap)
                enough = (
                    self._recovery.count() >= int(self.config["recovery_min_samples"])
                    and self._recovery.span() >= self._number("recovery_min_span_seconds")
                )
                if enough:
                    self.pending_profile_id = recovery.profile_id
                    commit_requested = True
                    commit_profile_id = recovery.profile_id
                    decision_status = STATUS_SWITCH_PENDING
                    decision_phase = PHASE_VALIDATING
                    self._last_reason = "RECOVERY_READY"
                else:
                    decision_status = STATUS_UNKNOWN
                    decision_phase = PHASE_SEARCHING
                    self._last_reason = SEARCH_REASON_WAITING_RECOVERY

        best = min(
            (item for item in candidates if item.enter_eligible),
            key=lambda item: (item.score, item.profile_id), default=None,
        )
        tested = tuple(tested_profile_ids or [item.profile_id for item in candidates])
        # 环境匹配可用性：有 active 参考且本 tick 可判断。
        active = self.selected_profile_id
        profile_match_available = bool(active) and bool(observable)
        prior_suitable = (
            True if self.prior_suitable_profile_ids is None
            else bool(active) and active in self.prior_suitable_profile_ids
        )
        prior_available = bool(current_hold) and bool(observable) and prior_suitable
        prior_profile_id = active if prior_available else None
        if prior_available:
            prior_unavailable_reason = None
        elif not observable:
            prior_unavailable_reason = PRIOR_UNAVAILABLE_NOT_OBSERVABLE
        elif not active:
            prior_unavailable_reason = PRIOR_UNAVAILABLE_NO_MATCH
        elif not prior_suitable:
            prior_unavailable_reason = PRIOR_UNAVAILABLE_UNSUITABLE
        else:
            prior_unavailable_reason = PRIOR_UNAVAILABLE_NOT_OBSERVABLE
        if self._prior_suitability_last is not None and (
            self._prior_suitability_last != bool(prior_available)
        ):
            # prior 能力发生变化（恢复或暂停）：按代际重新开始，不继承旧证据。
            self.prior_generation += 1
        self._prior_suitability_last = bool(prior_available)
        decision = SelectionDecision(
            bank_id=self.bank_id,
            bank_version=self.bank_version,
            view_id=self.view_id,
            selected_profile_id=self.selected_profile_id,
            candidate_profile_id=(
                commit_profile_id if commit_requested
                else (self._challenger.profile_id if self._challenger else None)
            ),
            status=decision_status,
            phase=decision_phase,
            prior_allowed=current_hold,
            active_profile_id=active,
            prior_profile_id=prior_profile_id,
            profile_match_available=profile_match_available,
            prior_available=prior_available,
            prior_unavailable_reason=prior_unavailable_reason,
            prior_generation=self.prior_generation,
            score_current=None if current is None else current.score,
            score_best=None if best is None else best.score,
            reason=self._last_reason,
            visible_fraction=float(visible_fraction),
            valid_fraction=float(valid_fraction),
            tested_profile_ids=tested,
            search_cursor=self._search.cursor,
            search_exhausted=self._search.exhausted(),
            budget_exhausted=bool(budget_exhausted and not current_hold),
            input_timestamp=float(timestamp),
            profile_generation=self.profile_generation,
            alignment_generation=self.alignment_generation,
            improvement=improvement,
            commit_requested=commit_requested,
            commit_profile_id=commit_profile_id,
            switch_count=self.switch_count,
            dwell_seconds=self.dwell_seconds,
            diagnostics={
                "tick": self._tick_index,
                "cooldown": {
                    key: round(value, 3) for key, value in self._cooldown_until.items()
                    if value > timestamp
                },
                "queue_length": len(self._search.queue),
                "search_round": self._search.round_index,
                **dict(diagnostics or {}),
            },
        )
        record = decision.as_dict()
        record.update({
            "observable": bool(observable),
            "non_observable_reason": str(non_observable_reason or ""),
            "observation_span_seconds": round(float(self._observation_span), 4),
            "prior_allowed": bool(current_hold) and bool(observable),
            "prior_available": prior_available,
            "prior_profile_id": prior_profile_id,
            "prior_unavailable_reason": prior_unavailable_reason,
            "prior_generation": self.prior_generation,
        })
        self._timeline.append({"kind": "decision", **record})
        # prior_allowed 是按“可判断且当前参考仍合格”记录的，决策对象本身同步。
        return decision

    # -- 提交与失败 -------------------------------------------------------- #

    def commit(self, *, profile_id: str, timestamp: float,
               verified: bool = True) -> dict[str, Any]:
        """全尺寸验证成功后的原子提交。未验证的结果一律拒绝。"""
        if not verified:
            raise SelectorError("未通过全尺寸验证的候选不能提交")
        if self.pending_profile_id != profile_id and self.selected_profile_id != profile_id:
            raise SelectorError("提交的 profile_id 与待验证意图不一致")
        previous = self.selected_profile_id
        same_profile_id = previous == profile_id
        self.selected_profile_id = profile_id
        self.pending_profile_id = None
        self.profile_generation += 1
        self._dwell_start = float(timestamp)
        self._challenger = None
        self._recovery = None
        if not same_profile_id:
            self.switch_count += 1
        if self._paused_since is not None:
            pause = float(timestamp) - self._paused_since
            self._paused_since = None
        else:
            pause = 0.0
        self._mark_matched(timestamp)
        record = {
            "kind": "commit", "profile_id": profile_id, "previous": previous,
            "same_profile_id": same_profile_id, "generation": self.profile_generation,
            "timestamp": float(timestamp), "pause_seconds": round(pause, 3),
            "switch_count": self.switch_count,
        }
        self._timeline.append(record)
        return record

    def fail(self, *, profile_id: str, timestamp: float,
             reason: str = "full_size_verification_failed") -> dict[str, Any]:
        """全尺寸验证失败：候选进入冷却，旧参考状态不变。"""
        cooldown = self._number("failed_candidate_cooldown_seconds")
        self._cooldown_until[profile_id] = float(timestamp) + cooldown
        if self.pending_profile_id == profile_id:
            self.pending_profile_id = None
        if self._challenger and self._challenger.profile_id == profile_id:
            self._challenger = None
        if self._recovery and self._recovery.profile_id == profile_id:
            self._recovery = None
        record = {
            "kind": "fail", "profile_id": profile_id, "reason": reason,
            "cooldown_seconds": cooldown, "timestamp": float(timestamp),
        }
        self._timeline.append(record)
        return record

    def discard_stale(self, *, profile_id: str, observed_at: float,
                      now: float) -> bool:
        """过期验证结果直接丢弃，不提交（方案二 §5.2）。"""
        age = float(now) - float(observed_at)
        if age > self._number("max_result_age_seconds"):
            self._timeline.append({
                "kind": "stale_discard", "profile_id": profile_id,
                "age_seconds": round(age, 3),
            })
            return True
        return False

    # -- 查询 -------------------------------------------------------------- #

    @property
    def dwell_seconds(self) -> float:
        if self._dwell_start is None or self._last_timestamp is None:
            return 0.0
        return max(0.0, self._last_timestamp - self._dwell_start)

    def is_cooling(self, profile_id: str, timestamp: float) -> bool:
        return self._cooldown_until.get(profile_id, 0.0) > timestamp

    def timeline(self) -> list[dict[str, Any]]:
        return list(self._timeline)

    def mark_non_observable(
        self, *, seconds: float, reason: str = "UNOBSERVED",
        timestamp: float | None = None,
    ) -> None:
        """登记一段**无法判断**的真实时间（录像空缺、解码失败、坏帧等）。

        这段既不计入有效覆盖，也不计入“Selector 暂停”：暂停是算法问题，
        不可观测是素材问题，混在一起会同时高估覆盖率与误判算法行为（R3）。
        """
        if seconds <= 0:
            return
        self._timeline.append({
            "kind": "gap",
            "reason": str(reason),
            "seconds": round(float(seconds), 4),
            "at": None if timestamp is None else float(timestamp),
        })

    def mark_alignment_change(self, generation: int) -> None:
        """几何代际变化：旧 warp 不能复用，重新验证（方案二 §5.2）。"""
        self.alignment_generation = int(generation)
        self._challenger = None
        self._recovery = None
        self._timeline.append({
            "kind": "alignment_generation", "generation": int(generation),
            "at": self._last_timestamp,
        })

    # -- 统计（方案一 §7.2 统一口径）-------------------------------------- #

    def prior_summary(self) -> dict[str, Any]:
        """prior 通道的可用性统计（与 match 覆盖分开）。

        判据完全来自决策记录：``prior_available`` 是 Selector 在**当时**根据
        "active 参考合格 + 可判断 + 该参考 prior_suitable" 得出的结论，因此
        这里的统计就是运行时真实会输出的 prior 能力，而不是事后推断。
        """
        decisions = [
            row for row in self._timeline if row.get("kind") == "decision"
        ]
        effective = 0.0
        observed = 0.0
        pause_runs: list[float] = []
        pause_total = 0.0
        intervals: list[dict[str, Any]] = []
        open_interval: dict[str, Any] | None = None
        for row in decisions:
            span = float(row.get("observation_span_seconds", 0.0) or 0.0)
            observable = bool(row.get("observable", True))
            if not observable:
                continue
            observed += span
            if row.get("prior_available"):
                effective += span
                if pause_total > 0.0:
                    pause_runs.append(pause_total)
                    pause_total = 0.0
                if open_interval is not None:
                    intervals.append(open_interval)
                    open_interval = None
                continue
            pause_total += span
            reason = str(
                row.get("prior_unavailable_reason") or "PRIOR_NOT_AVAILABLE"
            )
            if open_interval is not None and open_interval["reason"] == reason:
                open_interval["seconds"] += span
                open_interval["ticks"] += 1
            else:
                if open_interval is not None:
                    intervals.append(open_interval)
                open_interval = {
                    "start": float(row.get("input_timestamp", 0.0)),
                    "reason": reason, "seconds": span, "ticks": 1,
                }
        if pause_total > 0.0:
            pause_runs.append(pause_total)
        if open_interval is not None:
            intervals.append(open_interval)
        for row in intervals:
            row["seconds"] = round(float(row["seconds"]), 4)
        pauses_sorted = sorted(pause_runs)
        return {
            "prior_effective_seconds": round(effective, 4),
            "prior_effective_coverage": (
                round(effective / observed, 5) if observed > 0 else 0.0
            ),
            "prior_pause_max": round(max(pause_runs), 4) if pause_runs else 0.0,
            "prior_pause_p95": round(
                pauses_sorted[
                    max(0, math.ceil(len(pauses_sorted) * 0.95) - 1)
                ], 4,
            ) if pause_runs else 0.0,
            "observed_seconds": round(observed, 4),
            "semantic_only_seconds": round(
                sum(row["seconds"] for row in intervals), 4,
            ),
            "semantic_only_intervals": intervals,
            "semantic_only_ticks": sum(row["ticks"] for row in intervals),
            "decisions": len(decisions),
        }

    def summarise(self, *, join_gap_seconds: float | None = None) -> dict[str, Any]:
        """按真实观测区间积分动态覆盖（R3）。

        每个决策记录只覆盖 ``observation_span_seconds``（= min(计划采样周期,
        结果有效期)），不再把有效状态延续到下一次观测。两种缺口分开记账：

        * ``off_air_seconds``：与上一次观测的间隔超过 ``join_gap_seconds``，
          超出部分视为**录像空缺**（停机/换文件），不计入分母也不计为暂停；
        * ``non_observable_seconds``：显式登记的解码失败/坏帧/零可用/无共用视野
          等**不可判断**区间。

        有效覆盖只统计“可判断且当前参考仍合格”的区间。
        """
        decisions = [
            row for row in self._timeline if row.get("kind") == "decision"
        ]
        gaps = [row for row in self._timeline if row.get("kind") == "gap"]
        non_observable = sum(float(row.get("seconds", 0.0)) for row in gaps)
        gap_reasons: dict[str, float] = {}
        for row in gaps:
            key = str(row.get("reason", "UNOBSERVED"))
            gap_reasons[key] = round(
                gap_reasons.get(key, 0.0) + float(row.get("seconds", 0.0)), 4,
            )
        if not decisions:
            return {
                "decisions": 0, "effective_fraction": 0.0,
                "pause_p95": 0.0, "pause_max": 0.0, "switch_count": 0,
                "gaps": [], "unknown_reasons": {}, "off_air_seconds": 0.0,
                "observed_seconds": 0.0, "effective_seconds": 0.0,
                "non_observable_seconds": round(non_observable, 4),
                "non_observable_reasons": gap_reasons,
                "unobservable_ticks": 0, "wall_span_seconds": 0.0,
            }
        join_gap = (
            self._number("join_gap_seconds")
            if join_gap_seconds is None else float(join_gap_seconds)
        )
        join_gap = max(0.0, join_gap)
        # 一次观测最多证明到计划观察点；显式登记的空缺同样不能反向膨胀
        # “可观测”时间（否则 100s 无观测会被算成 100s 可判断时间）。
        nominal = self._number("tick_interval_seconds")
        effective = 0.0
        total = 0.0
        pauses: list[float] = []
        current_pause = 0.0
        off_air = 0.0
        unknown_reasons: dict[str, int] = {}
        unobservable_ticks = 0
        for index, row in enumerate(decisions):
            timestamp = float(row["input_timestamp"])
            span = float(row.get("observation_span_seconds", 0.0))
            gap_to_next = 0.0
            if index + 1 < len(decisions):
                gap_to_next = (
                    float(decisions[index + 1]["input_timestamp"]) - timestamp
                )
            if gap_to_next > join_gap:
                # 真正的录像空缺（停机/换文件）：超出 join_gap 的部分单独记账，
                # 本 tick 最多只能证明到 join_gap。
                off_air += gap_to_next - join_gap
                span = min(span, join_gap)
            if row.get("observable", True) and nominal > 0:
                # 区间 = min(结果有效期, 到下一个真实观测的时间)；
                # 无观测的那一段既不进分母也不算暂停。
                span = min(span, max(0.0, gap_to_next))
            span = max(0.0, span)
            if not row.get("observable", True):
                unobservable_ticks += 1
                reason = str(row.get("non_observable_reason", "UNOBSERVED"))
                unknown_reasons[reason] = unknown_reasons.get(reason, 0) + 1
                if current_pause > 0:
                    pauses.append(current_pause)
                    current_pause = 0.0
                continue
            total += span
            if row.get("prior_allowed"):
                effective += span
                if current_pause > 0:
                    pauses.append(current_pause)
                    current_pause = 0.0
            else:
                current_pause += span
                reason = str(row.get("reason", "UNKNOWN"))
                unknown_reasons[reason] = unknown_reasons.get(reason, 0) + 1
        if current_pause > 0:
            pauses.append(current_pause)
        # “可观测”只包含真实观测能覆盖的时间；显式登记的空缺单独列出，
        # 不并入 observed_seconds，避免同一段时间被两边重复计入。
        observed = total + off_air
        return {
            "decisions": len(decisions),
            "observed_seconds": round(total, 3),
            "off_air_seconds": round(off_air, 3),
            "non_observable_seconds": round(non_observable, 4),
            "non_observable_reasons": gap_reasons,
            "unobservable_ticks": unobservable_ticks,
            "effective_seconds": round(effective, 3),
            "effective_fraction": 0.0 if total <= 0 else round(effective / total, 5),
            "wall_span_seconds": round(observed, 3),
            "pause_p95": 0.0 if not pauses else round(
                float(np_percentile(pauses, 95)), 3
            ),
            "pause_max": 0.0 if not pauses else round(max(pauses), 3),
            "pause_intervals": [round(value, 3) for value in pauses],
            "switch_count": self.switch_count,
            "generation": self.profile_generation,
            "unknown_reasons": dict(sorted(unknown_reasons.items())),
            "budget_exhausted_ticks": sum(
                1 for row in decisions if row.get("budget_exhausted")
            ),
        }

    def last_current_hold_eligible(self, profile_id: str | None) -> bool:
        """上一 tick 中，给定 Profile 是否真的仍然可保持（R4）。

        调用方不能把“存在当前参考”当成“当前仍可保持”：那会让 Selector 走
        快速路径，扩展游标不推进，第 K+1 个可用参考永远查不到。
        """
        if not profile_id:
            return False
        for row in reversed(self._timeline):
            if row.get("kind") != "decision":
                continue
            if row.get("selected_profile_id") != profile_id:
                return False
            return bool(row.get("prior_allowed"))
        return False

    def _mark_matched(self, timestamp: float) -> None:
        self._last_matched_at = float(timestamp)

    def _expire_state(self, timestamp: float) -> None:
        for profile_id in list(self._cooldown_until):
            if self._cooldown_until[profile_id] <= timestamp:
                self._cooldown_until.pop(profile_id, None)

    def _improvement(self, current_score: float, best_score: float) -> float:
        return (float(current_score) - float(best_score)) / max(
            float(current_score), self.score_floor
        )

    def _pick_challenger(
        self, current: CandidateMatch, candidates: Sequence[CandidateMatch],
        timestamp: float,
    ) -> CandidateMatch | None:
        best: CandidateMatch | None = None
        for item in candidates:
            if item.profile_id == current.profile_id or not item.enter_eligible:
                continue
            if self.is_cooling(item.profile_id, timestamp):
                continue
            improvement = self._improvement(current.score, item.score)
            if improvement < self._number("switch_improvement_ratio"):
                continue
            if best is None or (item.score, item.profile_id) < (best.score, best.profile_id):
                best = item
        return best

    def _pick_recovery(
        self, candidates: Sequence[CandidateMatch], timestamp: float,
    ) -> CandidateMatch | None:
        best: CandidateMatch | None = None
        for item in candidates:
            if not item.enter_eligible or self.is_cooling(item.profile_id, timestamp):
                continue
            if best is None or (item.score, item.profile_id) < (best.score, best.profile_id):
                best = item
        return best


def np_percentile(values: Sequence[float], quantile: float) -> float:
    """不引入 numpy 依赖的线性插值分位数（与 numpy.percentile 默认一致）。"""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "CandidateMatch", "DEFAULT_SELECTION_CONFIG", "PHASE_PREPARING", "PHASE_SEARCHING",
    "PHASE_STEADY", "PHASE_VALIDATING", "ProfileSelector", "SEARCH_REASON_BUDGET_EXHAUSTED",
    "SEARCH_REASON_NO_ELIGIBLE", "SEARCH_REASON_OK", "SEARCH_REASON_WAITING_DWELL",
    "SEARCH_REASON_WAITING_RECOVERY", "SEARCH_REASON_WAITING_SWITCH", "STATUS_MATCHED",
    "STATUS_SWITCH_PENDING", "STATUS_UNKNOWN", "SelectionDecision", "SelectorError",
    "np_percentile",
]
