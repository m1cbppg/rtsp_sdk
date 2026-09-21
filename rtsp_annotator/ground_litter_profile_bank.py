"""Immutable Profile Bank assets shared by the offline factory and the runtime.

方案一 §2 资产契约与方案二 §3.4/§5.2 共用一份实现。本模块只负责：

* Bank 根目录解析与越界防护；
* 资产 SHA-256 计算、规范 JSON 序列化、原子落盘与原子版本发布；
* loader 校验（kind/schema/尺寸/散列/有效地面/阈值量纲）；
* 有界全尺寸上下文缓存（张数 + 估算字节，二者都必须受限）。

它不包含任何匹配、掩膜或时序逻辑——那些在 ``ground_litter_profile_match``、
``ground_litter_profile_analysis``、``ground_litter_profile_selector``。

设计约束（来自 r3 文档）：

* 版本目录不可变：发布一次后不再改写；需要重建时使用新版本号或先显式
  ``supersede`` 把旧目录改成 ``*.superseded-<utc>``，绝不静默覆盖。
* ``valid_mask`` 与离线 noise 在构建完成时冻结；本模块只做读取，不提供在线写回。
* 资产 SHA-256 一律小写十六进制；``profile.json`` 内的散列覆盖同目录资产文件，
  ``bank.json`` 的 ``assets_sha256`` 覆盖 ``camera_geometry.json`` / ``matcher.json`` /
  ``profile.json``（不是自身）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

BANK_KIND = "ground_litter_profile_bank"
SCHEMA_VERSION = 1
ASSET_NAMES = ("reference.png", "valid_mask.png", "noise.npz", "descriptor.npz")
# 兼容旧 V3.2 单 Profile：它是二值 tolerance，不是连续 noise，必须由 adapter 转换。
LEGACY_PROFILE_KIND = "ground_litter_clean_reference_v32"
LEGACY_TOLERANCE_NAME = "daylight_tolerance.png"

_VERSION_RE = re.compile(r"^v[0-9A-Za-z][0-9A-Za-z._-]*$")
_CAMERA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# 模块级硬上限：任何 loader 都不允许加载超过该像素数的参考。
MAX_REFERENCE_PIXELS = 4096 * 2160


class BankError(ValueError):
    """资产结构/校验失败。"""


# --------------------------------------------------------------------------- #
# 通用：规范 JSON、散列、原子写
# --------------------------------------------------------------------------- #


def canonical_json(payload: Any) -> str:
    """稳定、可复现的 JSON 文本（排序键 + 固定分隔符 + 无 NaN）。"""
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> str:
    """原子写文件并返回 sha256；同目录 ``.tmp`` + ``os.replace``。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    _fsync_directory(target.parent)
    return sha256_bytes(payload)


def atomic_write_text(path: str | Path, text: str) -> str:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Any) -> str:
    return atomic_write_text(path, canonical_json(payload) + "\n")


def _fsync_directory(directory: Path) -> None:
    try:
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - 平台不支持目录 fsync 时忽略
        return
    try:
        os.fsync(handle)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(handle)


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    """把数组序列化成确定性的 .npz 字节（固定时间戳）。"""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            inner = io.BytesIO()
            np.save(inner, np.ascontiguousarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, inner.getvalue())
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# 根目录解析
# --------------------------------------------------------------------------- #


def resolve_bank_version(
    root: str | Path, bank_id: str, version: str | None = None,
    *, must_exist: bool = True,
) -> Path:
    """把 ``bank_id``/``version`` 解析到受控根目录下的绝对路径，拒绝越界。"""
    if not _CAMERA_RE.match(str(bank_id or "")):
        raise BankError("bank_id非法")
    base = Path(root).expanduser().resolve()
    candidate = (base / bank_id).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:  # pragma: no cover - 防御
        raise BankError("bank_id路径越界") from exc
    if version is None:
        if must_exist and not candidate.is_dir():
            raise BankError(f"Bank 不存在: {bank_id}")
        return candidate
    if not _VERSION_RE.match(str(version)):
        raise BankError("version非法")
    resolved = (candidate / version).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:  # pragma: no cover - 防御
        raise BankError("version路径越界") from exc
    if must_exist and not resolved.is_dir():
        raise BankError(f"Bank 版本不存在: {bank_id}/{version}")
    return resolved


def list_versions(root: str | Path, bank_id: str) -> list[str]:
    base = resolve_bank_version(root, bank_id, must_exist=False)
    if not base.is_dir():
        return []
    return sorted(
        entry.name for entry in base.iterdir()
        if entry.is_dir() and _VERSION_RE.match(entry.name)
    )


def latest_version(root: str | Path, bank_id: str) -> str | None:
    versions = list_versions(root, bank_id)
    return versions[-1] if versions else None


# --------------------------------------------------------------------------- #
# 资产写入
# --------------------------------------------------------------------------- #


def write_png(path: str | Path, image: np.ndarray) -> str:
    """无损 PNG 写入（返回 sha256）。彩色按 BGR，掩膜按单通道 0/255。"""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise BankError("PNG 资产必须是 uint8 的二维或三维数组")
    ok, encoded = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise BankError("PNG 编码失败")
    return atomic_write_bytes(path, encoded.tobytes())


def _asset_arrays(payload: Mapping[str, Any], asset: str) -> dict[str, np.ndarray]:
    """把载荷规范成 .npz 数组；标量元数据用 0 维 ``np.array`` 保存。

    只接受数值/字符串/数组，拒绝 ``None`` 与 object dtype 载荷，避免
    ``read_npz`` 在 ``allow_pickle=False`` 下读出意外结构。
    """
    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise BankError(f"{asset} 的键必须是非空字符串")
        if value is None:
            raise BankError(f"{asset}.{key} 不允许为 None")
        if isinstance(value, (int, float, bool, str)):
            array = np.array(value)
        else:
            array = np.asarray(value)
        if array.dtype == object:
            raise BankError(f"{asset}.{key} 不能是 object dtype")
        arrays[key] = array
    if not arrays:
        raise BankError(f"{asset} 不能为空")
    return arrays


def write_noise(path: str | Path, payload: Mapping[str, Any]) -> str:
    """写 ``noise.npz``：阈值图、残差中心/MAD/Q95 诊断与标量元数据。"""
    return atomic_write_bytes(path, _npz_bytes(**_asset_arrays(payload, "noise.npz")))


def write_descriptor(path: str | Path, payload: Mapping[str, Any]) -> str:
    return atomic_write_bytes(
        path, _npz_bytes(**_asset_arrays(payload, "descriptor.npz"))
    )


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


# --------------------------------------------------------------------------- #
# 发布
# --------------------------------------------------------------------------- #


def publish_version(
    root: str | Path,
    bank_id: str,
    version: str,
    *,
    manifest: Mapping[str, Any],
    camera_geometry: Mapping[str, Any],
    matcher: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    supersede_existing: bool = False,
) -> Path:
    """原子发布一个不可变 Bank 版本。

    ``profiles`` 每一项必须已经包含 ``profile_id``、``profile_json``（dict）以及
    ``reference`` / ``valid`` / ``noise`` / ``descriptor`` 四个载荷：

    * ``reference``: uint8 (H, W, 3) BGR；
    * ``valid``: uint8 (H, W) 0/255；
    * ``noise``: dict[str, ndarray]；
    * ``descriptor``: dict[str, ndarray]。

    现有版本默认拒绝覆盖；``supersede_existing=True`` 时把旧目录改名保留为
    ``<version>.superseded-<utc>``，绝不删除。
    """
    if not _CAMERA_RE.match(str(bank_id or "")):
        raise BankError("bank_id非法")
    if not _VERSION_RE.match(str(version or "")):
        raise BankError("version非法")
    if not profiles:
        raise BankError("Bank 至少需要一个 Profile")

    base = Path(root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    final = resolve_bank_version(base, bank_id, version, must_exist=False)
    staging = final.with_name(f".{version}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        profile_records: list[dict[str, Any]] = []
        reference_size: tuple[int, int] | None = None
        for entry in profiles:
            profile_id = str(entry["profile_id"])
            if not _CAMERA_RE.match(profile_id):
                raise BankError(f"profile_id非法: {profile_id}")
            reference = np.asarray(entry["reference"])
            valid = np.asarray(entry["valid"])
            if reference.dtype != np.uint8 or reference.ndim != 3 or reference.shape[2] != 3:
                raise BankError(f"{profile_id} reference 必须是 uint8 BGR")
            if valid.dtype != np.uint8 or valid.ndim != 2:
                raise BankError(f"{profile_id} valid 必须是 uint8 二值图")
            if valid.shape != reference.shape[:2]:
                raise BankError(f"{profile_id} valid 与 reference 尺寸不一致")
            if reference.size // 3 > MAX_REFERENCE_PIXELS:
                raise BankError(f"{profile_id} 参考像素超过上限")
            if reference_size is None:
                reference_size = (reference.shape[1], reference.shape[0])
            elif reference_size != (reference.shape[1], reference.shape[0]):
                raise BankError("同一 Bank 内参考尺寸必须一致")

            directory = staging / "profiles" / profile_id
            directory.mkdir(parents=True, exist_ok=True)
            checksums = {
                "reference.png": write_png(directory / "reference.png", reference),
                "valid_mask.png": write_png(directory / "valid_mask.png", valid),
                "noise.npz": write_noise(directory / "noise.npz", entry["noise"]),
                "descriptor.npz": write_descriptor(
                    directory / "descriptor.npz", entry["descriptor"]
                ),
            }
            profile_json = dict(entry["profile_json"])
            profile_json.update({
                "kind": BANK_KIND,
                "schema_version": SCHEMA_VERSION,
                "profile_id": profile_id,
                "reference_size": [reference.shape[1], reference.shape[0]],
                "sha256": checksums,
            })
            profile_json.pop("profile_sha256", None)
            profile_sha = atomic_write_json(directory / "profile.json", profile_json)
            # profile.json 的自身散列不放在自己内部（会自我引用）；由 bank.json 记录。
            profile_records.append({
                "profile_id": profile_id,
                "relative_dir": f"profiles/{profile_id}",
                "reference_size": [reference.shape[1], reference.shape[0]],
                "profile_json_sha256": profile_sha,
                "assets_sha256": checksums,
                "sample_count": int(profile_json.get("sample_count", 0)),
                "valid_ground_fraction": float(
                    profile_json.get("valid_ground_fraction", 0.0)
                ),
                "support": profile_json.get("support", {}),
                "applicability": profile_json.get("applicability", {}),
                "source": profile_json.get("source", {}),
            })

        geometry_sha = atomic_write_json(staging / "camera_geometry.json", camera_geometry)
        matcher_sha = atomic_write_json(staging / "matcher.json", matcher)

        manifest_payload = dict(manifest)
        manifest_payload.update({
            "kind": BANK_KIND,
            "schema_version": SCHEMA_VERSION,
            "bank_id": bank_id,
            "version": version,
            "reference_size": list(reference_size or (0, 0)),
            "profile_count": len(profile_records),
            "profiles": profile_records,
            "assets_sha256": {
                "camera_geometry.json": geometry_sha,
                "matcher.json": matcher_sha,
                "profiles": {
                    record["profile_id"]: record["profile_json_sha256"]
                    for record in profile_records
                },
            },
            "created_utc": manifest_payload.get(
                "created_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ),
        })
        atomic_write_json(staging / "bank.json", manifest_payload)
        (staging / "reports").mkdir(exist_ok=True)

        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if not supersede_existing:
                raise BankError(f"Bank 版本已存在且不可变: {bank_id}/{version}")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archived = final.with_name(f"{version}.superseded-{stamp}")
            suffix = 1
            while archived.exists():
                archived = final.with_name(f"{version}.superseded-{stamp}-{suffix}")
                suffix += 1
            os.replace(final, archived)
        os.replace(staging, final)
        _fsync_directory(final.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    profile_id: str
    directory: Path
    metadata: dict[str, Any]
    reference_size: tuple[int, int]

    @property
    def sample_count(self) -> int:
        return int(self.metadata.get("sample_count", 0))

    @property
    def valid_ground_fraction(self) -> float:
        return float(self.metadata.get("valid_ground_fraction", 0.0))

    # -- 跨模块契约：能力字段（v4 复核后新增） ------------------------------ #

    @property
    def match_eligible(self) -> bool:
        """是否可参与环境匹配。"""
        return bool(self.metadata.get("match_eligible", True))

    @property
    def prior_suitable(self) -> bool:
        """是否允许产生 prior-only 候选。

        历史 Bank（v3 及更早）没有该字段；loader 会按保守值写入
        ``capability_source='legacy_conservative'``，这里缺省也返回 False。
        """
        return bool(self.metadata.get("prior_suitable", False))

    @property
    def calibration_state(self) -> str:
        state = str(self.metadata.get("calibration_state") or "")
        if state:
            return state
        return (
            "independent_matched" if self.prior_suitable
            else "reference_self_low_support"
        )

    @property
    def calibration_degradation_reason(self) -> str | None:
        value = self.metadata.get("prior_degradation_reason")
        return None if value in (None, "", "None") else str(value)

    @property
    def capability_source(self) -> str:
        return str(self.metadata.get("capability_source") or "declared")


@dataclass(frozen=True, slots=True)
class ProfileBank:
    """已校验的 Bank 版本（元数据级，不常驻全尺寸图像）。"""

    root: Path
    bank_id: str
    version: str
    manifest: dict[str, Any]
    geometry: dict[str, Any]
    matcher: dict[str, Any]
    profiles: tuple[ProfileRecord, ...]

    @property
    def reference_size(self) -> tuple[int, int]:
        width, height = self.manifest.get("reference_size", (0, 0))
        return int(width), int(height)

    @property
    def banner(self) -> str:
        return f"{self.bank_id}/{self.version}"

    @property
    def prior_suitable_ids(self) -> tuple[str, ...]:
        return tuple(r.profile_id for r in self.profiles if r.prior_suitable)

    @property
    def legacy_capabilities(self) -> bool:
        return any(
            r.capability_source == "legacy_conservative" for r in self.profiles
        )

    def profile(self, profile_id: str) -> ProfileRecord:
        for record in self.profiles:
            if record.profile_id == profile_id:
                return record
        raise BankError(f"Bank 内不存在 Profile: {profile_id}")

    def ids(self) -> tuple[str, ...]:
        return tuple(record.profile_id for record in self.profiles)

    def load_reference(self, profile_id: str) -> np.ndarray:
        record = self.profile(profile_id)
        image = cv2.imread(str(record.directory / "reference.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise BankError(f"reference.png 无法读取: {profile_id}")
        return image

    def load_valid(self, profile_id: str) -> np.ndarray:
        record = self.profile(profile_id)
        mask = cv2.imread(str(record.directory / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise BankError(f"valid_mask.png 无法读取: {profile_id}")
        return mask

    def load_noise(self, profile_id: str) -> dict[str, np.ndarray]:
        record = self.profile(profile_id)
        return read_npz(record.directory / "noise.npz")

    def load_descriptor(self, profile_id: str) -> dict[str, np.ndarray]:
        record = self.profile(profile_id)
        return read_npz(record.directory / "descriptor.npz")


def _require_file(path: Path, field: str) -> None:
    if not path.is_file():
        raise BankError(f"缺少资产文件: {field} ({path.name})")


def verify_checksum(path: Path, expected: Any, field: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise BankError(f"缺少 SHA-256: {field}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise BankError(f"资产校验失败: {field}")


MISSING_ENVELOPE = "MISSING_ENVELOPE"


def bank_envelope(
    matcher: Mapping[str, Any], profile_id: str,
) -> dict[str, Any] | None:
    """读取冻结包络；返回 None 表示该 Profile 没有校准包络。"""
    profiles = matcher.get("profiles")
    if not isinstance(profiles, Mapping):
        return None
    entry = profiles.get(profile_id)
    if not isinstance(entry, Mapping):
        return None
    envelope = entry.get("envelope")
    return dict(envelope) if isinstance(envelope, Mapping) else None


def validate_calibration(
    matcher: Mapping[str, Any], profile_ids: Sequence[str],
) -> list[str]:
    """返回缺失/未校准包络的 Profile 列表。

    评估与运行时必须拒绝「未校准」的 Bank：否则等于用默认阈值冒充校准结果，
    并且在待评数据上重新拟合会让盲测不再独立（R1）。
    """
    problems: list[str] = []
    calibration = matcher.get("calibration")
    if not isinstance(calibration, Mapping) or (
        str(calibration.get("source", "")) in ("", "uncalibrated_defaults")
    ):
        problems.append("BANK_UNCALIBRATED")
    for profile_id in profile_ids:
        envelope = bank_envelope(matcher, profile_id)
        if envelope is None:
            problems.append(f"{MISSING_ENVELOPE}:{profile_id}")
            continue
        if not bool(envelope.get("calibrated")):
            problems.append(f"UNCALIBRATED_ENVELOPE:{profile_id}")
    return problems


CALIBRATION_STATES = (
    "independent_matched", "reference_self_low_support",
    "no_independent_material",
)


def load_bank(
    root: str | Path, bank_id: str, version: str | None = None,
    *, verify: bool = True, require_calibration: bool = True,
    allow_legacy_profile_capabilities: bool = False,
) -> ProfileBank:
    """加载并校验一个 Bank 版本。``version=None`` 时取最新版本。

    ``require_calibration=True``（默认）时，缺少冻结包络的 Bank 直接拒绝加载；
    读取历史实验产物（例如未校准的 v1）需显式传 False，并在报告中标注。

    ``allow_legacy_profile_capabilities``（默认 False）：v4 复核后，
    ``prior_suitable`` / ``calibration_state`` 是跨模块契约的一部分，缺失即拒绝
    加载。只有显式打开该开关，才允许读取 v3 及更早的历史 Bank，此时
    **保守地把每个 Profile 视为 ``prior_suitable=false``**，
    并在 ``ProfileRecord.capability_source`` 标注 ``legacy_conservative``。
    """
    if version is None:
        version = latest_version(root, bank_id)
        if version is None:
            raise BankError(f"Bank 不存在或没有已发布版本: {bank_id}")
    directory = resolve_bank_version(root, bank_id, version)
    manifest_path = directory / "bank.json"
    if not manifest_path.is_file():
        raise BankError("缺少 bank.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != BANK_KIND:
        raise BankError("bank.json kind 无效")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BankError("bank.json schema_version 不受支持")
    if manifest.get("bank_id") != bank_id or manifest.get("version") != version:
        raise BankError("bank.json 身份与路径不一致")

    geometry_path = directory / "camera_geometry.json"
    matcher_path = directory / "matcher.json"
    _require_file(geometry_path, "camera_geometry.json")
    _require_file(matcher_path, "matcher.json")
    assets = manifest.get("assets_sha256")
    if not isinstance(assets, dict):
        raise BankError("bank.json 缺少 assets_sha256")
    if verify:
        verify_checksum(geometry_path, assets.get("camera_geometry.json"), "camera_geometry.json")
        verify_checksum(matcher_path, assets.get("matcher.json"), "matcher.json")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    matcher = json.loads(matcher_path.read_text(encoding="utf-8"))

    width, height = (int(value) for value in manifest.get("reference_size", (0, 0)))
    if width <= 0 or height <= 0 or width * height > MAX_REFERENCE_PIXELS:
        raise BankError("bank.json reference_size 非法")

    entries = manifest.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise BankError("bank.json 缺少 profiles")
    records: list[ProfileRecord] = []
    seen: set[str] = set()
    for entry in entries:
        profile_id = str(entry.get("profile_id", ""))
        if not _CAMERA_RE.match(profile_id):
            raise BankError(f"profile_id非法: {profile_id!r}")
        if profile_id in seen:
            raise BankError(f"profile_id重复: {profile_id}")
        seen.add(profile_id)
        directory_path = (directory / str(entry.get("relative_dir", ""))).resolve()
        try:
            directory_path.relative_to(directory)
        except ValueError as exc:
            raise BankError("profile relative_dir 越界") from exc
        profile_path = directory_path / "profile.json"
        _require_file(profile_path, f"{profile_id}/profile.json")
        if verify:
            verify_checksum(
                profile_path, entry.get("profile_json_sha256"), f"{profile_id}/profile.json"
            )
        metadata = json.loads(profile_path.read_text(encoding="utf-8"))
        if metadata.get("kind") != BANK_KIND or metadata.get("profile_id") != profile_id:
            raise BankError(f"profile.json 身份无效: {profile_id}")
        checksums = metadata.get("sha256")
        if not isinstance(checksums, dict):
            raise BankError(f"profile.json 缺少 sha256: {profile_id}")
        for name in ASSET_NAMES:
            asset_path = directory_path / name
            _require_file(asset_path, f"{profile_id}/{name}")
            if verify:
                verify_checksum(asset_path, checksums.get(name), f"{profile_id}/{name}")
        size = metadata.get("reference_size")
        if not isinstance(size, list) or [width, height] != [int(v) for v in size]:
            raise BankError(f"{profile_id} reference_size 与 Bank 不一致")
        declared = (
            "prior_suitable" in metadata and "calibration_state" in metadata
        )
        if not declared:
            if not allow_legacy_profile_capabilities:
                raise BankError(
                    f"{profile_id} 缺少能力字段 prior_suitable/calibration_state；"
                    "历史 Bank 必须显式传 allow_legacy_profile_capabilities=True，"
                    "且会被保守视为 prior_suitable=false"
                )
            metadata = dict(metadata)
            metadata["prior_suitable"] = False
            metadata["calibration_state"] = None
            metadata["capability_source"] = "legacy_conservative"
        else:
            state = str(metadata.get("calibration_state") or "")
            if state not in CALIBRATION_STATES:
                raise BankError(
                    f"{profile_id} calibration_state 非法: {state!r}"
                )
            suitable = bool(metadata.get("prior_suitable"))
            if suitable and state != "independent_matched":
                raise BankError(
                    f"{profile_id} prior_suitable=true 但 calibration_state={state}；"
                    "只有 independent_matched 才允许产生 prior 候选"
                )
            metadata = dict(metadata)
            metadata["capability_source"] = "declared"
        records.append(ProfileRecord(
            profile_id=profile_id,
            directory=directory_path,
            metadata=metadata,
            reference_size=(width, height),
        ))

    if verify:
        _verify_asset_semantics(records)
    if require_calibration:
        problems = validate_calibration(matcher, [r.profile_id for r in records])
        if problems:
            raise BankError(
                "Bank 缺少冻结校准包络，拒绝加载："
                + ", ".join(problems[:6])
                + ("…" if len(problems) > 6 else "")
            )
    return ProfileBank(
        root=Path(root).expanduser().resolve(),
        bank_id=bank_id,
        version=version,
        manifest=manifest,
        geometry=geometry,
        matcher=matcher,
        profiles=tuple(records),
    )


def _verify_asset_semantics(records: Iterable[ProfileRecord]) -> None:
    """PNG/npz 的可解码性与量纲校验（比散列更严格，比全量解码便宜）。"""
    for record in records:
        directory = record.directory
        valid = cv2.imread(str(directory / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(directory / "reference.png"), cv2.IMREAD_COLOR)
        if reference is None or valid is None:
            raise BankError(f"资产图像无法解码: {record.profile_id}")
        width, height = record.reference_size
        if reference.shape[:2] != (height, width) or valid.shape != (height, width):
            raise BankError(f"资产尺寸不一致: {record.profile_id}")
        if int(np.count_nonzero(valid)) < 5000:
            raise BankError(f"有效地面不足: {record.profile_id}")
        unique = np.unique(valid)
        if not np.all(np.isin(unique, (0, 255))):
            raise BankError(f"valid_mask 必须是 0/255: {record.profile_id}")
        noise = read_npz(directory / "noise.npz")
        for required in (
            "seed_signature_threshold", "seed_luminance_threshold",
            "support_signature_threshold", "support_luminance_threshold",
            "seed_signature_median", "seed_luminance_median",
            "seed_signature_cap", "seed_luminance_cap",
        ):
            if required not in noise:
                raise BankError(f"noise.npz 缺少 {required}: {record.profile_id}")
            array = noise[required]
            if array.shape != (height, width):
                raise BankError(f"noise.npz {required} 尺寸不一致: {record.profile_id}")
        descriptor = read_npz(directory / "descriptor.npz")
        for required in ("grid_luminance", "grid_chroma", "grid_structure", "grid_weight"):
            if required not in descriptor:
                raise BankError(f"descriptor.npz 缺少 {required}: {record.profile_id}")


# --------------------------------------------------------------------------- #
# 有界全尺寸上下文缓存（方案二 §5.2）
# --------------------------------------------------------------------------- #


def estimate_context_bytes(
    reference: np.ndarray, valid: np.ndarray, noise: Mapping[str, np.ndarray]
) -> int:
    """按真实 dtype/尺寸估算一个全尺寸上下文的内存占用。

    方案二 §5.2 明确要求不能只按 PNG 文件大小估算；这里同时计入参考、有效图、
    四项阈值图和 noise 诊断图。
    """
    total = reference.nbytes + valid.nbytes
    for array in noise.values():
        total += np.asarray(array).nbytes
    return int(total)


class BoundedContextCache:
    """LRU + 字节上限的全尺寸上下文缓存。

    正在使用/待提交的条目必须显式 ``pin``；淘汰只发生在未 pin 的条目上。
    容量不足时抛 ``BankError``，**不**在后台偷偷缩小分析尺寸。
    """

    def __init__(self, bank: ProfileBank, *, max_profiles: int = 4,
                 max_bytes: int = 256 * 1024 * 1024) -> None:
        if max_profiles < 1:
            raise BankError("max_profiles 必须为正")
        if max_bytes < 1:
            raise BankError("max_bytes 必须为正")
        self.bank = bank
        self.max_profiles = int(max_profiles)
        self.max_bytes = int(max_bytes)
        self._entries: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
        self._order: list[str] = []
        self._pins: dict[str, int] = {}
        self._bytes: dict[str, int] = {}
        self.evictions = 0
        self.disk_loads = 0

    @property
    def total_bytes(self) -> int:
        return sum(self._bytes.values())

    def __len__(self) -> int:
        return len(self._entries)

    def loaded_ids(self) -> tuple[str, ...]:
        return tuple(self._order)

    def pin(self, profile_id: str) -> None:
        if profile_id not in self._entries:
            self.get(profile_id)
        self._pins[profile_id] = self._pins.get(profile_id, 0) + 1

    def unpin(self, profile_id: str) -> None:
        remaining = self._pins.get(profile_id, 0) - 1
        if remaining > 0:
            self._pins[profile_id] = remaining
        else:
            self._pins.pop(profile_id, None)

    def get(self, profile_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if profile_id in self._entries:
            self._touch(profile_id)
            return self._entries[profile_id]
        reference = self.bank.load_reference(profile_id)
        valid = self.bank.load_valid(profile_id)
        noise = self.bank.load_noise(profile_id)
        size = estimate_context_bytes(reference, valid, noise)
        if size > self.max_bytes:
            raise BankError(
                f"单个上下文 {size} 字节超过缓存上限 {self.max_bytes}，资源不满足"
            )
        self._entries[profile_id] = (reference, valid, noise)
        self._bytes[profile_id] = size
        self._order.append(profile_id)
        self.disk_loads += 1
        self._enforce()
        return self._entries[profile_id]

    def _touch(self, profile_id: str) -> None:
        if profile_id in self._order:
            self._order.remove(profile_id)
        self._order.append(profile_id)

    def _enforce(self) -> None:
        while (len(self._entries) > self.max_profiles
               or self.total_bytes > self.max_bytes):
            victim = next(
                (pid for pid in self._order if not self._pins.get(pid)), None
            )
            if victim is None:
                raise BankError("缓存已满且全部条目被 pin，资源不满足")
            self._order.remove(victim)
            self._entries.pop(victim, None)
            self._bytes.pop(victim, None)
            self.evictions += 1

    def clear(self) -> None:
        if any(self._pins.values()):
            raise BankError("仍有条目被 pin，拒绝清空")
        self._entries.clear()
        self._order.clear()
        self._bytes.clear()


# --------------------------------------------------------------------------- #
# matcher / geometry 默认契约
# --------------------------------------------------------------------------- #


def default_camera_geometry(
    width: int, height: int, *, roi: Sequence[Sequence[float]] | None = None,
    exclude_zones: Sequence[Sequence[Sequence[float]]] | None = None,
    overlay_exclude_zones: Sequence[Sequence[Sequence[float]]] | None = None,
    view_id: str = "view_0", camera_id: str = "",
    geom_version: str = "r3",
) -> dict[str, Any]:
    """共同画布/ROI/排除区/稳定匹配区域。坐标一律归一化。"""
    if width <= 0 or height <= 0:
        raise BankError("画布尺寸必须为正")
    if roi is None:
        roi = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    return {
        "kind": "ground_litter_camera_geometry",
        "schema_version": SCHEMA_VERSION,
        "camera_id": camera_id,
        "view_id": view_id,
        "geometry_version": geom_version,
        "canvas_size": [int(width), int(height)],
        "roi": [list(map(float, point)) for point in roi],
        "exclude_zones": [
            [list(map(float, point)) for point in polygon]
            for polygon in (exclude_zones or [])
        ],
        "overlay_exclude_zones": [
            [list(map(float, point)) for point in polygon]
            for polygon in (overlay_exclude_zones or [])
        ],
        "registration": {"method": "sift_homography", "geom_basis": "canvas"},
    }


def default_matcher_config(**overrides: Any) -> dict[str, Any]:
    """方案二 §3.3/§3.4/§4.1 的起点参数；校准后冻结进 ``matcher.json``。"""
    config: dict[str, Any] = {
        "kind": "ground_litter_profile_matcher",
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": "matcher_r3",
        "grid": {"cols": 16, "rows": 9, "min_weight": 0.02},
        "score_weights": {
            "luminance": 0.4, "chroma": 0.2, "structure": 0.4,
            "q50": 0.5, "q90": 0.5, "compensation": 0.2, "missing": 1.0,
        },
        "scale_floor": {
            "luminance": 12.0, "chroma": 10.0, "structure": 0.06,
        },
        "compensation": {
            "gain_range": [0.65, 1.45], "bias_range": [-60.0, 60.0],
            "max_abs_gain_delta": 0.18, "max_abs_bias": 30.0,
        },
        "geometry": {
            "min_anchor_fraction": 0.35,
            "min_distinct_regions": 3,
            "max_reprojection_p95_px": 2.0,
            "min_inlier_hull_fraction": 0.02,
        },
        "envelope": {
            "enter_margin": 0.08, "hold_margin": 0.20,
            "fallback_score": 2.5, "fallback_enter": 1.6, "fallback_hold": 2.2,
            "min_calibration_samples": 8, "quantile": 0.95,
        },
        "selection": {
            "top_k": 3, "max_small_matches_per_tick": 4,
            "switch_improvement_ratio": 0.15,
            "switch_min_samples": 3, "switch_min_span_seconds": 4.0,
            "min_dwell_seconds": 10.0,
            "recovery_min_samples": 2, "recovery_min_span_seconds": 2.0,
            "failed_candidate_cooldown_seconds": 10.0,
            "max_result_age_seconds": 4.0,
            "max_observation_gap_seconds": 4.0,
            "cache_profiles": 4, "loader_inflight": 1,
            "score_floor": 0.25,
        },
        "calibration": {"source": "uncalibrated_defaults", "calibrated_utc": None},
        # 每个 Profile 的进入/保持包络在**校准集**上拟合后冻结在这里；
        # 运行时与评估都只读它，不得在待评数据上重新拟合（方案二 §3.4）。
        "profiles": {},
    }
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(config.get(key), Mapping):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def load_legacy_tolerance_as_noise(
    tolerance: np.ndarray, valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """旧 V3.2 二值 tolerance → 连续 noise 阈值图的兼容转换。

    旧图是 0/255 的二值「此处噪声更大」标记，新 Bank 需要连续阈值与诊断。
    转换规则：标记处使用现有 V3.2 噪声增量（``NOISE_*_INCREMENT``），
    其余位置使用基础阈值；MAD/中心没有旧证据，置 0 并在 ``support_blocks``
    标 0，明确表示低支持而不是伪造统计。
    """
    from .ground_litter_v32 import (
        LUMINANCE_THRESHOLD, SIGNATURE_THRESHOLD,
        SUPPORT_LUMINANCE_THRESHOLD, SUPPORT_SIGNATURE_THRESHOLD,
        NOISE_SEED_LUMINANCE_INCREMENT, NOISE_SEED_SIGNATURE_INCREMENT,
        NOISE_SUPPORT_LUMINANCE_INCREMENT, NOISE_SUPPORT_SIGNATURE_INCREMENT,
    )

    height, width = valid.shape
    if tolerance.shape != (height, width):
        raise BankError("tolerance 与 valid 尺寸不一致")
    noisy = (tolerance > 0).astype(np.float32)
    zeros = np.zeros((height, width), np.float32)
    return {
        "seed_signature_threshold": (
            SIGNATURE_THRESHOLD + noisy * NOISE_SEED_SIGNATURE_INCREMENT
        ).astype(np.float32),
        "seed_luminance_threshold": (
            LUMINANCE_THRESHOLD + noisy * NOISE_SEED_LUMINANCE_INCREMENT
        ).astype(np.float32),
        "support_signature_threshold": (
            SUPPORT_SIGNATURE_THRESHOLD + noisy * NOISE_SUPPORT_SIGNATURE_INCREMENT
        ).astype(np.float32),
        "support_luminance_threshold": (
            SUPPORT_LUMINANCE_THRESHOLD + noisy * NOISE_SUPPORT_LUMINANCE_INCREMENT
        ).astype(np.float32),
        "seed_signature_median": zeros.copy(),
        "seed_luminance_median": zeros.copy(),
        "seed_signature_mad": zeros.copy(),
        "seed_luminance_mad": zeros.copy(),
        "seed_signature_q95": zeros.copy(),
        "seed_luminance_q95": zeros.copy(),
        "seed_signature_cap": np.full(
            (height, width), SIGNATURE_THRESHOLD + NOISE_SEED_SIGNATURE_INCREMENT,
            np.float32,
        ),
        "seed_luminance_cap": np.full(
            (height, width), LUMINANCE_THRESHOLD + NOISE_SEED_LUMINANCE_INCREMENT,
            np.float32,
        ),
        "support_blocks": zeros.copy(),
        "over_cap_fraction": zeros.copy(),
        "bias_flag": zeros.astype(np.uint8),
        "legacy_tolerance_converted": np.array(1, np.uint8),
    }


__all__ = [
    "ASSET_NAMES", "BANK_KIND", "BankError", "BoundedContextCache",
    "LEGACY_PROFILE_KIND", "MAX_REFERENCE_PIXELS", "ProfileBank", "ProfileRecord",
    "SCHEMA_VERSION", "atomic_write_bytes", "atomic_write_json", "atomic_write_text",
    "canonical_json", "default_camera_geometry", "default_matcher_config",
    "estimate_context_bytes", "latest_version", "list_versions",
    "load_bank", "load_legacy_tolerance_as_noise", "publish_version",
    "read_npz", "resolve_bank_version", "sha256_bytes", "sha256_file",
    "verify_checksum", "write_descriptor", "write_noise", "write_png",
]
