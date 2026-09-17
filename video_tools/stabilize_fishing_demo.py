#!/usr/bin/env python3
"""Make a clean demo copy of a burned-in fishing-video recording.

The source recording already contains several red OSD rectangles.  This tool
removes those red overlay pixels, chooses one large vessel rectangle, applies
a causal smoother, and draws one red box plus a Chinese vessel-number label.
The number is intentionally a demo value (not an OCR claim); replace the
``--number`` argument when producing another demonstration.

Only OpenCV, NumPy and FFmpeg are required.  The input is never modified.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Include the dark maroon produced by H.264, but downstream only removes
    # long line-like structures.  We never inpaint an entire rectangle.
    red = (((hsv[:, :, 0] <= 20) | (hsv[:, :, 0] >= 160))
           & (hsv[:, :, 1] >= 45)
           & (hsv[:, :, 2] >= 35))
    red[: int(frame.shape[0] * 0.24)] = False  # keep the status panel
    return (red.astype(np.uint8) * 255)


def _candidates(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    result: list[tuple[int, int, int, int, int]] = []
    for x, y, w, h, area in stats[1:]:
        if y < height * 0.24 or w < width * 0.07 or h < height * 0.045:
            continue
        if area < max(2_000, width * height * 0.00015):
            continue
        result.append((int(x), int(y), int(w), int(h), int(area)))
    return result


def _iou(a: tuple[float, float, float, float], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _choose(candidates: list[tuple[int, int, int, int, int]], previous: tuple[float, float, float, float] | None):
    if not candidates:
        return None
    if previous is None:
        x, y, w, h, _ = max(candidates, key=lambda item: item[2] * item[3])
        return float(x), float(y), float(w), float(h)
    # Prefer continuity, with a mild preference for the outer/whole-vessel box.
    scored = []
    for x, y, w, h, area in candidates:
        continuity = _iou(previous, (x, y, w, h))
        cx, cy = x + w / 2, y + h / 2
        pcx, pcy = previous[0] + previous[2] / 2, previous[1] + previous[3] / 2
        distance = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
        score = continuity * 3.0 - distance / 900.0 + min(area / 1e6, 1.0) * 0.15
        scored.append((score, (x, y, w, h)))
    return tuple(float(v) for v in max(scored, key=lambda item: item[0])[1])


def _label_rgba(text: str) -> np.ndarray:
    # Draw once with Pillow so Chinese glyphs are available on macOS/Linux.
    from PIL import Image, ImageDraw, ImageFont

    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    )
    font_path = next((p for p in candidates if Path(p).exists()), None)
    if font_path is None:
        raise RuntimeError("找不到中文字体；请安装 Noto Sans CJK 或指定系统中文字体")
    font = ImageFont.truetype(font_path, 25)
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0] + 24, box[3] - box[1] + 16
    image = Image.new("RGBA", (width, height), (150, 0, 0, 235))
    ImageDraw.Draw(image).text((12, 6), text, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _ocr_panel_rgba(number: str, state: str) -> np.ndarray:
    """Create a conspicuous, but non-destructive OCR status card."""
    from PIL import Image, ImageDraw, ImageFont

    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    )
    font_path = next((p for p in candidates if Path(p).exists()), None)
    if font_path is None:
        raise RuntimeError("找不到中文字体；请安装 Noto Sans CJK 或指定系统中文字体")
    title_font = ImageFont.truetype(font_path, 30)
    locked = state == "已确认"
    number_font = ImageFont.truetype(font_path, 62 if locked else 40)
    width, height = 760, 184 if locked else 158
    image = Image.new("RGBA", (width, height), (8, 22, 35, 232))
    draw = ImageDraw.Draw(image)
    accent = (49, 224, 214, 255) if not locked else (255, 190, 45, 255)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=14,
                           outline=accent, width=4)
    draw.text((24, 16), "船号 OCR 演示", font=title_font, fill=(220, 240, 245, 255))
    draw.text((24, 62), "识别状态", font=title_font, fill=(150, 190, 200, 255))
    draw.text((180, 54), state, font=number_font, fill=accent)
    if locked:
        draw.text((24, 112), number, font=number_font, fill=(255, 255, 255, 255))
    else:
        draw.text((24, 112), "需连续清晰帧确认", font=title_font, fill=(195, 210, 215, 255))
    return np.asarray(image)


def _plate_is_clear(frame: np.ndarray, plate: tuple[int, int, int, int],
                    min_width: int, min_height: int, min_sharpness: float) -> bool:
    """Conservative quality gate for the demo OCR trigger."""
    x, y, w, h = plate
    if w < min_width or h < min_height:
        return False
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharpness >= min_sharpness


def _blend_rgba(frame: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    """Alpha blend a pre-rendered RGBA card in-place."""
    h, w = image.shape[:2]
    fh, fw = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    if x + w > fw or y + h > fh:
        return
    patch = frame[y : y + h, x : x + w]
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    # PIL stores RGB; OpenCV stores BGR.
    rgb = image[:, :, :3][:, :, ::-1].astype(np.float32)
    patch[:] = (patch.astype(np.float32) * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


def _find_blue_plate(
    frame: np.ndarray,
    vessel: tuple[float, float, float, float],
) -> tuple[int, int, int, int] | None:
    """Locate a plausible blue number board; text recognition is simulated."""
    frame_height, frame_width = frame.shape[:2]
    vx, vy, vw, vh = [int(round(value)) for value in vessel]
    vx, vy = max(vx, 0), max(vy, 0)
    vw = min(vw, frame_width - vx)
    vh = min(vh, frame_height - vy)
    crop = frame[vy : vy + vh, vx : vx + vw]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(
        hsv,
        np.array((90, 65, 35), dtype=np.uint8),
        np.array((135, 255, 255), dtype=np.uint8),
    )
    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
    )
    contours, _hierarchy = cv2.findContours(
        blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if w < 55 or h < 10 or not 2.8 <= aspect <= 12.0:
            continue
        if w > vw * 0.45 or h > vh * 0.28:
            continue
        if y + h / 2 < vh * 0.30:
            continue
        candidates.append((vx + x, vy + y, w, h))
    return max(candidates, key=lambda item: item[2] * item[3]) if candidates else None


def _find_blue_plate_global(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a plausible blue boat plate when the source has no OSD box."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array((90, 65, 35), np.uint8), np.array((135, 255, 255), np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if w < width * 0.025 or h < height * 0.012 or not 2.8 <= aspect <= 12:
            continue
        if y < height * 0.30 or y > height * 0.88:
            continue
        found.append((x, y, w, h))
    return max(found, key=lambda item: item[2] * item[3]) if found else None


def _vessel_from_plate(plate: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = plate
    # The plate sits near the covered middle section of this boat. Expand to
    # include bow, stern, and a little water margin for a stable demo box.
    x0 = max(0, x - int(2.4 * w))
    y0 = max(0, y - int(3.4 * h))
    x1 = min(width, x + int(4.8 * w))
    y1 = min(height, y + int(5.8 * h))
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def process(args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(args.start * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_width = source_width if args.width <= 0 else int(args.width)
    output_height = int(round(source_height * output_width / source_width))
    output_height -= output_height % 2
    output_fps = source_fps if args.fps <= 0 else min(float(args.fps), source_fps)
    reference_scale = output_width / 3334.0
    ocr_min_width = max(1, int(round(args.ocr_min_width * reference_scale)))
    ocr_min_height = max(1, int(round(args.ocr_min_height * reference_scale)))
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{output_width}x{output_height}",
        "-framerate", str(output_fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    searching_label = _label_rgba("追踪中｜船｜船牌校验中…")
    recognized_label = _label_rgba(f"追踪中｜船｜船号 {args.number}")
    waiting_panel = _ocr_panel_rgba(args.number, "等待清晰船牌")
    scanning_panel = _ocr_panel_rgba(args.number, "多帧校验中")
    locked_panel = _ocr_panel_rgba(args.number, "已确认")
    previous = None
    held = 0
    number_locked = False
    plate_evidence = 0.0
    first_clear_time: float | None = None
    # The capture is already positioned at ``start_frame``; keep the source
    # clock aligned so duration/cadence logic does not discard every frame.
    frame_index = start_frame
    next_output_time = 0.0
    alpha = float(args.smoothing)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_time = frame_index / source_fps
            if source_time < args.start:
                frame_index += 1
                continue
            elapsed = source_time - args.start
            if args.duration > 0 and elapsed >= args.duration:
                break
            if output_fps < source_fps - 0.01 and source_time + 1e-6 < args.start + next_output_time:
                frame_index += 1
                continue
            resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            mask = _red_mask(resized)
            candidates = _candidates(mask)
            chosen = None
            global_plate = None
            if chosen is None and previous is None:
                global_plate = _find_blue_plate_global(resized)
                if global_plate is not None:
                    chosen = _vessel_from_plate(global_plate, output_width, output_height)
            if previous is not None:
                chosen = _choose(candidates, previous)
            elif global_plate is None:
                chosen = _choose(candidates, previous)
            if chosen is not None:
                if previous is None:
                    previous = chosen
                else:
                    previous = tuple(alpha * n + (1.0 - alpha) * p for n, p in zip(chosen, previous))
                held = 0
            elif previous is not None and held < int(output_fps * args.hold):
                held += 1
            else:
                previous = None
            if candidates and not args.preserve_source:
                # Remove all lower-half red OSD components, including nested boxes.
                # Erase only line-like OSD pixels.  Do not fill the complete
                # rectangle: that destroys texture inside the boat.
                line_mask = np.zeros_like(mask)
                lines = cv2.HoughLinesP(
                    mask,
                    1,
                    np.pi / 180,
                    threshold=max(40, output_width // 80),
                    minLineLength=max(80, output_width // 35),
                    maxLineGap=max(12, output_width // 180),
                )
                if lines is not None:
                    for item in lines[:, 0]:
                        x1, y1, x2, y2 = [int(value) for value in item]
                        dx, dy = abs(x2 - x1), abs(y2 - y1)
                        if dy <= max(3, output_height // 450) or dx <= max(3, output_width // 700):
                            cv2.line(
                                line_mask,
                                (x1, y1),
                                (x2, y2),
                                255,
                                max(7, output_width // 420),
                            )
                erase = np.zeros_like(mask)
                label_erase = np.zeros_like(mask)
                for x, y, w, h, _area in candidates:
                    x0, y0 = max(0, x - 8), max(0, y - 8)
                    x1, y1 = min(output_width, x + w + 8), min(output_height, y + h + 8)
                    erase[y0:y1, x0:x1] = line_mask[y0:y1, x0:x1]
                    # Old tracker labels are compact solid blocks immediately
                    # above the top edge and, in this recording, lie on water.
                    # Repair them separately so no large mask touches the boat.
                    label_height = max(28, output_height // 28)
                    label_width = min(max(180, output_width // 8), w)
                    ly0 = max(0, y + 2)
                    ly1 = min(output_height, ly0 + label_height)
                    if ly1 > ly0 and label_width > 0:
                        label_erase[ly0:ly1, x:min(output_width, x + label_width)] = 255
                # The old overlay is baked into the pixels. Desaturating it is
                # lossless with respect to luminance/detail, unlike inpainting
                # which invents texture and creates visible blocks.
                suppress = cv2.bitwise_and(
                    cv2.bitwise_or(erase, label_erase), mask
                )
                if cv2.countNonZero(suppress):
                    pixels = suppress > 0
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    neutral = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    resized[pixels] = neutral[pixels]
            if previous is not None:
                plate = _find_blue_plate(resized, previous) or global_plate
                clear_plate = (
                    plate is not None
                    and elapsed >= args.ocr_earliest
                    and _plate_is_clear(
                        resized,
                        plate,
                        ocr_min_width,
                        ocr_min_height,
                        args.ocr_min_sharpness,
                    )
                )
                if clear_plate:
                    if first_clear_time is None:
                        first_clear_time = elapsed
                    plate_evidence = min(
                        plate_evidence + 1.0 / output_fps,
                        args.ocr_min_evidence * 1.5,
                    )
                else:
                    # Real OCR does not lock on one lucky frame.  Evidence
                    # decays slowly to tolerate detector flicker, but a long
                    # loss of a clear plate naturally prevents confirmation.
                    plate_evidence = max(
                        0.0, plate_evidence - 0.12 / output_fps
                    )
                if (
                    not number_locked
                    and first_clear_time is not None
                    and elapsed - first_clear_time >= args.ocr_stable
                    and plate_evidence >= args.ocr_min_evidence
                ):
                    number_locked = True
                    print(
                        f"OCR 演示结果在源视频 {source_time:.2f}s 锁定"
                        f"（清晰帧证据 {plate_evidence:.2f}s）",
                        flush=True,
                    )
                x, y, w, h = [int(round(v)) for v in previous]
                x, y = max(0, x), max(0, y)
                w, h = min(w, output_width - x - 1), min(h, output_height - y - 1)
                cv2.rectangle(resized, (x, y), (x + w, y + h), (35, 35, 235), 5, cv2.LINE_AA)
                label = recognized_label if number_locked else searching_label
                lh, lw = label.shape[:2]
                lx, ly = x, max(0, y - lh)
                patch = resized[ly:ly + lh, lx:lx + lw]
                if patch.shape[:2] == (lh, lw):
                    rgb, a = label[:, :, :3][:, :, ::-1], label[:, :, 3:4] / 255.0
                    patch[:] = (patch * (1 - a) + rgb * a).astype(np.uint8)
                if plate is not None:
                    px, py, pw, ph = plate
                    cv2.rectangle(
                        resized,
                        (px, py),
                        (px + pw, py + ph),
                        (255, 220, 0),
                        3,
                        cv2.LINE_AA,
                    )
                if number_locked:
                    _blend_rgba(resized, locked_panel, 48, 116)
                elif first_clear_time is not None or plate_evidence > 0:
                    _blend_rgba(resized, scanning_panel, 48, 116)
                else:
                    _blend_rgba(resized, waiting_panel, 48, 116)
            encoder.stdin.write(resized.tobytes())
            frame_index += 1
            next_output_time += 1.0 / output_fps
    finally:
        capture.release()
        if encoder.stdin:
            encoder.stdin.close()
        encoder.wait()
    if encoder.returncode:
        raise RuntimeError(f"FFmpeg 编码失败，退出码 {encoder.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--number", default="粤清城渔10032")
    parser.add_argument(
        "--preserve-source",
        action="store_true",
        help="不擦除原始 OSD，只叠加稳定框和演示船号（最高画质）",
    )
    parser.add_argument("--start", type=float, default=0.0, help="起始秒")
    parser.add_argument("--duration", type=float, default=0.0, help="处理时长，0=到结尾")
    parser.add_argument("--width", type=int, default=0, help="输出宽度，0=保持原始分辨率")
    parser.add_argument("--fps", type=float, default=0.0, help="输出帧率，0=保持原始帧率")
    parser.add_argument("--smoothing", type=float, default=0.28, help="框平滑系数，越小越稳")
    parser.add_argument("--hold", type=float, default=0.35, help="短暂漏检时保留框的秒数")
    parser.add_argument(
        "--ocr-stable", type=float, default=3.0,
        help="清晰船牌首次出现后，至少持续观察多少秒才允许锁定",
    )
    parser.add_argument(
        "--ocr-earliest", type=float, default=28.0,
        help="最早从第几秒开始累计 OCR 证据，避免远景/半遮挡误识别",
    )
    parser.add_argument(
        "--ocr-min-width", type=int, default=190,
        help="清晰船牌的最小像素宽度",
    )
    parser.add_argument(
        "--ocr-min-height", type=int, default=44,
        help="清晰船牌的最小像素高度",
    )
    parser.add_argument(
        "--ocr-min-sharpness", type=float, default=18.0,
        help="清晰船牌的 Laplacian 清晰度阈值",
    )
    parser.add_argument(
        "--ocr-min-evidence", type=float, default=1.7,
        help="锁定前需要累计的清晰帧证据秒数（允许短暂检测闪烁）",
    )
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--crf", type=int, default=14)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process(args)


if __name__ == "__main__":
    main()
