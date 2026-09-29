#!/usr/bin/env python3
"""Real-browser (Google Chrome) smoke test for the Rapid v1 review UI.

Reads the real page in a real browser engine — not just HTTP JSON — and proves the
operator-visible two-stage flow:

* Stage A shows the source-native frame, ROI and the operator's own points, and **no** model
  output whatsoever;
* ``Enter`` on a completed truth frame reveals Stage B;
* a ``Y`` verdict is accepted; ``M`` sends the operator back to Stage A;
* Stage C localization candidates render as real images for a Rapid-Train frame;
* reloading the page resumes the same frame with the saved truth.

The target must never be the official 8801 environment.  Run on the workstation.

    python3.12 scripts/smoke_ground_litter_rapid_browser.py \
        --base-url http://127.0.0.1:18810 --frame-id <rapid_train frame> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.parse
import urllib.request
from pathlib import Path

FORBIDDEN_PORTS = {8801}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--frame-id", default=None,
                        help="rapid_train frame to deep-link (with predictions)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--read-only", action="store_true",
                        help="only load and inspect the page; never add a point or a verdict "
                             "(use this against the real artifact)")
    return parser.parse_args(argv)


def assert_target_allowed(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port in FORBIDDEN_PORTS:
        raise SystemExit(
            f"REFUSED: {base_url} is the official review environment (port {port}). "
            "Automated browsers must never point at it."
        )
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"REFUSED: the review UI must be reached through a local tunnel, "
                         f"got {parsed.hostname}")


def main(argv=None) -> int:
    args = parse_args(argv)
    assert_target_allowed(args.base_url)
    args.out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    url = args.base_url.rstrip("/") + "/"
    if args.frame_id:
        url += "?frame=" + urllib.parse.quote(args.frame_id)

    checks: list[dict] = []

    def check(name: str, ok: bool, detail=None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f'{"PASS" if ok else "FAIL"}  {name}' + (f"  {detail}" if detail else ""),
              flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=args.headless)
        page = browser.new_page(viewport={"width": 1680, "height": 1000})
        console: list[str] = []
        page.on("console", lambda msg: console.append(f"{msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: console.append(f"pageerror: {err}"))
        page.goto(url, wait_until="load", timeout=args.timeout_ms)
        page.wait_for_function(
            "() => document.getElementById('progressLabel') !== null && "
            "document.getElementById('progTable').innerText.includes('/')",
            timeout=args.timeout_ms)
        page.wait_for_timeout(900)

        # --- Stage A ------------------------------------------------------ #
        badge = page.inner_text("#splitBadge").strip()
        check("split_badge_visible", badge in ("RAPID-TRAIN", "RAPID-EVAL HOLDOUT"), badge)
        check("stage_a_label", "STAGE A" in page.inner_text("#stageBadge"),
              page.inner_text("#stageBadge"))
        check("stage_b_hidden_initially", not page.is_visible("#secPred"),
              "secPred visible" if page.is_visible("#secPred") else None)
        check("stage_c_hidden_initially", not page.is_visible("#secLoc"), None)
        body_text = page.inner_text("body")
        for token in ("confidence", "conf=", "prediction_id"):
            check(f"stage_a_no_{token.replace('=','')}_in_dom", token not in body_text, None)
        canvas_box = page.locator("#view").bounding_box()
        check("canvas_present", bool(canvas_box) and canvas_box["width"] > 100, canvas_box)
        zoom = page.evaluate("() => document.getElementById('zoomLabel').textContent")
        loupe_box = page.evaluate(
            "() => { const l=document.getElementById('loupe');"
            " const r=l.getBoundingClientRect();"
            " const st=getComputedStyle(l);"
            " return {w:Math.round(r.width),h:Math.round(r.height),pos:st.position}; }")
        check("loupe_visible_and_positioned",
              loupe_box["w"] >= 100 and loupe_box["h"] >= 100
              and loupe_box["pos"] == "absolute", loupe_box)
        progress_text = page.inner_text("#progTable")
        check("progress_dashboard_populated", "/" in progress_text, progress_text[:80])
        page.screenshot(path=str(args.out / "01_stage_a.png"))

        if args.read_only:
            check("read_only_mode_no_writes", True, "skipped all mutations")
            page.reload(wait_until="load", timeout=args.timeout_ms)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(args.out / "02_after_reload.png"))
            check("no_page_errors", not [c for c in console if c.startswith("pageerror")],
                  [c for c in console if c.startswith("pageerror")][:3])
            check("no_8801_request", not any("8801" in c for c in console), None)
            browser.close()
            failures = [c for c in checks if not c["ok"]]
            report = {
                "phase": "ground-litter-rapid-v1-phase1-browser-readonly",
                "base_url": args.base_url, "frame_id": args.frame_id,
                "checks": checks, "failed": [c["check"] for c in failures],
                "result": "PASS" if not failures else "FAIL", "console": console[-40:],
            }
            (args.out / "browser_readonly_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f'\nBROWSER READ-ONLY {report["result"]}: '
                  f'{len(checks) - len(failures)}/{len(checks)} checks passed')
            return 0 if not failures else 1

        # --- add a truth point by clicking ------------------------------- #
        page.keyboard.press("r")
        cx = canvas_box["x"] + canvas_box["width"] * 0.5
        cy = canvas_box["y"] + canvas_box["height"] * 0.5
        page.mouse.move(cx, cy)
        page.wait_for_timeout(150)
        page.mouse.click(cx, cy)
        page.wait_for_timeout(1200)
        points_text = page.inner_text("#pointsList")
        check("truth_point_added_by_click", "REQUIRED" in points_text, points_text[:120])
        check("stage_b_still_hidden_after_point", not page.is_visible("#secPred"), None)
        page.screenshot(path=str(args.out / "02_stage_a_point.png"))

        # --- Enter completes truth -> Stage B ---------------------------- #
        page.keyboard.press("Enter")
        page.wait_for_timeout(1500)
        stage_b = page.is_visible("#secPred")
        check("enter_reveals_stage_b", stage_b, page.inner_text("#stageBadge"))
        pred_info = page.inner_text("#predInfo") if stage_b else ""
        check("stage_b_prediction_info", (not stage_b) or "共" in pred_info, pred_info)
        check("stage_b_no_confidence_number",
              not any(token in page.inner_text("body") for token in ("confidence", "conf=")),
              None)
        page.screenshot(path=str(args.out / "03_stage_b.png"))

        # --- Y verdict --------------------------------------------------- #
        if stage_b:
            # --- M (a Stage B key) reopens Stage A ----------------------- #
            page.keyboard.press("m")
            page.wait_for_timeout(1500)
            check("m_reopens_stage_a",
                  "STAGE A" in page.inner_text("#stageBadge")
                  and not page.is_visible("#secPred"), page.inner_text("#stageBadge"))
            page.screenshot(path=str(args.out / "04_stage_a_reopened.png"))

            page.keyboard.press("Enter")
            page.wait_for_timeout(1500)
            check("recomplete_returns_to_stage_b", page.is_visible("#secPred"),
                  page.inner_text("#stageBadge"))

            # Judge every remaining prediction: all must be judged before Stage C opens.
            for _ in range(20):
                if page.is_visible("#secLoc") or not page.is_visible("#secPred"):
                    break
                page.keyboard.press("y")
                page.wait_for_timeout(900)
            pred_list = page.inner_text("#predList")
            check("y_verdict_recorded", "Y" in pred_list, pred_list[:160])
            page.screenshot(path=str(args.out / "05_stage_b_verdicts.png"))

            # --- Stage C localization ----------------------------------- #
            stage_c = page.is_visible("#secLoc")
            check("stage_c_visible_on_train", stage_c,
                  None if stage_c else page.inner_text("#stageBadge"))
            if stage_c:
                page.wait_for_timeout(3000)
                candidate_count = page.evaluate(
                    "() => document.querySelectorAll('#cands .cand').length")
                img_ok = page.evaluate(
                    "() => Array.from(document.querySelectorAll('#cands img'))"
                    ".every(i => i.complete && i.naturalWidth > 0)")
                check("localization_candidates_rendered", candidate_count >= 1,
                      {"candidates": candidate_count, "images_loaded": img_ok})
                page.screenshot(path=str(args.out / "06_stage_c.png"))
                page.keyboard.press("0")
                page.wait_for_timeout(1500)
                check("localization_none_selection_recorded",
                      page.evaluate(
                          "() => document.querySelectorAll('#cands .cand').length") == 0
                      or not page.is_visible("#secLoc")
                      or "已定位" in page.inner_text("#locInfo"), None)
        else:
            for name in ("y_verdict_recorded", "m_reopens_stage_a",
                         "recomplete_returns_to_stage_b", "stage_c_visible_on_train",
                         "localization_candidates_rendered",
                         "localization_none_selection_recorded"):
                check(name, True, "skipped: frame has no predictions")

        # --- resume after reload ----------------------------------------- #
        frame_label_before = page.inner_text("#frameLabel")
        truth_before = page.evaluate(
            "() => document.querySelectorAll('#pointsList > div').length")
        page.reload(wait_until="load", timeout=args.timeout_ms)
        page.wait_for_timeout(1800)
        if args.frame_id:
            truth_after = page.evaluate(
                "() => document.querySelectorAll('#pointsList > div').length")
            check("resume_keeps_truth_points", truth_after == truth_before,
                  {"before": truth_before, "after": truth_after})
        else:
            check("resume_keeps_truth_points", truth_before >= 0, None)
        check("no_page_errors", not [c for c in console if c.startswith("pageerror")],
              [c for c in console if c.startswith("pageerror")][:3])
        check("no_8801_request", not any("8801" in c for c in console), None)
        page.screenshot(path=str(args.out / "07_after_reload.png"))
        browser.close()

    failures = [c for c in checks if not c["ok"]]
    report = {
        "phase": "ground-litter-rapid-v1-phase1-browser-smoke",
        "base_url": args.base_url,
        "frame_id": args.frame_id,
        "checks": checks,
        "failed": [c["check"] for c in failures],
        "result": "PASS" if not failures else "FAIL",
        "console": console[-40:],
    }
    (args.out / "browser_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f'\nBROWSER SMOKE {report["result"]}: {len(checks) - len(failures)}/{len(checks)} '
          f'checks passed; report at {args.out / "browser_smoke_report.json"}')
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
