#!/usr/bin/env python3
"""Read-only-ish diagnostic: capture exactly what Stage C localization submit does.

Drives a real Chrome against a review service and records every /api/ request and response
while the operator-style actions happen (click candidate A, press 1, press 0).  Prints a
verdict for the four possible failure layers:

    A. no request was sent
    B. request sent but the backend rejected it
    C. backend returned 200 but the state was not saved
    D. backend saved, but the frontend did not advance

Run against an isolated copy, never the official 8801 environment.

    python3.12 scripts/diag_ground_litter_rapid_localization.py \
        --base-url http://127.0.0.1:18813 --frame-id <frame> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

FORBIDDEN_PORTS = {8801}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--skip-click", action="store_true",
                        help="only press the number keys, do not click a candidate")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    parsed = urllib.parse.urlparse(args.base_url)
    port = parsed.port or 80
    if port in FORBIDDEN_PORTS:
        raise SystemExit(f"REFUSED: {args.base_url} is the official environment")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("REFUSED: use a local tunnel")
    args.out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    traffic: list[dict] = []
    console: list[str] = []

    def snapshot(page):
        return {
            "loc_info": (page.inner_text("#locInfo") if page.is_visible("#secLoc")
                         else "(secLoc hidden)"),
            "stage": page.inner_text("#stageBadge"),
            "candidate_cards": page.evaluate(
                "() => Array.from(document.querySelectorAll('#cands .cand'))"
                ".map(c => c.innerText.replace(/\\n/g,' | '))"),
            "active_card": page.evaluate(
                "() => { const a=document.querySelector('#cands .cand.act');"
                " return a ? a.innerText.replace(/\\n/g,' | ') : null; }"),
            "points_listed": page.evaluate(
                "() => document.querySelectorAll('#pointsList > div').length"),
            "progress": page.inner_text("#progTable"),
        }

    url = args.base_url.rstrip("/") + "/?frame=" + urllib.parse.quote(args.frame_id)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1000})
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        def on_request(request):
            if "/api/" in request.url:
                traffic.append({"dir": "request", "method": request.method,
                                "url": request.url.split("?")[0].split("/api/")[-1],
                                "query": urllib.parse.urlparse(request.url).query,
                                "post_data": request.post_data})

        def on_response(response):
            if "/api/" in response.url:
                body = None
                try:
                    body = response.text()[:1200]
                except Exception:                                 # noqa: BLE001
                    body = "(unreadable)"
                traffic.append({"dir": "response", "status": response.status,
                                "url": response.url.split("?")[0].split("/api/")[-1],
                                "body": body})

        page.on("request", on_request)
        page.on("response", on_response)
        page.goto(url, wait_until="load", timeout=args.timeout_ms)
        page.wait_for_selector("#secLoc", state="visible", timeout=args.timeout_ms)
        page.wait_for_timeout(3500)

        before = snapshot(page)
        observations = {"before": before}

        # --- action 1: click candidate A (or the first card) --------------- #
        clicked = False
        if not args.skip_click:
            cards = page.locator("#cands .cand")
            if cards.count() > 0:
                traffic.clear()
                cards.nth(0).click()
                page.wait_for_timeout(3500)
                clicked = True
                observations["after_click_A"] = snapshot(page)
                observations["click_A_traffic"] = list(traffic)

        # --- action 2: keyboard 1 ------------------------------------------ #
        traffic.clear()
        page.keyboard.press("1")
        page.wait_for_timeout(3500)
        observations["after_key_1"] = snapshot(page)
        observations["key_1_traffic"] = list(traffic)

        # --- action 3: keyboard 0 ------------------------------------------ #
        traffic.clear()
        page.keyboard.press("0")
        page.wait_for_timeout(3500)
        observations["after_key_0"] = snapshot(page)
        observations["key_0_traffic"] = list(traffic)

        observations["console"] = console[-40:]
        browser.close()

    def submitted(step: str) -> dict | None:
        for entry in observations.get(step) or []:
            if entry.get("dir") == "response" and entry.get("url") == "localize_select":
                return entry
        return None

    def requested(step: str) -> dict | None:
        for entry in observations.get(step) or []:
            if entry.get("dir") == "request" and entry.get("url") == "localize_select":
                return entry
        return None

    verdict = {}
    for step in ("click_A_traffic", "key_1_traffic", "key_0_traffic"):
        request = requested(step)
        response = submitted(step)
        if request is None:
            verdict[step] = "A: no localize_select request was sent"
        elif response is None:
            verdict[step] = "B?: request sent but no response captured"
        elif response["status"] != 200:
            verdict[step] = f'B: backend rejected with {response["status"]}: {response["body"]}'
        else:
            verdict[step] = f'C/D: backend 200 -> {response["body"]}'

    pending_before = observations["before"]["loc_info"]
    pending_after = observations["after_key_1"]["loc_info"]
    observations["verdict"] = verdict
    observations["advanced_after_key_1"] = pending_before != pending_after
    (args.out / "localization_diag.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Stage C state BEFORE ===")
    print(json.dumps(observations["before"], ensure_ascii=False, indent=1))
    if clicked:
        print("=== after click candidate A ===")
        print(json.dumps(observations["after_click_A"], ensure_ascii=False, indent=1))
    print("=== after key 1 ===")
    print(json.dumps(observations["after_key_1"], ensure_ascii=False, indent=1))
    print("=== after key 0 ===")
    print(json.dumps(observations["after_key_0"], ensure_ascii=False, indent=1))
    print("\n=== VERDICT ===")
    for step, text in verdict.items():
        print(f"{step:18} {text}")
    print("\nadvanced_after_key_1:", observations["advanced_after_key_1"])
    print("console tail:", json.dumps(observations["console"][-6:], ensure_ascii=False))
    print(f'\nreport: {args.out / "localization_diag.json"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
