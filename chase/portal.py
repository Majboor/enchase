#!/usr/bin/env python3
"""CHASE SSDC portal client: search the archive and mint signed download URLs.

The portal has a captcha, so you log in ONCE in a real browser window; the
session cookie is saved and every later command runs headless.

  # 1. one-time (repeat when the session expires)
  chase-portal login

  # 2. from then on, pure CLI
  chase-portal search --start "2023-03-29 02:00" --end "2023-03-29 03:00"
  chase-portal urls   --start "2023-03-29 02:00" --end "2023-03-29 03:00" -o links.txt

  # 3. feed the links straight into the pipeline
  chase links.txt --patch 940 1080 1870 2040

Signed URLs expire in ~24h, so mint them right before downloading.
State lives in ~/.chase_portal/ (chmod 600). Times are UTC.
The ``login`` step needs playwright (``pip install chasepy[portal]`` then
``playwright install chromium``); search/urls use plain requests.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

STATE_DIR = os.path.expanduser("~/.chase_portal")
COOKIES = os.path.join(STATE_DIR, "cookies.json")
BASE = "https://ssdc.nju.edu.cn/web-service/v1"
REFERER = "https://ssdc.nju.edu.cn/NdchaseSatellite"
# The portal parses this exact JS Date.toUTCString-ish format.
PORTAL_TIME = "%a %b %d %Y %H:%M:%S UTC"


def _session():
    import requests

    if not os.path.exists(COOKIES):
        sys.exit("No saved session. Run:  chase-portal login")
    cookies = {c["name"]: c["value"] for c in json.load(open(COOKIES))}
    s = requests.Session()
    s.cookies.update(cookies)
    s.headers.update({"Referer": REFERER, "Accept": "application/json, text/plain, */*",
                      "User-Agent": "Mozilla/5.0"})
    r = s.post(f"{BASE}/login/getUser", timeout=60)
    who = r.json().get("data") or {}
    if not who.get("isLogin"):
        sys.exit("Session expired. Run:  chase-portal login")
    return s, (who.get("user") or {}).get("mail", "?")


def _parse(t):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    sys.exit(f"bad time {t!r} — use 'YYYY-MM-DD HH:MM'")


def _search(s, start, end, line=None, page=1, limit=500):
    """One page of observation records for a UTC window."""
    files = {
        "beginTime": (None, start.strftime(PORTAL_TIME)),
        "endTime": (None, end.strftime(PORTAL_TIME)),
        "patternList": (None, "RSM"),
        "lineSpectrumList": (None, ""),
        "targetList": (None, ""),
        "dataType": (None, "chase"),
        "page": (None, str(page)),
        "limit": (None, str(limit)),
    }
    d = s.post(f"{BASE}/cl_observation_data/searchDataS", files=files, timeout=300).json()
    recs = d.get("data") or []
    if line:
        recs = [r for r in recs if r["fitsName"].endswith(f"_{line.upper()}.fits")]
    return recs, d.get("totalCount", 0)


def _search_all(s, start, end, line=None, chunk_hours=6):
    """Walk the window in chunks so no single query is truncated."""
    out, seen = [], set()
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(hours=chunk_hours), end)
        page = 1
        while True:
            recs, total = _search(s, cur, nxt, line, page=page)
            for r in recs:
                if r["fitsName"] not in seen:
                    seen.add(r["fitsName"])
                    out.append(r)
            if len(recs) < 500:
                break
            page += 1
        print(f"  {cur:%Y-%m-%d %H:%M} .. {nxt:%H:%M}  -> {len(seen)} files so far", file=sys.stderr)
        cur = nxt
    out.sort(key=lambda r: r["fitsName"])
    return out


def cmd_login(_):
    from playwright.sync_api import sync_playwright

    os.makedirs(STATE_DIR, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False)
        ctx = b.new_context()
        page = ctx.new_page()
        try:
            page.goto("https://ssdc.nju.edu.cn/NdchaseSatellite", timeout=120000,
                      wait_until="domcontentloaded")
        except Exception as e:
            print(f"(slow load: {e} — navigate manually in the window)")
        print("Log in (solve the captcha), then CLOSE the browser window.")
        while len(ctx.pages) > 0:
            time.sleep(1)
            try:
                json.dump(ctx.cookies(), open(COOKIES, "w"), indent=1)
            except Exception:
                pass
        json.dump(ctx.cookies(), open(COOKIES, "w"), indent=1)
        b.close()
    os.chmod(COOKIES, 0o600)
    s, who = _session()
    print(f"session saved for {who}")


def cmd_search(a):
    s, who = _session()
    recs = _search_all(s, _parse(a.start), _parse(a.end), a.line)
    ha = [r for r in recs if r["fitsName"].endswith("_HA.fits")]
    fe = [r for r in recs if r["fitsName"].endswith("_FE.fits")]
    size = sum(int(r.get("csize") or 0) for r in recs)
    print(f"user: {who}")
    print(f"files: {len(recs)}   HA: {len(ha)}   FE: {len(fe)}   total: {size/2**30:.2f} GiB")
    if recs:
        print(f"first: {recs[0]['fitsName']}\nlast:  {recs[-1]['fitsName']}")
    if a.list:
        for r in recs:
            print(f"  {r['fitsName']}  {int(r['csize'])/2**20:.0f} MiB")


def cmd_urls(a):
    s, who = _session()
    recs = _search_all(s, _parse(a.start), _parse(a.end), a.line)
    if not recs:
        sys.exit("no files in that window")
    urls = []
    for i in range(0, len(recs), a.batch):
        batch = recs[i:i + a.batch]
        r = s.post(f"{BASE}/web_download_log/url", json={"observationDatas": batch}, timeout=600)
        got = (r.json() or {}).get("data") or []
        urls += got
        print(f"  minted {len(urls)}/{len(recs)}", file=sys.stderr)
    out = open(a.output, "w") if a.output else sys.stdout
    for u in urls:
        print(u, file=out)
    if a.output:
        out.close()
        os.chmod(a.output, 0o600)
        size = sum(int(x.get("csize") or 0) for x in recs)
        print(f"wrote {len(urls)} signed URLs -> {a.output}  ({size/2**30:.2f} GiB, "
              f"{sum(1 for r in recs if r['fitsName'].endswith('_HA.fits'))} HA / "
              f"{sum(1 for r in recs if r['fitsName'].endswith('_FE.fits'))} FE)")
        print("URLs expire in ~24h — download soon.")


def main():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = P.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login", help="open a browser once to capture the session").set_defaults(fn=cmd_login)
    for name, fn, extra in (("search", cmd_search, False), ("urls", cmd_urls, True)):
        q = sub.add_parser(name, help="find observations" if not extra else "mint signed download URLs")
        q.add_argument("--start", required=True, help="UTC 'YYYY-MM-DD HH:MM'")
        q.add_argument("--end", required=True, help="UTC 'YYYY-MM-DD HH:MM'")
        q.add_argument("--line", choices=["HA", "FE"], help="only this spectral line")
        if extra:
            q.add_argument("-o", "--output", help="write URLs here (default stdout)")
            q.add_argument("--batch", type=int, default=100, help="records per mint request")
        else:
            q.add_argument("--list", action="store_true", help="print every filename")
        q.set_defaults(fn=fn)
    A = P.parse_args()
    A.fn(A)


if __name__ == "__main__":
    main()
