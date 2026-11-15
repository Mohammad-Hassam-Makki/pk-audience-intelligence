"""Collect public video + comment data for Pakistani lawn brands with the YouTube Data API v3.

Setup (once): Google Cloud Console -> new project -> enable "YouTube Data API v3" -> Credentials -> API key.
    export YOUTUBE_API_KEY=your_key            (Windows PowerShell:  $env:YOUTUBE_API_KEY="your_key")
Run:
    python src/collect_youtube.py --brands brands.txt --max-videos 100 --max-comments 100 --lawn-only

Writes data/posts.csv and data/comments.csv (Tables A and B of the brief) with provenance columns. Quota: default 10,000
units/day; this script costs ~100 per brand to find the channel (search.list, cached in data/channels.json) and ~1 unit per
50 videos / per page of 100 comments. 10 brands x 100 videos x 100 comments is roughly 1,000 + 30 + 1,000 units.
Only collect what the YouTube API Terms and each channel's content allow, keep the data for coursework, and do not republish comments.
"""
import argparse, csv, json, os, re, sys, time, datetime as dt
import requests
sys.path.insert(0, os.path.dirname(__file__)); from nlp import normalize
from nlp import classify_post
BASE = "https://www.googleapis.com/youtube/v3"; KEY = os.environ.get("YOUTUBE_API_KEY")
LAWN = re.compile(r"lawn|suit|unstitched|stitched|embroidered|printed|pret|3[- ]?piece|2[- ]?piece|collection|kurta|shalwar", re.I)

class Quota(Exception): pass
def api(endpoint, **params):
    params["key"] = KEY
    for attempt in range(3):
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        if r.status_code == 200: return r.json()
        err = (r.json().get("error", {}) if r.headers.get("content-type", "").startswith("application/json") else {})
        reason = (err.get("errors") or [{}])[0].get("reason", "")
        if reason in ("quotaExceeded", "dailyLimitExceeded"): raise Quota("YouTube API daily quota used up; rerun tomorrow or request more quota.")
        if reason in ("commentsDisabled", "videoNotFound", "forbidden"): return {"items": [], "_skipped": reason}
        if r.status_code >= 500: time.sleep(2 ** attempt); continue
        raise RuntimeError(f"{endpoint} failed: {r.status_code} {err.get('message', r.text[:200])}")
    raise RuntimeError(f"{endpoint}: repeated server errors")

def resolve_channel(name, cache):
    if name in cache: return cache[name]
    j = api("search", part="snippet", type="channel", q=f"{name} official", maxResults=1)
    if not j["items"]: print(f"  ! no channel found for {name}"); return None
    it = j["items"][0]; cache[name] = dict(id=it["snippet"]["channelId"], title=it["snippet"]["channelTitle"]); return cache[name]

def iso_dur(s):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    d, h, mi, se = (int(x or 0) for x in m.groups()) if m else (0, 0, 0, 0); return d * 86400 + h * 3600 + mi * 60 + se

def uploads(channel_id, max_videos):
    pl = api("channels", part="contentDetails", id=channel_id)["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]; ids, tok = [], None
    while len(ids) < max_videos:
        j = api("playlistItems", part="contentDetails", playlistId=pl, maxResults=50, **({"pageToken": tok} if tok else {}))
        ids += [i["contentDetails"]["videoId"] for i in j["items"]]; tok = j.get("nextPageToken")
        if not tok: break
    return ids[:max_videos]

def video_rows(ids, brand, today):
    rows = []
    for i in range(0, len(ids), 50):
        for v in api("videos", part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50]))["items"]:
            sn, st = v["snippet"], v.get("statistics", {}); dur = iso_dur(v["contentDetails"].get("duration"))
            pub = dt.datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00")); ctype, camp = classify_post(sn["title"], dur)
            text = sn["title"] + " " + sn.get("description", "")[:300]
            rows.append(dict(post_id=v["id"], brand=brand, platform="YouTube", campaign=camp, content_type=ctype,
                product_category="Lawn suit" if LAWN.search(text) else "Other", likes=st.get("likeCount"), comments_count=st.get("commentCount"),
                shares="", views=st.get("viewCount"), days_since_post=(today - pub.date()).days, title=sn["title"], published_at=sn["publishedAt"],
                duration_s=dur, source_url=f"https://www.youtube.com/watch?v={v['id']}", collected_at=today.isoformat(), data_provenance="youtube_api"))
    return rows

def comments_for(video_id, limit):
    out, tok = [], None
    while len(out) < limit:
        j = api("commentThreads", part="snippet", videoId=video_id, maxResults=min(100, limit - len(out)), order="relevance", textFormat="plainText", **({"pageToken": tok} if tok else {}))
        for it in j["items"]:
            s = it["snippet"]["topLevelComment"]["snippet"]
            out.append(dict(comment_id=it["id"], post_id=video_id, comment_text=s["textDisplay"], like_count=s.get("likeCount", 0), published_at=s["publishedAt"]))
        tok = j.get("nextPageToken")
        if not tok: break
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--brands", default="brands.txt"); ap.add_argument("--max-videos", type=int, default=100)
    ap.add_argument("--max-comments", type=int, default=100); ap.add_argument("--lawn-only", action="store_true"); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    if not KEY: sys.exit("Set the YOUTUBE_API_KEY environment variable first (see the top of this file).")
    brands = [b.strip() for b in open(a.brands, encoding="utf-8") if b.strip() and not b.startswith("#")]; os.makedirs(a.out, exist_ok=True)
    cpath = f"{a.out}/channels.json"; cache = json.load(open(cpath)) if os.path.exists(cpath) else {}; today = dt.date.today(); P, C = [], []
    try:
        for b in brands:
            ch = resolve_channel(b, cache); json.dump(cache, open(cpath, "w"), indent=1)
            if not ch: continue
            print(f"{b} -> channel '{ch['title']}' (check this is the official channel)")
            rows = video_rows(uploads(ch["id"], a.max_videos), b, today)
            if a.lawn_only: rows = [r for r in rows if r["product_category"] == "Lawn suit"]
            for r in rows:
                if int(r["comments_count"] or 0) > 0: C += comments_for(r["post_id"], a.max_comments)
            P += rows; print(f"  {len(rows)} videos, {len(C)} comments so far")
    except Quota as e: print("STOPPED:", e)
    if not P: sys.exit("No data collected.")
    for name, rows in (("posts", P), ("comments", C)):
        with open(f"{a.out}/{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Saved {len(P)} posts and {len(C)} comments to {a.out}/")
if __name__ == "__main__": main()
