import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

BASE = "https://arctic-shift.photon-reddit.com"
HEADERS = {"User-Agent": "ECON7055-Research (education)"}

# Keyword
DISPUTE_KEYWORDS = [
    "refund", "return", "logistics", "customs",
    "fraud", "delay", "scam", "never received", "damaged", "wrong item"
]

# Target Subreddits
SUBREDDITS = [
    "Amazon", "SkincareAddiction", "BeautyGuruChatter",
    "MakeupAddiction", "ecommerce", "OnlineShopping", "beauty"
]

YEARS_RANGE = 2

def get_json(path: str, params: dict, timeout=40, retries=4, backoff=1.6):
    """API requests with retry mechanisms"""
    url = f"{BASE}{path}"
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code} {url} :: {r.text[:250]}")
        except Exception as e:
            last_err = e
        time.sleep(backoff ** i)
    raise last_err


def fetch_posts_for_query(subreddit: str, query: str, start_ts: int, end_ts: int):
    """Pagination capture of posts - Use API supported valid fields"""
    all_rows = []
    cursor = start_ts

    while True:
        params = {
            "subreddit": subreddit,
            "query": query,
            "after": cursor,
            "before": end_ts,
            "sort": "asc",
            "limit": 100,
            # ✅ Only use the valid fields supported by the API
            "fields": "id,created_utc,title,selftext,author,score,num_comments,url,subreddit"
        }
        try:
            js = get_json("/api/posts/search", params)
            data = js.get("data", [])
            if not data:
                break
            all_rows.extend(data)
            last_created = data[-1].get("created_utc")
            if last_created is None:
                break
            nxt = int(last_created) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.35)
        except Exception as e:
            print(f"      rqi request interrupt: {e}")
            break

    return all_rows


def fetch_comments_for_post(post_id: str, start_ts: int, end_ts: int):
    """Pagination capture of comments"""
    all_rows = []
    cursor = start_ts

    while True:
        params = {
            "link_id": post_id,
            "after": cursor,
            "before": end_ts,
            "sort": "asc",
            "limit": 100,
            "fields": "id,author,body,created_utc,score,parent_id,link_id"
        }
        try:
            js = get_json("/api/comments/search", params)
            data = js.get("data", [])
            if not data:
                break
            all_rows.extend(data)
            last_created = data[-1].get("created_utc")
            if last_created is None:
                break
            nxt = int(last_created) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.30)
        except Exception:
            break

    return all_rows


def main():
    # Time Settings
    # START：2021-01-01 00:00:00 UTC
    start_dt = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    # END：2023-12-31 23:59:59 UTC
    end_dt = datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print("=" * 70)
    print("📊 Reddit Data capture of e-commerce disputes | ECON7055 Research Project")
    print("=" * 70)
    print(f"⏰ Time Range：{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')} ({YEARS_RANGE})")
    print(f"🔑 Controversial key words：{len(DISPUTE_KEYWORDS)} ")
    print(f"📂 Target Subreddits：{len(SUBREDDITS)} ")
    print("=" * 70 + "\n")

    all_posts = []

    # Crawl posts
    for subreddit in SUBREDDITS:
        print(f"📂 handle r/{subreddit}...")
        subreddit_posts = {}

        for keyword in DISPUTE_KEYWORDS:
            try:
                rows = fetch_posts_for_query(subreddit, keyword, start_ts, end_ts)
                for p in rows:
                    pid = p.get("id")
                    if pid:
                        subreddit_posts[pid] = p
                if rows:
                    print(f"   → '{keyword}': +{len(rows)} 条")
            except Exception as e:
                print(f"   ✗ '{keyword}' failed: {type(e).__name__}")
                continue
            time.sleep(0.5)

        print(f"   ✅ r/{subreddit}: Total {len(subreddit_posts)} posts\n")
        all_posts.extend(subreddit_posts.values())
        time.sleep(1)

    # Handle empty data
    if not all_posts:
        print("⚠️ No posts were captured. Create an empty file and exit")
        pd.DataFrame(columns=[
            "post_id", "created_utc", "title", "selftext", "author",
            "score", "num_comments", "url", "subreddit", "reddit_link"
        ]).to_csv(".venv/Reddit_ecommerce_disputes_posts.csv", index=False)
        pd.DataFrame(columns=[
            "post_id", "comment_id", "author", "body",
            "created_utc", "score", "parent_id", "link_id"
        ]).to_csv("Reddit_ecommerce_disputes_comments.csv", index=False)
        return

    # Construct DataFrame
    df_posts = pd.DataFrame([{
        "post_id": p.get("id"),
        "created_utc": p.get("created_utc"),
        "title": p.get("title"),
        "selftext": p.get("selftext"),
        "author": p.get("author"),
        "score": p.get("score"),
        "num_comments": p.get("num_comments"),
        "url": p.get("url"),
        "subreddit": p.get("subreddit"),
        "reddit_link": f"https://www.reddit.com/comments/{p.get('id')}/" if p.get("id") else None
    } for p in all_posts])

    df_posts = df_posts.dropna(subset=["post_id"]).drop_duplicates(subset=["post_id"])

    # Save
    posts_file = ".venv/Reddit_ecommerce_disputes_posts.csv"
    df_posts.to_csv(posts_file, index=False, encoding='utf-8')
    print(f"\n✅ The post has been saved.：{posts_file}")
    print(f"   📊 Total：{len(df_posts)} ")

    # Grab comments
    if len(df_posts) == 0:
        print("⚠️ There are no valid posts. Skip the comment capture")
        pd.DataFrame().to_csv("Reddit_ecommerce_disputes_comments.csv", index=False)
        return

    print(f"\n🔄 Start collecting comments (expected {len(df_posts)} posts）...")
    comment_rows = []
    post_ids = df_posts["post_id"].tolist()

    for i, pid in enumerate(tqdm(post_ids, desc="Grab comments")):
        try:
            cs = fetch_comments_for_post(pid, start_ts, end_ts)
            for c in cs:
                cid = c.get("id")
                if cid:
                    comment_rows.append({
                        "post_id": pid,
                        "comment_id": cid,
                        "author": c.get("author"),
                        "body": c.get("body"),
                        "created_utc": c.get("created_utc"),
                        "score": c.get("score"),
                        "parent_id": c.get("parent_id"),
                        "link_id": c.get("link_id"),
                    })
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            time.sleep(2)

    df_comments = pd.DataFrame(comment_rows).dropna(subset=["comment_id"]) if comment_rows else pd.DataFrame()
    comments_file = "Reddit_ecommerce_disputes_comments.csv"
    df_comments.to_csv(comments_file, index=False, encoding='utf-8')
    print(f"\n✅ The comment has been saved：{comments_file}")
    print(f"   📊 Total：{len(df_comments)} ")

    # Summary statistics
    print("\n" + "=" * 70)
    print("📈 Data Summarization")
    print("=" * 70)
    print(f"🔹 Total number of posts：{len(df_posts)}")
    print(f"🔹 Total number of comments：{len(df_comments)}")
    if len(df_posts) > 0:
        print(f"🔹 The average number of comments per post：{len(df_comments) / len(df_posts):.2f}")

    if 'subreddit' in df_posts.columns and len(df_posts) > 0:
        print(f"\n📂 Subreddit distribution：")
        print(df_posts['subreddit'].value_counts().to_string())

    if len(df_posts) > 0:
        df_posts['full_text'] = (df_posts['title'].fillna('') + ' ' + df_posts['selftext'].fillna('')).str.lower()
        print(f"\n🔍 Keyword hit statistics：")
        for kw in DISPUTE_KEYWORDS:
            count = df_posts['full_text'].str.contains(kw, na=False).sum()
            if count > 0:
                print(f"   • {kw}: {count}")

    print(f"\n💾 Output：")
    print(f"   1. {posts_file}")
    print(f"   2. {comments_file}")
    print(f"\n✅ Completed")


if __name__ == "__main__":
    main()