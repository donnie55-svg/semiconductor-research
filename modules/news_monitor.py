import os
import feedparser
import pandas as pd
from datetime import datetime

RSS_FEEDS = [
    {"name": "Reuters Tech", "url": "https://feeds.reuters.com/reuters/technologyNews"},
    {"name": "EE Times", "url": "https://www.eetimes.com/feed/"},
    {"name": "Tom's Hardware", "url": "https://www.tomshardware.com/feeds/all"},
    {"name": "AnandTech", "url": "https://www.anandtech.com/rss/"},
    {"name": "The Register HW", "url": "https://www.theregister.com/hardware/graphics/rss"},
    {"name": "Seeking Alpha Semi", "url": "https://seekingalpha.com/tag/semiconductors.xml"},
]

KEYWORDS = {
    "high": [
        "HBM", "CoWoS", "CapEx", "AI chip", "NVIDIA", "NVDA", "TSM", "TSMC",
        "Micron", "guidance", "earnings beat", "revenue miss", "data center",
        "AI server", "Blackwell", "HBM3E", "HBM4", "export control", "chip ban",
        "Broadcom", "AVGO", "MI300", "GB200", "NVL72",
    ],
    "medium": [
        "semiconductor", "chip", "wafer", "foundry", "memory", "DRAM", "NAND",
        "GPU", "CPU", "AI", "machine learning", "hyperscaler", "cloud",
        "Microsoft", "Google", "Meta", "Amazon", "AMD",
    ],
    "positive": [
        "beat", "surge", "record", "growth", "strong demand", "upgrade",
        "outperform", "raise guidance", "expand", "breakthrough", "win contract",
    ],
    "negative": [
        "ban", "restrict", "sanction", "shortage", "miss", "cut guidance",
        "reduce", "decline", "inventory correction", "oversupply", "concern",
    ],
}


def rate_impact(title: str, summary: str = "") -> tuple:
    text = (title + " " + summary).lower()
    is_high = any(kw.lower() in text for kw in KEYWORDS["high"])
    is_medium = any(kw.lower() in text for kw in KEYWORDS["medium"])
    pos = sum(1 for kw in KEYWORDS["positive"] if kw.lower() in text)
    neg = sum(1 for kw in KEYWORDS["negative"] if kw.lower() in text)
    sentiment_score = pos - neg
    sentiment_icon = "📈" if sentiment_score > 0 else ("📉" if sentiment_score < 0 else "➡️")
    if is_high:
        return "🔴 High", sentiment_icon
    elif is_medium:
        return "🟡 Medium", sentiment_icon
    return "⚪ Low", sentiment_icon


def _parse_date(entry) -> str:
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return entry.get("published", "")[:16]


def fetch_news_feed(max_items_per_feed: int = 10) -> pd.DataFrame:
    all_keywords = KEYWORDS["high"] + KEYWORDS["medium"]
    articles = []

    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:max_items_per_feed]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = (title + " " + summary).lower()

                if not any(kw.lower() in text for kw in all_keywords):
                    continue

                impact, sentiment = rate_impact(title, summary)
                articles.append({
                    "Date": _parse_date(entry),
                    "Source": feed_info["name"],
                    "Impact": impact,
                    "Sentiment": sentiment,
                    "Title": title,
                    "Link": entry.get("link", ""),
                    "Summary": summary[:250] + "..." if len(summary) > 250 else summary,
                })
        except Exception as e:
            print(f"RSS error [{feed_info['name']}]: {e}")

    if not articles:
        return pd.DataFrame(columns=["Date", "Source", "Impact", "Sentiment", "Title", "Link", "Summary"])

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset="Title")
    # Sort: High first, then by date desc
    impact_order = {"🔴 High": 0, "🟡 Medium": 1, "⚪ Low": 2}
    df["_sort"] = df["Impact"].map(impact_order).fillna(3)
    df = df.sort_values(["_sort", "Date"], ascending=[True, False]).drop(columns="_sort")
    return df.reset_index(drop=True)
