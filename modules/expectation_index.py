"""
expectation_index.py — 分析师预期综合指数
计算每只股票 0-100 分的预期指数，评估市场预期变化方向：
  EPS 预期修正(20) + 目标价变化(20) + 评级变化(20) + Guidance 新闻(20) + CapEx 新闻(20)
基础分 50，正向信号最高 +50 → 100，负向最大约 -45 → 5。
结果缓存到 data/expectation_cache.json（TTL 6 小时）。
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import feedparser
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.normpath(os.path.join(_THIS_DIR, '..', 'data', 'expectation_cache.json'))
CACHE_TTL  = 6 * 3600  # 6 hours in seconds

# ── 股票专属 CapEx/需求关键词 ──────────────────────────────────────────────────
CAPEX_KEYWORDS: Dict[str, List[str]] = {
    'MU':   ['HBM', 'DRAM price', 'memory demand', 'HBM3E', 'HBM4', 'NAND price'],
    'NVDA': ['Blackwell', 'GB200', 'data center', 'NVL72', 'H100', 'H200', 'B200'],
    'TSM':  ['CoWoS', 'advanced packaging', 'N3', 'N2', 'capacity expansion'],
    'AVGO': ['XPU', 'TPU', 'hyperscaler', 'custom chip', 'AI ASIC'],
}
_CAPEX_DEFAULT = ['AI demand', 'capex increase', 'capital expenditure', 'AI investment']

_GUID_POSITIVE = [
    'beat', 'raised guidance', 'strong demand', 'raise guidance',
    'exceeded expectations', 'outperform', 'record revenue', 'robust demand',
    'positive outlook', 'raised outlook', 'better than expected',
]
_GUID_NEGATIVE = [
    'missed', 'lowered', 'weak demand', 'below expectations',
    'cut guidance', 'lowered guidance', 'softer demand', 'cautious outlook',
    'disappointing', 'reduced forecast', 'below estimates',
]


# ── 磁盘缓存 ──────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Cache save failed: {e}")


def _is_fresh(entry: dict) -> bool:
    try:
        dt = datetime.fromisoformat(entry.get('timestamp', ''))
        return (datetime.now() - dt).total_seconds() < CACHE_TTL
    except Exception:
        return False


# ── 各维度评分 ────────────────────────────────────────────────────────────────

def _score_eps_revision(tk: yf.Ticker) -> Tuple[int, str]:
    """EPS 预期修正（30 天）：上调 +20，下调 -10。"""
    try:
        rev = tk.eps_revisions
        if rev is None or (hasattr(rev, 'empty') and rev.empty):
            return 0, ''
        up30 = down30 = 0
        for idx_val in rev.index:
            idx_s = str(idx_val).lower()
            try:
                row = rev.loc[idx_val]
                val = row.iloc[0] if hasattr(row, 'iloc') else row
                val = int(float(val)) if (val is not None and str(val) not in ('nan', 'None', '')) else 0
            except (TypeError, ValueError):
                val = 0
            if 'up' in idx_s and '30' in idx_s:
                up30 = val
            elif 'down' in idx_s and '30' in idx_s:
                down30 = val
        if up30 > down30 and up30 > 0:
            return 20, f'EPS预期上调{up30}次(30天)'
        if down30 > up30 and down30 > 0:
            return -10, f'EPS预期下调{down30}次(30天)'
    except Exception as e:
        logger.debug(f"EPS revision {e}")
    return 0, ''


def _score_target_price(tk: yf.Ticker) -> Tuple[int, str]:
    """分析师平均目标价 vs 现价：>5% 上行 +20，>5% 下行 -10。"""
    try:
        price = None
        try:
            fi = tk.fast_info
            price = getattr(fi, 'last_price', None) or getattr(fi, 'lastPrice', None)
        except Exception:
            pass
        if not price:
            info_d = tk.info or {}
            price = info_d.get('currentPrice') or info_d.get('regularMarketPrice')

        mean_target = None
        try:
            pt = tk.analyst_price_targets
            if pt is not None:
                if isinstance(pt, dict):
                    mean_target = pt.get('mean') or pt.get('targetMeanPrice')
                elif isinstance(pt, pd.Series):
                    mean_target = pt.get('mean')
        except Exception:
            pass
        if mean_target is None:
            info_d = tk.info or {}
            mean_target = (info_d.get('targetMeanPrice')
                           or info_d.get('targetMedianPrice'))

        if mean_target and price and float(price) > 0:
            upside = (float(mean_target) - float(price)) / float(price)
            if upside > 0.05:
                return 20, f'目标价上行空间{upside*100:.0f}%'
            if upside < -0.05:
                return -10, f'目标价低于现价{abs(upside)*100:.0f}%'
    except Exception as e:
        logger.debug(f"Target price: {e}")
    return 0, ''


def _score_rating_change(tk: yf.Ticker) -> Tuple[int, str]:
    """评级变化（30 天）：升级 +20，降级 -15。"""
    try:
        ud = tk.upgrades_downgrades
        if ud is None or (hasattr(ud, 'empty') and ud.empty):
            return 0, ''
        # 去时区
        try:
            if hasattr(ud.index, 'tz') and ud.index.tz is not None:
                ud = ud.copy()
                ud.index = ud.index.tz_localize(None)
        except Exception:
            pass
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=30))
        try:
            recent = ud[ud.index >= cutoff]
        except Exception:
            recent = ud.tail(10)
        if recent.empty:
            return 0, ''
        action_col = next((c for c in ['Action', 'action'] if c in recent.columns), None)
        if action_col is None:
            return 0, ''
        acts = recent[action_col].astype(str).str.lower()
        upgrades   = acts.str.contains(r'\bup\b|upgrade', na=False, regex=True).sum()
        downgrades = acts.str.contains(r'\bdown\b|downgrade', na=False, regex=True).sum()
        if upgrades > downgrades and upgrades > 0:
            return 20, f'30天评级上调{upgrades}次'
        if downgrades > upgrades and downgrades > 0:
            return -15, f'30天评级下调{downgrades}次'
    except Exception as e:
        logger.debug(f"Rating change: {e}")
    return 0, ''


def _yahoo_entries(ticker: str) -> list:
    """抓取 Yahoo Finance 新闻 RSS（最近条目）。"""
    try:
        url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
               f"?s={ticker}&region=US&lang=en-US")
        feed = feedparser.parse(url)
        return list(feed.entries or [])
    except Exception:
        return []


def _filter_recent(entries: list, days: int = 30) -> list:
    cutoff = datetime.now() - timedelta(days=days)
    result = []
    for e in entries:
        try:
            if hasattr(e, 'published_parsed') and e.published_parsed:
                pub = datetime(*e.published_parsed[:6])
                if pub < cutoff:
                    continue
        except Exception:
            pass
        result.append(e)
    return result


def _score_guidance_news(ticker: str, entries: list) -> Tuple[int, str]:
    """Guidance/需求关键词扫描：正面 +20，负面 -10。"""
    recent = _filter_recent(entries)
    pos = neg = 0
    pos_hit = neg_hit = ''
    for e in recent[:30]:
        text = (getattr(e, 'title', '') + ' ' + getattr(e, 'summary', '')).lower()
        for kw in _GUID_POSITIVE:
            if kw.lower() in text:
                pos += 1
                if not pos_hit:
                    pos_hit = kw
                break
        for kw in _GUID_NEGATIVE:
            if kw.lower() in text:
                neg += 1
                if not neg_hit:
                    neg_hit = kw
                break
    if pos > 0 and pos >= neg:
        return 20, f'正面指引:{pos_hit or "利好"}'
    if neg > 0 and neg > pos:
        return -10, f'负面指引:{neg_hit or "利空"}'
    return 0, ''


def _score_capex_news(ticker: str, entries: list) -> Tuple[int, str]:
    """股票专属 CapEx/需求关键词：命中 2+ 次 +20，1 次 +10。"""
    kws = CAPEX_KEYWORDS.get(ticker.upper(), _CAPEX_DEFAULT)
    recent = _filter_recent(entries)
    hits = 0
    hit_kw = ''
    for e in recent[:30]:
        text = (getattr(e, 'title', '') + ' ' + getattr(e, 'summary', '')).lower()
        for kw in kws:
            if kw.lower() in text:
                hits += 1
                if not hit_kw:
                    hit_kw = kw
                break
    if hits >= 2:
        return 20, f'CapEx信号:{hit_kw or "AI需求"}(×{hits})'
    if hits == 1:
        return 10, f'CapEx信号:{hit_kw or "AI需求"}'
    return 0, ''


# ── 对外 API ──────────────────────────────────────────────────────────────────

def compute_expectation_index(ticker: str, force_refresh: bool = False) -> dict:
    """
    计算单只股票预期指数 (0-100)。
    基础分 50，5 个维度各 ±20 分范围。
    Returns: {ticker, score, components, reasons, timestamp}
    """
    cache = _load_cache()
    if not force_refresh and ticker in cache and _is_fresh(cache[ticker]):
        return cache[ticker]

    result: dict = {
        'ticker':     ticker,
        'score':      50,
        'components': {},
        'reasons':    [],
        'timestamp':  datetime.now().isoformat(),
    }
    try:
        tk = yf.Ticker(ticker)
        eps_s,   eps_r   = _score_eps_revision(tk)
        tp_s,    tp_r    = _score_target_price(tk)
        rat_s,   rat_r   = _score_rating_change(tk)
        entries          = _yahoo_entries(ticker)
        guid_s,  guid_r  = _score_guidance_news(ticker, entries)
        capex_s, capex_r = _score_capex_news(ticker, entries)

        raw = 50 + eps_s + tp_s + rat_s + guid_s + capex_s
        result.update({
            'score':      int(max(0, min(100, raw))),
            'components': {
                'eps_revision':  eps_s,
                'target_price':  tp_s,
                'rating_change': rat_s,
                'guidance_news': guid_s,
                'capex_news':    capex_s,
            },
            'reasons': [r for r in [eps_r, tp_r, rat_r, guid_r, capex_r] if r][:3],
        })
    except Exception as e:
        logger.warning(f"compute_expectation_index {ticker}: {e}")

    cache[ticker] = result
    _save_cache(cache)
    return result


def get_expectation_scores(
    tickers: List[str],
    force_refresh: bool = False,
) -> Dict[str, dict]:
    """
    批量获取预期指数，优先读磁盘缓存（TTL 6 小时）。
    Returns {ticker: result_dict}
    """
    cache = _load_cache()
    results: Dict[str, dict] = {}
    for ticker in tickers:
        try:
            if not force_refresh and ticker in cache and _is_fresh(cache[ticker]):
                results[ticker] = cache[ticker]
            else:
                results[ticker] = compute_expectation_index(ticker, force_refresh)
        except Exception as e:
            logger.warning(f"get_expectation_scores {ticker}: {e}")
            results[ticker] = {
                'ticker': ticker, 'score': 50, 'components': {},
                'reasons': [], 'timestamp': datetime.now().isoformat(),
            }
    return results
