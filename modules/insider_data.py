"""
modules/insider_data.py — 内幕交易（Form 4）数据源
============================================================
数据来源：SEC EDGAR 官方免费接口，不需要API Key。
  1) ticker -> CIK 映射：   https://www.sec.gov/files/company_tickers.json
  2) 该CIK最近的Form4列表： https://www.sec.gov/cgi-bin/browse-edgar?...&output=atom
  3) 单笔Form4完整内容：     https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}.txt

SEC要求所有请求必须带描述性 User-Agent（含联系方式），否则会被403拦截。
==> 请把下面 _SEC_USER_AGENT 改成你自己的项目名+邮箱，否则可能被SEC拒绝请求。

输出信号约定（与 realtime_radar.py 的 _score_risks(insider_signal=...) 对应）：
  "heavy_sell" : 近期(lookback_days内) C-suite/董事密集净卖出，无买入对冲 → 风险扣分
  "ceo_buy"    : CEO/COO/CFO 中有人净买入（买入抵消卖出或单纯买入） → 中性/略加分
  None         : 无数据 / 无显著内幕交易 / 抓取失败（不参与评分，安全默认）

设计上完全可以单独运行测试：
    python -m modules.insider_data NVDA
"""
import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# ⚠️ 改成你自己的：项目名 + 联系邮箱（SEC强制要求，否则会403）
_SEC_USER_AGENT = "ai-trading-brain research tool donnie55@gmail.com"

_CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", ".cache")
_CIK_CACHE_FILE     = os.path.join(_CACHE_DIR, "sec_cik_map.json")
_INSIDER_CACHE_FILE = os.path.join(_CACHE_DIR, "insider_signal_cache.json")

_CIK_TTL_SECONDS     = 7 * 24 * 3600     # ticker->CIK 映射，一周刷新一次就够
_SIGNAL_TTL_SECONDS  = 12 * 3600          # 内幕信号缓存12小时，跟其他模块的cache节奏一致

# 判定"C-suite/重要内部人"的职位关键词（不区分大小写）
_C_SUITE_KEYWORDS = [
    "chief executive", "ceo",
    "chief operating", "coo",
    "chief financial", "cfo",
    "president",
    "chief legal", "general counsel",
    "chairman",
]

LOOKBACK_DAYS_DEFAULT = 30   # 默认看最近30天的Form4


# ── 底层HTTP请求（带SEC要求的User-Agent + 简单重试）─────────────────────────
def _http_get(url: str, retries: int = 2, timeout: int = 10) -> Optional[str]:
    headers = {
        "User-Agent": _SEC_USER_AGENT,
        "Accept-Encoding": "identity",
    }
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(1.0 + attempt)
                continue
            return None
        except Exception:
            if attempt < retries:
                time.sleep(0.5)
                continue
            return None
    return None


# ── 磁盘缓存通用工具 ────────────────────────────────────────────────────────
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def _is_fresh(entry: dict, ttl_seconds: int) -> bool:
    ts = entry.get("_ts")
    if ts is None:
        return False
    return (time.time() - ts) < ttl_seconds


# ── 1. ticker -> CIK 映射 ───────────────────────────────────────────────────
def _get_cik_map() -> Dict[str, str]:
    """返回 {TICKER: '0000123456'} 全市场映射，本地缓存7天。"""
    cache = _load_json(_CIK_CACHE_FILE)
    if cache and _is_fresh(cache, _CIK_TTL_SECONDS):
        return cache.get("map", {})

    raw = _http_get("https://www.sec.gov/files/company_tickers.json")
    if raw is None:
        # 抓取失败时，如果有旧缓存（即使过期）也先用旧的，好过没有
        return cache.get("map", {}) if cache else {}

    try:
        data = json.loads(raw)
    except Exception:
        return cache.get("map", {}) if cache else {}

    mapping: Dict[str, str] = {}
    for _, item in data.items():
        try:
            ticker = str(item["ticker"]).upper()
            cik = str(item["cik_str"]).zfill(10)
            mapping[ticker] = cik
        except Exception:
            continue

    _save_json(_CIK_CACHE_FILE, {"_ts": time.time(), "map": mapping})
    return mapping


def _cik_for(ticker: str) -> Optional[str]:
    m = _get_cik_map()
    return m.get(ticker.upper())


# ── 2. 该CIK最近的Form4文件列表（Atom feed） ─────────────────────────────────
_ACCESSION_RE = re.compile(r"/Archives/edgar/data/\d+/([\w-]+)-index\.htm")


def _recent_form4_accessions(cik: str, count: int = 12) -> List[str]:
    """返回最近 count 笔 Form4 的 accession number（带横线格式，如 0001234567-26-001234）。"""
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={cik}&type=4&dateb=&owner=include&count={count}&output=atom"
    )
    raw = _http_get(url)
    if raw is None:
        return []
    return _ACCESSION_RE.findall(raw)


# ── 3. 单笔Form4完整内容 → 解析交易明细 ───────────────────────────────────────
_OFFICER_TITLE_RE   = re.compile(r"<officerTitle>(.*?)</officerTitle>", re.S | re.I)
_IS_OFFICER_RE       = re.compile(r"<isOfficer>1</isOfficer>", re.I)
_IS_DIRECTOR_RE      = re.compile(r"<isDirector>1</isDirector>", re.I)
_OWNER_NAME_RE       = re.compile(r"<rptOwnerName>(.*?)</rptOwnerName>", re.S | re.I)
_FILING_DATE_RE      = re.compile(r"<periodOfReport>(.*?)</periodOfReport>", re.S | re.I)

# 每个 nonDerivativeTransaction 块单独解析
_TXN_BLOCK_RE   = re.compile(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", re.S | re.I)
_TXN_CODE_RE    = re.compile(r"<transactionCode>(.*?)</transactionCode>", re.S | re.I)
_TXN_AD_RE      = re.compile(r"<transactionAcquiredDisposedCode>\s*<value>(.*?)</value>", re.S | re.I)
_TXN_SHARES_RE  = re.compile(r"<transactionShares>\s*<value>(.*?)</value>", re.S | re.I)
_TXN_PRICE_RE   = re.compile(r"<transactionPricePerShare>\s*<value>(.*?)</value>", re.S | re.I)


def _fetch_form4_text(cik: str, accession_dashes: str) -> Optional[str]:
    accession_nodashes = accession_dashes.replace("-", "")
    cik_int = str(int(cik))  # SEC在Archives路径里CIK不带前导0
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}.txt"
    return _http_get(url)


def _parse_form4(text: str) -> Optional[dict]:
    """从单笔Form4完整文本里解析出关键字段，失败返回None。"""
    if not text:
        return None

    title_match = _OFFICER_TITLE_RE.search(text)
    officer_title = title_match.group(1).strip() if title_match else ""
    is_officer = bool(_IS_OFFICER_RE.search(text))
    is_director = bool(_IS_DIRECTOR_RE.search(text))

    owner_match = _OWNER_NAME_RE.search(text)
    owner_name = owner_match.group(1).strip() if owner_match else "未知"

    date_match = _FILING_DATE_RE.search(text)
    period = date_match.group(1).strip() if date_match else None

    is_c_suite = is_officer and any(kw in officer_title.lower() for kw in _C_SUITE_KEYWORDS)

    net_acquired_value = 0.0
    net_disposed_value = 0.0
    has_open_market_txn = False

    for block in _TXN_BLOCK_RE.findall(text):
        code_m   = _TXN_CODE_RE.search(block)
        ad_m     = _TXN_AD_RE.search(block)
        shares_m = _TXN_SHARES_RE.search(block)
        price_m  = _TXN_PRICE_RE.search(block)

        code = (code_m.group(1).strip() if code_m else "").upper()
        # 只看公开市场买卖：P=买入 S=卖出。其他(A/G/F/M等行权/赠与/税务扣缴噪音大，跳过)
        if code not in ("P", "S"):
            continue
        has_open_market_txn = True

        ad_code = (ad_m.group(1).strip() if ad_m else "").upper()  # A=增持 D=减持
        try:
            shares = float((shares_m.group(1).strip() if shares_m else "0") or 0)
        except ValueError:
            shares = 0.0
        try:
            price = float((price_m.group(1).strip() if price_m else "0") or 0)
        except ValueError:
            price = 0.0
        value = shares * price

        if ad_code == "A" or code == "P":
            net_acquired_value += value
        elif ad_code == "D" or code == "S":
            net_disposed_value += value

    if not has_open_market_txn:
        return None  # 这笔Form4没有公开市场买卖（可能纯是行权/赠与），不参与统计

    return {
        "owner_name":   owner_name,
        "officer_title": officer_title,
        "is_c_suite":   is_c_suite,
        "is_director":  is_director,
        "period":       period,
        "acquired_value": net_acquired_value,
        "disposed_value": net_disposed_value,
        "net_value":    net_acquired_value - net_disposed_value,
    }


# ── 4. 主入口：单只ticker的内幕信号 ──────────────────────────────────────────
def get_insider_signal(
    ticker: str,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    force_refresh: bool = False,
) -> Tuple[Optional[str], dict]:
    """
    返回 (signal, detail)
    signal: "heavy_sell" | "ceo_buy" | None
    detail: 调试用的明细字典，包含每个内部人的净买卖金额，方便UI展示原因
    """
    cache = _load_json(_INSIDER_CACHE_FILE)
    entry = cache.get(ticker)
    if entry and not force_refresh and _is_fresh(entry, _SIGNAL_TTL_SECONDS):
        return entry.get("signal"), entry.get("detail", {})

    detail = {"transactions": [], "error": None}
    signal = None

    cik = _cik_for(ticker)
    if cik is None:
        detail["error"] = "未找到CIK（可能是ETF或非美股注册主体）"
        _write_cache(cache, ticker, signal, detail)
        return signal, detail

    accessions = _recent_form4_accessions(cik, count=12)
    if not accessions:
        detail["error"] = "近期无Form4文件或SEC接口请求失败"
        _write_cache(cache, ticker, signal, detail)
        return signal, detail

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    c_suite_net = 0.0      # C-suite 净买卖金额（正=净买入，负=净卖出）
    c_suite_sellers: List[str] = []
    c_suite_buyers: List[str] = []
    any_offsetting_buy = False

    for acc in accessions:
        text = _fetch_form4_text(cik, acc)
        parsed = _parse_form4(text) if text else None
        if parsed is None:
            continue

        # 日期过滤（period格式通常是 YYYY-MM-DD）
        if parsed["period"]:
            try:
                pdate = datetime.strptime(parsed["period"][:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if pdate < cutoff:
                    continue
            except Exception:
                pass

        detail["transactions"].append({
            "owner": parsed["owner_name"],
            "title": parsed["officer_title"],
            "net_value": round(parsed["net_value"], 0),
            "period": parsed["period"],
        })

        if not parsed["is_c_suite"]:
            continue

        c_suite_net += parsed["net_value"]
        if parsed["net_value"] > 0:
            c_suite_buyers.append(parsed["owner_name"])
            any_offsetting_buy = True
        elif parsed["net_value"] < 0:
            c_suite_sellers.append(parsed["owner_name"])

        time.sleep(0.15)  # 礼貌性限速，避免触发SEC的请求频率限制

    detail["c_suite_net_value"] = round(c_suite_net, 0)
    detail["c_suite_sellers"] = list(set(c_suite_sellers))
    detail["c_suite_buyers"]  = list(set(c_suite_buyers))

    # ── 判定逻辑 ──────────────────────────────────────────────────────────
    # heavy_sell：至少2位C-suite净卖出，且没有任何C-suite买入对冲
    if len(set(c_suite_sellers)) >= 2 and not any_offsetting_buy:
        signal = "heavy_sell"
    # ceo_buy：只要有C-suite净买入（哪怕同时有人卖，只要买入方存在即视为正面信号占位）
    elif c_suite_buyers:
        signal = "ceo_buy"
    else:
        signal = None

    _write_cache(cache, ticker, signal, detail)
    return signal, detail


def _write_cache(cache: dict, ticker: str, signal: Optional[str], detail: dict) -> None:
    cache[ticker] = {"_ts": time.time(), "signal": signal, "detail": detail}
    _save_json(_INSIDER_CACHE_FILE, cache)


# ── 5. 批量刷新（配合dashboard侧边栏"刷新内幕数据"按钮用） ───────────────────
def refresh_insider_data(tickers: List[str], force_refresh: bool = True) -> Dict[str, Optional[str]]:
    """批量跑一遍，返回 {ticker: signal}，用于侧边栏按钮触发的批量刷新。"""
    results: Dict[str, Optional[str]] = {}
    for t in tickers:
        try:
            sig, _ = get_insider_signal(t, force_refresh=force_refresh)
            results[t] = sig
        except Exception:
            results[t] = None
        time.sleep(0.2)
    return results


if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    sig, det = get_insider_signal(tk, force_refresh=True)
    print(f"{tk} -> signal={sig}")
    print(json.dumps(det, indent=2, ensure_ascii=False))
