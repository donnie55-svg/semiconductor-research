"""财报雷达 — 扫描未来 N 天即将发布财报的重点成分股，AI 评分筛选 Top 20"""
import os

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.futu_data import get_kline, get_info_partial, is_futu_connected

# ── 扫描范围：标普500 + 纳斯达克100 重点成分股 ───────────────────────────────────
SCAN_TICKERS = [
    # 科技 / 半导体
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD", "INTC",
    "QCOM", "TXN", "MU",   "AMAT",  "LRCX", "KLAC", "ASML", "ADI",  "MRVL", "ORCL",
    # 软件 / 云
    "CRM",  "ADBE", "NOW",  "INTU",  "SNOW", "PLTR", "PANW", "CRWD", "ZS",   "FTNT",
    "NET",  "DDOG", "MDB",  "WDAY",  "TEAM", "HUBS", "SHOP", "SQ",   "PYPL",
    # 消费科技 / 出行
    "UBER", "ABNB", "NFLX", "DIS",   "CMCSA",
    # 金融
    "JPM",  "BAC",  "WFC",  "GS",   "MS",   "C",    "BLK",  "SCHW", "V",    "MA",
    "AXP",  "COF",  "SPGI", "MCO",  "ICE",  "CME",
    # 医疗 / 生物
    "UNH",  "JNJ",  "LLY",  "ABBV", "MRK",  "PFE",  "BMY",  "AMGN", "GILD", "REGN",
    "VRTX", "MRNA", "ISRG", "MDT",  "ABT",  "TMO",  "DHR",
    # 消费 / 零售
    "COST", "WMT",  "TGT",  "HD",   "LOW",  "NKE",  "SBUX", "MCD",  "CMG",  "LULU",
    "PG",   "KO",   "PEP",  "MNST",
    # 能源 / 工业
    "XOM",  "CVX",  "COP",  "CAT",  "DE",   "HON",  "GE",   "RTX",  "LMT",  "BA",
    "UPS",  "FDX",  "CSX",  "UNP",
    # 通信
    "T",    "VZ",   "TMUS",
    # 房地产
    "AMT",  "PLD",  "EQIX",
]
SCAN_TICKERS = list(dict.fromkeys(SCAN_TICKERS))  # 去重

# 各行业 Forward PE 参考中位值
_SECTOR_FWD_PE = {
    "Technology":             28,
    "Communication Services": 22,
    "Consumer Cyclical":      24,
    "Consumer Defensive":     22,
    "Healthcare":             20,
    "Financial Services":     14,
    "Financial":              14,
    "Industrials":            20,
    "Energy":                 12,
    "Utilities":              18,
    "Real Estate":            35,
    "Basic Materials":        14,
}
_DEFAULT_PE = 20


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _upcoming_earn_date(info: dict, today, window_days: int):
    cutoff = today + timedelta(days=window_days)
    for key in ("earningsTimestampStart", "earningsTimestamp", "earningsTimestampEnd"):
        ts = info.get(key)
        if not ts:
            continue
        try:
            dt = datetime.fromtimestamp(int(ts)).date()
            if today <= dt <= cutoff:
                return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _consecutive_beats(qe) -> int:
    if qe is None or qe.empty:
        return 0
    if "Actual" not in qe.columns or "Estimate" not in qe.columns:
        return 0
    count = 0
    for _, row in qe.head(4).iterrows():
        a, e = row["Actual"], row["Estimate"]
        if pd.notna(a) and pd.notna(e) and abs(float(e)) > 0.001 and float(a) > float(e):
            count += 1
        else:
            break
    return count


# ── 核心评分函数 ───────────────────────────────────────────────────────────────

def _score_one(ticker: str, window_days: int) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # 富途覆盖实时价格 / EPS / 营收增速
        if is_futu_connected():
            partial = get_info_partial(ticker)
            for k, v in partial.items():
                if v is not None:
                    info[k] = v

        if info.get("quoteType", "EQUITY") != "EQUITY":
            return {}

        today = datetime.now().date()
        earn_date = _upcoming_earn_date(info, today, window_days)
        if not earn_date:
            return {}

        score = 0
        reasons = []

        # ① EPS 预期增速 > 20%  → +20
        fwd_eps   = float(info.get("forwardEps")  or 0)
        trail_eps = float(info.get("trailingEps") or 0)
        eps_growth = None
        if abs(trail_eps) > 0.01:
            eps_growth = (fwd_eps - trail_eps) / abs(trail_eps) * 100
            if eps_growth > 20:
                score += 20
                reasons.append(f"EPS预期+{eps_growth:.0f}%")

        # ② 营收增速 > 15%  → +20
        rg_raw = info.get("revenueGrowth")
        rev_growth = round(float(rg_raw) * 100, 1) if rg_raw is not None else None
        if rev_growth is not None and rev_growth > 15:
            score += 20
            reasons.append(f"营收+{rev_growth:.0f}%")

        # ③ 连续 EPS beat（季度历史财报富途不提供，始终用 yfinance）
        beats = 0
        try:
            qe = t.quarterly_earnings
            beats = _consecutive_beats(qe)
        except Exception:
            pass
        if beats >= 4:
            score += 20
            reasons.append("连续4季beat")
        elif beats >= 3:
            score += 12
            reasons.append("连续3季beat")

        # ④ Forward PE < 行业均值  → +15
        fwd_pe = info.get("forwardPE")
        sector = info.get("sector", "")
        sec_pe = _SECTOR_FWD_PE.get(sector, _DEFAULT_PE)
        if fwd_pe and 0 < float(fwd_pe) < sec_pe:
            score += 15
            reasons.append(f"FwdPE{float(fwd_pe):.1f}<行业{sec_pe}")

        # ⑤ 分析师评级
        rec_mean = info.get("recommendationMean")
        if rec_mean is not None:
            rm = float(rec_mean)
            if rm <= 2.0:
                score += 15
                reasons.append("分析师强烈看多")
            elif rm <= 2.5:
                score += 8
                reasons.append("分析师偏多")

        # ⑥ 机构持仓 > 60%  → +10
        inst = info.get("heldPercentInstitutions")
        if inst and float(inst) > 0.60:
            score += 10
            reasons.append(f"机构持仓{float(inst)*100:.0f}%")

        cur_price = info.get("currentPrice") or info.get("regularMarketPrice")
        tgt_price = info.get("targetMeanPrice")
        upside = None
        if cur_price and tgt_price and float(cur_price) > 0:
            upside = round((float(tgt_price) - float(cur_price)) / float(cur_price) * 100, 1)

        # 近30天涨跌幅：优先富途 K 线，回退 yfinance
        pullback_30d = None
        try:
            if is_futu_connected():
                h = get_kline(ticker, "1mo")
                if h.empty:
                    h = t.history(period="1mo")
            else:
                h = t.history(period="1mo")

            if not h.empty and cur_price:
                p0 = float(h["Close"].iloc[0])
                pullback_30d = round((float(cur_price) - p0) / p0 * 100, 1) if p0 else None
        except Exception:
            pass

        flags = []
        if score >= 70 and pullback_30d is not None and pullback_30d < -5:
            flags.append("🔥")
        if beats >= 3:
            flags.append("⚡")

        reason_str = " · ".join(reasons[:2]) if reasons else "基本面稳健"

        return {
            "标记":       " ".join(flags),
            "Ticker":     ticker,
            "公司名":     info.get("shortName", ticker),
            "财报日期":   earn_date,
            "评分":       score,
            "预期EPS":    round(fwd_eps, 2) if fwd_eps else None,
            "上季EPS":    round(trail_eps, 2) if trail_eps else None,
            "EPS增速%":   round(eps_growth, 1) if eps_growth is not None else None,
            "营收增速%":  rev_growth,
            "Forward PE": round(float(fwd_pe), 1) if fwd_pe else None,
            "行业PE均值": sec_pe,
            "当前价":     round(float(cur_price), 2) if cur_price else None,
            "目标价":     round(float(tgt_price), 2) if tgt_price else None,
            "上行空间%":  upside,
            "30天涨跌%":  pullback_30d,
            "连续beat季": beats,
            "推荐理由":   reason_str,
        }
    except Exception:
        return {}


# ── 公开接口 ───────────────────────────────────────────────────────────────────

def get_earnings_radar(days: int = 14, max_workers: int = 20) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, tk, days): tk for tk in SCAN_TICKERS}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("评分", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    df.index = range(1, len(df) + 1)
    return df


def get_top_picks(df: pd.DataFrame, n: int = 3) -> list:
    picks = []
    for _, row in df.head(n).iterrows():
        flags   = row.get("标记", "")
        details = []

        if "🔥" in flags:
            pull = row.get("30天涨跌%")
            details.append(
                f"高评分({row['评分']}分) + 近30天回调{abs(pull):.1f}%，提供买入窗口"
                if pull is not None else f"高评分({row['评分']}分)，近期出现回调"
            )
        if "⚡" in flags:
            details.append(f"连续 {row.get('连续beat季', 0)} 季度超预期，business momentum强")

        upside = row.get("上行空间%")
        if upside and upside > 15:
            details.append(f"分析师目标价较现价上行空间 {upside:.0f}%")

        rev = row.get("营收增速%")
        if rev and rev > 15:
            details.append(f"预期营收增速 {rev:.0f}%，成长性突出")

        eps_g = row.get("EPS增速%")
        if eps_g and eps_g > 20:
            details.append(f"预期 EPS 增速 {eps_g:.0f}%")

        if not details:
            details.append(row.get("推荐理由", "基本面稳健，综合评分较高"))

        picks.append({
            "ticker":   row["Ticker"],
            "company":  row["公司名"],
            "score":    row["评分"],
            "date":     row["财报日期"],
            "upside":   upside,
            "pullback": row.get("30天涨跌%"),
            "flags":    flags,
            "details":  details,
        })
    return picks
