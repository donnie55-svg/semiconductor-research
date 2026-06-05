"""
high_conviction.py — 严选模式 / High-Conviction Mode
Historical A+ signal validation and classification.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hc_validation_cache.json")
CACHE_TTL_HOURS = 24

MIN_STOCK_SAMPLES  = 30
MIN_SECTOR_SAMPLES = 80
MIN_MARKET_SAMPLES = 150

MIN_WIN_RATE    = 0.65
MIN_RR_RATIO    = 1.5
MIN_EV          = 0.0
MAX_CONSEC_LOSS = 3

# A+ thresholds (automated conditions)
APLUS_RSI5_MAX      = 28
APLUS_DROP3_MIN     = 0.06
APLUS_VOL_RATIO_MIN = 1.5
APLUS_EXP_SCORE_MIN = 85
APLUS_VIX_MAX       = 25
APLUS_CLOSE_REBOUND = 1.03    # close >= low * 1.03
APLUS_CLOSE_CAP     = 1.015   # close <= bb_lower * 1.015


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 5) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0
               ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std


# ── A+ Bollinger condition ─────────────────────────────────────────────────────

def check_aplus_bollinger(close_val: float, low_val: float, lower_val: float) -> bool:
    """
    A+ Bollinger gate:
      1. Low < BB_Lower  (盘中跌破下轨)
      2. Close >= Low * 1.03  (收盘反弹≥3%)
      3. Close <= BB_Lower * 1.015  (没有完全涨飞)
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in [close_val, low_val, lower_val]):
        return False
    return (
        low_val < lower_val
        and close_val >= low_val * APLUS_CLOSE_REBOUND
        and close_val <= lower_val * APLUS_CLOSE_CAP
    )


# ── A+ full evaluation (automated conditions only) ────────────────────────────

def evaluate_aplus_conditions(
    signal: dict,
    exp_score: Optional[float],
    vix_val: Optional[float],
    spy_close: Optional[pd.Series],
    qqq_close: Optional[pd.Series],
) -> Tuple[bool, List[str], List[str]]:
    """
    Check all automated A+ conditions.
    Returns (all_pass, passed_list, failed_list).
    Manual conditions (news, earnings, macro events) are flagged as '需人工确认'.
    """
    passed, failed = [], []

    def chk(name: str, ok: bool):
        (passed if ok else failed).append(name)

    # 1. expectation_score > 85
    if exp_score is not None:
        chk("预期指数>85", exp_score > APLUS_EXP_SCORE_MIN)
    else:
        failed.append("预期指数未计算")

    # 2. RSI(5) < 28
    rsi5 = signal.get("rsi5")
    chk("RSI(5)<28", rsi5 is not None and rsi5 < APLUS_RSI5_MAX)

    # 3-5. A+ Bollinger (low < lower, close rebound, close cap)
    bb_ok = check_aplus_bollinger(
        signal.get("price", float("nan")),
        signal.get("latest_low", float("nan")),
        signal.get("bb_lower", float("nan")),
    )
    chk("布林带A+形态", bb_ok)

    # 6. 3日跌幅 > 6%
    drop3 = signal.get("drop3_pct")
    chk("3日跌幅>6%", drop3 is not None and drop3 > APLUS_DROP3_MIN * 100)

    # 7. Volume > 20d avg * 1.5
    vol_r = signal.get("vol_ratio")
    chk("放量>1.5x", vol_r is not None and vol_r > APLUS_VOL_RATIO_MIN)

    # 8. Not simultaneously below 20/50/200 MA
    ma20  = signal.get("ma20")
    ma50  = signal.get("ma50")
    ma200 = signal.get("ma200")
    price = signal.get("price", float("nan"))
    if ma20 and ma50:
        below_all = (
            price < ma20 and price < ma50 and
            (ma200 is None or price < ma200)
        )
        chk("未同跌破三均线", not below_all)
    else:
        failed.append("均线数据不足")

    # 9. SPY > 200MA
    if spy_close is not None and len(spy_close) >= 200:
        spy_ma200 = float(spy_close.rolling(200).mean().iloc[-1])
        chk("SPY>200MA", float(spy_close.iloc[-1]) > spy_ma200)
    else:
        failed.append("SPY数据不足")

    # 10. QQQ > 50MA
    if qqq_close is not None and len(qqq_close) >= 50:
        qqq_ma50 = float(qqq_close.rolling(50).mean().iloc[-1])
        chk("QQQ>50MA", float(qqq_close.iloc[-1]) > qqq_ma50)
    else:
        failed.append("QQQ数据不足")

    # 11. VIX < 25
    if vix_val is not None:
        chk("VIX<25", vix_val < APLUS_VIX_MAX)
    else:
        failed.append("VIX数据不足")

    # 12. SPY近5天无连续大阴线
    if spy_close is not None and len(spy_close) >= 6:
        spy_ret = spy_close.pct_change().iloc[-5:]
        consec_red = (spy_ret < -0.01).all()
        chk("SPY无连续阴线", not consec_red)
    else:
        failed.append("SPY历史不足")

    # 13-15: manual confirmation required
    passed.append("⚠️ 无重大利空（需人工确认）")
    passed.append("⚠️ 财报窗口外（需人工确认）")
    passed.append("⚠️ 非CPI/FOMC/非农日（需人工确认）")

    return len(failed) == 0, passed, failed


# ── Historical validation ──────────────────────────────────────────────────────

def _scan_aplus_history(ticker: str, period: str = "2y") -> List[dict]:
    """Scan 2y of OHLCV for A+ signal occurrences and compute 5-day outcomes."""
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 60:
            return []

        close  = raw["Close"].squeeze().dropna()
        low    = raw["Low"].squeeze().dropna()
        volume = raw["Volume"].squeeze().dropna()

        rsi5      = _rsi(close, 5)
        _, _, bbl = _bollinger(close)
        drop3     = (1 - close / close.shift(3)).clip(lower=0)
        vol20     = volume.rolling(20).mean()
        ma20      = close.rolling(20).mean()
        ma50      = close.rolling(50).mean()
        ma200     = close.rolling(200).mean()

        results = []
        aligned = close.index.intersection(low.index)
        for i in range(25, len(close) - 5):
            dt = close.index[i]
            if dt not in aligned:
                continue

            c    = float(close.iloc[i])
            l    = float(low.loc[dt])
            bb_l = float(bbl.iloc[i])
            rsi_v = float(rsi5.iloc[i]) if not pd.isna(rsi5.iloc[i]) else np.nan
            d3    = float(drop3.iloc[i]) if not pd.isna(drop3.iloc[i]) else np.nan
            vr    = float(volume.iloc[i] / vol20.iloc[i]) if vol20.iloc[i] > 0 else np.nan
            m20   = float(ma20.iloc[i])
            m50   = float(ma50.iloc[i])
            m200  = float(ma200.iloc[i]) if i >= 200 else np.nan

            if not check_aplus_bollinger(c, l, bb_l):  continue
            if np.isnan(rsi_v) or rsi_v >= APLUS_RSI5_MAX: continue
            if np.isnan(d3) or d3 <= APLUS_DROP3_MIN: continue
            if np.isnan(vr) or vr <= APLUS_VOL_RATIO_MIN: continue
            below_all = c < m20 and c < m50 and (not np.isnan(m200) and c < m200)
            if below_all: continue

            future = close.iloc[i + 1: i + 6]
            if len(future) < 5:
                continue
            max_gain  = float((future.max() - c) / c)
            final_ret = float((future.iloc[-1] - c) / c)

            results.append({
                "date":       str(dt.date()),
                "buy_price":  round(c, 2),
                "win":        max_gain >= 0.03,
                "return_5d":  round(final_ret * 100, 2),
                "max_gain_5d": round(max_gain * 100, 2),
            })
        return results
    except Exception as e:
        logger.debug(f"_scan_aplus_history {ticker}: {e}")
        return []


def _compute_stats(signals: List[dict]) -> dict:
    if not signals:
        return {
            "n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "rr_ratio": 0.0, "ev": 0.0, "max_consec_losses": 0, "recent_20_wr": 0.0,
        }
    wins   = [s for s in signals if s["win"]]
    losses = [s for s in signals if not s["win"]]
    n      = len(signals)
    wr     = len(wins) / n
    avg_w  = float(np.mean([s["max_gain_5d"] for s in wins]))  if wins   else 0.0
    avg_l  = float(np.mean([abs(s["return_5d"]) for s in losses])) if losses else 0.0
    rr     = round(avg_w / avg_l, 2) if avg_l > 0 else 99.0
    ev     = round(wr * avg_w - (1 - wr) * avg_l, 2)

    cl = max_cl = 0
    for s in signals:
        if not s["win"]:
            cl += 1
            max_cl = max(max_cl, cl)
        else:
            cl = 0

    recent  = signals[-20:] if len(signals) >= 20 else signals
    r20_wr  = round(sum(1 for s in recent if s["win"]) / len(recent), 4)

    return {
        "n": n, "win_rate": round(wr, 4), "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2), "rr_ratio": rr, "ev": ev,
        "max_consec_losses": max_cl, "recent_20_wr": r20_wr,
    }


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    try:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(data: dict):
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"hc_save_cache: {e}")


def _fresh(entry: dict) -> bool:
    ts = entry.get("ts")
    if not ts:
        return False
    try:
        age_h = (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 3600
        return age_h < CACHE_TTL_HOURS
    except Exception:
        return False


# ── Public stats APIs ──────────────────────────────────────────────────────────

def get_stock_stats(ticker: str, force: bool = False) -> dict:
    cache = _load_cache()
    k = f"s_{ticker}"
    if not force and k in cache and _fresh(cache[k]):
        return cache[k]["d"]
    sigs  = _scan_aplus_history(ticker)
    stats = _compute_stats(sigs)
    stats["ticker"] = ticker
    cache[k] = {"ts": datetime.now().isoformat(), "d": stats}
    _save_cache(cache)
    return stats


def get_sector_hc_stats(tickers: List[str], sector: str, force: bool = False) -> dict:
    cache = _load_cache()
    k = f"sec_{sector}"
    if not force and k in cache and _fresh(cache[k]):
        return cache[k]["d"]
    all_sigs: List[dict] = []
    for t in tickers:
        all_sigs.extend(_scan_aplus_history(t))
    all_sigs.sort(key=lambda x: x["date"])
    stats = _compute_stats(all_sigs)
    stats["sector"] = sector
    cache[k] = {"ts": datetime.now().isoformat(), "d": stats}
    _save_cache(cache)
    return stats


def get_market_hc_stats(tickers: List[str], force: bool = False) -> dict:
    cache = _load_cache()
    k = "market"
    if not force and k in cache and _fresh(cache[k]):
        return cache[k]["d"]
    all_sigs: List[dict] = []
    for t in tickers:
        all_sigs.extend(_scan_aplus_history(t))
    all_sigs.sort(key=lambda x: x["date"])
    stats = _compute_stats(all_sigs)
    cache[k] = {"ts": datetime.now().isoformat(), "d": stats}
    _save_cache(cache)
    return stats


def classify_aplus(
    ticker: str,
    sector: str,
    stock_st: dict,
    sector_st: dict,
    market_st: dict,
) -> dict:
    """
    Determine signal_level (A+ or A候选) and tradable flag.
    Uses best available validation level (stock ≥ sector ≥ market).
    """
    base = {
        "stock_n":  stock_st.get("n", 0),
        "sector_n": sector_st.get("n", 0),
        "market_n": market_st.get("n", 0),
    }

    if stock_st.get("n", 0) >= MIN_STOCK_SAMPLES:
        stats, val_level = stock_st, "个股级"
    elif sector_st.get("n", 0) >= MIN_SECTOR_SAMPLES:
        stats, val_level = sector_st, "板块级"
    elif market_st.get("n", 0) >= MIN_MARKET_SAMPLES:
        stats, val_level = market_st, "全市场级"
    else:
        return {
            **base,
            "signal_level": "A候选", "tradable": False,
            "reason": (
                f"历史样本不足，不能升级为A+"
                f"（个股{base['stock_n']}笔/板块{base['sector_n']}笔/全市场{base['market_n']}笔）"
            ),
            "validation_level": "不足", "stats_used": stock_st,
        }

    fails = []
    if stats.get("win_rate", 0) <= MIN_WIN_RATE:
        fails.append(f"胜率{stats['win_rate']*100:.1f}%≤65%")
    if stats.get("rr_ratio", 0) <= MIN_RR_RATIO:
        fails.append(f"盈亏比{stats['rr_ratio']:.2f}≤1.5")
    if stats.get("ev", 0) <= MIN_EV:
        fails.append(f"期望值{stats['ev']:.2f}≤0")
    if stats.get("max_consec_losses", 0) > MAX_CONSEC_LOSS:
        fails.append(f"最大连亏{stats['max_consec_losses']}笔>3")

    if fails:
        return {
            **base,
            "signal_level": "A候选", "tradable": False,
            "reason": "历史验证未通过：" + "、".join(fails),
            "validation_level": val_level, "stats_used": stats,
        }

    return {
        **base,
        "signal_level": "A+", "tradable": True,
        "reason": f"通过{val_level}历史验证（{stats['n']}笔，胜率{stats['win_rate']*100:.1f}%）",
        "validation_level": val_level, "stats_used": stats,
    }


def check_auto_shutdown(market_st: dict, monthly_drawdown_pct: float = 0.0) -> dict:
    """
    Returns dict(shutdown, reasons) if strict mode should auto-disable.
    """
    reasons = []
    r20_wr = market_st.get("recent_20_wr", 1.0)
    ev     = market_st.get("ev", 1.0)

    if r20_wr < 0.60:
        reasons.append(f"最近20笔胜率{r20_wr*100:.1f}%<60%，自动关闭严选模式")
    if ev <= 0:
        reasons.append(f"期望值{ev:.2f}≤0，自动关闭严选模式")
    if monthly_drawdown_pct > 3.0:
        reasons.append(f"月度回撤{monthly_drawdown_pct:.1f}%>3%，关闭短线交易系统")

    return {"shutdown": bool(reasons), "reasons": reasons}
