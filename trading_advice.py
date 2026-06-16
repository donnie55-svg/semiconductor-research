# trading_advice.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. 默认板块联动池
# =========================================================

SECTOR_POOLS = {
    "market": ["SPY", "QQQ", "IWM"],
    "semi_ai": ["SOXX", "SMH", "NVDA", "AMD", "AVGO", "TSM"],
    "MU": ["SOXX", "SMH", "NVDA", "AMD", "WDC", "STX"],
    "ETN": ["XLI", "GRID", "PAVE", "VRT", "GE", "EMR"],
    "default": ["SPY", "QQQ"],
}


def get_sector_pool(ticker: str) -> list[str]:
    ticker = ticker.upper()
    if ticker in SECTOR_POOLS:
        return SECTOR_POOLS[ticker]
    semi_names = {"NVDA", "AMD", "AVGO", "TSM", "AMAT", "LRCX", "ASML", "KLAC", "QCOM", "MRVL"}
    if ticker in semi_names:
        return SECTOR_POOLS["semi_ai"]
    return SECTOR_POOLS["default"]


# =========================================================
# 2. 工具函数
# =========================================================

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def get_quote_row(quote_df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if quote_df is None or quote_df.empty:
        return None
    ticker_col = find_col(quote_df, ["ticker", "symbol", "code", "stock", "股票"])
    if ticker_col is None:
        return None
    rows = quote_df[quote_df[ticker_col].astype(str).str.upper() == ticker.upper()]
    if rows.empty:
        return None
    return rows.iloc[0]


def get_quote_value(quote_df, ticker, candidates, default=np.nan):
    row = get_quote_row(quote_df, ticker)
    if row is None:
        return default
    for c in candidates:
        for real_col in row.index:
            if real_col.lower() == c.lower():
                return safe_float(row[real_col], default)
    return default


def get_price(quote_df, ticker, fallback=np.nan):
    return get_quote_value(quote_df, ticker,
        ["price", "current_price", "last", "last_price", "close", "当前价", "现价"], fallback)


def get_prev_close(quote_df, ticker, fallback=np.nan):
    return get_quote_value(quote_df, ticker,
        ["prev_close", "previous_close", "last_close", "昨收", "昨收价"], fallback)


def pct_change(price, prev_close):
    price = safe_float(price)
    prev_close = safe_float(prev_close)
    if np.isnan(price) or np.isnan(prev_close) or prev_close == 0:
        return np.nan
    return price / prev_close - 1


# =========================================================
# 3. 日K技术面：MA5/10/20/50
# =========================================================

def prepare_kline(kline_df: pd.DataFrame) -> pd.DataFrame:
    if kline_df is None or kline_df.empty:
        return pd.DataFrame()
    df = kline_df.copy()
    date_col = find_col(df, ["date", "time_key", "datetime", "time", "日期"])
    close_col = find_col(df, ["close", "收盘", "收盘价"])
    volume_col = find_col(df, ["volume", "vol", "成交量"])
    if close_col is None:
        return pd.DataFrame()
    if date_col:
        df = df.sort_values(date_col)
    df["close_"] = pd.to_numeric(df[close_col], errors="coerce")
    for n in [5, 10, 20, 50]:
        df[f"MA{n}"] = df["close_"].rolling(n).mean()
    if volume_col:
        df["volume_"] = pd.to_numeric(df[volume_col], errors="coerce")
        df["VOL20"] = df["volume_"].rolling(20).mean()
    else:
        df["volume_"] = np.nan
        df["VOL20"] = np.nan
    return df


def get_tech_status(ticker: str, kline_map: Dict[str, pd.DataFrame], quote_df: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "price": np.nan, "ma5": np.nan, "ma10": np.nan, "ma20": np.nan, "ma50": np.nan,
        "above_ma20": False, "above_ma50": False, "ma_bullish": False,
        "below_ma20": False, "below_ma50": False,
        "volume_breakout": False, "distance_ma20_pct": np.nan,
    }
    _kdf = kline_map.get(ticker.upper())
    if _kdf is None:
        _kdf = kline_map.get(ticker)
    kline_df = _kdf
    df = prepare_kline(kline_df)
    if df.empty:
        result["price"] = get_price(quote_df, ticker)
        return result
    latest = df.iloc[-1]
    price = get_price(quote_df, ticker, fallback=safe_float(latest["close_"]))
    ma5 = safe_float(latest.get("MA5"))
    ma10 = safe_float(latest.get("MA10"))
    ma20 = safe_float(latest.get("MA20"))
    ma50 = safe_float(latest.get("MA50"))
    result.update({
        "price": price, "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma50": ma50,
        "above_ma20": not np.isnan(price) and not np.isnan(ma20) and price > ma20,
        "above_ma50": not np.isnan(price) and not np.isnan(ma50) and price > ma50,
        "below_ma20": not np.isnan(price) and not np.isnan(ma20) and price < ma20,
        "below_ma50": not np.isnan(price) and not np.isnan(ma50) and price < ma50,
        "ma_bullish": not any(np.isnan(x) for x in [ma5, ma10, ma20]) and ma5 > ma10 > ma20,
    })
    if not np.isnan(price) and not np.isnan(ma20) and ma20 != 0:
        result["distance_ma20_pct"] = price / ma20 - 1
    latest_vol = safe_float(latest.get("volume_"))
    vol20 = safe_float(latest.get("VOL20"))
    if not np.isnan(latest_vol) and not np.isnan(vol20) and vol20 > 0:
        result["volume_breakout"] = latest_vol > vol20 * 1.5
    return result


# =========================================================
# 4. 板块联动
# =========================================================

def calc_sector_strength(ticker: str, quote_df: pd.DataFrame) -> Dict[str, Any]:
    pool = get_sector_pool(ticker)
    changes = []
    for t in pool:
        price = get_price(quote_df, t)
        prev_close = get_prev_close(quote_df, t)
        chg = pct_change(price, prev_close)
        if not np.isnan(chg):
            changes.append(chg)
    if len(changes) == 0:
        return {"sector_status": "unknown", "sector_score": 0,
                "red_ratio": np.nan, "avg_change": np.nan, "pool": pool}
    red_ratio = sum(x > 0 for x in changes) / len(changes)
    avg_change = float(np.mean(changes))
    if red_ratio >= 0.6 and avg_change > 0:
        status, score = "strong", 15
    elif red_ratio >= 0.4:
        status, score = "neutral", 5
    else:
        status, score = "weak", -15
    return {"sector_status": status, "sector_score": score,
            "red_ratio": red_ratio, "avg_change": avg_change, "pool": pool}


# =========================================================
# 5. 盘中VWAP / fallback
# =========================================================

def calc_vwap(intraday_df: pd.DataFrame) -> float:
    if intraday_df is None or intraday_df.empty:
        return np.nan
    df = intraday_df.copy()
    high_col = find_col(df, ["high", "最高", "最高价"])
    low_col = find_col(df, ["low", "最低", "最低价"])
    close_col = find_col(df, ["close", "收盘", "收盘价"])
    volume_col = find_col(df, ["volume", "vol", "成交量"])
    if close_col is None or volume_col is None:
        return np.nan
    close = pd.to_numeric(df[close_col], errors="coerce")
    volume = pd.to_numeric(df[volume_col], errors="coerce")
    if high_col and low_col:
        high = pd.to_numeric(df[high_col], errors="coerce")
        low = pd.to_numeric(df[low_col], errors="coerce")
        typical_price = (high + low + close) / 3
    else:
        typical_price = close
    total_volume = volume.sum()
    if total_volume <= 0:
        return np.nan
    return float((typical_price * volume).sum() / total_volume)


def intraday_confirm_status(ticker, quote_df, intraday_map=None):
    price = get_price(quote_df, ticker)
    prev_close = get_prev_close(quote_df, ticker)
    open_price = get_quote_value(quote_df, ticker, ["open", "open_price", "开盘", "开盘价"], np.nan)
    day_low = get_quote_value(quote_df, ticker, ["low", "day_low", "最低", "日内低点"], np.nan)
    result = {"intraday_status": "unknown", "intraday_score": 0,
              "vwap": np.nan, "above_vwap": False, "high_open_low_close": False}
    if intraday_map:
        intraday_df = intraday_map.get(ticker.upper()) or intraday_map.get(ticker)
        vwap = calc_vwap(intraday_df)
        if not np.isnan(vwap) and not np.isnan(price):
            result["vwap"] = vwap
            result["above_vwap"] = price >= vwap
            if price >= vwap:
                result["intraday_status"] = "confirmed"
                result["intraday_score"] = 15
            else:
                result["intraday_status"] = "below_vwap"
                result["intraday_score"] = -5
    if result["intraday_status"] == "unknown":
        if not np.isnan(price) and not np.isnan(open_price) and not np.isnan(prev_close):
            if price > open_price and price > prev_close:
                result["intraday_status"] = "confirmed_fallback"
                result["intraday_score"] = 10
            elif price < open_price and open_price > prev_close * 1.005:
                result["intraday_status"] = "high_open_fading"
                result["intraday_score"] = -10
                result["high_open_low_close"] = True
        if not np.isnan(price) and not np.isnan(day_low):
            if price > day_low * 1.01 and result["intraday_score"] >= 0:
                result["intraday_score"] += 5
    return result


# =========================================================
# 6. 买入评分
# =========================================================

def normalize_signal_grade(x):
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in {"A+", "APLUS", "A_PLUS"}:
        return "A+"
    if s == "A":
        return "A"
    return s


def score_buy_candidate(row, quote_df, kline_map, intraday_map=None):
    ticker = str(row["ticker"]).upper()
    signal_grade = normalize_signal_grade(row["signal_grade"])
    if signal_grade not in {"A", "A+"}:
        return {"ticker": ticker, "signal_grade": signal_grade, "score": 0,
                "action": "不进入买入池", "reason": "系统信号不是 A/A+"}
    score = 0
    reasons = []
    risks = []
    if signal_grade == "A+":
        score += 35
        reasons.append("系统 A+")
    else:
        score += 25
        reasons.append("系统 A")
    sector = calc_sector_strength(ticker, quote_df)
    score += sector["sector_score"]
    if sector["sector_status"] == "strong":
        reasons.append("板块联动强")
    elif sector["sector_status"] == "neutral":
        reasons.append("板块中性")
    elif sector["sector_status"] == "weak":
        risks.append("板块偏弱")
    tech = get_tech_status(ticker, kline_map, quote_df)
    if tech["above_ma20"]:
        score += 15
        reasons.append("站上 MA20")
    else:
        score -= 15
        risks.append("未站上 MA20")
    if tech["above_ma50"]:
        score += 10
        reasons.append("站上 MA50")
    else:
        score -= 20
        risks.append("未站上 MA50")
    if tech["ma_bullish"]:
        score += 10
        reasons.append("MA5>MA10>MA20")
    if tech["below_ma50"]:
        score -= 15
        risks.append("跌破 MA50")
    if tech["volume_breakout"]:
        score += 10
        reasons.append("放量突破")
    dist_ma20 = tech["distance_ma20_pct"]
    if not np.isnan(dist_ma20) and dist_ma20 > 0.15:
        score -= 10
        risks.append("短线离 MA20 偏远，谨慎追高")
    intraday = intraday_confirm_status(ticker, quote_df, intraday_map)
    score += intraday["intraday_score"]
    if intraday["intraday_status"] in {"confirmed", "confirmed_fallback"}:
        reasons.append("盘中确认通过")
    elif intraday["high_open_low_close"]:
        risks.append("高开回落")
    if signal_grade == "A+":
        if score >= 75:
            action = "强买入候选"
        elif score >= 60:
            action = "可分批买入"
        elif score >= 45:
            action = "观察，不追"
        else:
            action = "暂不买"
    else:
        if score >= 60:
            action = "A信号：可小仓/分批"
        elif score >= 45:
            action = "A信号：观察候选"
        else:
            action = "A信号：暂不买"
    return {
        "ticker": ticker, "signal_grade": signal_grade,
        "price": tech["price"], "score": int(round(score)),
        "action": action,
        "sector_status": sector["sector_status"],
        "sector_red_ratio": sector["red_ratio"],
        "sector_avg_change": sector["avg_change"],
        "ma20": tech["ma20"], "ma50": tech["ma50"],
        "above_ma20": tech["above_ma20"], "above_ma50": tech["above_ma50"],
        "ma_bullish": tech["ma_bullish"],
        "distance_ma20_pct": tech["distance_ma20_pct"],
        "volume_breakout": tech["volume_breakout"],
        "intraday_status": intraday["intraday_status"],
        "vwap": intraday["vwap"],
        "reasons": "；".join(reasons),
        "risks": "；".join(risks),
    }


def build_buy_advice(signals_df, quote_df, kline_map, intraday_map=None):
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()
    df = signals_df.copy()
    ticker_col = find_col(df, ["ticker", "symbol", "code", "股票", "标的"])
    signal_col = find_col(df, ["signal_grade", "grade", "rating", "signal", "信号", "等级"])
    if ticker_col is None or signal_col is None:
        raise ValueError("signals_df 必须包含 ticker/symbol 和 signal_grade/grade 字段")
    df = df.rename(columns={ticker_col: "ticker", signal_col: "signal_grade"})
    df["signal_grade"] = df["signal_grade"].apply(normalize_signal_grade)
    df = df[df["signal_grade"].isin(["A", "A+"])].copy()
    if df.empty:
        return pd.DataFrame()
    rows = [score_buy_candidate(row, quote_df, kline_map, intraday_map) for _, row in df.iterrows()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "signal_grade"], ascending=[False, True])


# =========================================================
# 7. 持仓卖出建议
# =========================================================

DEFAULT_POSITIONS = {
    "MU": {"cost": None, "shares": None, "target_price": None},
    "ETN": {"cost": None, "shares": None, "target_price": None},
}


def load_positions_config(path="positions_config.json"):
    p = Path(path)
    if not p.exists():
        return DEFAULT_POSITIONS.copy()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_POSITIONS.copy()


def save_positions_config(data, path="positions_config.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def score_position_advice(ticker, position, quote_df, kline_map):
    ticker = ticker.upper()
    cost = safe_float(position.get("cost"))
    shares = safe_float(position.get("shares"))
    target_price = safe_float(position.get("target_price"))
    tech = get_tech_status(ticker, kline_map, quote_df)
    price = tech["price"]
    pnl_pct = np.nan
    pnl_amount = np.nan
    distance_to_target_pct = np.nan
    if not np.isnan(price) and not np.isnan(cost) and cost > 0:
        pnl_pct = price / cost - 1
    if not np.isnan(price) and not np.isnan(cost) and not np.isnan(shares):
        pnl_amount = (price - cost) * shares
    if not np.isnan(price) and not np.isnan(target_price) and target_price > 0:
        distance_to_target_pct = target_price / price - 1
    action = "继续持有"
    reasons = []
    risks = []
    if tech["below_ma50"]:
        action = "强减仓 / 清仓观察"
        risks.append("跌破 MA50，趋势明显转弱")
    elif tech["below_ma20"]:
        action = "减仓提醒"
        risks.append("跌破 MA20，短线转弱")
    stop_loss = -0.08 if ticker in {"MU", "NVDA", "AMD", "PL", "ASTS"} else -0.06
    if not np.isnan(pnl_pct) and pnl_pct <= stop_loss:
        action = "触发止损 / 至少减仓"
        risks.append(f"亏损达到 {stop_loss:.0%} 止损线")
    if not np.isnan(pnl_pct):
        if pnl_pct >= 0.25:
            action = "止盈至少 1/2"
            reasons.append("盈利超过 25%")
        elif pnl_pct >= 0.15 and action == "继续持有":
            action = "止盈 1/3 或上移止损"
            reasons.append("盈利超过 15%")
    if not np.isnan(distance_to_target_pct):
        if distance_to_target_pct <= 0:
            action = "到达目标价，分批止盈"
            reasons.append("已经达到或超过目标价")
        elif distance_to_target_pct <= 0.05:
            reasons.append("距离目标价小于 5%，接近止盈区")
    if ticker == "MU":
        if not np.isnan(tech["distance_ma20_pct"]) and tech["distance_ma20_pct"] > 0.18:
            risks.append("MU 离 MA20 偏远，周期股短线容易回撤")
    elif ticker == "ETN":
        if action == "减仓提醒":
            reasons.append("ETN 可比 MU 更耐心，先减仓不必直接清仓")
    return {
        "ticker": ticker, "price": price, "cost": cost, "shares": shares,
        "pnl_pct": pnl_pct, "pnl_amount": pnl_amount,
        "target_price": target_price, "distance_to_target_pct": distance_to_target_pct,
        "ma20": tech["ma20"], "ma50": tech["ma50"],
        "above_ma20": tech["above_ma20"], "above_ma50": tech["above_ma50"],
        "action": action, "reasons": "；".join(reasons), "risks": "；".join(risks),
    }


def build_position_advice(positions, quote_df, kline_map):
    rows = [score_position_advice(ticker, pos, quote_df, kline_map)
            for ticker, pos in positions.items()]
    return pd.DataFrame(rows)


# =========================================================
# 8. Streamlit 渲染
# =========================================================

def render_trading_advice_tab(
    signals_df: pd.DataFrame,
    quote_df: pd.DataFrame,
    kline_map: Dict[str, pd.DataFrame],
    intraday_map: Optional[Dict[str, pd.DataFrame]] = None,
    positions_config_path: str = "positions_config.json",
):
    st.header("交易建议")
    st.caption("规则：A/A+ 宽松入池；板块联动、均线、VWAP、风险项做评分；A 信号不做强买，只作为观察/小仓候选。")

    # 持仓设置
    with st.expander("持仓设置：MU / ETN 成本价", expanded=False):
        positions = load_positions_config(positions_config_path)
        for ticker in ["MU", "ETN"]:
            if ticker not in positions:
                positions[ticker] = {"cost": None, "shares": None, "target_price": None}
            st.subheader(ticker)
            c1, c2, c3 = st.columns(3)
            with c1:
                cost = st.number_input(f"{ticker} 成本价", min_value=0.0,
                    value=float(positions[ticker].get("cost") or 0.0), step=0.1, key=f"{ticker}_cost")
            with c2:
                shares = st.number_input(f"{ticker} 股数", min_value=0.0,
                    value=float(positions[ticker].get("shares") or 0.0), step=1.0, key=f"{ticker}_shares")
            with c3:
                target_price = st.number_input(f"{ticker} 目标价", min_value=0.0,
                    value=float(positions[ticker].get("target_price") or 0.0), step=0.1, key=f"{ticker}_target_price")
            positions[ticker]["cost"] = cost if cost > 0 else None
            positions[ticker]["shares"] = shares if shares > 0 else None
            positions[ticker]["target_price"] = target_price if target_price > 0 else None
        if st.button("保存持仓设置"):
            save_positions_config(positions, positions_config_path)
            st.success("持仓设置已保存")

    # 今日买入候选
    st.subheader("今日可买候选")
    try:
        buy_df = build_buy_advice(signals_df=signals_df, quote_df=quote_df,
                                   kline_map=kline_map, intraday_map=intraday_map)
    except Exception as e:
        st.error(f"买入建议生成失败：{e}")
        buy_df = pd.DataFrame()

    if buy_df.empty:
        st.info("今天没有 A/A+ 买入候选。")
    else:
        display_buy_df = buy_df.copy()
        for col in ["sector_red_ratio", "sector_avg_change", "distance_ma20_pct"]:
            if col in display_buy_df.columns:
                display_buy_df[col] = display_buy_df[col].apply(
                    lambda x: "" if pd.isna(x) else f"{x:.2%}")
        show_cols = ["ticker", "signal_grade", "price", "score", "action",
                     "sector_status", "sector_red_ratio", "above_ma20", "above_ma50",
                     "ma_bullish", "distance_ma20_pct", "intraday_status", "reasons", "risks"]
        show_cols = [c for c in show_cols if c in display_buy_df.columns]
        st.dataframe(display_buy_df[show_cols], use_container_width=True, hide_index=True)

    # 现有持仓建议
    st.subheader("现有持仓建议")
    positions = load_positions_config(positions_config_path)
    pos_df = build_position_advice(positions=positions, quote_df=quote_df, kline_map=kline_map)
    if pos_df.empty:
        st.info("暂无持仓配置。")
    else:
        display_pos_df = pos_df.copy()
        for col in ["pnl_pct", "distance_to_target_pct"]:
            if col in display_pos_df.columns:
                display_pos_df[col] = display_pos_df[col].apply(
                    lambda x: "" if pd.isna(x) else f"{x:.2%}")
        if "pnl_amount" in display_pos_df.columns:
            display_pos_df["pnl_amount"] = display_pos_df["pnl_amount"].apply(
                lambda x: "" if pd.isna(x) else f"{x:,.2f}")
        show_cols = ["ticker", "price", "cost", "shares", "pnl_pct", "pnl_amount",
                     "target_price", "distance_to_target_pct", "above_ma20", "above_ma50",
                     "action", "reasons", "risks"]
        show_cols = [c for c in show_cols if c in display_pos_df.columns]
        st.dataframe(display_pos_df[show_cols], use_container_width=True, hide_index=True)

    # 仓位规则
    st.subheader("仓位建议规则")
    st.markdown("""
| 条件 | 建议仓位 |
|---|---:|
| A+ 且评分 ≥ 75 | 8%–12% |
| A+ 且评分 60–74 | 5%–8% |
| A 且评分 ≥ 60 | 3%–5% |
| A 且评分 45–59 | 观察，不追 |
| 高波动题材股 | 2%–5% |
| 跌破 MA20 | 减仓 |
| 跌破 MA50 | 强减仓 / 清仓观察 |
""")
