"""
realtime_radar.py — 实时信号扫描雷达
每30秒刷新，满足1条即触发，按强度分级
🔴 强信号（3-4条）/ 🟡 中信号（2条）/ 🟢 弱信号（1条）
信号条件：RSI(5)<35 / 布林带距下轨5%以内 / 3日跌幅>5% / 成交量>20日均量×1.5
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple

VIX_HIGH   = 30.0
RADAR_TICKERS_BY_SECTOR: Dict[str, List[str]] = {
    "AI算力/GPU/芯片": ["NVDA","MU","ARM","QCOM","INTC","ADI","TXN","NXPI","ON","STM","MPWR","ALAB","CRDO"],
    "半导体设备":       ["ASML","AMAT","KLAC","LRCX","ONTO","TER"],
    "AI服务器/电力":    ["VRT","ETN","NVT","POWL","VST","CEG"],
    "AI云/软件":        ["MSFT","GOOGL","META","AMZN","PLTR","CRWD","NET","SNOW","DDOG"],
    "存储/HBM":         ["MU","WDC","STX"],
    "AI网络":           ["ANET","CSCO","LITE","COHR"],
    "核能/新能源":      ["CCJ","NNE","FSLR","VST"],
    "航天/国防":        ["RKLB","ASTS","LMT","RTX"],
    "中概股":           ["BABA","PDD","JD","NIO","XPEV","LI"],
    "ETF":              ["SMH","SOXX","SPY","QQQ"],
    "医药/生物科技":    ["LLY","NVO","JNJ","MRK","ABBV","PFE","AMGN","GILD","MRNA"],
    "能源/核能":        ["CEG","VST","NNE","SMR","OKLO"],
    "AI基础设施":       ["VRT","SMCI","APLD","CRWV","IREN"],
    "机器人/自动化":    ["ISRG","ABB","ACHR"],
    "防御对冲":         ["V","MA","JNJ","PFE"],
    "消费零售":         ["WMT","AMZN","COST","TGT","HD","NKE","MCD","SBUX","KO"],
}

ALL_RADAR_TICKERS: List[str] = list(dict.fromkeys(
    t for tickers in RADAR_TICKERS_BY_SECTOR.values() for t in tickers
))

_TICKER_SECTOR: Dict[str, str] = {}
for _sec, _tks in RADAR_TICKERS_BY_SECTOR.items():
    for _t in _tks:
        _TICKER_SECTOR.setdefault(_t, _sec)


# ── Indicators (same as backtest_pro) ─────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 5) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period-1, min_periods=period).mean()
    avg_l = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0
               ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + std_dev*std, sma, sma - std_dev*std


# ── Signal Strength Score ──────────────────────────────────────────────────────
def _signal_strength(
    rsi_val: float,
    close_val: float,
    lower_val: float,
    mid_val: float,
    drop3_val: float,
    vol_ratio_val: float,
) -> Tuple[int, List[str]]:
    """Return (score 0-100, triggered_labels)"""
    triggered = []
    score = 0

    # RSI(5) < 35
    if not np.isnan(rsi_val) and rsi_val < 35:
        triggered.append("RSI超卖")
        base = 25
        if rsi_val < 20: base += 10
        elif rsi_val < 25: base += 5
        elif rsi_val < 30: base += 2
        score += base

    # 布林带距下轨5%以内（含跌破）
    if not (np.isnan(close_val) or np.isnan(lower_val)) and close_val < lower_val * 1.05:
        triggered.append("布林带下轨")
        if close_val < lower_val:
            base = 25
            pct_below = (lower_val - close_val) / lower_val
            if pct_below > 0.05: base += 8
            elif pct_below > 0.02: base += 4
        else:
            base = 15  # 距下轨5%以内未跌破，信号稍弱
        score += base

    # 3日跌幅 >5%
    if not np.isnan(drop3_val) and drop3_val > 0.05:
        triggered.append("3日急跌")
        base = 25
        if drop3_val > 0.12: base += 10
        elif drop3_val > 0.08: base += 5
        elif drop3_val > 0.06: base += 2
        score += base

    # 成交量 >20日均量×1.5
    if not np.isnan(vol_ratio_val) and vol_ratio_val > 1.5:
        triggered.append("放量")
        base = 15
        if vol_ratio_val > 4: base += 10
        elif vol_ratio_val > 3: base += 7
        elif vol_ratio_val > 2: base += 4
        score += base

    return min(score, 100), triggered


def _signal_tier(n: int) -> Tuple[str, str]:
    """Return (tier_label, tier_emoji) by triggered condition count."""
    if n >= 3: return "强信号", "🔴"
    if n == 2: return "中信号", "🟡"
    return "弱信号", "🟢"


def _stars(score: int) -> str:
    if score >= 76: return "⭐⭐⭐⭐"
    if score >= 51: return "⭐⭐⭐"
    if score >= 26: return "⭐⭐"
    return "⭐"


def grade_signal(sig: dict, exp_score: Optional[float]) -> str:
    """
    Classify a signal into A+/A/B/C.

    A+: exp>70  AND  RSI超卖+布林带下轨+3日急跌 全触发  AND  RR>1.5  AND  MA50>MA200
    A:  exp>55  AND  ≥2条核心技术触发  AND  RR>1.2
    B:  ≥1条触发  AND  RR>1.0
    C:  有触发但 RR≤1.0 或 exp<40（仍显示，标注仅观察）
    """
    triggered = sig.get("triggered", [])
    rr        = sig.get("rr_ratio") or 0.0
    exp       = exp_score  # may be None

    # Count the three core technical conditions
    tech3 = sum([
        "RSI超卖"   in triggered,
        "布林带下轨" in triggered,
        "3日急跌"   in triggered,
    ])

    # Trend structure intact: MA50 > MA200 (mid-term uptrend)
    ma50  = sig.get("ma50")
    ma200 = sig.get("ma200")
    trend_ok = (ma50 is not None and ma200 is not None and float(ma50) > float(ma200))

    if exp is not None and exp > 70 and tech3 >= 3 and rr > 1.5 and trend_ok:
        return "A+"

    if exp is not None and exp > 55 and tech3 >= 2 and rr > 1.2:
        return "A"

    if len(triggered) >= 1 and rr > 1.0:
        return "B"

    return "C"


def _market_state_from_spy(spy_close: Optional[pd.Series]) -> dict:
    if spy_close is None or spy_close.empty:
        return {"state": "unknown", "emoji": "⚪", "label": "未知"}
    try:
        ma200 = spy_close.rolling(200, min_periods=100).mean().iloc[-1]
        price = float(spy_close.iloc[-1])
        if pd.isna(ma200):
            return {"state": "unknown", "emoji": "⚪", "label": "数据不足"}
        ratio = price / float(ma200) - 1
        if ratio > 0.05:
            return {"state": "bull",     "emoji": "🟢", "label": "牛市"}
        elif ratio >= -0.05:
            return {"state": "sideways", "emoji": "🟡", "label": "震荡"}
        else:
            return {"state": "bear",     "emoji": "🔴", "label": "熊市"}
    except Exception:
        return {"state": "unknown", "emoji": "⚪", "label": "未知"}


def _get_vix_level() -> Optional[float]:
    try:
        vix_raw = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
        if vix_raw.empty: return None
        vc = vix_raw["Close"]
        vix_s = (vc.iloc[:, 0] if isinstance(vc, pd.DataFrame) else vc).dropna()
        return round(float(vix_s.iloc[-1]), 1) if not vix_s.empty else None
    except Exception:
        return None


# ── Main Scanner ───────────────────────────────────────────────────────────────
def scan_signals(
    tickers: Optional[List[str]] = None,
    min_conditions: int = 1,
    backtest_grades: Optional[Dict[str, str]] = None,
) -> Tuple[List[dict], dict, Optional[float], Optional[pd.Series], Optional[pd.Series]]:
    """
    扫描实时信号。
    Returns (signals_list, market_state_dict, vix_value, spy_close, qqq_close)
    spy_close / qqq_close are passed back for A+ condition checks in the dashboard.
    """
    if tickers is None:
        tickers = ALL_RADAR_TICKERS

    signals: List[dict] = []

    # Download recent data (need ~60 days for BB / 200MA calculation)
    _extra = [t for t in ["SPY", "QQQ"] if t not in tickers]
    raw = yf.download(
        tickers + _extra,
        period="1y",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # SPY / QQQ series for market condition checks
    spy_close: Optional[pd.Series] = None
    qqq_close: Optional[pd.Series] = None
    try:
        if hasattr(raw.columns, "levels"):
            lvl1 = raw.columns.get_level_values(1)
            if "SPY" in lvl1:
                spy_close = raw["Close"]["SPY"].dropna()
            if "QQQ" in lvl1:
                qqq_close = raw["Close"]["QQQ"].dropna()
        elif not hasattr(raw.columns, "levels"):
            spy_close = raw["Close"].dropna()
    except Exception:
        pass

    mkt_state = _market_state_from_spy(spy_close)
    vix_val   = _get_vix_level()

    for ticker in tickers:
        try:
            # Extract OHLCV
            if hasattr(raw.columns, "levels"):
                lvl1 = raw.columns.get_level_values(1)
                if ticker not in lvl1: continue
                close  = raw["Close"][ticker].dropna()
                low    = raw["Low"][ticker].dropna()   if "Low"    in raw.columns.get_level_values(0) else pd.Series(dtype=float)
                volume = raw["Volume"][ticker].dropna() if "Volume" in raw.columns.get_level_values(0) else pd.Series(dtype=float)
            else:
                close  = raw["Close"].dropna()
                low    = raw["Low"].dropna()   if "Low"    in raw.columns else pd.Series(dtype=float)
                volume = raw["Volume"].dropna() if "Volume" in raw.columns else pd.Series(dtype=float)

            if len(close) < 25: continue

            # Compute indicators
            rsi5          = _rsi(close, 5)
            _, mid, lower = _bollinger(close)
            drop3         = (1 - close / close.shift(3)).clip(lower=0)
            ma20_s        = close.rolling(20).mean()
            ma50_s        = close.rolling(50).mean()
            ma200_s       = close.rolling(200).mean()

            # Volume ratio
            if not volume.empty and len(volume) >= 20:
                vol_ratio = volume / volume.rolling(20).mean()
            else:
                vol_ratio = pd.Series(np.nan, index=close.index)

            # Latest values
            latest_price     = float(close.iloc[-1])
            latest_low       = float(low.iloc[-1])       if not low.empty       else np.nan
            latest_rsi       = float(rsi5.iloc[-1])      if not rsi5.empty      else np.nan
            latest_lower     = float(lower.iloc[-1])      if not lower.empty      else np.nan
            latest_mid       = float(mid.iloc[-1])        if not mid.empty        else np.nan
            latest_drop3     = float(drop3.iloc[-1])      if not drop3.empty      else np.nan
            latest_vol_ratio = float(vol_ratio.iloc[-1])  if not vol_ratio.empty  else np.nan
            latest_1d_chg    = float(close.pct_change().iloc[-1]) if len(close) >= 2 else 0.0
            latest_ma20      = float(ma20_s.iloc[-1])  if len(close) >= 20  else np.nan
            latest_ma50      = float(ma50_s.iloc[-1])  if len(close) >= 50  else np.nan
            latest_ma200     = float(ma200_s.iloc[-1]) if len(close) >= 200 else np.nan

            score, triggered = _signal_strength(
                latest_rsi, latest_price, latest_lower, latest_mid,
                latest_drop3, latest_vol_ratio,
            )

            if len(triggered) < min_conditions:
                continue

            tier_label, tier_emoji = _signal_tier(len(triggered))

            # Determine best strategy and stop/target
            # If RSI+BB both triggered → strategy 3 (combo): stop -6%, target +8%
            # If only dip conditions → strategy 4: stop -8%, target +10%
            # Otherwise → strategy 1/2: stop -8%
            if "RSI超卖" in triggered and "布林带下轨" in triggered:
                strat_label = "RSI+布林带复合"
                stop_pct    = 0.06
                target_pct  = 0.08
            elif "3日急跌" in triggered and "RSI超卖" in triggered:
                strat_label = "急跌反弹"
                stop_pct    = 0.08
                target_pct  = 0.10
            elif "布林带下轨" in triggered:
                strat_label = "布林带下轨反弹"
                stop_pct    = 0.08
                target_pct  = (latest_mid / latest_price - 1) if latest_mid > 0 else 0.08
            else:
                strat_label = "RSI超卖反弹"
                stop_pct    = 0.08
                target_pct  = 0.10

            stop_price   = round(latest_price * (1 - stop_pct), 2)
            target_price = round(latest_price * (1 + target_pct), 2)
            rr_ratio     = round(target_pct / stop_pct, 2) if stop_pct > 0 else 1.0

            # Position range (±0.5%)
            buy_low  = round(latest_price * 0.995, 2)
            buy_high = round(latest_price * 1.005, 2)

            # Extreme flags
            extreme_flags = []
            if vix_val and vix_val > VIX_HIGH:
                extreme_flags.append(f"VIX={vix_val:.0f}⚠️")
            if abs(latest_1d_chg) > 0.08:
                extreme_flags.append(f"单日波动{latest_1d_chg*100:+.1f}%⚠️")
            # Near earnings proxy: any >5% move in last 3 days
            if len(close) >= 4:
                recent_moves = close.pct_change().iloc[-4:].abs()
                if (recent_moves > 0.05).any():
                    extreme_flags.append("疑似财报前后⚡")

            # Strategy grade from backtest (if available)
            grade_info = "未回测"
            if backtest_grades:
                strat_key_map = {
                    "RSI+布林带复合":  "rsi_boll_combo",
                    "急跌反弹":       "dip_bounce",
                    "布林带下轨反弹": "boll_lower",
                    "RSI超卖反弹":    "rsi_oversold",
                }
                bg_key = f"{ticker}_{strat_key_map.get(strat_label,'')}"
                grade_info = backtest_grades.get(bg_key, "未回测")

            signals.append({
                "ticker":        ticker,
                "sector":        _TICKER_SECTOR.get(ticker, "Other"),
                "tier":          tier_label,
                "tier_emoji":    tier_emoji,
                "price":         latest_price,
                "latest_low":    round(latest_low, 2) if not np.isnan(latest_low) else None,
                "change_1d_pct": round(latest_1d_chg * 100, 2),
                "rsi5":          round(latest_rsi, 1) if not np.isnan(latest_rsi) else None,
                "bb_lower":      round(latest_lower, 2) if not np.isnan(latest_lower) else None,
                "bb_mid":        round(latest_mid, 2) if not np.isnan(latest_mid) else None,
                "drop3_pct":     round(latest_drop3 * 100, 2) if not np.isnan(latest_drop3) else None,
                "vol_ratio":     round(latest_vol_ratio, 2) if not np.isnan(latest_vol_ratio) else None,
                "ma20":          round(latest_ma20, 2) if not np.isnan(latest_ma20) else None,
                "ma50":          round(latest_ma50, 2) if not np.isnan(latest_ma50) else None,
                "ma200":         round(latest_ma200, 2) if not np.isnan(latest_ma200) else None,
                "score":         score,
                "stars":         _stars(score),
                "triggered":     triggered,
                "strategy":      strat_label,
                "buy_low":       buy_low,
                "buy_high":      buy_high,
                "stop_price":    stop_price,
                "stop_pct":      round(stop_pct * 100, 1),
                "target_price":  target_price,
                "target_pct":    round(target_pct * 100, 1),
                "rr_ratio":      rr_ratio,
                "market_state":  mkt_state["label"],
                "market_emoji":  mkt_state["emoji"],
                "extreme_flags": extreme_flags,
                "grade_info":    grade_info,
                "scan_time":     datetime.now().strftime("%H:%M:%S"),
                # A+ evaluation fields (populated in dashboard after scan)
                "signal_level":  None,
                "tradable":      None,
                "aplus_passed":  [],
                "aplus_failed":  [],
                "hc_reason":     "",
            })

        except Exception:
            continue

    # Sort by score descending
    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals, mkt_state, vix_val, spy_close, qqq_close


def get_current_vix_and_market() -> Tuple[Optional[float], dict]:
    """Fast check: VIX level + SPY market state. Used for top status bar."""
    try:
        raw = yf.download(["SPY","^VIX"], period="1y", auto_adjust=True, progress=False)
        spy_close: Optional[pd.Series] = None
        vix_val: Optional[float] = None

        if hasattr(raw.columns, "levels"):
            lvl1 = raw.columns.get_level_values(1)
            if "SPY" in lvl1:
                spy_close = raw["Close"]["SPY"].dropna()
            if "^VIX" in lvl1:
                vc = raw["Close"]["^VIX"].dropna()
                if not vc.empty:
                    vix_val = round(float(vc.iloc[-1]), 1)
        elif "SPY" in ["SPY","^VIX"]:
            spy_close = raw["Close"].dropna()

        mkt_state = _market_state_from_spy(spy_close)
        return vix_val, mkt_state
    except Exception:
        return None, {"state": "unknown", "emoji": "⚪", "label": "数据不可用"}


def build_signal_card_html(sig: dict) -> str:
    """Render a signal card as HTML string for st.markdown()"""
    score      = sig["score"]
    score_c    = "#00CC96" if score >= 70 else ("#FFA500" if score >= 45 else "#EF553B")
    chg        = sig["change_1d_pct"]
    chg_c      = "#EF553B" if chg < 0 else "#00CC96"
    chg_icon   = "📉" if chg < 0 else "📈"
    flags_html = "".join(f"<span style='color:#FF6B35;font-size:0.78rem'>⚡ {f}</span><br>" for f in sig["extreme_flags"])
    tier_emoji = sig.get("tier_emoji", "⚪")
    tier_label = sig.get("tier", "")

    # Signal level drives border color; C-level gets gray
    signal_level = sig.get("signal_level") or ""
    if signal_level == "A+":
        border_c = "#00CC96"
    elif signal_level == "A":
        border_c = "#FFA500"
    elif signal_level == "B":
        border_c = "#FFD700"
    elif signal_level == "C":
        border_c = "#555566"
    else:
        border_c = "#EF553B" if tier_label == "强信号" else ("#FFA500" if tier_label == "中信号" else "#00CC96")

    market_label = sig.get("market_label", sig.get("market_state",""))
    mkt_emoji    = sig.get("market_emoji","⚪")

    c_warning = (
        "<div style='margin-top:8px;padding:6px 10px;background:#2a2a2a;"
        "border-radius:6px;font-size:0.80rem;color:#888'>"
        "⚠️ 仅观察，不建议操作</div>"
    ) if signal_level == "C" else ""

    return f"""
<div style='background:#1a1a2e;border:1px solid {border_c};border-radius:10px;
padding:14px 16px;margin:8px 0;'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
    <span style='font-size:1.3rem;font-weight:700;color:#fff'>{tier_emoji} {sig['ticker']}</span>
    <span style='font-size:1.1rem;font-weight:600;color:#ccc'>${sig['price']:.2f}
      <span style='color:{chg_c}'> {chg_icon} {chg:+.2f}%</span></span>
  </div>
  <div style='font-size:1.0rem;margin-bottom:4px'>
    {sig['stars']} <span style='color:{score_c};font-weight:600'>信号强度 {score}/100</span>
    &nbsp;<span style='color:{border_c};font-size:0.88rem;font-weight:600'>{tier_emoji} {tier_label}</span>
  </div>
  <div style='font-size:0.85rem;color:#aaa;margin-bottom:8px'>
    触发：<span style='color:#82C8FF'>{'·'.join(sig['triggered'])}</span> &nbsp;|&nbsp;
    策略：<span style='color:#DDA0DD'>{sig['strategy']}</span>
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.82rem;color:#ccc'>
    <div>市场状态：{mkt_emoji} <b>{sig['market_state']}</b></div>
    <div>评级参考：<b>{sig['grade_info']}</b></div>
    <div>买入区间：<span style='color:#82C8FF'>${sig['buy_low']}-${sig['buy_high']}</span></div>
    <div>风险收益比：<b>1:{sig['rr_ratio']}</b></div>
    <div>止损：<span style='color:#EF553B'>${sig['stop_price']} ({'-' + str(sig['stop_pct'])}%)</span></div>
    <div>目标：<span style='color:#00CC96'>${sig['target_price']} (+{sig['target_pct']}%)</span></div>
  </div>
  {('<div style="margin-top:8px">'+flags_html+'</div>') if sig['extreme_flags'] else ''}
  {c_warning}
  <div style='font-size:0.72rem;color:#666;margin-top:6px'>扫描时间: {sig['scan_time']}</div>
</div>
"""
