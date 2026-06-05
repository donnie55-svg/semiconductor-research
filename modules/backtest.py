"""
Backtesting module — 防过拟合真实交易验证版
4 strategies, 15 sectors, 2020-present data
Includes: real costs, stress tests, OOS validation, market state, VIX filter, composite scoring
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
COST_PER_SIDE = 0.0015        # 0.1% commission + 0.05% slippage
DATA_START    = "2020-01-01"  # extended for stress tests

STRESS_PERIODS: Dict[str, Tuple[str, str]] = {
    "2020年3月崩盘": ("2020-02-20", "2020-03-23"),
    "2022年熊市":    ("2022-01-01", "2022-12-31"),
    "2023年银行危机": ("2023-03-08", "2023-03-31"),
}

TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31"
OOS_START   = "2024-01-01"
OOS_END     = "2024-12-31"

MIN_RELIABLE      = 25    # below this: sample warning
MIN_DISPLAY       = 10    # below this: skip entirely
VIX_HIGH          = 30.0
BIG_DROP_THRESH   = -0.08  # -8% single day

# ── Stock Universe ─────────────────────────────────────────────────────────────
BACKTEST_SECTORS: Dict[str, List[str]] = {
    "AI算力/GPU/芯片": [
        "NVDA", "MU", "ARM", "QCOM", "INTC", "ADI", "TXN", "NXPI", "ON", "STM",
        "MCHP", "LSCC", "ALGM", "NVTS", "MPWR", "ALAB", "CRDO", "RMBS", "SITM",
        "VICR", "MTSI", "SKYT", "TSEM", "GFS",
    ],
    "半导体设备/材料": [
        "ASML", "AMAT", "KLAC", "LRCX", "ONTO", "NVMI", "ACLS", "FORM", "ACMR",
        "MKSI", "CAMT", "VECO", "COHR", "LASR", "IPGP", "COHU", "TER", "DIOD",
        "LFUS", "AEIS",
    ],
    "AI服务器/数据中心/电力": [
        "VRT", "ETN", "NVT", "ENS", "POWL", "HUBB", "GEV", "VST", "CEG", "BE",
        "CAT", "CMI",
    ],
    "存储/HBM": ["MU", "WDC", "STX", "SIMO", "RMBS"],
    "AI网络/光模块": ["ANET", "CSCO", "FTNT", "ERIC", "NOK", "LITE", "COHR", "IPGP"],
    "数据中心EMS": ["JBL", "FLEX", "SANM", "PLXS", "TTMI", "FN"],
    "AI云/软件": [
        "CRWV", "DOCN", "APLD", "SOUN", "NBIS", "IONQ", "MSFT", "GOOGL", "META",
        "AMZN", "CRM", "NOW", "SNOW", "MDB", "PLTR", "AI", "PATH", "DDOG", "NET",
        "ZS", "CRWD", "PANW",
    ],
    "核能/新能源": [
        "CCJ", "BWXT", "NNE", "FSLR", "PLUG", "FCEL", "BLDP", "VST", "CEG", "ENPH",
    ],
    "航天/卫星/国防": [
        "RKLB", "ASTS", "PL", "BKSY", "ESLT", "LMT", "BA", "GE", "CW", "ATRO",
        "VSAT", "KTOS", "RDW", "LUNR", "NOC", "RTX",
    ],
    "工业自动化": ["ST", "CTS", "CGNX", "NDSN", "PPG", "MMM", "EMR", "GLW"],
    "原材料/铜/能源": ["SCCO", "FCX", "OXY", "DOW", "CC", "KALU", "XOM", "CVX"],
    "医药/消费/金融": [
        "KO", "PG", "MRK", "AMGN", "JPM", "BAC", "MA", "MCO", "LLY", "NVO",
        "MRNA", "VRTX",
    ],
    "航空&旅游": [
        "AAL", "DAL", "UAL", "LUV", "ALK", "CCL", "RCL", "NCLH", "ABNB", "BKNG",
        "DIS", "LVS", "WYNN", "MGM",
    ],
    "中概股ADR": [
        "BABA", "PDD", "JD", "BIDU", "NIO", "XPEV", "LI", "TCOM", "FUTU",
    ],
    "ETF/指数": ["SMH", "SOXX", "XSD", "DRAM", "SPY", "QQQ", "EWT"],
}

ALL_BACKTEST_TICKERS: List[str] = list(dict.fromkeys(
    t for tickers in BACKTEST_SECTORS.values() for t in tickers
))

STRATEGY_NAMES: Dict[str, str] = {
    "rsi":      "RSI超买超卖",
    "macd":     "MACD金叉死叉",
    "boll":     "布林带通道",
    "pullback": "回调RSI复合",
}

_TICKER_SECTOR: Dict[str, str] = {}
for _sec, _tks in BACKTEST_SECTORS.items():
    for _t in _tks:
        _TICKER_SECTOR.setdefault(_t, _sec)


# ── Technical Indicators ───────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0
               ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std


# ── Market State ───────────────────────────────────────────────────────────────
def compute_market_states(spy_close: pd.Series) -> pd.Series:
    """Bull / sideways / bear classification based on SPY vs 200-day MA."""
    ma200 = spy_close.rolling(200, min_periods=100).mean()
    ratio = (spy_close / ma200 - 1.0)
    states = pd.Series("unknown", index=spy_close.index, dtype=object)
    states[ratio > 0.03]                          = "bull"
    states[(ratio >= -0.03) & (ratio <= 0.03)]    = "sideways"
    states[ratio < -0.03]                          = "bear"
    states[ma200.isna()]                           = "unknown"
    return states


def get_current_market_state(spy_close: pd.Series) -> dict:
    """Return dict with state/emoji/label for dashboard indicator."""
    if spy_close.empty:
        return {"state": "unknown", "emoji": "⚪", "label": "数据不可用",
                "spy_price": None, "ma200": None, "ratio_pct": None}
    ma200_val = spy_close.rolling(200, min_periods=100).mean().iloc[-1]
    spy_price = float(spy_close.iloc[-1])
    if pd.isna(ma200_val):
        return {"state": "unknown", "emoji": "⚪", "label": "数据不足",
                "spy_price": spy_price, "ma200": None, "ratio_pct": None}
    ratio = spy_price / float(ma200_val) - 1.0
    ratio_pct = round(ratio * 100, 2)
    if ratio > 0.03:
        return {"state": "bull",     "emoji": "🟢", "label": "牛市-正常交易",
                "spy_price": spy_price, "ma200": round(float(ma200_val), 2), "ratio_pct": ratio_pct}
    elif ratio >= -0.03:
        return {"state": "sideways", "emoji": "🟡", "label": "震荡-谨慎交易",
                "spy_price": spy_price, "ma200": round(float(ma200_val), 2), "ratio_pct": ratio_pct}
    else:
        return {"state": "bear",     "emoji": "🔴", "label": "熊市-禁止做多",
                "spy_price": spy_price, "ma200": round(float(ma200_val), 2), "ratio_pct": ratio_pct}


# ── Trade Labeling Helper ──────────────────────────────────────────────────────
def _entry_labels(
    entry_idx: int,
    close: pd.Series,
    daily_rets: pd.Series,
    market_states: Optional[pd.Series],
    vix: Optional[pd.Series],
) -> Tuple[str, List[str]]:
    """Return (market_state, risk_flags) for a given entry index."""
    entry_dt = close.index[entry_idx]

    mstate = "unknown"
    if market_states is not None:
        try:
            ms = market_states.asof(entry_dt)
            if ms and str(ms) not in ("nan", "None"):
                mstate = str(ms)
        except Exception:
            pass

    flags: List[str] = []

    # VIX > 30
    if vix is not None:
        try:
            vv = float(vix.asof(entry_dt))
            if not np.isnan(vv) and vv > VIX_HIGH:
                flags.append("vix_high")
        except Exception:
            pass

    # Post big-drop (接飞刀)
    try:
        pre = daily_rets.iloc[max(0, entry_idx - 3):entry_idx]
        if (pre < BIG_DROP_THRESH).any():
            flags.append("post_crash_buy")
    except Exception:
        pass

    # Near earnings proxy: >5% single-day move within ±3 days
    try:
        window = daily_rets.iloc[max(0, entry_idx - 3):min(len(daily_rets), entry_idx + 4)]
        if (window.abs() > 0.05).any():
            flags.append("near_earnings")
    except Exception:
        pass

    return mstate, flags


# ── Core Simulator ─────────────────────────────────────────────────────────────
def _simulate(
    close: pd.Series,
    buy: pd.Series,
    sell: pd.Series,
    market_states: Optional[pd.Series] = None,
    vix: Optional[pd.Series] = None,
) -> List[dict]:
    """Long-only, one-position-at-a-time simulator with cost deduction."""
    trades: List[dict] = []
    entry_idx: Optional[int] = None
    raw_entry: float = 0.0
    adj_entry: float = 0.0

    c = close.values
    b = buy.values.astype(bool)
    s = sell.values.astype(bool)
    dates = close.index
    daily_rets = close.pct_change()

    for i in range(len(c)):
        if np.isnan(c[i]):
            continue
        if entry_idx is None:
            if b[i]:
                raw_entry = c[i]
                adj_entry = c[i] * (1.0 + COST_PER_SIDE)
                entry_idx = i
        else:
            if s[i]:
                adj_exit = c[i] * (1.0 - COST_PER_SIDE)
                net_ret = (adj_exit / adj_entry - 1.0) * 100.0
                hold_days = (dates[i] - dates[entry_idx]).days
                mstate, flags = _entry_labels(entry_idx, close, daily_rets, market_states, vix)
                trades.append({
                    "entry_date":    dates[entry_idx],
                    "exit_date":     dates[i],
                    "net_return_pct": net_ret,
                    "hold_days":     hold_days,
                    "market_state":  mstate,
                    "risk_flags":    flags,
                    "forced":        False,
                })
                entry_idx = None

    if entry_idx is not None:
        adj_exit = c[-1] * (1.0 - COST_PER_SIDE)
        net_ret = (adj_exit / adj_entry - 1.0) * 100.0
        hold_days = (dates[-1] - dates[entry_idx]).days
        mstate, flags = _entry_labels(entry_idx, close, daily_rets, market_states, vix)
        trades.append({
            "entry_date":    dates[entry_idx],
            "exit_date":     dates[-1],
            "net_return_pct": net_ret,
            "hold_days":     hold_days,
            "market_state":  mstate,
            "risk_flags":    flags,
            "forced":        True,
        })
    return trades


# ── Strategy Functions ─────────────────────────────────────────────────────────
def _strat_rsi(close: pd.Series, market_states=None, vix=None) -> List[dict]:
    rsi = _rsi(close)
    return _simulate(close, rsi < 35, rsi > 65, market_states, vix)


def _strat_macd(close: pd.Series, market_states=None, vix=None) -> List[dict]:
    macd_line, signal = _macd(close)
    above = (macd_line > signal).astype(int)
    prev  = above.shift(1).fillna(0).astype(int)
    return _simulate(close, (above == 1) & (prev == 0), (above == 0) & (prev == 1), market_states, vix)


def _strat_boll(close: pd.Series, market_states=None, vix=None) -> List[dict]:
    upper, _, lower = _bollinger(close)
    return _simulate(close, close < lower, close > upper, market_states, vix)


def _strat_pullback(close: pd.Series, market_states=None, vix=None) -> List[dict]:
    """Buy: pullback >8% from 30-day high AND RSI<45. Exit: +15% gross or RSI>70."""
    rsi = _rsi(close)
    rolling_high = close.rolling(30, min_periods=5).max()
    pullback_pct  = (close / rolling_high - 1.0) * 100.0
    daily_rets    = close.pct_change()

    trades: List[dict] = []
    entry_idx: Optional[int] = None
    raw_entry: float = 0.0
    adj_entry: float = 0.0

    c    = close.values
    r    = rsi.values
    pb   = pullback_pct.values
    dates = close.index

    for i in range(len(c)):
        if np.isnan(c[i]):
            continue
        curr_rsi = r[i]
        curr_pb  = pb[i]

        if entry_idx is None:
            if (not np.isnan(curr_pb) and curr_pb < -8.0
                    and not np.isnan(curr_rsi) and curr_rsi < 45.0):
                raw_entry = c[i]
                adj_entry = c[i] * (1.0 + COST_PER_SIDE)
                entry_idx = i
        else:
            gross_gain = (c[i] / raw_entry - 1.0) * 100.0
            if gross_gain >= 15.0 or (not np.isnan(curr_rsi) and curr_rsi > 70.0):
                adj_exit = c[i] * (1.0 - COST_PER_SIDE)
                net_ret  = (adj_exit / adj_entry - 1.0) * 100.0
                hold_days = (dates[i] - dates[entry_idx]).days
                mstate, flags = _entry_labels(entry_idx, close, daily_rets, market_states, vix)
                trades.append({
                    "entry_date":    dates[entry_idx],
                    "exit_date":     dates[i],
                    "net_return_pct": net_ret,
                    "hold_days":     hold_days,
                    "market_state":  mstate,
                    "risk_flags":    flags,
                    "forced":        False,
                })
                entry_idx = None

    if entry_idx is not None:
        adj_exit  = c[-1] * (1.0 - COST_PER_SIDE)
        net_ret   = (adj_exit / adj_entry - 1.0) * 100.0
        hold_days = (dates[-1] - dates[entry_idx]).days
        mstate, flags = _entry_labels(entry_idx, close, daily_rets, market_states, vix)
        trades.append({
            "entry_date":    dates[entry_idx],
            "exit_date":     dates[-1],
            "net_return_pct": net_ret,
            "hold_days":     hold_days,
            "market_state":  mstate,
            "risk_flags":    flags,
            "forced":        True,
        })
    return trades


_STRAT_FUNCS: Dict[str, Callable] = {
    "rsi":      _strat_rsi,
    "macd":     _strat_macd,
    "boll":     _strat_boll,
    "pullback": _strat_pullback,
}


# ── Analytics ──────────────────────────────────────────────────────────────────
def _stress_win_rate(trades: List[dict], start: str, end: str) -> Optional[float]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sub = [t for t in trades if s <= t["entry_date"] <= e]
    if not sub:
        return None
    wins = sum(1 for t in sub if t["net_return_pct"] > 0)
    return round(wins / len(sub) * 100.0, 1)


def _oos_validation(trades: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    train_s, train_e = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    oos_s,   oos_e   = pd.Timestamp(OOS_START),   pd.Timestamp(OOS_END)
    train_t = [t for t in trades if train_s <= t["entry_date"] <= train_e]
    oos_t   = [t for t in trades if oos_s   <= t["entry_date"] <= oos_e]

    def _wr(ts):
        if not ts:
            return None
        return round(sum(1 for t in ts if t["net_return_pct"] > 0) / len(ts) * 100.0, 1)

    return _wr(train_t), _wr(oos_t)


def _metrics(trades: List[dict], spy_return: float) -> dict:
    EMPTY = {
        "trade_count": 0, "win_rate": None, "avg_return": None,
        "max_loss": None, "max_drawdown": None, "sharpe": None,
        "avg_hold_days": None, "max_hold_days": None,
        "hold_1d_pct": None, "hold_1w_pct": None,
        "hold_1m_pct": None, "hold_long_pct": None, "hold_style": None,
        "annualized_return": None, "vs_spy": None,
        "high_risk_count": 0, "state_counts": {},
    }
    if not trades:
        return EMPTY

    rets      = [t["net_return_pct"] for t in trades]
    hold_days = [max(t["hold_days"], 0) for t in trades]
    cnt       = len(trades)
    wins      = sum(1 for r in rets if r > 0)

    # Equity curve & max drawdown
    equity = [1.0]
    for r in rets:
        equity.append(equity[-1] * (1.0 + r / 100.0))
    running_max = max_dd = 0.0
    running_max = equity[0]
    for v in equity:
        if v > running_max:
            running_max = v
        dd = (running_max - v) / running_max if running_max > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    total_return = (equity[-1] - 1.0) * 100.0

    # Sharpe (trade-level, annualized)
    avg_hold = float(np.mean(hold_days)) if hold_days else 1.0
    if len(rets) > 1 and np.std(rets) > 1e-9:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1)
                       * np.sqrt(252.0 / max(avg_hold, 1.0)))
    else:
        sharpe = 0.0

    # Hold distribution
    h1d   = sum(1 for d in hold_days if d <= 1)
    h1w   = sum(1 for d in hold_days if 1 < d <= 7)
    h1m   = sum(1 for d in hold_days if 7 < d <= 30)
    hlong = sum(1 for d in hold_days if d > 30)

    if avg_hold <= 2:
        hold_style = "适合日内/超短线"
    elif avg_hold <= 7:
        hold_style = "适合短线"
    elif avg_hold <= 30:
        hold_style = "适合波段"
    else:
        hold_style = "不适合短线"

    # Market state breakdown
    state_counts: dict = {}
    for t in trades:
        ms = t.get("market_state", "unknown")
        state_counts[ms] = state_counts.get(ms, 0) + 1

    high_risk_cnt = sum(1 for t in trades if t.get("risk_flags"))

    return {
        "trade_count":      cnt,
        "win_rate":         round(wins / cnt * 100.0, 1),
        "avg_return":       round(float(np.mean(rets)), 2),
        "max_loss":         round(float(min(rets)), 2),
        "max_drawdown":     round(max_dd * 100.0, 1),
        "sharpe":           round(sharpe, 2),
        "avg_hold_days":    round(avg_hold, 1),
        "max_hold_days":    int(max(hold_days)),
        "hold_1d_pct":      round(h1d   / cnt * 100, 1),
        "hold_1w_pct":      round(h1w   / cnt * 100, 1),
        "hold_1m_pct":      round(h1m   / cnt * 100, 1),
        "hold_long_pct":    round(hlong / cnt * 100, 1),
        "hold_style":       hold_style,
        "annualized_return": round(total_return, 2),
        "vs_spy":           round(total_return - spy_return, 2),
        "high_risk_count":  high_risk_cnt,
        "state_counts":     state_counts,
    }


def _composite_score(m: dict) -> float:
    cnt = m.get("trade_count", 0)
    if cnt < MIN_DISPLAY:
        return 0.0

    wr       = m.get("win_rate",    0) or 0
    sharpe   = m.get("sharpe",      0) or 0
    bear_wr  = m.get("bear_win_rate")
    max_dd   = m.get("max_drawdown", 100) or 100

    # Win rate: 0-35 pts (50%→0, 85%→35)
    wr_score = min(max((wr - 50) / 35 * 35, 0), 35)

    # Sample size: 0-20 pts (40+ trades → full)
    sample_score = min(cnt / 40.0, 1.0) * 20

    # Bear performance: 0-25 pts (None→10, 30%→0, 80%→25)
    if bear_wr is not None:
        bear_score = min(max((bear_wr - 30) / 50 * 25, 0), 25)
    else:
        bear_score = 10.0  # neutral when no 2022 data

    # Sharpe: 0-15 pts (Sharpe 2.0 → 15)
    sharpe_score = min(max(sharpe / 2.0 * 15, 0), 15)

    # Max drawdown: 0-5 pts (DD=0%→5, DD=30%→0)
    dd_score = max(5.0 - max_dd / 6.0, 0)

    return round(min(wr_score + sample_score + bear_score + sharpe_score + dd_score, 100), 1)


def _verdict(m: dict) -> str:
    cnt      = m.get("trade_count", 0)
    score    = m.get("score", 0)
    bear_wr  = m.get("bear_win_rate")
    oos_wr   = m.get("oos_win_rate")

    if cnt < MIN_DISPLAY:
        return "数据不足"
    if cnt < MIN_RELIABLE:
        return f"样本不足⚠️ ({cnt}笔)"

    ok = (
        score > 70
        and (bear_wr is not None and bear_wr > 50)
        and cnt >= MIN_RELIABLE
        and (oos_wr is not None and oos_wr > 60)
    )
    return "✅值得交易" if ok else "❌不建议实盘"


def reliability_label(trade_count: int) -> str:
    if trade_count >= MIN_RELIABLE:
        return "高可信 ✅"
    if trade_count >= MIN_DISPLAY:
        return f"样本不足⚠️"
    return "低可信 ❌"


# ── Core Runner ────────────────────────────────────────────────────────────────
def run_full_backtest(
    tickers: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[dict], float, dict]:
    """
    Download 2020-present OHLCV data, run 4 strategies with full anti-overfitting analysis.
    Returns (results_list, spy_return_pct, current_market_state_dict).
    """
    if tickers is None:
        tickers = ALL_BACKTEST_TICKERS

    end_str = datetime.today().strftime("%Y-%m-%d")

    # ── Batch download ──────────────────────────────────────────────────────
    raw = yf.download(
        tickers,
        start=DATA_START,
        end=end_str,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # ── VIX ─────────────────────────────────────────────────────────────────
    vix_series: Optional[pd.Series] = None
    try:
        vix_raw = yf.download("^VIX", start=DATA_START, end=end_str,
                              auto_adjust=True, progress=False)
        if not vix_raw.empty:
            vc = vix_raw["Close"]
            vix_series = (vc.iloc[:, 0] if isinstance(vc, pd.DataFrame) else vc).dropna()
    except Exception:
        pass

    # ── SPY series & market state ────────────────────────────────────────────
    spy_close: pd.Series = pd.Series(dtype=float)
    spy_return = 0.0
    try:
        if hasattr(raw.columns, "levels") and "SPY" in raw.columns.get_level_values(1):
            spy_close = raw["Close"]["SPY"].dropna()
        elif "SPY" in tickers and not hasattr(raw.columns, "levels"):
            spy_close = raw["Close"].dropna()
        if len(spy_close) >= 2:
            spy_return = (spy_close.iloc[-1] / spy_close.iloc[0] - 1.0) * 100.0
    except Exception:
        pass

    market_states: Optional[pd.Series] = None
    current_state = {"state": "unknown", "emoji": "⚪", "label": "数据不足",
                     "spy_price": None, "ma200": None, "ratio_pct": None}
    if len(spy_close) >= 200:
        market_states = compute_market_states(spy_close)
        current_state = get_current_market_state(spy_close)

    # ── Per-ticker processing ────────────────────────────────────────────────
    results: List[dict] = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        try:
            if hasattr(raw.columns, "levels"):
                lvl1 = raw.columns.get_level_values(1)
                if ticker not in lvl1:
                    close = pd.Series(dtype=float)
                else:
                    close = raw["Close"][ticker].dropna()
            else:
                close = raw["Close"].dropna() if len(tickers) == 1 else pd.Series(dtype=float)

            if len(close) < 60:
                results.append({"ticker": ticker,
                                 "sector": _TICKER_SECTOR.get(ticker, "Unknown"),
                                 "error": "data_insufficient"})
                if progress_callback:
                    progress_callback(idx + 1, total, ticker)
                continue

            bah_return = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
            res: dict = {
                "ticker":     ticker,
                "sector":     _TICKER_SECTOR.get(ticker, "Unknown"),
                "bah_return": round(bah_return, 2),
                "spy_return": round(spy_return, 2),
                "error":      None,
            }

            for strat_id, strat_fn in _STRAT_FUNCS.items():
                try:
                    trades = strat_fn(close, market_states=market_states, vix=vix_series)
                    m = _metrics(trades, spy_return)

                    # Stress test win rates
                    for period_key, (ps, pe) in STRESS_PERIODS.items():
                        m[f"stress_{period_key}"] = _stress_win_rate(trades, ps, pe)

                    m["bear_win_rate"] = m.get("stress_2022年熊市")

                    # OOS validation
                    m["train_win_rate"], m["oos_win_rate"] = _oos_validation(trades)

                    # Score & verdict
                    m["score"]   = _composite_score(m)
                    m["verdict"] = _verdict(m)
                    m["reliability"] = reliability_label(m["trade_count"])

                    res[strat_id] = m

                except Exception as exc:
                    res[strat_id] = {"trade_count": 0, "verdict": "数据不足",
                                     "score": 0.0, "error": str(exc)}

            results.append(res)

        except Exception as exc:
            results.append({"ticker": ticker,
                             "sector": _TICKER_SECTOR.get(ticker, "Unknown"),
                             "error": str(exc)})

        if progress_callback:
            progress_callback(idx + 1, total, ticker)

    return results, spy_return, current_state


# ── DataFrame Builders ─────────────────────────────────────────────────────────
def build_flat_df(results: List[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        if res.get("error") and not any(k in res for k in _STRAT_FUNCS):
            continue
        ticker = res["ticker"]
        sector = res.get("sector", "Unknown")
        for strat_id in _STRAT_FUNCS:
            m = res.get(strat_id)
            if not m or m.get("trade_count", 0) < MIN_DISPLAY:
                continue
            stress_cols = {f"stress_{k}": m.get(f"stress_{k}") for k in STRESS_PERIODS}
            rows.append({
                "ticker":           ticker,
                "sector":           sector,
                "strategy":         strat_id,
                "strategy_name":    STRATEGY_NAMES[strat_id],
                "trade_count":      m["trade_count"],
                "win_rate":         m.get("win_rate"),
                "avg_return":       m.get("avg_return"),
                "max_loss":         m.get("max_loss"),
                "max_drawdown":     m.get("max_drawdown"),
                "sharpe":           m.get("sharpe"),
                "avg_hold_days":    m.get("avg_hold_days"),
                "max_hold_days":    m.get("max_hold_days"),
                "hold_1d_pct":      m.get("hold_1d_pct"),
                "hold_1w_pct":      m.get("hold_1w_pct"),
                "hold_1m_pct":      m.get("hold_1m_pct"),
                "hold_long_pct":    m.get("hold_long_pct"),
                "hold_style":       m.get("hold_style"),
                "annualized_return": m.get("annualized_return"),
                "vs_spy":           m.get("vs_spy"),
                "bah_return":       res.get("bah_return"),
                "spy_return":       res.get("spy_return"),
                "bear_win_rate":    m.get("bear_win_rate"),
                "train_win_rate":   m.get("train_win_rate"),
                "oos_win_rate":     m.get("oos_win_rate"),
                "score":            m.get("score"),
                "verdict":          m.get("verdict"),
                "reliability":      m.get("reliability"),
                "high_risk_count":  m.get("high_risk_count", 0),
                **stress_cols,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_stock_summary(results: List[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        if res.get("error") and not any(k in res for k in _STRAT_FUNCS):
            continue
        ticker = res["ticker"]
        sector = res.get("sector", "Unknown")
        best_strat, best_wr, best_ann, best_score = None, -1.0, None, 0.0
        strat_cells: dict = {}
        for strat_id in _STRAT_FUNCS:
            m = res.get(strat_id)
            if not m:
                strat_cells[strat_id + "_wr"] = None
                strat_cells[strat_id + "_cnt"] = 0
                continue
            wr  = m.get("win_rate")
            cnt = m.get("trade_count", 0)
            sc  = m.get("score", 0) or 0
            strat_cells[strat_id + "_wr"]  = round(wr, 1) if wr is not None else None
            strat_cells[strat_id + "_cnt"] = cnt
            if wr is not None and cnt >= MIN_DISPLAY and wr > best_wr:
                best_wr    = wr
                best_strat = STRATEGY_NAMES[strat_id]
                best_ann   = m.get("annualized_return")
                best_score = sc
        row = {
            "ticker":       ticker,
            "sector":       sector,
            "best_strategy": best_strat,
            "best_win_rate": round(best_wr, 1) if best_wr >= 0 else None,
            "best_ann_ret":  round(best_ann, 1) if best_ann is not None else None,
            "best_score":    best_score,
            "bah_return":   res.get("bah_return"),
            "spy_return":   res.get("spy_return"),
        }
        row.update(strat_cells)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_sector_stats(flat_df: pd.DataFrame) -> pd.DataFrame:
    if flat_df.empty:
        return pd.DataFrame()
    g = flat_df.dropna(subset=["win_rate"]).groupby("sector")
    stats = pd.DataFrame({
        "avg_win_rate":    g["win_rate"].mean().round(1),
        "avg_trade_count": g["trade_count"].mean().round(1),
        "avg_ann_return":  g["annualized_return"].mean().round(1),
        "avg_vs_spy":      g["vs_spy"].mean().round(1),
        "avg_score":       g["score"].mean().round(1),
        "stock_count":     flat_df.groupby("sector")["ticker"].nunique(),
    }).reset_index().sort_values("avg_win_rate", ascending=False)
    return stats


def get_strategy_stats(flat_df: pd.DataFrame) -> pd.DataFrame:
    if flat_df.empty:
        return pd.DataFrame()
    g = flat_df.dropna(subset=["win_rate"]).groupby("strategy_name")
    stats = pd.DataFrame({
        "avg_win_rate":    g["win_rate"].mean().round(1),
        "avg_trade_count": g["trade_count"].mean().round(1),
        "avg_ann_return":  g["annualized_return"].mean().round(1),
        "avg_vs_spy":      g["vs_spy"].mean().round(1),
        "avg_score":       g["score"].mean().round(1),
        "total_signals":   g["trade_count"].sum(),
    }).reset_index().sort_values("avg_win_rate", ascending=False)
    return stats


def get_best_trades(flat_df: pd.DataFrame, spy_return: float, n: int = 10) -> pd.DataFrame:
    """Return top-N by score where verdict == ✅值得交易."""
    if flat_df.empty:
        return pd.DataFrame()
    df = flat_df[flat_df["verdict"] == "✅值得交易"].copy()
    if df.empty:
        return pd.DataFrame()
    return (
        df.nlargest(n, "score")
        .drop_duplicates(subset="ticker")
        .reset_index(drop=True)
    )


def get_top_recommendations(flat_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Broader filter: score >= 50, win_rate >= 55."""
    if flat_df.empty:
        return pd.DataFrame()
    df = flat_df.dropna(subset=["win_rate", "score"]).copy()
    df = df[(df["win_rate"] >= 55) & (df["score"] >= 50)]
    if df.empty:
        return pd.DataFrame()
    return (
        df.nlargest(n, "score")
        .drop_duplicates(subset="ticker")
        .reset_index(drop=True)
    )


def get_sector_top5(flat_df: pd.DataFrame, min_trades: int = MIN_DISPLAY) -> pd.DataFrame:
    if flat_df.empty:
        return pd.DataFrame()
    filtered = flat_df[flat_df["trade_count"] >= min_trades].dropna(subset=["win_rate"]).copy()
    if filtered.empty:
        return pd.DataFrame()
    return (
        filtered.sort_values("win_rate", ascending=False)
        .groupby("sector", sort=False)
        .head(5)
        .reset_index(drop=True)
    )
