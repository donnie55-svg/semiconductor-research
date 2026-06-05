"""
backtest_pro.py — Pro Backtest Engine v2
训练期: 2022-2023  验证期: 2024-2025
4种短线策略 + 动态门槛 + Kelly仓位 + Sortino + Profit Factor
完整7条推荐门槛 + 100分制评级
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
COST_PER_SIDE  = 0.0015       # 0.10% 手续费 + 0.05% 滑点
DATA_START     = "2019-01-01" # 覆盖2020疫情压力测试 + 200MA预热
TRAIN_START    = "2022-01-01"
TRAIN_END      = "2023-12-31"
OOS_START      = "2024-01-01"
OOS_END        = "2025-05-01"
VIX_HIGH       = 30.0
BIG_DROP_1D    = 0.08         # 单日跌幅 >8% → 极端标记

STRESS_PERIODS: Dict[str, Tuple[str, str]] = {
    "A_2022熊市":    ("2022-01-01", "2022-12-31"),
    "B_2020疫情崩盘": ("2020-02-20", "2020-05-31"),
    "C_2023银行危机": ("2023-03-01", "2023-05-31"),
}

# ── 动态交易次数门槛 ───────────────────────────────────────────────────────────
VOL_HIGH_THRESH = 0.50   # 年化波动率 >50% → 需要25笔
VOL_MED_THRESH  = 0.30   # 年化波动率 30-50% → 需要15笔
MIN_DISPLAY_RATIO = 0.50 # 低于门槛50% → 不显示

def get_min_trades(annual_vol: float) -> int:
    if annual_vol > VOL_HIGH_THRESH: return 25
    if annual_vol > VOL_MED_THRESH:  return 15
    return 10

def get_vol_type(annual_vol: float) -> str:
    if annual_vol > VOL_HIGH_THRESH: return "高波动(>50%)"
    if annual_vol > VOL_MED_THRESH:  return "中波动(30~50%)"
    return "低波动(<30%)"

def compute_annual_vol(close: pd.Series) -> float:
    if len(close) < 20: return 0.5
    return float(close.pct_change().dropna().std() * np.sqrt(252))

# ── Stock Universe ─────────────────────────────────────────────────────────────
BACKTEST_SECTORS: Dict[str, List[str]] = {
    "AI算力/GPU/芯片": [
        "NVDA","MU","ARM","QCOM","INTC","ADI","TXN","NXPI","ON","STM",
        "MCHP","LSCC","ALGM","NVTS","MPWR","ALAB","CRDO","RMBS",
        "SITM","VICR","MTSI","SKYT","TSEM","GFS",
    ],
    "半导体设备/材料": [
        "ASML","AMAT","KLAC","LRCX","ONTO","NVMI","ACLS","FORM",
        "ACMR","MKSI","CAMT","VECO","COHR","LASR","IPGP","COHU",
        "TER","DIOD","LFUS","AEIS",
    ],
    "AI服务器/数据中心/电力": [
        "VRT","ETN","NVT","ENS","POWL","HUBB","GEV","VST","CEG","BE","CAT","CMI",
    ],
    "存储/HBM": ["MU","WDC","STX","SIMO","RMBS"],
    "AI网络/光模块": ["ANET","CSCO","FTNT","ERIC","NOK","LITE","COHR","IPGP"],
    "数据中心EMS": ["JBL","FLEX","SANM","PLXS","TTMI","FN"],
    "AI云/软件": [
        "CRWV","DOCN","APLD","SOUN","NBIS","IONQ","MSFT","GOOGL",
        "META","AMZN","CRM","NOW","SNOW","MDB","PLTR","AI","PATH",
        "DDOG","NET","ZS","CRWD","PANW",
    ],
    "核能/新能源": ["CCJ","BWXT","NNE","FSLR","PLUG","FCEL","BLDP","VST","CEG","ENPH"],
    "航天/卫星/国防": [
        "RKLB","ASTS","PL","BKSY","ESLT","LMT","BA","GE","CW",
        "ATRO","VSAT","KTOS","RDW","LUNR","NOC","RTX",
    ],
    "工业自动化": ["ST","CTS","CGNX","NDSN","PPG","MMM","EMR","GLW"],
    "原材料/铜/能源": ["SCCO","FCX","OXY","DOW","CC","KALU","XOM","CVX"],
    "医药/消费/金融": ["KO","PG","MRK","AMGN","JPM","BAC","MA","MCO","LLY","NVO","MRNA","VRTX"],
    "航空&旅游": [
        "AAL","DAL","UAL","LUV","ALK","CCL","RCL","NCLH",
        "ABNB","BKNG","DIS","LVS","WYNN","MGM",
    ],
    "中概股ADR": ["BABA","PDD","JD","BIDU","NIO","XPEV","LI","TCOM","FUTU"],
    "太空/SpaceX概念": ["ASTS","RKLB","LUNR","RDW","KTOS","SPCE","VSAT"],
    "ETF/指数": ["SMH","SOXX","XSD","DRAM","SPY","QQQ","EWT"],
}

ALL_BACKTEST_TICKERS: List[str] = list(dict.fromkeys(
    t for tickers in BACKTEST_SECTORS.values() for t in tickers
))

STRATEGY_NAMES: Dict[str, str] = {
    "rsi_oversold":   "RSI超卖反弹",
    "boll_lower":     "布林带下轨反弹",
    "rsi_boll_combo": "RSI+布林带复合",
    "dip_bounce":     "急跌反弹",
}

_TICKER_SECTOR: Dict[str, str] = {}
for _sec, _tks in BACKTEST_SECTORS.items():
    for _t in _tks:
        _TICKER_SECTOR.setdefault(_t, _sec)


# ── Technical Indicators ───────────────────────────────────────────────────────
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


def _n_day_drop(close: pd.Series, n: int = 3) -> pd.Series:
    """Positive value = price dropped by that fraction over n days."""
    return (1.0 - close / close.shift(n)).clip(lower=0)


# ── Market State (两层机制) ────────────────────────────────────────────────────
def compute_market_states(spy_close: pd.Series) -> pd.Series:
    """
    第一层：统计层 — 历史市场状态，用于计算牛/熊胜率
    Bull: SPY/200MA > 1.05
    Sideways: |SPY/200MA - 1| ≤ 0.05
    Bear: SPY/200MA < 0.95
    """
    ma200  = spy_close.rolling(200, min_periods=100).mean()
    ratio  = spy_close / ma200 - 1.0
    states = pd.Series("unknown", index=spy_close.index, dtype=object)
    states[ratio > 0.05]                           = "bull"
    states[(ratio >= -0.05) & (ratio <= 0.05)]     = "sideways"
    states[ratio < -0.05]                          = "bear"
    states[ma200.isna()]                           = "unknown"
    return states


def get_current_market_state(spy_close: pd.Series) -> dict:
    """第二层：实盘执行层 — 当前市场状态 + 仓位建议"""
    if spy_close.empty:
        return {"state":"unknown","emoji":"⚪","label":"数据不可用",
                "spy_price":None,"ma200":None,"ratio_pct":None,"pos_advice":"N/A"}
    ma200_val = spy_close.rolling(200, min_periods=100).mean().iloc[-1]
    spy_price = float(spy_close.iloc[-1])
    if pd.isna(ma200_val):
        return {"state":"unknown","emoji":"⚪","label":"数据不足",
                "spy_price":spy_price,"ma200":None,"ratio_pct":None,"pos_advice":"N/A"}
    ratio = spy_price / float(ma200_val) - 1.0
    ratio_pct = round(ratio * 100, 2)
    if ratio > 0.05:
        return {"state":"bull","emoji":"🟢","label":"牛市-全部策略开启",
                "spy_price":spy_price,"ma200":round(float(ma200_val),2),
                "ratio_pct":ratio_pct,"pos_advice":"标准仓位"}
    elif ratio >= -0.05:
        return {"state":"sideways","emoji":"🟡","label":"震荡-仅策略3/仓位×50%",
                "spy_price":spy_price,"ma200":round(float(ma200_val),2),
                "ratio_pct":ratio_pct,"pos_advice":"减半仓位"}
    else:
        return {"state":"bear","emoji":"🔴","label":"熊市-仅观察模式⚠️",
                "spy_price":spy_price,"ma200":round(float(ma200_val),2),
                "ratio_pct":ratio_pct,"pos_advice":"建议仓位0%"}


# ── Extreme Filter ─────────────────────────────────────────────────────────────
def _get_entry_flags(
    entry_idx: int,
    close: pd.Series,
    daily_rets: pd.Series,
    market_states: Optional[pd.Series],
    vix: Optional[pd.Series],
) -> Tuple[str, List[str]]:
    """返回 (market_state, extreme_flags)"""
    entry_dt = close.index[entry_idx]
    mstate = "unknown"
    if market_states is not None:
        try:
            ms = market_states.asof(entry_dt)
            if ms and str(ms) not in ("nan","None"):
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

    # 前3天单日跌幅 >8%
    try:
        pre = daily_rets.iloc[max(0, entry_idx-3):entry_idx]
        if (pre < -BIG_DROP_1D).any():
            flags.append("after_crash")
    except Exception:
        pass

    # 近3天±内 >5%单日大幅波动 (财报代理)
    try:
        win = daily_rets.iloc[max(0, entry_idx-3):min(len(daily_rets), entry_idx+4)]
        if (win.abs() > 0.05).any():
            flags.append("near_earnings")
    except Exception:
        pass

    return mstate, flags


# ── Unified Trade Simulator ────────────────────────────────────────────────────
def _simulate(
    close: pd.Series,
    buy_signal: pd.Series,
    sell_signal: Optional[pd.Series],
    stop_pct: float,
    profit_target_pct: Optional[float] = None,
    market_states: Optional[pd.Series] = None,
    vix: Optional[pd.Series] = None,
) -> List[dict]:
    """
    多空一头、一次一仓、含止损/止盈/信号平仓的完整模拟器
    """
    trades: List[dict] = []
    daily_rets = close.pct_change()
    c = close.values
    b = buy_signal.fillna(False).values.astype(bool)
    s = sell_signal.fillna(False).values.astype(bool) if sell_signal is not None else np.zeros(len(c), bool)

    in_pos      = False
    raw_entry   = 0.0
    adj_entry   = 0.0
    entry_idx   = 0
    dates       = close.index

    for i in range(len(c)):
        if np.isnan(c[i]):
            continue

        if not in_pos:
            if b[i]:
                raw_entry = c[i]
                adj_entry = c[i] * (1.0 + COST_PER_SIDE)
                entry_idx = i
                in_pos    = True
        else:
            triggered = False
            exit_type = "signal"

            # 止损
            if c[i] <= raw_entry * (1.0 - stop_pct):
                triggered = True
                exit_type = "stop"
            # 止盈
            elif profit_target_pct and c[i] >= raw_entry * (1.0 + profit_target_pct):
                triggered = True
                exit_type = "profit_target"
            # 信号平仓
            elif s[i]:
                triggered = True
                exit_type = "signal"

            if triggered:
                adj_exit   = c[i] * (1.0 - COST_PER_SIDE)
                net_ret    = (adj_exit / adj_entry - 1.0) * 100.0
                hold_days  = int((dates[i] - dates[entry_idx]).days)
                mstate, flags = _get_entry_flags(entry_idx, close, daily_rets, market_states, vix)
                trades.append({
                    "entry_date":    dates[entry_idx],
                    "exit_date":     dates[i],
                    "net_return_pct": net_ret,
                    "hold_days":     hold_days,
                    "market_state":  mstate,
                    "risk_flags":    flags,
                    "exit_type":     exit_type,
                    "forced":        False,
                })
                in_pos = False

    # 强制平仓未结仓位
    if in_pos:
        adj_exit   = c[-1] * (1.0 - COST_PER_SIDE)
        net_ret    = (adj_exit / adj_entry - 1.0) * 100.0
        hold_days  = int((dates[-1] - dates[entry_idx]).days)
        mstate, flags = _get_entry_flags(entry_idx, close, daily_rets, market_states, vix)
        trades.append({
            "entry_date":    dates[entry_idx],
            "exit_date":     dates[-1],
            "net_return_pct": net_ret,
            "hold_days":     hold_days,
            "market_state":  mstate,
            "risk_flags":    flags,
            "exit_type":     "forced",
            "forced":        True,
        })
    return trades


# ── Strategy Functions ─────────────────────────────────────────────────────────
def _strat_rsi_oversold(close, market_states=None, vix=None) -> List[dict]:
    """策略1: RSI(5)<25买入, RSI(5)>60卖出, 止损-8%"""
    rsi = _rsi(close, 5)
    return _simulate(close, rsi < 25, rsi > 60, 0.08, None, market_states, vix)


def _strat_boll_lower(close, market_states=None, vix=None) -> List[dict]:
    """策略2: 跌破布林带下轨买入, 回到中轨卖出, 止损-8%"""
    _, mid, lower = _bollinger(close)
    return _simulate(close, close < lower, close >= mid, 0.08, None, market_states, vix)


def _strat_rsi_boll_combo(close, market_states=None, vix=None) -> List[dict]:
    """策略3: RSI(5)<30 AND 低于下轨买入, 回到中轨OR盈利8%卖出, 止损-6%"""
    rsi = _rsi(close, 5)
    _, mid, lower = _bollinger(close)
    buy = (rsi < 30) & (close < lower)
    sell = close >= mid
    return _simulate(close, buy, sell, 0.06, 0.08, market_states, vix)


def _strat_dip_bounce(close, market_states=None, vix=None) -> List[dict]:
    """策略4: 3日跌幅>10% AND RSI(5)<30买入, 盈利10%卖出, 止损-8%"""
    rsi   = _rsi(close, 5)
    drop3 = _n_day_drop(close, 3)
    buy   = (drop3 > 0.10) & (rsi < 30)
    return _simulate(close, buy, None, 0.08, 0.10, market_states, vix)


_STRAT_FUNCS: Dict[str, Callable] = {
    "rsi_oversold":   _strat_rsi_oversold,
    "boll_lower":     _strat_boll_lower,
    "rsi_boll_combo": _strat_rsi_boll_combo,
    "dip_bounce":     _strat_dip_bounce,
}


# ── Analytics ──────────────────────────────────────────────────────────────────
def _period_win_rate(trades: List[dict], start: str, end: str) -> Optional[float]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sub  = [t for t in trades if s <= t["entry_date"] <= e]
    if not sub: return None
    wins = sum(1 for t in sub if t["net_return_pct"] > 0)
    return round(wins / len(sub) * 100.0, 1)


def _split_win_rates(trades: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    train = _period_win_rate(trades, TRAIN_START, TRAIN_END)
    oos   = _period_win_rate(trades, OOS_START, OOS_END)
    return train, oos


def _market_state_win_rate(trades: List[dict], state: str) -> Optional[float]:
    sub = [t for t in trades if t.get("market_state") == state]
    if not sub: return None
    wins = sum(1 for t in sub if t["net_return_pct"] > 0)
    return round(wins / len(sub) * 100.0, 1)


def _metrics(trades: List[dict], annual_vol: float) -> dict:
    if not trades:
        return {"trade_count": 0}

    rets      = [t["net_return_pct"] for t in trades]
    hold_days = [max(t["hold_days"], 0) for t in trades]
    cnt       = len(trades)
    wins      = [r for r in rets if r > 0]
    losses    = [r for r in rets if r <= 0]
    win_cnt   = len(wins)

    # ── Equity curve & max drawdown ──────────────────────────────────────────
    equity      = [1.0]
    running_max = 1.0
    max_dd      = 0.0
    for r in rets:
        equity.append(equity[-1] * (1.0 + r / 100.0))
    for v in equity:
        if v > running_max: running_max = v
        dd = (running_max - v) / running_max if running_max > 0 else 0
        if dd > max_dd: max_dd = dd

    total_return = (equity[-1] - 1.0) * 100.0
    avg_hold     = float(np.mean(hold_days)) if hold_days else 1.0

    # ── Sharpe ───────────────────────────────────────────────────────────────
    if len(rets) > 1 and np.std(rets) > 1e-9:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1)
                       * np.sqrt(252.0 / max(avg_hold, 1.0)))
    else:
        sharpe = 0.0

    # ── Sortino ──────────────────────────────────────────────────────────────
    neg_rets = [r for r in rets if r < 0]
    if neg_rets and len(neg_rets) > 1:
        down_std = float(np.std(neg_rets, ddof=1))
        sortino  = float(np.mean(rets) / down_std * np.sqrt(252.0 / max(avg_hold, 1.0)))
    else:
        sortino = sharpe * 1.5  # 无下行波动时估算

    # ── Profit Factor ────────────────────────────────────────────────────────
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    # ── Kelly Fraction ───────────────────────────────────────────────────────
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = float(np.mean([abs(l) for l in losses])) if losses else 1e-9
    wr_dec   = win_cnt / cnt
    if avg_loss > 1e-9 and avg_win > 1e-9:
        b = avg_win / avg_loss
        kelly = round((b * wr_dec - (1 - wr_dec)) / b * 100, 1)
        kelly = max(kelly, 0.0)
    else:
        kelly = 0.0

    # ── Hold distribution ────────────────────────────────────────────────────
    h1d   = sum(1 for d in hold_days if d <= 1)
    h1w   = sum(1 for d in hold_days if 1 < d <= 7)
    h1m   = sum(1 for d in hold_days if 7 < d <= 30)
    hlong = sum(1 for d in hold_days if d > 30)

    if avg_hold < 3:    hold_style = "日内/超短线"
    elif avg_hold < 10: hold_style = "短线波段"
    elif avg_hold < 30: hold_style = "波段"
    else:               hold_style = "不适合短线"

    # ── Market state breakdown ───────────────────────────────────────────────
    high_risk_cnt = sum(1 for t in trades if t.get("risk_flags"))

    return {
        "trade_count":      cnt,
        "win_rate":         round(win_cnt / cnt * 100.0, 1),
        "avg_return":       round(float(np.mean(rets)), 2),
        "max_loss":         round(float(min(rets)), 2),
        "max_drawdown":     round(max_dd * 100.0, 1),
        "sharpe":           round(sharpe, 2),
        "sortino":          round(sortino, 2),
        "profit_factor":    profit_factor,
        "kelly":            kelly,
        "avg_hold_days":    round(avg_hold, 1),
        "max_hold_days":    int(max(hold_days)),
        "hold_1d_pct":      round(h1d  / cnt * 100, 1),
        "hold_1w_pct":      round(h1w  / cnt * 100, 1),
        "hold_1m_pct":      round(h1m  / cnt * 100, 1),
        "hold_long_pct":    round(hlong / cnt * 100, 1),
        "hold_style":       hold_style,
        "total_return":     round(total_return, 2),
        "high_risk_count":  high_risk_cnt,
    }


def _compute_all_metrics(trades: List[dict], annual_vol: float) -> dict:
    m = _metrics(trades, annual_vol)
    if m["trade_count"] == 0:
        return m

    # Stress periods
    for pk, (ps, pe) in STRESS_PERIODS.items():
        m[f"stress_{pk}"] = _period_win_rate(trades, ps, pe)
    m["bear_win_rate"]    = m.get("stress_A_2022熊市")
    m["bull_win_rate"]    = _market_state_win_rate(trades, "bull")
    m["sideways_win_rate"] = _market_state_win_rate(trades, "sideways")

    # Train / OOS split
    m["train_win_rate"], m["oos_win_rate"] = _split_win_rates(trades)

    # Dynamic threshold
    min_trades = get_min_trades(annual_vol)
    m["min_required_trades"] = min_trades
    m["vol_type"]            = get_vol_type(annual_vol)
    m["annual_vol_pct"]      = round(annual_vol * 100, 1)

    # Sample adequacy label
    cnt = m["trade_count"]
    if cnt < min_trades * MIN_DISPLAY_RATIO:
        m["sample_label"] = "不显示"
    elif cnt < min_trades:
        pct = round(cnt / min_trades * 100)
        m["sample_label"] = f"⚠️样本不足 (当前{cnt}笔/需要{min_trades}笔/可信度{pct}%)"
    else:
        m["sample_label"] = f"✅充足 ({cnt}笔≥{min_trades}笔)"

    # Kelly position sizing
    k = m.get("kelly", 0) or 0
    m["kelly_conservative"] = round(min(k * 0.25, 15.0), 1)
    m["kelly_standard"]     = round(min(k * 0.50, 15.0), 1)
    m["kelly_aggressive"]   = round(min(k * 1.00, 15.0), 1)

    # Composite score
    m["score"] = _composite_score(m)
    m["grade"] = _grade(m["score"])

    # 7-gate verdict
    m["verdict"]        = _verdict(m)
    m["verdict_detail"] = _verdict_detail(m)

    return m


# ── Scoring System (100分制) ──────────────────────────────────────────────────
def _composite_score(m: dict) -> float:
    cnt       = m.get("trade_count", 0)
    min_req   = m.get("min_required_trades", 25)
    sharpe    = m.get("sharpe", 0) or 0
    bear_wr   = m.get("bear_win_rate")
    train_wr  = m.get("train_win_rate")
    oos_wr    = m.get("oos_win_rate")
    max_dd    = m.get("max_drawdown", 100) or 100

    # 1. 样本量充足度 20分
    sample_score = min(cnt / max(min_req, 1), 1.0) * 20.0

    # 2. 夏普比率 25分 (Sharpe 2.0 → 25分)
    sharpe_score = min(max(sharpe / 2.0, 0), 1.0) * 25.0

    # 3. 熊市表现 20分 (50%→0, 80%→20)
    if bear_wr is not None:
        bear_score = min(max((bear_wr - 50) / 30, 0), 1.0) * 20.0
    else:
        bear_score = 8.0  # 无数据时中性

    # 4. 样本外验证一致性 25分
    if train_wr and oos_wr:
        oos_base    = min(max(oos_wr / 70, 0), 1.0)
        consistency = max(1.0 - abs(oos_wr - train_wr) / 30.0, 0.0)
        oos_score   = oos_base * consistency * 25.0
    else:
        oos_score = 0.0

    # 5. 最大回撤控制 10分 (DD=0%→10, DD=25%→0)
    dd_score = max(10.0 - max_dd / 2.5, 0.0)

    return round(min(sample_score + sharpe_score + bear_score + oos_score + dd_score, 100.0), 1)


def _grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 75: return "A"
    if score >= 60: return "B"
    if score >= 40: return "C"
    return "D"


def _verdict(m: dict) -> str:
    cnt      = m.get("trade_count", 0)
    min_req  = m.get("min_required_trades", 25)
    if cnt < min_req * MIN_DISPLAY_RATIO:
        return "数据不足"
    gates = [
        m.get("score", 0) >= 70,
        cnt >= min_req,
        (m.get("sharpe") or 0) >= 1.0,
        (m.get("profit_factor") or 0) >= 1.5,
        (m.get("bear_win_rate") or 0) >= 50,
        (m.get("max_drawdown") or 100) <= 25,
        (m.get("oos_win_rate") or 0) >= 55,
    ]
    return "✅值得实盘测试" if all(gates) else "❌不建议实盘"


def _verdict_detail(m: dict) -> List[str]:
    """返回7条门槛的逐条检查结果"""
    details = [
        ("评分≥70",         m.get("score",0) >= 70,            f"{m.get('score',0):.1f}"),
        ("达到动态门槛",     m.get("trade_count",0) >= m.get("min_required_trades",25),
         f"{m.get('trade_count',0)}/{m.get('min_required_trades',25)}"),
        ("夏普≥1.0",        (m.get("sharpe") or 0) >= 1.0,     f"{m.get('sharpe',0):.2f}"),
        ("PF≥1.5",          (m.get("profit_factor") or 0) >= 1.5, f"{m.get('profit_factor',0):.2f}"),
        ("熊市胜率≥50%",    (m.get("bear_win_rate") or 0) >= 50, f"{m.get('bear_win_rate','N/A')}"),
        ("最大回撤≤25%",    (m.get("max_drawdown") or 100) <= 25, f"{m.get('max_drawdown',0):.1f}%"),
        ("验证期胜率≥55%",  (m.get("oos_win_rate") or 0) >= 55, f"{m.get('oos_win_rate','N/A')}"),
    ]
    return [f"{'✅' if ok else '❌'} {name}：{val}" for name, ok, val in details]


# ── Core Runner ────────────────────────────────────────────────────────────────
def run_full_backtest(
    tickers: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[dict], float, dict]:
    """
    下载2019-至今OHLCV数据，跑4种策略+完整防过拟合分析。
    返回 (results_list, spy_return_pct, current_market_state_dict)
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
    current_state = {"state":"unknown","emoji":"⚪","label":"数据不足",
                     "spy_price":None,"ma200":None,"ratio_pct":None,"pos_advice":"N/A"}
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
                                 "sector": _TICKER_SECTOR.get(ticker,"Unknown"),
                                 "error": "data_insufficient"})
                if progress_callback:
                    progress_callback(idx + 1, total, ticker)
                continue

            annual_vol = compute_annual_vol(close)
            bah_return = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
            res: dict = {
                "ticker":     ticker,
                "sector":     _TICKER_SECTOR.get(ticker, "Unknown"),
                "bah_return": round(bah_return, 2),
                "spy_return": round(spy_return, 2),
                "annual_vol": round(annual_vol * 100, 1),
                "vol_type":   get_vol_type(annual_vol),
                "error":      None,
            }

            for strat_id, strat_fn in _STRAT_FUNCS.items():
                try:
                    trades = strat_fn(close, market_states=market_states, vix=vix_series)
                    m = _compute_all_metrics(trades, annual_vol)
                    if m.get("sample_label") == "不显示":
                        res[strat_id] = {"trade_count": 0, "sample_label": "不显示",
                                         "verdict": "数据不足", "score": 0.0}
                    else:
                        res[strat_id] = m
                except Exception as exc:
                    res[strat_id] = {"trade_count": 0, "verdict": "数据不足",
                                     "score": 0.0, "error": str(exc)}

            results.append(res)

        except Exception as exc:
            results.append({"ticker": ticker,
                             "sector": _TICKER_SECTOR.get(ticker,"Unknown"),
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
        ticker  = res["ticker"]
        sector  = res.get("sector", "Unknown")
        vol_pct = res.get("annual_vol", 0)
        vol_type = res.get("vol_type", "N/A")

        for strat_id in _STRAT_FUNCS:
            m = res.get(strat_id)
            if not m or m.get("trade_count", 0) == 0:
                continue
            if m.get("sample_label") == "不显示":
                continue
            row = {
                "ticker":            ticker,
                "sector":            sector,
                "strategy":          strat_id,
                "strategy_name":     STRATEGY_NAMES[strat_id],
                "annual_vol_pct":    vol_pct,
                "vol_type":          vol_type,
                "trade_count":       m.get("trade_count"),
                "min_required":      m.get("min_required_trades"),
                "sample_label":      m.get("sample_label"),
                "win_rate":          m.get("win_rate"),
                "avg_return":        m.get("avg_return"),
                "max_loss":          m.get("max_loss"),
                "max_drawdown":      m.get("max_drawdown"),
                "profit_factor":     m.get("profit_factor"),
                "sharpe":            m.get("sharpe"),
                "sortino":           m.get("sortino"),
                "kelly":             m.get("kelly"),
                "kelly_conservative": m.get("kelly_conservative"),
                "kelly_standard":    m.get("kelly_standard"),
                "bull_win_rate":     m.get("bull_win_rate"),
                "bear_win_rate":     m.get("bear_win_rate"),
                "sideways_win_rate": m.get("sideways_win_rate"),
                "train_win_rate":    m.get("train_win_rate"),
                "oos_win_rate":      m.get("oos_win_rate"),
                "avg_hold_days":     m.get("avg_hold_days"),
                "hold_style":        m.get("hold_style"),
                "score":             m.get("score"),
                "grade":             m.get("grade"),
                "verdict":           m.get("verdict"),
                "verdict_detail":    m.get("verdict_detail", []),
                "high_risk_count":   m.get("high_risk_count", 0),
                "bah_return":        res.get("bah_return"),
            }
            # Stress test columns
            for pk in STRESS_PERIODS:
                row[f"stress_{pk}"] = m.get(f"stress_{pk}")
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_stock_summary(results: List[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        if res.get("error") and not any(k in res for k in _STRAT_FUNCS):
            continue
        ticker = res["ticker"]
        sector = res.get("sector", "Unknown")
        best_strat, best_wr, best_score = None, -1.0, 0.0
        strat_cells: dict = {}
        for strat_id in _STRAT_FUNCS:
            m = res.get(strat_id)
            if not m:
                strat_cells[strat_id+"_wr"]  = None
                strat_cells[strat_id+"_cnt"] = 0
                continue
            wr  = m.get("win_rate")
            cnt = m.get("trade_count", 0)
            sc  = m.get("score", 0) or 0
            strat_cells[strat_id+"_wr"]  = round(wr, 1) if wr is not None else None
            strat_cells[strat_id+"_cnt"] = cnt
            if wr is not None and cnt > 0 and wr > best_wr:
                best_wr    = wr
                best_strat = STRATEGY_NAMES[strat_id]
                best_score = sc
        row = {
            "ticker":        ticker,
            "sector":        sector,
            "vol_type":      res.get("vol_type",""),
            "best_strategy": best_strat,
            "best_win_rate": round(best_wr,1) if best_wr >= 0 else None,
            "best_score":    best_score,
            "bah_return":    res.get("bah_return"),
        }
        row.update(strat_cells)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_sector_stats(flat_df: pd.DataFrame) -> pd.DataFrame:
    if flat_df.empty: return pd.DataFrame()
    g = flat_df.dropna(subset=["win_rate"]).groupby("sector")
    return pd.DataFrame({
        "avg_win_rate":    g["win_rate"].mean().round(1),
        "avg_trade_count": g["trade_count"].mean().round(1),
        "avg_sharpe":      g["sharpe"].mean().round(2),
        "avg_profit_factor": g["profit_factor"].mean().round(2),
        "avg_score":       g["score"].mean().round(1),
        "stock_count":     flat_df.groupby("sector")["ticker"].nunique(),
    }).reset_index().sort_values("avg_win_rate", ascending=False)


def get_strategy_stats(flat_df: pd.DataFrame) -> pd.DataFrame:
    if flat_df.empty: return pd.DataFrame()
    g = flat_df.dropna(subset=["win_rate"]).groupby("strategy_name")
    return pd.DataFrame({
        "avg_win_rate":    g["win_rate"].mean().round(1),
        "avg_trade_count": g["trade_count"].mean().round(1),
        "avg_sharpe":      g["sharpe"].mean().round(2),
        "avg_profit_factor": g["profit_factor"].mean().round(2),
        "avg_score":       g["score"].mean().round(1),
        "total_signals":   g["trade_count"].sum(),
    }).reset_index().sort_values("avg_win_rate", ascending=False)


def get_top_recommendations(flat_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if flat_df.empty: return pd.DataFrame()
    df = flat_df[flat_df["verdict"] == "✅值得实盘测试"].copy()
    if df.empty:
        df = flat_df.dropna(subset=["score","win_rate"]).copy()
        df = df[(df["win_rate"] >= 55) & (df["score"] >= 50)]
    if df.empty: return pd.DataFrame()
    return df.nlargest(n, "score").reset_index(drop=True)


def get_best_trades(flat_df: pd.DataFrame, spy_return: float, n: int = 10) -> pd.DataFrame:
    if flat_df.empty: return pd.DataFrame()
    df = flat_df[flat_df["verdict"] == "✅值得实盘测试"].copy()
    if df.empty: return pd.DataFrame()
    return df.nlargest(n, "score").drop_duplicates(subset="ticker").reset_index(drop=True)


def get_sector_top5(flat_df: pd.DataFrame, min_trades: int = 5) -> pd.DataFrame:
    if flat_df.empty: return pd.DataFrame()
    filtered = flat_df[flat_df["trade_count"] >= min_trades].dropna(subset=["win_rate"]).copy()
    if filtered.empty: return pd.DataFrame()
    return (filtered.sort_values("win_rate", ascending=False)
            .groupby("sector", sort=False).head(5)
            .reset_index(drop=True))


def reliability_label(count: int, min_req: int = 15) -> str:
    if count >= min_req:    return "高可信 ✅"
    if count >= min_req//2: return "样本不足⚠️"
    return "低可信 ❌"
