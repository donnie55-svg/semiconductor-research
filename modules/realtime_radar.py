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
    "AI云/软件":        ["MSFT","GOOGL","META","AMZN","PLTR","CRWD","NET","SNOW","DDOG","NOW","ZS","PANW","WDAY"],
    "存储/HBM":         ["MU","WDC","STX"],
    "AI网络":           ["ANET","CSCO","LITE","COHR"],
    "核能/新能源":      ["CCJ","NNE","FSLR","VST"],
    "航天/国防":        ["RKLB","ASTS","LMT","RTX"],
    "太空":             ["ASTS","RKLB","LUNR","PL","IRDM","BKSY"],
    "中概股":           ["BABA","PDD","JD","NIO","XPEV","LI"],
    "ETF":              ["SMH","SOXX","SPY","QQQ"],
    "医药/生物科技":    ["LLY","NVO","JNJ","MRK","ABBV","PFE","AMGN","GILD","MRNA"],
    "能源/核能":        ["CEG","VST","NNE","SMR","OKLO"],
    "AI基础设施":       ["VRT","SMCI","APLD","CRWV","IREN","KULR","IONQ","DELL"],
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
def _score_fundamentals(exp_score: Optional[float], fpe: Optional[float], peg: Optional[float], eps_growth: Optional[float], upside_pct: Optional[float]) -> Tuple[int, List[str]]:
    """
    基本面评分 0-100。
    exp_score: 预期指数(0-100)，EPS修正+目标价+评级+Guidance+CapEx
    fpe: Forward PE
    peg: PEG ratio
    eps_growth: EPS增长率（小数，如0.30表示30%）
    upside_pct: 分析师目标价上行空间（小数）
    """
    score = 0
    reasons = []

    # 1. 预期指数（最重要，占40分）
    if exp_score is not None:
        if exp_score >= 75:
            score += 40; reasons.append(f"预期指数优秀({exp_score:.0f})")
        elif exp_score >= 60:
            score += 30; reasons.append(f"预期指数良好({exp_score:.0f})")
        elif exp_score >= 50:
            score += 20; reasons.append(f"预期指数中性({exp_score:.0f})")
        elif exp_score >= 40:
            score += 10
        else:
            score -= 10; reasons.append(f"预期指数偏弱({exp_score:.0f})")
    else:
        score += 30  # 无预期数据时给较高中性分，避免误杀

    # 2. EPS增长率（占25分）
    if eps_growth is not None:
        if eps_growth >= 0.50:
            score += 25; reasons.append(f"EPS增长{eps_growth*100:.0f}%")
        elif eps_growth >= 0.25:
            score += 20; reasons.append(f"EPS增长{eps_growth*100:.0f}%")
        elif eps_growth >= 0.15:
            score += 15; reasons.append(f"EPS增长{eps_growth*100:.0f}%")
        elif eps_growth >= 0.05:
            score += 8
        elif eps_growth < 0:
            score -= 15; reasons.append("EPS负增长")

    # 3. 估值（FPE/PEG，占20分）
    if peg is not None and peg > 0:
        if peg < 1.0:
            score += 20; reasons.append(f"PEG低估({peg:.1f})")
        elif peg < 1.5:
            score += 15; reasons.append(f"PEG合理({peg:.1f})")
        elif peg < 2.0:
            score += 8
        elif peg > 3.0:
            score -= 10
    elif fpe is not None and fpe > 0:
        if fpe < 20:
            score += 20; reasons.append(f"FPE低估({fpe:.0f}x)")
        elif fpe < 30:
            score += 15; reasons.append(f"FPE合理({fpe:.0f}x)")
        elif fpe < 40:
            score += 8
        elif fpe > 60:
            score -= 10; reasons.append(f"FPE偏高({fpe:.0f}x)")

    # 4. 分析师目标价上行空间（占15分）
    if upside_pct is not None:
        if upside_pct >= 0.20:
            score += 15; reasons.append(f"上行空间{upside_pct*100:.0f}%")
        elif upside_pct >= 0.10:
            score += 10
        elif upside_pct >= 0.05:
            score += 5
        elif upside_pct < 0:
            score -= 10; reasons.append("目标价低于现价")

    return int(max(0, min(100, score))), reasons


def _score_technical_timing(
    rsi_val: float,
    close_val: float,
    lower_val: float,
    mid_val: float,
    upper_val: float,
    drop3_val: float,
    vol_ratio_val: float,
    ma20_val: float,
    ma50_val: float,
    ma200_val: float,
    mkt_state: str,
) -> Tuple[int, List[str], str]:
    """
    技术面买点评分 0-100，根据市场环境选择不同买点逻辑。
    返回 (score, triggered_labels, scenario)
    scenario: 'bull_breakout' | 'pullback_support' | 'panic_oversold'
    """
    triggered = []
    score = 0
    scenario = "观察"

    price = close_val
    has_ma20  = not np.isnan(ma20_val)  if not np.isnan(ma20_val)  else False
    has_ma50  = not np.isnan(ma50_val)  if not np.isnan(ma50_val)  else False
    has_ma200 = not np.isnan(ma200_val) if not np.isnan(ma200_val) else False

    # ── 场景1：牛市回踩支撑买点 ────────────────────────────────────────────
    # 适合：趋势向上，短期回调到均线附近企稳
    if has_ma20 and has_ma50:
        above_ma50 = price > ma50_val
        above_ma20 = price > ma20_val
        near_ma20  = abs(price - ma20_val) / ma20_val < 0.05  # 距MA20在5%以内（放宽）
        near_ma50  = abs(price - ma50_val) / ma50_val < 0.05  # 距MA50在5%以内
        rsi_ok     = not np.isnan(rsi_val) and 35 <= rsi_val <= 70  # RSI范围放宽

        if above_ma50 and near_ma20 and rsi_ok:
            score += 50; triggered.append("回踩MA20企稳")
            scenario = "回踩支撑"
        elif above_ma50 and near_ma50 and rsi_ok:
            score += 40; triggered.append("回踩MA50企稳")
            scenario = "回踩支撑"
        elif above_ma20 and above_ma50:
            # 站稳双均线，趋势良好
            score += 25; triggered.append("站稳均线")
            scenario = "趋势持有"
        elif near_ma20 or near_ma50:
            score += 20; triggered.append("均线附近")
            scenario = "回踩支撑"

    # ── 场景2：放量突破买点 ─────────────────────────────────────────────────
    if not np.isnan(upper_val) and not np.isnan(vol_ratio_val):
        if price > upper_val and vol_ratio_val > 1.5:
            score += 45; triggered.append("放量突破上轨")
            scenario = "突破"
        elif price > upper_val:
            score += 25; triggered.append("突破布林上轨")
            scenario = "突破"

    # ── 场景3：恐慌错杀买点（仅在基本面好时有意义）────────────────────────
    if not np.isnan(rsi_val) and rsi_val < 40 and drop3_val > 0.05:
        score += 40; triggered.append("恐慌错杀")
        scenario = "错杀低吸"
    elif not np.isnan(rsi_val) and rsi_val < 40:
        score += 25; triggered.append("RSI超卖")
        scenario = "错杀低吸"

    # ── 加分项 ──────────────────────────────────────────────────────────────
    # 布林带下轨支撑
    if not np.isnan(lower_val) and price < lower_val * 1.08:
        score += 15; triggered.append("布林下轨支撑")

    # 放量
    if not np.isnan(vol_ratio_val) and vol_ratio_val > 2.0:
        score += 15; triggered.append("显著放量")
    elif not np.isnan(vol_ratio_val) and vol_ratio_val > 1.5:
        score += 8; triggered.append("放量")

    # MA趋势向好（MA50 > MA200）
    if has_ma50 and has_ma200 and ma50_val > ma200_val:
        score += 10; triggered.append("长期趋势向上")

    # 短期回调适度
    if 0.02 < drop3_val < 0.10:
        score += 8; triggered.append("短期回调适度")

    return int(max(0, min(100, score))), triggered, scenario


def _score_risks(
    vix_val: Optional[float],
    drop3_val: float,
    change_1d: float,
    eps_growth: Optional[float],
    insider_signal: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """
    风险扣分项，返回 (risk_score, risk_labels)，risk_score越高风险越大。

    insider_signal: 来自 modules/insider_data.py 的内幕交易信号（SEC Form 4）。
      - "heavy_sell" : C-suite密集净卖出、无买入对冲 → 扣分
      - "ceo_buy"    : C-suite有净买入 → 不扣分（占位，暂不加分，避免双重计分）
      - None         : 无数据或中性，不参与评分
    """
    risk = 0
    risks = []

    if vix_val and vix_val > 30:
        risk += 20; risks.append(f"VIX高恐慌({vix_val:.0f})")
    elif vix_val and vix_val > 25:
        risk += 10; risks.append(f"VIX偏高({vix_val:.0f})")

    if drop3_val > 0.15:
        risk += 20; risks.append(f"3日急跌{drop3_val*100:.0f}%")
    elif drop3_val > 0.10:
        risk += 10

    if abs(change_1d) > 0.08:
        risk += 15; risks.append(f"单日波动{change_1d*100:+.0f}%")

    if eps_growth is not None and eps_growth < 0:
        risk += 15; risks.append("EPS负增长")

    if insider_signal == "heavy_sell":
        risk += 15; risks.append("内幕密集卖出⚠️")
    # insider_signal == "ceo_buy" 暂不加分，先占位，避免双重计分（exp_score里可能已含管理层信心维度）

    return int(min(50, risk)), risks


def grade_signal(sig: dict, exp_score: Optional[float]) -> str:
    """
    新版信号分级：基本面门槛 + 技术面买点 + 风险过滤。

    A+: 基本面≥70 且 技术面≥60 且 风险低
    A:  基本面≥55 且 技术面≥45
    B:  基本面≥40 且 有技术触发
    C:  基本面不足或技术面弱
    """
    fund_score  = sig.get("fundamental_score", 50)
    tech_score  = sig.get("technical_score", 0)
    risk_score  = sig.get("risk_score", 0)
    triggered   = sig.get("triggered", [])

    if fund_score >= 70 and tech_score >= 60 and risk_score < 20:
        return "A+"
    if fund_score >= 55 and tech_score >= 45:
        return "A"
    if fund_score >= 40 and len(triggered) >= 1:
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

    # ── 数据源：富途优先，yfinance单股fallback ────────────────────────────────
    try:
        from modules.futu_data import get_kline, is_futu_connected
        _futu_ok = is_futu_connected()
    except Exception:
        _futu_ok = False

    def _get_ohlcv(sym: str):
        """返回 (close, low, volume) 三个 pd.Series，失败返回 None。"""
        # 1. 富途优先
        if _futu_ok:
            try:
                df = get_kline(sym, "1y")
                if df is not None and not df.empty and len(df) >= 25:
                    close  = df["Close"].dropna()
                    low    = df["Low"].dropna()   if "Low"    in df.columns else pd.Series(dtype=float)
                    volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
                    if len(close) >= 25:
                        return close, low, volume
            except Exception:
                pass
        # 2. yfinance 单股fallback
        try:
            df = yf.Ticker(sym).history(period="1y", auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 25:
                close  = df["Close"].dropna()
                low    = df["Low"].dropna()   if "Low"    in df.columns else pd.Series(dtype=float)
                volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
                if len(close) >= 25:
                    return close, low, volume
        except Exception:
            pass
        return None

    # SPY / QQQ for market state
    spy_close: Optional[pd.Series] = None
    qqq_close: Optional[pd.Series] = None
    _spy = _get_ohlcv("SPY")
    if _spy:
        spy_close = _spy[0]
    _qqq = _get_ohlcv("QQQ")
    if _qqq:
        qqq_close = _qqq[0]

    mkt_state = _market_state_from_spy(spy_close)
    vix_val   = _get_vix_level()

    # ── 基本面缓存（批量读取，不在循环里逐个调用）────────────────────────────
    _exp_cache: dict = {}
    _val_cache: dict = {}
    try:
        from modules.expectation_index import _load_cache, _is_fresh
        disk = _load_cache()
        _exp_cache = {t: disk[t] for t in tickers if t in disk and _is_fresh(disk[t])}
    except Exception:
        pass
    try:
        from modules.valuation import get_valuation_metrics
        val_df = get_valuation_metrics(tickers)
        if not val_df.empty:
            for _, row in val_df.iterrows():
                t = row.get("Ticker")
                if t:
                    _val_cache[t] = row.to_dict()
    except Exception:
        pass

    for ticker in tickers:
        try:
            ohlcv = _get_ohlcv(ticker)
            if ohlcv is None:
                continue
            close, low, volume = ohlcv

            if len(close) < 25: continue

            # Compute indicators
            rsi5           = _rsi(close, 5)
            upper, mid, lower = _bollinger(close)
            drop3          = (1 - close / close.shift(3)).clip(lower=0)
            ma20_s         = close.rolling(20).mean()
            ma50_s         = close.rolling(50).mean()
            ma200_s        = close.rolling(200).mean()

            # Volume ratio
            if not volume.empty and len(volume) >= 20:
                vol_ratio = volume / volume.rolling(20).mean()
            else:
                vol_ratio = pd.Series(np.nan, index=close.index)

            # Latest values
            latest_price     = float(close.iloc[-1])
            latest_low       = float(low.iloc[-1])        if not low.empty        else np.nan
            latest_rsi       = float(rsi5.iloc[-1])       if not rsi5.empty       else np.nan
            latest_lower     = float(lower.iloc[-1])       if not lower.empty       else np.nan
            latest_upper     = float(upper.iloc[-1])       if not upper.empty       else np.nan
            latest_mid       = float(mid.iloc[-1])         if not mid.empty         else np.nan
            latest_drop3     = float(drop3.iloc[-1])       if not drop3.empty       else np.nan
            latest_vol_ratio = float(vol_ratio.iloc[-1])   if not vol_ratio.empty   else np.nan
            latest_1d_chg    = float(close.pct_change().iloc[-1]) if len(close) >= 2 else 0.0
            latest_ma20      = float(ma20_s.iloc[-1])   if len(close) >= 20  else np.nan
            latest_ma50      = float(ma50_s.iloc[-1])   if len(close) >= 50  else np.nan
            latest_ma200     = float(ma200_s.iloc[-1])  if len(close) >= 200 else np.nan

            # ── 基本面评分 ───────────────────────────────────────────────────
            # 从预期指数缓存读取（不在此实时计算，避免阻塞）
            exp_data   = _exp_cache.get(ticker, {})
            exp_score  = exp_data.get("score") if exp_data else None

            # 从valuation缓存读取FPE/PEG/EPS
            val_data   = _val_cache.get(ticker, {})
            fpe        = val_data.get("Forward PE")
            peg        = val_data.get("PEG")
            eps_growth = val_data.get("eps_growth")
            target     = val_data.get("Analyst Target")
            upside_pct = ((target - latest_price) / latest_price) if (target and latest_price > 0) else None

            # val_cache为空时给eps_growth一个中性默认值，避免所有股票被过滤
            if eps_growth is None and fpe is None and peg is None:
                eps_growth = 0.15  # 默认假设15%增长，中性处理

            fund_score, fund_reasons = _score_fundamentals(
                exp_score, fpe, peg, eps_growth, upside_pct
            )

            # 基本面低于40分直接跳过，不进入信号池
            # 注：val_cache为空时eps_growth/fpe/peg全None，默认给中性分避免过滤
            if fund_score < 40:
                continue

            # ── 技术面买点评分 ────────────────────────────────────────────────
            tech_score, triggered, scenario = _score_technical_timing(
                latest_rsi, latest_price, latest_lower, latest_mid, latest_upper,
                latest_drop3, latest_vol_ratio,
                latest_ma20, latest_ma50, latest_ma200,
                mkt_state.get("state", "unknown"),
            )

            # 技术面低于10分不产生信号
            if tech_score < 10:
                continue

            # ── 内幕交易信号（SEC Form 4，带本地缓存，正常情况下很快）──────────────
            insider_signal = None
            try:
                from modules.insider_data import get_insider_signal
                insider_signal, _insider_detail = get_insider_signal(ticker)
            except Exception:
                pass

            # ── 风险评分 ──────────────────────────────────────────────────────
            risk_score, risk_reasons = _score_risks(
                vix_val, latest_drop3, latest_1d_chg, eps_growth, insider_signal
            )

            # ── 综合评分 ──────────────────────────────────────────────────────
            final_score = int(fund_score * 0.6 + tech_score * 0.4 - risk_score * 0.3)
            final_score = max(0, min(100, final_score))

            # 止损止盈（按场景）
            if scenario == "突破":
                stop_pct, target_pct = 0.06, 0.10
            elif scenario == "回踩支撑":
                stop_pct, target_pct = 0.05, 0.08
            elif scenario == "错杀低吸":
                stop_pct, target_pct = 0.08, 0.12
            else:
                stop_pct, target_pct = 0.07, 0.09

            stop_price   = round(latest_price * (1 - stop_pct), 2)
            target_price = round(latest_price * (1 + target_pct), 2)
            rr_ratio     = round(target_pct / stop_pct, 2)
            buy_low      = round(latest_price * 0.995, 2)
            buy_high     = round(latest_price * 1.005, 2)

            # 极端风险标记
            extreme_flags = []
            if vix_val and vix_val > VIX_HIGH:
                extreme_flags.append(f"VIX={vix_val:.0f}⚠️")
            if abs(latest_1d_chg) > 0.08:
                extreme_flags.append(f"单日波动{latest_1d_chg*100:+.1f}%⚠️")
            if len(close) >= 4:
                recent_moves = close.pct_change().iloc[-4:].abs()
                if (recent_moves > 0.05).any():
                    extreme_flags.append("疑似财报前后⚡")

            # 回测评级
            strat_key_map = {
                "突破":    "rsi_boll_combo",
                "回踩支撑": "dip_bounce",
                "错杀低吸": "rsi_oversold",
            }
            grade_info = "未回测"
            if backtest_grades:
                bg_key = f"{ticker}_{strat_key_map.get(scenario, '')}"
                grade_info = backtest_grades.get(bg_key, "未回测")

            tier_label, tier_emoji = _signal_tier(len(triggered))

            signals.append({
                "ticker":             ticker,
                "sector":             _TICKER_SECTOR.get(ticker, "Other"),
                "tier":               tier_label,
                "tier_emoji":         tier_emoji,
                "price":              latest_price,
                "latest_low":         round(latest_low, 2) if not np.isnan(latest_low) else None,
                "change_1d_pct":      round(latest_1d_chg * 100, 2),
                "rsi5":               round(latest_rsi, 1) if not np.isnan(latest_rsi) else None,
                "bb_lower":           round(latest_lower, 2) if not np.isnan(latest_lower) else None,
                "bb_mid":             round(latest_mid, 2) if not np.isnan(latest_mid) else None,
                "drop3_pct":          round(latest_drop3 * 100, 2) if not np.isnan(latest_drop3) else None,
                "vol_ratio":          round(latest_vol_ratio, 2) if not np.isnan(latest_vol_ratio) else None,
                "ma20":               round(latest_ma20, 2) if not np.isnan(latest_ma20) else None,
                "ma50":               round(latest_ma50, 2) if not np.isnan(latest_ma50) else None,
                "ma200":              round(latest_ma200, 2) if not np.isnan(latest_ma200) else None,
                "score":              final_score,
                "fundamental_score":  fund_score,
                "technical_score":    tech_score,
                "risk_score":         risk_score,
                "stars":              _stars(final_score),
                "triggered":          triggered,
                "strategy":           scenario,
                "fund_reasons":       fund_reasons,
                "risk_reasons":       risk_reasons,
                "insider_signal":     insider_signal,
                # 估值字段（用于UI显示）
                "fpe":                fpe,
                "peg":                peg,
                "eps_growth":         round(eps_growth * 100, 1) if eps_growth else None,
                "exp_score":          exp_score,
                # 买卖参数
                "buy_low":            buy_low,
                "buy_high":           buy_high,
                "stop_price":         stop_price,
                "stop_pct":           round(stop_pct * 100, 1),
                "target_price":       target_price,
                "target_pct":         round(target_pct * 100, 1),
                "rr_ratio":           rr_ratio,
                "market_state":       mkt_state["label"],
                "market_emoji":       mkt_state["emoji"],
                "extreme_flags":      extreme_flags,
                "grade_info":         grade_info,
                "scan_time":          datetime.now().strftime("%H:%M:%S"),
                "signal_level":       None,
                "tradable":           None,
                "aplus_passed":       [],
                "aplus_failed":       [],
                "hc_reason":          "",
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
