import os

import yfinance as yf
import pandas as pd
import numpy as np

from modules.futu_data import get_kline, get_market_snapshot, is_futu_connected


def _hist(ticker: str, period: str) -> pd.DataFrame:
    """OHLCV 历史：优先富途 K 线，断线或不支持时回退 yfinance。"""
    if is_futu_connected():
        df = get_kline(ticker, period)
        if not df.empty:
            return df
    return yf.Ticker(ticker).history(period=period)


def get_daily_summary(tickers: list) -> pd.DataFrame:
    rows = []

    # 批量获取富途快照（仅含富途支持的 ticker，^SOX / 2330.TW 等自动跳过）
    futu_snap: dict = {}
    if is_futu_connected():
        snap = get_market_snapshot(tickers)
        if not snap.empty:
            for _, r in snap.iterrows():
                tk = r.get("Ticker")
                if tk:
                    futu_snap[tk] = r

    for ticker in tickers:
        try:
            snap_row = futu_snap.get(ticker)
            if snap_row is not None:
                price     = float(snap_row.get("last_price") or 0)
                change_pct = float(snap_row.get("change_rate") or 0)
                volume    = int(snap_row.get("volume") or 0)
                high_52w  = float(snap_row.get("52week_high") or 0)
                low_52w   = float(snap_row.get("52week_low") or 0)

                # 20 日均量需要 K 线
                vol_ratio = 1.0
                hist = get_kline(ticker, "3mo")
                if not hist.empty and len(hist) >= 20:
                    avg_vol = float(hist["Volume"].tail(20).mean())
                    vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0
            else:
                # 回退：yfinance（指数 / 台股 / 未连接）
                hist = yf.Ticker(ticker).history(period="3mo")
                if hist.empty or len(hist) < 2:
                    continue
                latest    = hist.iloc[-1]
                prev      = hist.iloc[-2]
                price     = float(latest["Close"])
                change_pct = (price - float(prev["Close"])) / float(prev["Close"]) * 100
                volume    = int(latest["Volume"])
                avg_vol   = float(hist["Volume"].tail(20).mean())
                vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0
                high_52w  = float(hist["High"].max())
                low_52w   = float(hist["Low"].min())

            pct_from_high = (price - high_52w) / high_52w * 100 if high_52w else 0
            rows.append({
                "Ticker":      ticker,
                "Price":       round(price, 2),
                "Change%":     round(change_pct, 2),
                "Volume":      volume,
                "Vol/AvgVol":  round(vol_ratio, 2),
                "52W High":    round(high_52w, 2),
                "52W Low":     round(low_52w, 2),
                "% from High": round(pct_from_high, 2),
            })
        except Exception:
            rows.append({
                "Ticker": ticker, "Price": None, "Change%": None,
                "Volume": None, "Vol/AvgVol": None,
                "52W High": None, "52W Low": None, "% from High": None,
            })
    return pd.DataFrame(rows)


def get_price_history(tickers: list, period: str = "6mo") -> dict:
    result = {}
    for ticker in tickers:
        try:
            hist = _hist(ticker, period)
            if not hist.empty:
                result[ticker] = hist
        except Exception:
            pass
    return result


def get_sox_beta(tickers: list, period: str = "3mo") -> dict:
    # ^SOX 指数富途不支持，始终用 yfinance
    try:
        sox_raw = yf.Ticker("^SOX").history(period=period)["Close"].pct_change().dropna()
    except Exception:
        return {}

    betas = {}
    for ticker in tickers:
        if ticker in ["^SOX", "SMH"]:
            continue
        try:
            hist = _hist(ticker, period)
            if hist.empty:
                continue
            stk_raw = hist["Close"].pct_change().dropna()

            # 统一索引为 date（去掉 timezone 差异）
            sox_idx = sox_raw.index.normalize() if hasattr(sox_raw.index, "normalize") else sox_raw.index
            stk_idx = stk_raw.index.normalize() if hasattr(stk_raw.index, "normalize") else stk_raw.index
            sox_s = pd.Series(sox_raw.values, index=sox_idx)
            stk_s = pd.Series(stk_raw.values, index=stk_idx)

            aligned = pd.concat([stk_s, sox_s], axis=1, join="inner")
            aligned.columns = ["stock", "sox"]
            if len(aligned) < 20:
                continue

            cov = np.cov(aligned["stock"].values, aligned["sox"].values)
            beta = float(cov[0][1] / cov[1][1]) if cov[1][1] != 0 else None
            betas[ticker] = round(beta, 2) if beta is not None else None
        except Exception:
            pass
    return betas
