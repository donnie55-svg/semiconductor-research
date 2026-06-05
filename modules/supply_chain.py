import os

import yfinance as yf
import pandas as pd

from modules.futu_data import get_kline, is_futu_connected

# 1 TSM ADR (NYSE) represents 5 shares of 2330.TW
ADR_RATIO = 5

HYPERSCALERS = {
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "META":  "Meta",
    "AMZN": "Amazon",
}


def _hist(ticker: str, period: str) -> pd.Series:
    """获取 Close 序列：TSM 等美股优先富途，台股/外汇始终 yfinance。"""
    if is_futu_connected():
        df = get_kline(ticker, period)
        if not df.empty:
            return df["Close"]
    return yf.Ticker(ticker).history(period=period)["Close"]


def get_adr_premium(period: str = "3mo") -> pd.DataFrame:
    """Track TSM ADR premium/discount vs 2330.TW on Taiwan exchange."""
    try:
        # TSM ADR (美股)：优先富途实时 K 线
        tsm = _hist("TSM", period)
        # 台积电台股（富途不支持台股）、台币汇率（富途不支持外汇）：yfinance
        tw      = yf.Ticker("2330.TW").history(period=period)["Close"]
        usdtwd  = yf.Ticker("TWD=X").history(period=period)["Close"]

        # Align trading days across markets
        common = (
            tsm.index.normalize()
            .intersection(tw.index.normalize())
            .intersection(usdtwd.index.normalize())
        )

        if len(common) == 0:
            return pd.DataFrame()

        def reindex_to_dates(series, dates):
            s = series.copy()
            s.index = s.index.normalize()
            return s[~s.index.duplicated()].reindex(dates, method="ffill")

        tsm_a  = reindex_to_dates(tsm, common)
        tw_a   = reindex_to_dates(tw, common)
        rate_a = reindex_to_dates(usdtwd, common)

        tw_usd_per_adr = (tw_a / rate_a) * ADR_RATIO
        premium = (tsm_a / tw_usd_per_adr - 1) * 100

        return pd.DataFrame({
            "TSM_ADR":           tsm_a.round(2).values,
            "2330_TW_equiv_USD": tw_usd_per_adr.round(2).values,
            "ADR_Premium%":      premium.round(2).values,
        }, index=common)
    except Exception as e:
        print(f"ADR premium error: {e}")
        return pd.DataFrame()


def get_hyperscaler_capex() -> pd.DataFrame:
    """Pull quarterly CapEx from MSFT/GOOGL/META/AMZN cash flow statements.
    季度财务报表富途不提供，始终使用 yfinance。
    """
    rows = []
    for ticker, name in HYPERSCALERS.items():
        try:
            cf = yf.Ticker(ticker).quarterly_cashflow
            if cf.empty:
                continue

            capex_row = None
            for idx in cf.index:
                idx_lower = str(idx).lower()
                if "capital expenditure" in idx_lower or "capex" in idx_lower:
                    capex_row = idx
                    break

            if capex_row is None:
                continue

            capex_series = cf.loc[capex_row].dropna()
            for col, val in capex_series.items():
                rows.append({
                    "Company":  name,
                    "Ticker":   ticker,
                    "Quarter":  pd.Timestamp(col).to_period("Q").strftime("Q%q-%Y"),
                    "CapEx_B":  round(abs(float(val)) / 1e9, 2),
                })
        except Exception as e:
            print(f"CapEx error {ticker}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["Quarter", "Company"], ascending=[False, True]).reset_index(drop=True)


def get_hbm_manual_data(filepath: str = "data/supply_chain_manual.csv") -> pd.DataFrame:
    """Load manually maintained HBM / CoWoS data."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame({
            "Date":               ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"],
            "HBM_MU_Rev_B":       [None, None, None, None],
            "HBM_Share%":         [None, None, None, None],
            "CoWoS_Util%":        [None, None, None, None],
            "DRAM_ASP_Change%":   [None, None, None, None],
            "Source":             ["TrendForce"] * 4,
            "Notes":              [""] * 4,
        })
