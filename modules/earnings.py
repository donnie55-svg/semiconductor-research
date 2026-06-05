import os

import yfinance as yf
import pandas as pd
from datetime import datetime

from modules.futu_data import get_info_partial, is_futu_connected

_INFO_FIELDS = {
    "shortName":        "Name",
    "trailingEps":      "EPS (TTM)",
    "forwardEps":       "EPS Forward",
    "totalRevenue":     "_rev_raw",
    "revenueGrowth":    "_rev_growth_raw",
    "grossMargins":     "_gross_raw",
    "operatingMargins": "_op_raw",
    "earningsTimestamp":"_ts",
}

# 富途快照支持的覆盖字段
_FUTU_OVERRIDES = {"shortName", "trailingEps", "revenueGrowth", "currentPrice"}


def get_earnings_summary(tickers: list) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info

            # 富途覆盖实时价格 / EPS / 营收增速
            if is_futu_connected():
                partial = get_info_partial(ticker)
                for yf_key in _FUTU_OVERRIDES:
                    if partial.get(yf_key) is not None:
                        info[yf_key] = partial[yf_key]

            row = {"Ticker": ticker}
            for key, label in _INFO_FIELDS.items():
                row[label] = info.get(key)

            row["Revenue TTM (B)"] = round(row.pop("_rev_raw") / 1e9, 2) if row.get("_rev_raw") else None
            rg = row.pop("_rev_growth_raw", None)
            row["Revenue Growth YoY%"] = round(rg * 100, 1) if rg else None
            gm = row.pop("_gross_raw", None)
            row["Gross Margin%"] = round(gm * 100, 1) if gm else None
            om = row.pop("_op_raw", None)
            row["Operating Margin%"] = round(om * 100, 1) if om else None

            ts = row.pop("_ts", None)
            if ts:
                try:
                    row["Next Earnings"] = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
                except Exception:
                    row["Next Earnings"] = None
            else:
                row["Next Earnings"] = None

            # 季度 EPS beat：富途不提供，始终用 yfinance
            row["EPS Beat%"] = None
            try:
                qe = t.quarterly_earnings
                if qe is not None and not qe.empty and "Actual" in qe.columns and "Estimate" in qe.columns:
                    last = qe.iloc[0]
                    actual, estimate = last["Actual"], last["Estimate"]
                    if pd.notna(actual) and pd.notna(estimate) and estimate != 0:
                        row["EPS Beat%"] = round((actual - estimate) / abs(estimate) * 100, 1)
            except Exception:
                pass

            rows.append(row)
        except Exception:
            rows.append({"Ticker": ticker, "Name": ticker})

    return pd.DataFrame(rows)


def get_quarterly_revenue(tickers: list) -> dict:
    # 季度营收数据富途不提供，始终使用 yfinance
    result = {}
    for ticker in tickers:
        try:
            qf = yf.Ticker(ticker).quarterly_financials
            if qf.empty:
                continue
            rev_row = next(
                (idx for idx in qf.index if "revenue" in str(idx).lower() and "total" in str(idx).lower()),
                None,
            )
            if rev_row is None:
                continue
            rev = qf.loc[rev_row].dropna().sort_index()
            df = pd.DataFrame({
                "Quarter":   pd.PeriodIndex(rev.index, freq="Q").strftime("Q%q-%Y"),
                "Revenue_B": (rev.values / 1e9).round(2),
            })
            df["QoQ%"] = df["Revenue_B"].pct_change().mul(100).round(1)
            result[ticker] = df
        except Exception:
            pass
    return result


def get_guidance_tracker(filepath: str = "data/earnings_guidance.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame({
            "Ticker":               ["MU", "TSM", "NVDA", "AVGO", "AMD"],
            "Quarter":              ["", "", "", "", ""],
            "Revenue_Guidance_B":   [None, None, None, None, None],
            "Gross_Margin_Guide%":  [None, None, None, None, None],
            "EPS_Guide":            [None, None, None, None, None],
            "Guidance_Direction":   ["", "", "", "", ""],
            "Notes":                ["", "", "", "", ""],
            "Updated":              ["", "", "", "", ""],
        })
