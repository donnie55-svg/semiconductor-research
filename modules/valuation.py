import os

import yfinance as yf
import pandas as pd

from modules.futu_data import get_info_partial, is_futu_connected

_FIELDS = {
    "trailingPE":                    "Trailing PE",
    "forwardPE":                     "Forward PE",
    "pegRatio":                      "PEG",
    "priceToBook":                   "P/B",
    "priceToSalesTrailing12Months":  "P/S",
    "enterpriseToEbitda":            "EV/EBITDA",
    "marketCap":                     "Market Cap",
    "targetMeanPrice":               "Analyst Target",
    "currentPrice":                  "_price",
    "shortName":                     "Name",
}

# 富途快照可提供的字段（会覆盖 yfinance 同名字段，提供实时价格/市值）
_FUTU_OVERRIDES = {
    "trailingPE", "priceToBook", "marketCap", "currentPrice", "shortName", "trailingEps",
}


def get_valuation_metrics(tickers: list) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            # yfinance 提供完整基本面数据（Forward PE / PEG / P/S / EV-EBITDA / 目标价等）
            info = yf.Ticker(ticker).info

            # 富途快照覆盖实时价格相关字段（PE_TTM / PB / 市值 / 当前价）
            if is_futu_connected():
                partial = get_info_partial(ticker)
                for yf_key in _FUTU_OVERRIDES:
                    if partial.get(yf_key) is not None:
                        info[yf_key] = partial[yf_key]

            row = {"Ticker": ticker}
            for key, label in _FIELDS.items():
                row[label] = info.get(key)

            mc = row.get("Market Cap")
            row["Market Cap (B)"] = round(mc / 1e9, 1) if mc else None

            price = row.pop("_price", None)
            target = row.get("Analyst Target")
            if price and target:
                row["Upside%"] = round((target - price) / price * 100, 1)
            else:
                row["Upside%"] = None

            rows.append(row)
        except Exception:
            rows.append({"Ticker": ticker, "Name": ticker})

    return pd.DataFrame(rows)
