"""
富途 OpenAPI 数据连接器。
需要 FutuOpenD 在本地运行（127.0.0.1:11111）。
美股代码格式：US.NVDA  港股代码格式：HK.00700
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List

# 富途连接不需要代理（本地 socket）
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

_HOST = "127.0.0.1"
_PORT = 11111

_ctx = None
_connected = False
_last_check: float = 0
_CHECK_TTL = 30  # 秒内不重复探活


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _to_futu_code(ticker: str) -> Optional[str]:
    """把 yfinance ticker 转为富途代码；不支持的（指数/台股/外汇）返回 None。"""
    if ticker.startswith("^"):   # ^SOX, ^GSPC 等指数
        return None
    if ticker.endswith(".TW"):   # 台股 2330.TW
        return None
    if "=" in ticker:            # 外汇 TWD=X
        return None
    return f"US.{ticker}"


def _port_open() -> bool:
    """快速检查 FutuOpenD 端口是否可达（1 秒超时，CONNREFUSED 立即返回）。"""
    import socket
    try:
        with socket.create_connection((_HOST, _PORT), timeout=1):
            return True
    except OSError:
        return False


def _try_connect() -> bool:
    """尝试连接 FutuOpenD，成功返回 True。先用 socket 预检，避免 futu 库内置重试延迟。"""
    global _ctx, _connected
    if not _port_open():
        _connected = False
        return False
    try:
        from futu import OpenQuoteContext, RET_OK
        ctx = OpenQuoteContext(host=_HOST, port=_PORT)
        ret, _ = ctx.get_global_state()
        if ret == RET_OK:
            if _ctx is not None:
                try:
                    _ctx.close()
                except Exception:
                    pass
            _ctx = ctx
            _connected = True
            return True
        ctx.close()
    except Exception:
        pass
    _connected = False
    return False


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def is_futu_connected() -> bool:
    """
    检查富途连接。结果缓存 30 秒避免频繁探活；
    断线时自动重连一次。
    """
    global _ctx, _connected, _last_check
    now = time.time()
    # 缓存命中：30 秒内已确认连通
    if _connected and _ctx is not None and (now - _last_check) < _CHECK_TTL:
        return True
    _last_check = now
    # 探活已有连接
    if _connected and _ctx is not None:
        try:
            from futu import RET_OK
            ret, _ = _ctx.get_global_state()
            if ret == RET_OK:
                return True
        except Exception:
            pass
        _connected = False
    # 重连
    return _try_connect()


def get_realtime_quote(symbols: List[str]) -> pd.DataFrame:
    """
    实时报价。
    返回含 Ticker / Price / Change% / Volume / Open / High / Low / PrevClose 的 DataFrame。
    不支持的 ticker（指数/台股/外汇）会被跳过；连接失败返回空 DataFrame。
    """
    codes, rev = [], {}
    for sym in symbols:
        c = _to_futu_code(sym)
        if c:
            codes.append(c)
            rev[c] = sym

    if not codes or not is_futu_connected():
        return pd.DataFrame()

    try:
        from futu import RET_OK
        ret, data = _ctx.get_market_snapshot(codes)
        if ret != RET_OK or data.empty:
            return pd.DataFrame()

        rows = []
        for _, r in data.iterrows():
            sym = rev.get(r["code"], r["code"])
            rows.append({
                "Ticker":    sym,
                "Price":     float(r.get("last_price") or 0),
                "Change%":   float(r.get("change_rate") or 0),
                "Volume":    int(r.get("volume") or 0),
                "Open":      float(r.get("open_price") or 0),
                "High":      float(r.get("high_price") or 0),
                "Low":       float(r.get("low_price") or 0),
                "PrevClose": float(r.get("prev_close_price") or 0),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def get_kline(symbol: str, period: str = "3mo") -> pd.DataFrame:
    """
    日 K 线数据（前复权）。
    period 与 yfinance 格式一致：1mo / 3mo / 6mo / 1y / 2y。
    返回以日期为索引、含 Open / High / Low / Close / Volume 的 DataFrame；
    连接失败或代码不支持时返回空 DataFrame。
    """
    code = _to_futu_code(symbol)
    if not code or not is_futu_connected():
        return pd.DataFrame()

    days = {"1mo": 35, "3mo": 95, "6mo": 185, "1y": 370, "2y": 740}.get(period, 95)
    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        from futu import RET_OK, KLType, AuType
        ret, data, _ = _ctx.get_history_kline(
            code,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            ktype=KLType.K_DAY,
            autype=AuType.QFQ,
            max_count=1000,
        )
        if ret != RET_OK or data.empty:
            return pd.DataFrame()

        df = data[["time_key", "open", "close", "high", "low", "volume"]].copy()
        df.columns = ["Date", "Open", "Close", "High", "Low", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")
    except Exception:
        return pd.DataFrame()


def get_market_snapshot(symbols: List[str]) -> pd.DataFrame:
    """
    市场快照（含价格、涨跌幅、成交量、PE、PB、市值等）。
    返回富途原始 DataFrame，附加 Ticker 列（yfinance 格式）。
    不支持的 ticker 会被过滤；连接失败返回空 DataFrame。
    """
    codes, rev = [], {}
    for sym in symbols:
        c = _to_futu_code(sym)
        if c:
            codes.append(c)
            rev[c] = sym

    if not codes or not is_futu_connected():
        return pd.DataFrame()

    try:
        from futu import RET_OK
        ret, data = _ctx.get_market_snapshot(codes)
        if ret != RET_OK or data.empty:
            return pd.DataFrame()

        data = data.copy()
        data["Ticker"] = data["code"].map(rev)
        return data
    except Exception:
        return pd.DataFrame()


def get_info_partial(ticker: str) -> dict:
    """
    从富途快照提取与 yfinance .info 字段名对齐的 dict（仅含富途支持的字段）。
    调用方负责用 yfinance 补充缺失字段。
    """
    snap = get_market_snapshot([ticker])
    if snap.empty:
        return {}

    r = snap.iloc[0]

    def _f(key) -> Optional[float]:
        v = r.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    rev_growth_raw = _f("revenue_growth_rate")
    return {
        "shortName":    r.get("stock_name"),
        "currentPrice": _f("last_price"),
        "regularMarketPrice": _f("last_price"),
        "trailingPE":   _f("pe_ttm"),
        "priceToBook":  _f("pb_ratio"),
        "marketCap":    _f("total_market_val"),
        "trailingEps":  _f("basic_eps"),
        # 富途存储的是百分比数值，yfinance 存的是小数（0.15 表示 15%）
        "revenueGrowth": rev_growth_raw / 100.0 if rev_growth_raw is not None else None,
        "volume": int(r.get("volume") or 0),
    }
