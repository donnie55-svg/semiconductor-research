"""Streamlit dashboard — run with: streamlit run dashboard.py"""
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from modules.backtest_pro import (
    BACKTEST_SECTORS, ALL_BACKTEST_TICKERS, STRATEGY_NAMES, STRESS_PERIODS,
    run_full_backtest, build_flat_df, build_stock_summary,
    get_sector_stats, get_strategy_stats, get_top_recommendations,
    get_best_trades, get_sector_top5, reliability_label,
    get_current_market_state, compute_market_states,
)
from modules.realtime_radar import (
    scan_signals, build_signal_card_html, grade_signal,
    ALL_RADAR_TICKERS, RADAR_TICKERS_BY_SECTOR,
    get_current_vix_and_market,
)
from modules.earnings import get_earnings_summary, get_guidance_tracker, get_quarterly_revenue
from modules.earnings_radar import SCAN_TICKERS, get_earnings_radar, get_top_picks
from modules.futu_data import is_futu_connected
from modules.market_report import get_daily_summary, get_price_history, get_sox_beta
from modules.news_monitor import KEYWORDS, fetch_news_feed
from modules.supply_chain import get_adr_premium, get_hbm_manual_data, get_hyperscaler_capex
from modules.valuation import get_valuation_metrics
from modules.expectation_index import get_expectation_scores
from modules.high_conviction import (
    evaluate_aplus_conditions, classify_aplus,
    get_stock_stats, get_sector_hc_stats, get_market_hc_stats,
    check_auto_shutdown,
    MIN_STOCK_SAMPLES, MIN_SECTOR_SAMPLES, MIN_MARKET_SAMPLES,
)

st.set_page_config(
    page_title="AI 半导体研究系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached data loaders (TTL in seconds) ──────────────────────────────────────

@st.cache_data(ttl=60)
def c_daily(tickers):
    return get_daily_summary(list(tickers))

@st.cache_data(ttl=60)
def c_history(tickers, period):
    return get_price_history(list(tickers), period)

@st.cache_data(ttl=60)
def c_beta(tickers, period):
    return get_sox_beta(list(tickers), period)

@st.cache_data(ttl=3600)
def c_valuation(tickers):
    return get_valuation_metrics(list(tickers))

@st.cache_data(ttl=3600)
def c_earnings(tickers):
    return get_earnings_summary(list(tickers))

@st.cache_data(ttl=1800)
def c_adr(period):
    return get_adr_premium(period)

@st.cache_data(ttl=3600)
def c_capex():
    return get_hyperscaler_capex()

@st.cache_data(ttl=600)
def c_news():
    return fetch_news_feed(max_items_per_feed=15)

@st.cache_data(ttl=3600)
def c_qrev(tickers):
    return get_quarterly_revenue(list(tickers))

@st.cache_data(ttl=3600)
def c_radar(days: int):
    return get_earnings_radar(days=days)

@st.cache_data(ttl=21600)
def c_expectation(tickers: tuple) -> dict:
    """预期指数（只读磁盘缓存，秒级返回，不阻塞 UI）"""
    try:
        from modules.expectation_index import _load_cache, _is_fresh
        disk = _load_cache()
        return {t: disk[t] for t in tickers if t in disk and _is_fresh(disk[t])}
    except Exception:
        return {}

@st.cache_data(ttl=300)
def c_spy_state():
    """Live SPY market state + VIX for the top status bar."""
    try:
        import yfinance as yf
        spy_raw = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        if spy_raw.empty:
            return None, None
        sc = spy_raw["Close"]
        spy_close = (sc.iloc[:, 0] if isinstance(sc, pd.DataFrame) else sc).dropna()
        mkt_state = get_current_market_state(spy_close)
        # VIX
        vix_val = None
        try:
            vix_raw = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
            if not vix_raw.empty:
                vc = vix_raw["Close"]
                vs = (vc.iloc[:, 0] if isinstance(vc, pd.DataFrame) else vc).dropna()
                if not vs.empty:
                    vix_val = round(float(vs.iloc[-1]), 1)
        except Exception:
            pass
        return mkt_state, vix_val
    except Exception:
        return None, None


@st.cache_data(ttl=30)
def c_radar_signals(tickers_key: str, backtest_grades_key: str = ""):
    """实时信号扫描（30秒缓存）"""
    tickers = ALL_RADAR_TICKERS
    try:
        sigs, mkt, vix, _, _ = scan_signals(tickers, min_conditions=1)
        return sigs, mkt, vix
    except Exception:
        return [], {"state":"unknown","emoji":"⚪","label":"扫描失败"}, None

# ── Helpers ────────────────────────────────────────────────────────────────────

_DEFAULT_SECTOR = "AI算力/GPU/芯片"
_ETF_SECTOR     = "ETF/指数"

def load_watchlist() -> pd.DataFrame:
    try:
        df = pd.read_csv("watchlist.csv", encoding="utf-8")
    except FileNotFoundError:
        df = pd.DataFrame({
            "ticker":   ["NVDA", "AMD", "AVGO", "MU", "TSM", "AMAT", "LRCX", "KLAC",
                         "ASML", "ADI", "MRVL", "TXN", "INTC", "QCOM",
                         "AAL", "DAL", "UAL", "LUV", "ALK", "CCL", "RCL", "NCLH",
                         "ABNB", "BKNG", "DIS", "LVS", "WYNN", "MGM",
                         "BABA", "PDD", "JD", "BIDU", "NIO", "XPEV", "LI", "TCOM", "FUTU",
                         "SMH", "SOXX", "XSD", "DRAM", "SPY", "QQQ", "NASA", "EWT",
                         "WMT", "AMZN", "COST", "TGT", "HD", "LOW",
                         "NKE", "MCD", "SBUX", "KO", "PEP", "PG", "LULU", "TJX", "EBAY"],
            "name":     ["NVIDIA", "AMD", "Broadcom", "Micron", "TSMC",
                         "Applied Materials", "Lam Research", "KLA Corp",
                         "ASML", "Analog Devices", "Marvell Tech", "Texas Instruments",
                         "Intel", "Qualcomm",
                         "American Airlines", "Delta Air Lines", "United Airlines",
                         "Southwest Airlines", "Alaska Airlines",
                         "Carnival Corp", "Royal Caribbean", "Norwegian Cruise",
                         "Airbnb", "Booking Holdings", "Disney",
                         "Las Vegas Sands", "Wynn Resorts", "MGM Resorts",
                         "阿里巴巴", "拼多多", "京东", "百度", "蔚来",
                         "小鹏汽车", "理想汽车", "携程", "富途控股",
                         "VanEck Semiconductor ETF", "iShares Semiconductor ETF",
                         "SPDR Semiconductor ETF", "DRAM ETF",
                         "S&P 500 ETF", "Nasdaq 100 ETF",
                         "Procure Space ETF", "iShares MSCI Taiwan ETF",
                         "沃尔玛", "亚马逊", "好市多 Costco", "塔吉特 Target",
                         "Home Depot", "Lowe's", "耐克 Nike", "麦当劳", "星巴克",
                         "可口可乐", "百事可乐", "宝洁 P&G", "Lululemon", "TJ Maxx", "eBay"],
            "sector":   [_DEFAULT_SECTOR] * 14 +
                        ["空运&旅游"] * 14 +
                        ["中概股ADR"] * 9 +
                        [_ETF_SECTOR] * 8 +
                        ["消费零售"] * 15,
            "priority": ["high"] * 37 + ["medium"] * 8 + ["medium"] * 15,
        })
    # 兼容旧格式：补充缺失的 sector 列
    if "sector" not in df.columns:
        df["sector"] = _DEFAULT_SECTOR
    return df


def sign_color(val):
    if pd.isna(val):
        return ""
    return "color: #00CC96" if float(val) > 0 else ("color: #EF553B" if float(val) < 0 else "")


def fmt_safe(fmt, val):
    try:
        return fmt.format(val) if pd.notna(val) else "N/A"
    except Exception:
        return "N/A"


CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
)

# ── Legacy: kept for reference only, not called ───────────────────────────────
_MIN_RELIABLE = 15  # legacy constant still referenced in dead-code below

def _bt_antioverfitting_view(display_df: pd.DataFrame, spy_ret: float) -> None:  # unused
    """Render the full anti-overfitting validation report."""
    import plotly.graph_objects as go

    # ── Final verdict split ──────────────────────────────────────────────────
    worthy_df  = display_df[display_df["verdict"] == "✅值得交易"].copy()
    other_df   = display_df[display_df["verdict"] != "✅值得交易"].copy()

    # Shared columns for result table
    RESULT_COLS = {
        "ticker":          "股票",
        "strategy_name":   "策略",
        "trade_count":     "交易次数",
        "win_rate":        "胜率%",
        "avg_return":      "平均收益%",
        "max_loss":        "最大亏损%",
        "max_drawdown":    "最大回撤%",
        "sharpe":          "夏普比率",
        "bear_win_rate":   "熊市胜率%",
        "oos_win_rate":    "样本外胜率%",
        "avg_hold_days":   "平均持仓天",
        "score":           "可信评分",
        "verdict":         "最终结论",
        "hold_style":      "持仓风格",
    }
    FMT = {
        "胜率%":        "{:.1f}%",
        "平均收益%":    "{:+.2f}%",
        "最大亏损%":    "{:.2f}%",
        "最大回撤%":    "{:.1f}%",
        "夏普比率":     "{:.2f}",
        "熊市胜率%":    "{:.1f}%",
        "样本外胜率%":  "{:.1f}%",
        "平均持仓天":   "{:.1f}",
        "可信评分":     "{:.1f}",
    }

    def _color_verdict(val):
        if "✅" in str(val):
            return "color: #00CC96; font-weight:bold"
        if "❌" in str(val):
            return "color: #EF553B"
        return "color: #FFA500"

    def _score_bg(val):
        if pd.isna(val):
            return ""
        v = float(val)
        if v >= 70:
            return "background-color: rgba(0,204,150,0.2)"
        if v >= 50:
            return "background-color: rgba(255,165,0,0.15)"
        return ""

    def _render_table(df: pd.DataFrame) -> None:
        avail = {k: v for k, v in RESULT_COLS.items() if k in df.columns}
        disp = df[list(avail.keys())].rename(columns=avail)
        # Sample warning: color trade_count < MIN_RELIABLE red
        styled = disp.style.format(
            {k: v for k, v in FMT.items() if k in disp.columns}, na_rep="N/A"
        )
        styled = styled.map(sign_color,    subset=[c for c in ["胜率%","平均收益%","夏普比率"] if c in disp.columns])
        styled = styled.map(_color_verdict, subset=["最终结论"] if "最终结论" in disp.columns else [])
        styled = styled.map(_score_bg,      subset=["可信评分"] if "可信评分" in disp.columns else [])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Worthy section ───────────────────────────────────────────────────────
    if worthy_df.empty:
        st.warning("🔍 当前筛选条件下无策略同时满足全部 4 个条件（评分>70 + 熊市胜率>50% + 样本>25笔 + 样本外胜率>60%）")
        st.caption("放宽板块/策略选择，或等待更多历史数据积累。")
    else:
        st.markdown(
            f"<div style='background:#0e3320;border:1px solid #00CC96;border-radius:8px;"
            f"padding:10px 16px;margin-bottom:12px'>"
            f"<span style='color:#00CC96;font-size:1.1rem;font-weight:700'>✅ 值得交易</span>"
            f"<span style='color:#aaa;font-size:0.9rem;margin-left:12px'>共 {len(worthy_df)} 个策略通过所有筛选</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        _render_table(worthy_df.sort_values("score", ascending=False))

    st.divider()

    # ── Stress test summary ──────────────────────────────────────────────────
    st.subheader("⚡ 特殊时期压力测试")
    st.caption("只有在危机期间仍保持 >50% 胜率，策略才算真正经得起考验")

    period_labels = list(STRESS_PERIODS.keys())
    stress_rows = []
    for _, row in display_df.iterrows():
        for pk in period_labels:
            col = f"stress_{pk}"
            wr = row.get(col)
            stress_rows.append({
                "股票":     row.get("ticker", ""),
                "策略":     row.get("strategy_name", ""),
                "时期":     pk,
                "胜率%":    wr,
                "状态":     ("✅通过" if (wr is not None and wr > 50) else
                             ("⚠️无数据" if wr is None else "❌不达标")),
            })
    if stress_rows:
        stress_summary = pd.DataFrame(stress_rows)
        # Pivot for compact display
        pivot = stress_summary.pivot_table(
            index=["股票", "策略"],
            columns="时期",
            values="胜率%",
            aggfunc="first",
        ).reset_index()
        pivot.columns.name = None
        # Merge status
        pass_counts = (
            stress_summary[stress_summary["状态"] == "✅通过"]
            .groupby(["股票", "策略"])
            .size()
            .reset_index(name="通过压力测试数")
        )
        pivot = pivot.merge(pass_counts, on=["股票", "策略"], how="left").fillna({"通过压力测试数": 0})
        pivot["通过压力测试数"] = pivot["通过压力测试数"].astype(int)
        st.dataframe(
            pivot.style.format(
                {pk: "{:.1f}%" for pk in period_labels if pk in pivot.columns},
                na_rep="无数据",
            ).map(
                lambda v: "background-color:rgba(0,204,150,0.2)" if isinstance(v, float) and v > 50 else
                          ("background-color:rgba(239,85,59,0.15)" if isinstance(v, float) and v <= 50 else ""),
                subset=[pk for pk in period_labels if pk in pivot.columns],
            ),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ── OOS Validation summary ───────────────────────────────────────────────
    st.subheader("📐 样本外验证 (训练期 2021-2023 vs 验证期 2024)")
    oos_df = display_df.dropna(subset=["train_win_rate", "oos_win_rate"]).copy()
    if oos_df.empty:
        st.info("暂无2024年数据（样本外验证需要2024年有交易信号）")
    else:
        oos_df["OOS状态"] = oos_df["oos_win_rate"].apply(
            lambda v: "✅ >60% 样本外有效" if v > 60 else "❌ ≤60% 过拟合风险"
        )
        oos_show = oos_df[["ticker","strategy_name","train_win_rate","oos_win_rate","OOS状态"]].rename(columns={
            "ticker": "股票", "strategy_name": "策略",
            "train_win_rate": "训练期胜率%(21-23)", "oos_win_rate": "样本外胜率%(2024)",
        })
        st.dataframe(
            oos_show.style.format({
                "训练期胜率%(21-23)": "{:.1f}%",
                "样本外胜率%(2024)":  "{:.1f}%",
            }, na_rep="N/A").map(
                lambda v: "color:#00CC96;font-weight:bold" if "✅" in str(v) else
                          ("color:#EF553B" if "❌" in str(v) else ""),
                subset=["OOS状态"],
            ),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ── Not recommended (collapsed) ─────────────────────────────────────────
    if not other_df.empty:
        with st.expander(f"❌ 不建议实盘（{len(other_df)} 个策略未通过筛选）", expanded=False):
            st.caption(
                "以下策略未同时满足：评分>70 · 熊市胜率>50% · 样本>25笔 · 样本外胜率>60%，"
                "仅供研究参考"
            )
            _render_table(other_df.sort_values("score", ascending=False))

    # ── Scoring guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📖 评分体系说明 & 风险提示"):
        st.markdown(f"""
**可信评分 (0-100分) 构成**

| 维度 | 权重 | 说明 |
|------|------|------|
| 胜率 | 0-35分 | 50%→0分, 85%→35分 |
| 样本量 | 0-20分 | 40+笔→满分 |
| 熊市表现 | 0-25分 | 2022年全年胜率, 无数据→10分中性 |
| 夏普比率 | 0-15分 | Sharpe 2.0→15分 |
| 最大回撤 | 0-5分 | 回撤越小越高分 |

**✅值得交易 需同时满足**
1. 可信评分 > 70
2. 2022熊市胜率 > 50%
3. 交易次数 > {_MIN_RELIABLE} 笔
4. 2024年样本外胜率 > 60%

**成本说明**: 每笔交易已扣除 0.1%手续费 + 0.05%滑点 = 共0.3%往返成本

**高风险信号⚡ (已标注但不影响策略计分)**
- VIX > 30：恐慌市场信号
- 接飞刀：前3天出现>8%单日跌幅
- 疑似财报：前后3天内出现>5%单日大幅波动

**市场状态建议**
- 🟢 牛市（SPY > 200MA +3%）：正常使用信号
- 🟡 震荡（200MA ±3%）：建议减半仓位
- 🔴 熊市（SPY < 200MA -3%）：禁止做多信号

*数据来源: Yahoo Finance · 历史表现不代表未来收益 · 仅供研究参考*
""")


# ── Main app ───────────────────────────────────────────────────────────────────

def _check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("🔐 AI 半导体研究系统")
    st.markdown("请输入访问密码以继续使用")
    pwd = st.text_input("密码", type="password", key="_pwd_input")
    if st.button("确认", type="primary"):
        try:
            correct = st.secrets["WYF197858"]
        except Exception:
            correct = ""
        if pwd == correct and correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ 密码错误，拒绝访问")
    st.stop()


def main():
    _check_password()
    wl = load_watchlist()
    all_tickers  = wl["ticker"].tolist()
    etf_tickers  = set(wl[wl["sector"] == _ETF_SECTOR]["ticker"].tolist())

    # 保留板块顺序（CSV 中出现的顺序）
    sectors_ordered = list(dict.fromkeys(wl["sector"].tolist()))
    sector_options  = ["全部"] + sectors_ordered

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 AI 半导体研究")
        st.caption(f"更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.divider()

        if st.button("🔄 刷新数据", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # 板块下拉选择器，默认 AI算力/GPU/芯片
        default_idx = sector_options.index(_DEFAULT_SECTOR) if _DEFAULT_SECTOR in sector_options else 0
        selected_sector = st.selectbox("板块选择", sector_options, index=default_idx)

        period = st.selectbox("时间区间", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

        st.divider()
        st.markdown("**核心研究逻辑**")
        notes = [
            "股价领先业绩 1-2 季度",
            "HBM 出货量 = MU 核心变量",
            "CoWoS 利用率 = TSM 领先指标",
            "大厂 CapEx 驱动整个供应链",
            "台股 ADR 溢价监控套利信号",
            "SOX 作为板块 beta 基准",
        ]
        for n in notes:
            st.caption(f"• {n}")

        # ── 数据源指示器（左下角）────────────────────────────────────────────
        st.divider()
        if is_futu_connected():
            st.markdown(
                "<div style='font-size:0.82rem;font-weight:600;color:#00CC96;'>"
                "✅ 数据来源: 富途牛牛实时</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.82rem;font-weight:600;color:#FFA500;'>"
                "⚠️ 数据来源: Yahoo Finance延迟15分钟</div>",
                unsafe_allow_html=True,
            )

    # ── 板块过滤 ─────────────────────────────────────────────────────────────
    if selected_sector == "全部":
        selected = all_tickers
    else:
        selected = wl[wl["sector"] == selected_sector]["ticker"].tolist()

    # Beta 计算只用普通股（排除 ETF / 指数）
    stock_tickers = [t for t in selected if t not in etf_tickers and not t.startswith("^")]

    sector_label = selected_sector if selected_sector != "全部" else "全部板块"
    st.title("🔬 AI 半导体股票研究系统")
    st.caption(f"当前板块: {sector_label}（{len(selected)} 只）  ·  数据来源: 富途牛牛实时 ✅")

    # ── 严选模式横幅 ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(90deg,#1a1a2e,#16213e);border:1px solid #4a9eff;"
        "border-radius:8px;padding:10px 16px;margin-bottom:8px;"
        "display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>"
        "<div>"
        "<span style='font-size:1.05rem;font-weight:700;color:#4a9eff'>交易模式：严选模式 / High-Conviction Mode</span>"
        "</div>"
        "<div style='font-size:0.82rem;color:#aaa;max-width:600px'>"
        "本模式目标是提高信号质量、降低回撤和强化纪律，不保证盈利。"
        "<span style='color:#00CC96'>A+</span> 可交易 · "
        "<span style='color:#FFA500'>A</span> 候选仅观察 · "
        "<span style='color:#FFD700'>B</span> 参考 · "
        "<span style='color:#888'>C</span> 仅观察不建议操作"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 顶部状态栏 (常驻) ──────────────────────────────────────────────────────
    _spy_result = c_spy_state()
    _live_state, _live_vix = (_spy_result if isinstance(_spy_result, tuple) else (_spy_result, None))
    _radar_count  = len(st.session_state.get("radar_signals", []))
    _radar_sigs_s = st.session_state.get("radar_signals", [])
    _aplus_count  = sum(1 for s in _radar_sigs_s if s.get("signal_level") == "A+")
    _a_count      = sum(1 for s in _radar_sigs_s if s.get("signal_level") == "A")
    _b_count      = sum(1 for s in _radar_sigs_s if s.get("signal_level") == "B")
    _c_count      = sum(1 for s in _radar_sigs_s if s.get("signal_level") == "C")

    _s_price = _live_state.get("spy_price") if _live_state else None
    _ma200   = _live_state.get("ma200")     if _live_state else None
    _ratio   = _live_state.get("ratio_pct") if _live_state else None
    _st_emoji = _live_state.get("emoji","⚪") if _live_state else "⚪"
    _st_label = _live_state.get("label","未知") if _live_state else "未知"
    _st_key   = _live_state.get("state","unknown") if _live_state else "unknown"
    _st_color = {"bull":"#00CC96","sideways":"#FFA500","bear":"#EF553B"}.get(_st_key,"#888")

    _vix_str  = f"{_live_vix:.1f}" if _live_vix else "N/A"
    _vix_c    = "#EF553B" if (_live_vix or 0) > 30 else ("#FFA500" if (_live_vix or 0) > 20 else "#00CC96")
    _price_str = f"SPY ${_s_price:.2f}" if _s_price else "SPY N/A"
    _ma_str    = f"| 200MA ${_ma200:.2f}" if _ma200 else ""
    _rat_str   = f"| 偏离 {_ratio:+.2f}%" if _ratio is not None else ""

    st.markdown(
        f"<div style='background:#12122a;border:1px solid {_st_color};border-radius:8px;"
        f"padding:8px 16px;margin-bottom:12px;"
        f"display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>"
        f"<span style='font-size:1.0rem;font-weight:700;color:{_st_color}'>"
        f"{_st_emoji} {_st_label}</span>"
        f"<span style='font-size:0.85rem;color:#aaa'>{_price_str} {_ma_str} {_rat_str}</span>"
        f"<span style='font-size:0.85rem'>VIX: <b style='color:{_vix_c}'>{_vix_str}</b></span>"
        f"<span style='font-size:0.85rem;color:#82C8FF'>📡 今日信号: <b>{_radar_count}</b>"
        f"&nbsp;|&nbsp;<b style='color:#00CC96'>A+:{_aplus_count}</b>"
        f"&nbsp;|&nbsp;<b style='color:#FFA500'>A:{_a_count}</b>"
        f"&nbsp;|&nbsp;<b style='color:#FFD700'>B:{_b_count}</b>"
        f"&nbsp;|&nbsp;<b style='color:#888'>C:{_c_count}</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 市场概览", "💰 估值分析", "🔗 供应链追踪", "📈 财报分析",
        "📰 新闻监控", "📡 财报雷达", "🎯 策略回测", "📻 实时雷达",
        "🎯 历史验证", "🔎 单股查询",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1  市场概览
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header(f"每日市场概览 · {sector_label}")

        with st.spinner("加载行情数据..."):
            daily_df = c_daily(tuple(selected))

        if not daily_df.empty:
            # Metric cards：最多显示前 6 只
            cards = daily_df.head(6)
            cols = st.columns(min(len(cards), 6))
            for i, (_, row) in enumerate(cards.iterrows()):
                price = row.get("Price")
                chg = row.get("Change%")
                with cols[i]:
                    st.metric(
                        label=row["Ticker"],
                        value=f"${price:.2f}" if pd.notna(price) else "N/A",
                        delta=f"{chg:+.2f}%" if pd.notna(chg) else None,
                    )

            st.divider()

            # Full table
            show_cols = [c for c in ["Ticker", "Price", "Change%", "Vol/AvgVol", "% from High", "52W High", "52W Low"] if c in daily_df.columns]
            num_fmt = {
                "Price": "${:.2f}", "Change%": "{:+.2f}%",
                "Vol/AvgVol": "{:.2f}x", "% from High": "{:.1f}%",
                "52W High": "${:.2f}", "52W Low": "${:.2f}",
            }
            fmt = {k: v for k, v in num_fmt.items() if k in show_cols}
            color_cols = [c for c in ["Change%", "% from High"] if c in show_cols]

            styled = daily_df[show_cols].style.format(fmt, na_rep="N/A")
            if color_cols:
                styled = styled.map(sign_color, subset=color_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("归一化价格走势")

        with st.spinner("加载历史行情..."):
            price_hist = c_history(tuple(selected), period)

        if price_hist:
            fig = go.Figure()
            for tk, hist in price_hist.items():
                if hist.empty:
                    continue
                base = hist["Close"].iloc[0]
                norm = (hist["Close"] / base * 100).round(2)
                fig.add_trace(go.Scatter(x=hist.index, y=norm, name=tk, mode="lines",
                                         hovertemplate=f"{tk}: %{{y:.1f}}<extra></extra>"))
            fig.update_layout(
                title=f"归一化价格 (基准=100, {period})",
                yaxis_title="相对价格",
                hovermode="x unified",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("SOX Beta 系数")
        with st.spinner("计算 Beta..."):
            betas = c_beta(tuple(stock_tickers), period)

        if betas:
            beta_df = (
                pd.DataFrame(list(betas.items()), columns=["Ticker", "Beta vs SOX"])
                .dropna()
                .sort_values("Beta vs SOX", ascending=False)
            )
            fig_b = px.bar(beta_df, x="Ticker", y="Beta vs SOX",
                           color="Beta vs SOX", color_continuous_scale="RdYlGn",
                           text="Beta vs SOX", title=f"Beta vs SOX ({period})")
            fig_b.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_b.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="β=1")
            fig_b.update_layout(height=340, showlegend=False, **CHART_LAYOUT)
            st.plotly_chart(fig_b, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2  估值分析
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("估值分析")
        st.caption("Trailing PE / Forward PE / PEG — 横向比较，关注 Forward PE 折价幅度")

        with st.spinner("加载估值数据..."):
            val_df = c_valuation(tuple(selected))

        if not val_df.empty:
            show = [c for c in ["Ticker", "Name", "Trailing PE", "Forward PE", "PEG",
                                 "P/B", "P/S", "EV/EBITDA", "Market Cap (B)", "Analyst Target", "Upside%"]
                    if c in val_df.columns]
            fmt_v = {
                "Trailing PE": "{:.1f}", "Forward PE": "{:.1f}", "PEG": "{:.2f}",
                "P/B": "{:.1f}", "P/S": "{:.1f}", "EV/EBITDA": "{:.1f}",
                "Market Cap (B)": "${:.0f}B", "Analyst Target": "${:.2f}", "Upside%": "{:+.1f}%",
            }
            fmt_v = {k: v for k, v in fmt_v.items() if k in show}
            styled_v = val_df[show].style.format(fmt_v, na_rep="N/A")
            if "Upside%" in show:
                styled_v = styled_v.map(sign_color, subset=["Upside%"])
            st.dataframe(styled_v, use_container_width=True, hide_index=True)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                try:
                    for col in ["Trailing PE", "Forward PE"]:
                        if col not in val_df.columns:
                            val_df[col] = None
                    pe_melt = val_df[["Ticker", "Trailing PE", "Forward PE"]].copy()
                    pe_melt = pe_melt.melt(id_vars="Ticker", var_name="Type", value_name="PE")
                    pe_melt = pe_melt.dropna(subset=["PE"])
                    if not pe_melt.empty:
                        fig_pe = px.bar(pe_melt, x="Ticker", y="PE", color="Type", barmode="group",
                                        title="Trailing PE vs Forward PE",
                                        color_discrete_map={"Trailing PE": "#636EFA", "Forward PE": "#EF553B"})
                        fig_pe.update_layout(height=380, **CHART_LAYOUT)
                        st.plotly_chart(fig_pe, use_container_width=True)
                    else:
                        st.caption("PE 数据暂不可用")
                except Exception as e:
                    st.caption(f"PE 图表暂不可用: {e}")

            with col2:
                try:
                    if "PEG" not in val_df.columns:
                        val_df["PEG"] = None
                    peg_d = val_df[["Ticker", "PEG"]].dropna(subset=["PEG"])
                    if not peg_d.empty:
                        fig_peg = px.bar(peg_d, x="Ticker", y="PEG",
                                         color="PEG", color_continuous_scale="RdYlGn_r",
                                         text="PEG", title="PEG Ratio (< 1 低估 / < 2 合理)")
                        fig_peg.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        fig_peg.add_hline(y=1, line_dash="dash", line_color="#EF553B", annotation_text="PEG=1")
                        fig_peg.add_hline(y=2, line_dash="dot", line_color="orange", annotation_text="PEG=2")
                        fig_peg.update_layout(height=380, showlegend=False, **CHART_LAYOUT)
                        st.plotly_chart(fig_peg, use_container_width=True)
                    else:
                        st.caption("PEG 数据暂不可用")
                except Exception as e:
                    st.caption(f"PEG 图表暂不可用: {e}")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3  供应链追踪
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("AI 供应链追踪")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏭 TSM ADR 溢价监控")
            st.caption("TSM (NYSE) vs 台积电 2330.TW · ADR=5股 · 汇率实时换算")

            with st.spinner("计算 ADR 溢价..."):
                adr_df = c_adr(period)

            if not adr_df.empty:
                latest_p = adr_df["ADR_Premium%"].iloc[-1]
                avg_p = adr_df["ADR_Premium%"].mean()
                m1, m2 = st.columns(2)
                m1.metric("当前 ADR 溢价", f"{latest_p:+.2f}%",
                          f"{latest_p - avg_p:+.2f}% vs 均值")
                m2.metric(f"均值溢价 ({period})", f"{avg_p:+.2f}%")

                fig_adr = go.Figure()
                fig_adr.add_trace(go.Scatter(
                    x=adr_df.index, y=adr_df["ADR_Premium%"],
                    mode="lines", fill="tozeroy",
                    line=dict(color="#00CC96", width=1.5),
                    name="ADR溢价%",
                ))
                fig_adr.add_hline(y=avg_p, line_dash="dash", line_color="orange",
                                   annotation_text=f"均值 {avg_p:.1f}%")
                fig_adr.add_hline(y=0, line_color="gray", line_width=1)
                fig_adr.update_layout(title="TSM ADR 溢价率", height=300,
                                       yaxis_title="溢价 (%)", **CHART_LAYOUT)
                st.plotly_chart(fig_adr, use_container_width=True)
            else:
                st.warning("ADR 数据暂不可用（可能是市场休市或网络问题）")

        with col2:
            st.subheader("💰 超大规模云厂 AI CapEx")
            st.caption("MSFT / GOOGL / META / AMZN 季度资本支出 (十亿美元)")

            with st.spinner("加载 CapEx..."):
                capex_df = c_capex()

            if not capex_df.empty:
                fig_cx = px.bar(capex_df.sort_values("Quarter"),
                                x="Quarter", y="CapEx_B", color="Company",
                                barmode="group", text="CapEx_B",
                                title="季度 CapEx (B$)")
                fig_cx.update_traces(texttemplate="$%{text:.1f}B", textposition="outside")
                fig_cx.update_layout(height=310, xaxis_tickangle=-30, **CHART_LAYOUT)
                st.plotly_chart(fig_cx, use_container_width=True)

                # Total trend
                total = capex_df.groupby("Quarter")["CapEx_B"].sum().reset_index()
                total.columns = ["Quarter", "Total_CapEx_B"]
                fig_tot = px.line(total.sort_values("Quarter"),
                                  x="Quarter", y="Total_CapEx_B",
                                  markers=True, title="4家合计 CapEx 趋势 (B$)")
                fig_tot.update_layout(height=250, **CHART_LAYOUT)
                st.plotly_chart(fig_tot, use_container_width=True)
            else:
                st.warning("CapEx 数据加载中，请稍后刷新")

        st.divider()
        st.subheader("📋 HBM / CoWoS 手动追踪表")
        st.caption("请在财报/TrendForce发布后手动填写，保存后写入 data/supply_chain_manual.csv")

        hbm_df = get_hbm_manual_data()
        c_left, c_right = st.columns([4, 1])
        with c_left:
            edited_hbm = st.data_editor(
                hbm_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Date": st.column_config.TextColumn("季度", width="small"),
                    "HBM_MU_Rev_B": st.column_config.NumberColumn("MU HBM营收(B$)", format="$%.2f"),
                    "HBM_Share%": st.column_config.NumberColumn("HBM市占率%", format="%.1f%%"),
                    "CoWoS_Util%": st.column_config.NumberColumn("CoWoS利用率%", format="%.1f%%"),
                    "DRAM_ASP_Change%": st.column_config.NumberColumn("DRAM ASP变化%", format="%.1f%%"),
                },
            )
        with c_right:
            st.info(
                "**信号解读**\n\n"
                "CoWoS > 90% → TSM AI收入强\n\n"
                "HBM份额↑ → MU估值提升\n\n"
                "DRAM ASP↑ → MU毛利改善"
            )
        if st.button("💾 保存供应链数据"):
            os.makedirs("data", exist_ok=True)
            edited_hbm.to_csv("data/supply_chain_manual.csv", index=False)
            st.success("✅ 已保存到 data/supply_chain_manual.csv")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4  财报分析
    # ════════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("财报分析")

        with st.spinner("加载财报数据..."):
            earn_df = c_earnings(tuple(selected))

        if not earn_df.empty:
            # KPI row
            m1, m2, m3, m4 = st.columns(4)
            rg = earn_df["Revenue Growth YoY%"].mean() if "Revenue Growth YoY%" in earn_df else None
            gm = earn_df["Gross Margin%"].mean() if "Gross Margin%" in earn_df else None
            beats = earn_df[earn_df.get("EPS Beat%", pd.Series(dtype=float)).fillna(0).gt(0)].shape[0] if "EPS Beat%" in earn_df.columns else 0
            m1.metric("平均营收增速 YoY", f"{rg:.1f}%" if pd.notna(rg) else "N/A")
            m2.metric("平均毛利率", f"{gm:.1f}%" if pd.notna(gm) else "N/A")
            m3.metric("EPS Beat", f"{beats}/{len(earn_df)}")
            next_earn = earn_df["Next Earnings"].dropna().iloc[0] if "Next Earnings" in earn_df.columns and earn_df["Next Earnings"].notna().any() else "N/A"
            m4.metric("最近财报日", str(next_earn))

            st.divider()

            earn_cols = [c for c in [
                "Ticker", "Name", "Revenue TTM (B)", "Revenue Growth YoY%",
                "Gross Margin%", "Operating Margin%", "EPS (TTM)", "EPS Forward",
                "EPS Beat%", "Next Earnings",
            ] if c in earn_df.columns]
            fmt_e = {
                "Revenue TTM (B)": "${:.2f}B",
                "Revenue Growth YoY%": "{:+.1f}%",
                "Gross Margin%": "{:.1f}%",
                "Operating Margin%": "{:.1f}%",
                "EPS (TTM)": "${:.2f}",
                "EPS Forward": "${:.2f}",
                "EPS Beat%": "{:+.1f}%",
            }
            fmt_e = {k: v for k, v in fmt_e.items() if k in earn_cols}
            styled_e = earn_df[earn_cols].style.format(fmt_e, na_rep="N/A")
            color_e = [c for c in ["Revenue Growth YoY%", "EPS Beat%"] if c in earn_cols]
            if color_e:
                styled_e = styled_e.map(sign_color, subset=color_e)
            st.dataframe(styled_e, use_container_width=True, hide_index=True)

            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                try:
                    for col in ["Gross Margin%", "Operating Margin%"]:
                        if col not in earn_df.columns:
                            earn_df[col] = None
                    m_melt = earn_df[["Ticker", "Gross Margin%", "Operating Margin%"]].copy()
                    m_melt = m_melt.melt(id_vars="Ticker", var_name="Type", value_name="Margin%")
                    m_melt = m_melt.dropna(subset=["Margin%"])
                    if not m_melt.empty:
                        fig_m = px.bar(m_melt, x="Ticker", y="Margin%", color="Type",
                                       barmode="group", title="毛利率 vs 营业利润率")
                        fig_m.update_layout(height=360, **CHART_LAYOUT)
                        st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.caption("利润率数据暂不可用")
                except Exception as e:
                    st.caption(f"利润率图表暂不可用: {e}")

            with c2:
                try:
                    if "Revenue Growth YoY%" not in earn_df.columns:
                        earn_df["Revenue Growth YoY%"] = None
                    rg_d = earn_df[["Ticker", "Revenue Growth YoY%"]].dropna(subset=["Revenue Growth YoY%"])
                    if not rg_d.empty:
                        fig_rg = px.bar(rg_d, x="Ticker", y="Revenue Growth YoY%",
                                        color="Revenue Growth YoY%",
                                        color_continuous_scale="RdYlGn",
                                        text="Revenue Growth YoY%",
                                        title="营收同比增速 YoY%")
                        fig_rg.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
                        fig_rg.add_hline(y=0, line_color="gray")
                        fig_rg.update_layout(height=360, showlegend=False, **CHART_LAYOUT)
                        st.plotly_chart(fig_rg, use_container_width=True)
                    else:
                        st.caption("营收增速数据暂不可用")
                except Exception as e:
                    st.caption(f"营收增速图表暂不可用: {e}")

        st.divider()
        st.subheader("📋 Forward Guidance 追踪")
        st.caption("每次财报后手动更新 — 这是股价领先指标的核心输入")
        guid_df = get_guidance_tracker()
        edited_guid = st.data_editor(
            guid_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Guidance_Direction": st.column_config.SelectboxColumn(
                    "方向", options=["beat", "inline", "miss", "raised", "cut"]
                )
            },
        )
        if st.button("💾 保存 Guidance 数据"):
            os.makedirs("data", exist_ok=True)
            edited_guid.to_csv("data/earnings_guidance.csv", index=False)
            st.success("✅ 已保存到 data/earnings_guidance.csv")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 5  新闻监控
    # ════════════════════════════════════════════════════════════════════════════
    with tab5:
        st.header("新闻监控")

        col_search, col_filter = st.columns([3, 1])
        with col_search:
            search = st.text_input("🔍 关键词过滤", placeholder="HBM / CoWoS / guidance ...")
        with col_filter:
            impact_f = st.selectbox("影响级别", ["全部", "High", "Medium", "Low"])

        with st.spinner("抓取 RSS 新闻..."):
            news_df = c_news()

        if not news_df.empty:
            filtered = news_df.copy()
            if search:
                mask = (filtered["Title"].str.contains(search, case=False, na=False) |
                        filtered["Summary"].str.contains(search, case=False, na=False))
                filtered = filtered[mask]
            if impact_f != "全部":
                filtered = filtered[filtered["Impact"].str.contains(impact_f, na=False)]

            st.caption(f"显示 {len(filtered)} / {len(news_df)} 条")

            for _, row in filtered.head(40).iterrows():
                impact_icon = row["Impact"].split()[0]
                with st.container():
                    ic, ct = st.columns([1, 11])
                    with ic:
                        st.markdown(f"### {impact_icon}")
                        st.caption(row["Sentiment"])
                    with ct:
                        link = row.get("Link", "")
                        title_md = f"**[{row['Title']}]({link})**" if link else f"**{row['Title']}**"
                        st.markdown(title_md)
                        st.caption(f"{row.get('Date', '')}  ·  {row.get('Source', '')}  ·  {row['Impact']}")
                        summary = row.get("Summary", "")
                        if summary:
                            with st.expander("摘要"):
                                st.write(summary)
                st.divider()
        else:
            st.warning("新闻加载失败，请检查网络连接")
            st.subheader("监控关键词参考")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**高影响**")
                for kw in KEYWORDS["high"]:
                    st.caption(f"• {kw}")
            with c2:
                st.markdown("**中影响**")
                for kw in KEYWORDS["medium"][:12]:
                    st.caption(f"• {kw}")


    # ════════════════════════════════════════════════════════════════════════════
    # TAB 6  财报雷达
    # ════════════════════════════════════════════════════════════════════════════
    with tab6:
        st.header("财报雷达")
        st.caption(
            f"自动扫描 {len(SCAN_TICKERS)} 只标普500+纳斯达克100重点成分股，"
            "AI 评分筛选即将发布财报的投资机会 — 仅供研究参考，不构成投资建议"
        )

        # ── 顶部控制栏 ───────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
        with ctrl1:
            scan_days = st.radio(
                "扫描窗口", [7, 14], index=1, horizontal=True,
                format_func=lambda x: f"未来 {x} 天",
            )
        with ctrl2:
            st.caption("")
            st.caption(
                "🔥 高分 + 回调>5% = 最佳窗口  "
                "⚡ 连续3季以上beat"
            )
        with ctrl3:
            if st.button("🔄 重新扫描", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        # ── 数据加载 ─────────────────────────────────────────────────────────
        with st.spinner(f"扫描财报日历（约需 30–60 秒，数据缓存1小时）…"):
            radar_df = c_radar(scan_days)

        if radar_df.empty:
            st.warning(
                f"未来 {scan_days} 天内暂无重点成分股财报，"
                "或数据加载失败（请检查网络/代理后重新扫描）"
            )
        else:
            st.success(f"共发现 **{len(radar_df)}** 只即将发布财报的股票，按综合评分排序")

            # ── 评分排行榜表格 ────────────────────────────────────────────────
            st.subheader("评分排行榜")

            def _score_bg(val):
                if pd.isna(val):
                    return ""
                v = float(val)
                if v >= 70:
                    return "background-color: rgba(0,204,150,0.25)"
                if v >= 50:
                    return "background-color: rgba(255,200,0,0.15)"
                return ""

            show_r = [c for c in [
                "标记", "Ticker", "公司名", "财报日期", "评分",
                "预期EPS", "上季EPS", "EPS增速%", "营收增速%",
                "Forward PE", "当前价", "目标价", "上行空间%",
                "30天涨跌%", "推荐理由",
            ] if c in radar_df.columns]

            fmt_r = {
                "评分":        "{:.0f}",
                "预期EPS":     "${:.2f}",
                "上季EPS":     "${:.2f}",
                "EPS增速%":    "{:+.1f}%",
                "营收增速%":   "{:+.1f}%",
                "Forward PE":  "{:.1f}",
                "当前价":      "${:.2f}",
                "目标价":      "${:.2f}",
                "上行空间%":   "{:+.1f}%",
                "30天涨跌%":   "{:+.1f}%",
            }
            fmt_r = {k: v for k, v in fmt_r.items() if k in show_r}

            styled_r = radar_df[show_r].style.format(fmt_r, na_rep="N/A")
            if "评分" in show_r:
                styled_r = styled_r.map(_score_bg, subset=["评分"])
            for col in ["上行空间%", "EPS增速%", "营收增速%", "30天涨跌%"]:
                if col in show_r:
                    styled_r = styled_r.map(sign_color, subset=[col])

            st.dataframe(styled_r, use_container_width=True, hide_index=False)

            st.divider()

            # ── 图表区 ────────────────────────────────────────────────────────
            col_c1, col_c2 = st.columns(2)

            with col_c1:
                bar_h = max(320, len(radar_df) * 24)
                fig_bar = px.bar(
                    radar_df.sort_values("评分", ascending=True),
                    x="评分", y="Ticker",
                    orientation="h",
                    color="评分",
                    color_continuous_scale="RdYlGn",
                    text="评分",
                    title="AI 综合评分分布",
                )
                fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                fig_bar.update_layout(height=bar_h, showlegend=False, **CHART_LAYOUT)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_c2:
                try:
                    sc_data = radar_df[
                        ["Ticker", "评分", "上行空间%", "营收增速%", "公司名"]
                    ].dropna(subset=["上行空间%"])
                    if not sc_data.empty:
                        fig_sc = px.scatter(
                            sc_data,
                            x="上行空间%", y="评分",
                            text="Ticker",
                            color="营收增速%",
                            color_continuous_scale="RdYlGn",
                            hover_data=["公司名", "营收增速%"],
                            title="评分 vs 分析师目标上行空间",
                        )
                        fig_sc.update_traces(
                            textposition="top center",
                            marker=dict(size=10),
                        )
                        fig_sc.add_vline(x=0, line_color="gray", line_width=1)
                        fig_sc.update_layout(height=bar_h, **CHART_LAYOUT)
                        st.plotly_chart(fig_sc, use_container_width=True)
                    else:
                        st.caption("散点图需要目标价数据，当前暂无")
                except Exception as e:
                    st.caption(f"散点图暂不可用: {e}")

            st.divider()

            # ── 重点推荐 Top 3 ────────────────────────────────────────────────
            st.subheader("🎯 重点推荐 Top 3")

            picks = get_top_picks(radar_df, n=3)
            for idx, pick in enumerate(picks, 1):
                score_color = "#00CC96" if pick["score"] >= 70 else (
                    "#FFA500" if pick["score"] >= 50 else "#EF553B"
                )
                p_col1, p_col2 = st.columns([1, 5])

                with p_col1:
                    st.markdown(
                        f"<div style='text-align:center;font-size:2.4rem;"
                        f"font-weight:bold;color:{score_color};line-height:1.1'>"
                        f"{pick['score']}</div>"
                        f"<div style='text-align:center;font-size:0.65rem;color:#888'>综合评分</div>",
                        unsafe_allow_html=True,
                    )
                    if pick["flags"]:
                        st.markdown(
                            f"<div style='text-align:center;font-size:1.6rem;margin-top:4px'>"
                            f"{pick['flags']}</div>",
                            unsafe_allow_html=True,
                        )

                with p_col2:
                    header = f"**#{idx}  {pick['ticker']} — {pick['company']}**"
                    if pick["upside"] is not None:
                        header += f"　目标价上行 **{pick['upside']:+.1f}%**"
                    st.markdown(header)
                    st.caption(f"财报日期：{pick['date']}")
                    for detail in pick["details"]:
                        st.caption(f"• {detail}")

                if idx < len(picks):
                    st.divider()

            # ── 风险提示 ──────────────────────────────────────────────────────
            st.divider()
            with st.expander("⚠️ 风险提示 & 使用说明"):
                st.markdown("""
**财报前后波动风险**
- 财报窗口通常伴随高期权隐含波动率（IV），股价可能大幅偏离预期方向
- 即便业绩超预期，若前瞻指引低于市场预期，股价仍可能下跌（"buy the rumor, sell the news"）
- 🔥 标记代表高评分 + 近30天回调 > 5%，可能是较好的风险收益比入场点，但需结合整体市场环境判断

**评分说明**
| 维度 | 权重 | 判断标准 |
|------|------|---------|
| EPS 预期增速 | +20 | Forward EPS 同比 > 20% |
| 营收增速 | +20 | YoY > 15% |
| 连续 EPS beat | +20 | 过去4季均超预期 |
| Forward PE 估值 | +15 | 低于行业中位 PE |
| 分析师看多 | +15 | 综合评级均值 ≤ 2.0 |
| 机构持仓 | +10 | 机构持仓比例 > 60% |

**数据局限性**
- 财报日期基于 Yahoo Finance 预估，可能因公司临时调整而变更
- 分析师一致预期可能与最终结果存在较大偏差
- 本工具仅供研究参考，不构成任何投资建议
""")


    # ════════════════════════════════════════════════════════════════════════════
    # TAB 7  策略回测 Pro
    # ════════════════════════════════════════════════════════════════════════════
    with tab7:
        st.header("🎯 策略回测 Pro · AI时代短线策略验证")
        st.caption(
            f"训练期 2022-2023 | 验证期 2024-2025 · {len(ALL_BACKTEST_TICKERS)} 只股票 · "
            "4种短线策略 · 动态门槛/Kelly仓位/Sortino/Profit Factor · 7条实盘门槛 — 仅供研究参考"
        )

        # ── Session state 初始化 ──────────────────────────────────────────────
        for _k, _v in [("bt_flat_df",None),("bt_stock_df",None),
                        ("bt_spy_return",0.0),("bt_raw_results",None),
                        ("bt_market_state",None)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        # ── 今日市场状态 + 第二层执行建议 ────────────────────────────────────
        _spy_r = c_spy_state()
        live_state = st.session_state.get("bt_market_state") or (
            _spy_r[0] if isinstance(_spy_r, tuple) else _spy_r)
        if live_state:
            state_k   = live_state.get("state","unknown")
            emoji     = live_state.get("emoji","⚪")
            label     = live_state.get("label","未知")
            s_price   = live_state.get("spy_price")
            ma200     = live_state.get("ma200")
            ratio     = live_state.get("ratio_pct")
            pos_adv   = live_state.get("pos_advice","N/A")
            bg_color  = {"bull":"#0e3320","sideways":"#332a00","bear":"#330e0e"}.get(state_k,"#1a1a2e")
            border    = {"bull":"#00CC96","sideways":"#FFA500","bear":"#EF553B"}.get(state_k,"#888")
            detail    = ""
            if s_price: detail = f"SPY ${s_price:.2f}"
            if ma200:   detail += f" | 200MA ${ma200:.2f}"
            if ratio is not None: detail += f" | 偏离 {ratio:+.2f}%"
            exec_msg = {
                "bull":     "🟢 第二层执行：4种策略全开，标准仓位",
                "sideways": "🟡 第二层执行：仅策略3(RSI+布林带复合)，仓位×50%",
                "bear":     "🔴 第二层执行：仅观察模式，建议仓位0%⚠️",
            }.get(state_k, "⚪ 市场状态未知")
            st.markdown(
                f"<div style='background:{bg_color};border:1px solid {border};"
                f"border-radius:10px;padding:12px 20px;margin-bottom:16px'>"
                f"<div style='display:flex;align-items:center;gap:16px'>"
                f"<span style='font-size:2rem'>{emoji}</span>"
                f"<div style='flex:1'>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{border}'>今日市场状态：{label}</div>"
                f"<div style='font-size:0.8rem;color:#aaa;margin-top:2px'>{detail}</div>"
                f"<div style='font-size:0.82rem;color:{border};margin-top:4px'>{exec_msg}</div>"
                f"</div>"
                f"<div style='text-align:right;font-size:0.85rem;color:#ccc'>"
                f"仓位建议：<b style='color:{border}'>{pos_adv}</b></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # ── 控制栏 ────────────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 1, 1])
        with ctrl1:
            bt_sector_sel = st.selectbox(
                "扫描板块",
                ["全部板块（~{}只）".format(len(ALL_BACKTEST_TICKERS))]
                + [f"{s}（{len(v)}只）" for s, v in BACKTEST_SECTORS.items()],
                key="bt_sector_sel",
            )
        with ctrl2:
            bt_view_strat = st.selectbox(
                "展示策略",
                ["全部策略"] + list(STRATEGY_NAMES.values()),
                key="bt_view_strat",
            )
        with ctrl3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run_bt = st.button("🚀 运行回测", type="primary", use_container_width=True, key="bt_run")
        with ctrl4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            only_worthy = st.toggle("仅显示推荐", value=False, key="bt_only_worthy")

        st.divider()

        # ── 运行回测 ──────────────────────────────────────────────────────────
        if run_bt:
            # 解析板块选择
            if bt_sector_sel.startswith("全部板块"):
                bt_tickers = ALL_BACKTEST_TICKERS
            else:
                chosen_sector = bt_sector_sel.split("（")[0]
                bt_tickers = list(dict.fromkeys(BACKTEST_SECTORS.get(chosen_sector, [])))

            total_n = len(bt_tickers)
            st.info(f"🔍 开始扫描 **{total_n}** 只股票，每只跑 4 种策略，预计需要 1-2 分钟...")

            progress_bar = st.progress(0.0)
            status_box = st.empty()
            status_box.markdown("📡 **批量下载历史数据中…** 请稍候")

            def _on_progress(done: int, total: int, ticker: str) -> None:
                frac = 0.15 + 0.85 * done / total
                progress_bar.progress(min(frac, 1.0))
                status_box.markdown(
                    f"🔎 分析中 **({done}/{total})** — `{ticker}`"
                    f"&nbsp;&nbsp;{'█' * int(frac * 20)}{'░' * (20 - int(frac * 20))} "
                    f"**{frac * 100:.0f}%**"
                )

            raw_results, spy_ret, mkt_state = run_full_backtest(
                tickers=bt_tickers,
                progress_callback=_on_progress,
            )

            progress_bar.progress(1.0)
            valid = sum(1 for r in raw_results if not r.get("error"))
            status_box.success(
                f"✅ 回测完成！共分析 **{valid}/{total_n}** 只股票，"
                f"SPY 同期买入持有: **{spy_ret:+.1f}%**"
            )

            st.session_state.bt_raw_results  = raw_results
            st.session_state.bt_spy_return   = spy_ret
            st.session_state.bt_market_state = mkt_state
            st.session_state.bt_flat_df      = build_flat_df(raw_results)
            st.session_state.bt_stock_df     = build_stock_summary(raw_results)
            st.rerun()

        # ── 结果展示 ──────────────────────────────────────────────────────────
        flat_df  = st.session_state.bt_flat_df
        stock_df = st.session_state.bt_stock_df
        spy_ret  = st.session_state.bt_spy_return

        if flat_df is None:
            st.markdown(
                "<div style='text-align:center;padding:60px 0;color:#888'>"
                "<div style='font-size:3rem'>🎯</div>"
                "<div style='font-size:1.2rem;margin-top:12px'>点击上方「🚀 运行回测」开始扫描</div>"
                "<div style='font-size:0.85rem;margin-top:8px;color:#aaa'>"
                "训练期 2022-2023 | 验证期 2024-2025 · 首次约需 1-2 分钟下载数据</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif flat_df.empty:
            st.warning("回测完成，但未产生有效交易信号。请尝试更宽泛的板块选择。")
        else:
            # 策略 + 推荐过滤
            display_df = flat_df.copy()
            if bt_view_strat != "全部策略":
                strat_id = {v: k for k, v in STRATEGY_NAMES.items()}.get(bt_view_strat)
                if strat_id:
                    display_df = display_df[display_df["strategy"] == strat_id]
            if only_worthy:
                display_df = display_df[display_df["verdict"] == "✅值得实盘测试"]

            # ── 总览指标 ──────────────────────────────────────────────────────
            valid_wr = display_df.dropna(subset=["win_rate"])
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("扫描股票数", f"{display_df['ticker'].nunique()} 只")
            avg_wr = valid_wr["win_rate"].mean()
            m2.metric("平均胜率", f"{avg_wr:.1f}%" if not pd.isna(avg_wr) else "N/A")
            worthy_cnt = len(display_df[display_df["verdict"] == "✅值得实盘测试"])
            m3.metric("值得实盘策略", f"{worthy_cnt} 个")
            avg_pf = valid_wr["profit_factor"].mean() if "profit_factor" in valid_wr else None
            m4.metric("平均Profit Factor", f"{avg_pf:.2f}" if avg_pf and not pd.isna(avg_pf) else "N/A")
            avg_sharpe = valid_wr["sharpe"].mean() if "sharpe" in valid_wr else None
            m5.metric("平均夏普", f"{avg_sharpe:.2f}" if avg_sharpe and not pd.isna(avg_sharpe) else "N/A",
                      f"SPY持有: {spy_ret:+.1f}%")

            st.divider()

            # ── 子视图选择 ────────────────────────────────────────────────────
            view = st.radio(
                "查看视图",
                ["🏆 推荐排行榜", "📋 完整输出表", "📊 板块对比", "📈 策略比较",
                 "🔬 验证期分析", "⚡ 压力测试"],
                horizontal=True,
                key="bt_view_radio",
            )

            PRO_COLS = {
                "ticker":         "股票",
                "sector":         "板块",
                "strategy_name":  "策略",
                "vol_type":       "波动率类型",
                "trade_count":    "交易次数",
                "min_required":   "所需门槛",
                "win_rate":       "胜率%",
                "avg_return":     "均收益%",
                "max_loss":       "最大亏损%",
                "max_drawdown":   "最大回撤%",
                "profit_factor":  "PF",
                "sharpe":         "夏普",
                "sortino":        "Sortino",
                "kelly_conservative": "保守仓位%",
                "kelly_standard":     "标准仓位%",
                "bull_win_rate":  "牛市胜率%",
                "bear_win_rate":  "熊市胜率%",
                "train_win_rate": "训练期胜率%",
                "oos_win_rate":   "验证期胜率%",
                "avg_hold_days":  "均持仓天",
                "hold_style":     "持仓风格",
                "score":          "评分",
                "grade":          "评级",
                "verdict":        "最终结论",
            }
            PRO_FMT = {
                "胜率%": "{:.1f}%", "均收益%": "{:+.2f}%", "最大亏损%": "{:.2f}%",
                "最大回撤%": "{:.1f}%", "PF": "{:.2f}", "夏普": "{:.2f}", "Sortino": "{:.2f}",
                "保守仓位%": "{:.1f}%", "标准仓位%": "{:.1f}%",
                "牛市胜率%": "{:.1f}%", "熊市胜率%": "{:.1f}%",
                "训练期胜率%": "{:.1f}%", "验证期胜率%": "{:.1f}%",
                "均持仓天": "{:.1f}", "评分": "{:.1f}",
            }

            def _render_pro_table(df: pd.DataFrame, highlight_verdict: bool = True) -> None:
                avail = {k: v for k, v in PRO_COLS.items() if k in df.columns}
                disp  = df[list(avail.keys())].rename(columns=avail)
                fmt   = {k: v for k, v in PRO_FMT.items() if k in disp.columns}
                styled = disp.style.format(fmt, na_rep="N/A")
                for col in ["胜率%","均收益%","牛市胜率%","熊市胜率%","训练期胜率%","验证期胜率%"]:
                    if col in disp.columns:
                        styled = styled.map(sign_color, subset=[col])
                if highlight_verdict and "最终结论" in disp.columns:
                    def _verdict_color(v):
                        if "✅" in str(v): return "color:#00CC96;font-weight:bold"
                        if "❌" in str(v): return "color:#EF553B"
                        return "color:#FFA500"
                    styled = styled.map(_verdict_color, subset=["最终结论"])
                if "评分" in disp.columns:
                    def _score_bg(v):
                        if pd.isna(v): return ""
                        if float(v) >= 75: return "background-color:rgba(0,204,150,0.2)"
                        if float(v) >= 60: return "background-color:rgba(255,165,0,0.15)"
                        return ""
                    styled = styled.map(_score_bg, subset=["评分"])
                if "评级" in disp.columns:
                    def _grade_color(v):
                        return {"A+":"color:#00CC96;font-weight:700","A":"color:#82C8FF;font-weight:600",
                                "B":"color:#FFA500","C":"color:#EF553B","D":"color:#888"}.get(str(v),"")
                    styled = styled.map(_grade_color, subset=["评级"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

            # ════════════════════════════════════════════════════════════════
            # VIEW 0: 推荐排行榜
            # ════════════════════════════════════════════════════════════════
            if view == "🏆 推荐排行榜":
                worthy_df = display_df[display_df["verdict"] == "✅值得实盘测试"].copy()

                if worthy_df.empty:
                    st.warning("🔍 当前无策略同时满足全部 7 条门槛。以下显示评分 Top 20：")
                    top_df = display_df.dropna(subset=["score"]).nlargest(20,"score")
                else:
                    st.success(f"✅ 共 **{len(worthy_df)}** 个策略通过全部7条门槛")
                    top_df = worthy_df.sort_values("score", ascending=False).head(20)

                # 7条门槛说明卡片
                with st.expander("📋 7条实盘门槛", expanded=False):
                    st.markdown("""
| 条件 | 门槛 | 说明 |
|------|------|------|
| 可信评分 | ≥70分 | 样本量/夏普/熊市/验证/回撤综合评分 |
| 动态交易次数 | 依波动率 | 高波动≥25笔 / 中波动≥15笔 / 低波动≥10笔 |
| 夏普比率 | ≥1.0 | 风险调整收益 |
| Profit Factor | ≥1.5 | 总盈利/总亏损比值 |
| 熊市胜率 | ≥50% | 2022全年熊市期间表现 |
| 最大回撤 | ≤25% | 历史最大回撤控制 |
| 验证期胜率 | ≥55% | 2024-2025样本外验证 |
                    """)

                if not top_df.empty:
                    # 信号卡片式展示 Top 5
                    st.subheader("🥇 Top 推荐")
                    cols5 = st.columns(min(5, len(top_df)))
                    for ci, (_, row) in enumerate(top_df.head(5).iterrows()):
                        score = row.get("score", 0) or 0
                        sc = "#00CC96" if score >= 75 else ("#FFA500" if score >= 60 else "#EF553B")
                        grade = row.get("grade","D")
                        with cols5[ci]:
                            st.markdown(
                                f"<div style='border:1px solid {sc};border-radius:8px;padding:10px;text-align:center'>"
                                f"<div style='font-size:1.3rem;font-weight:700;color:#fff'>{row['ticker']}</div>"
                                f"<div style='font-size:0.75rem;color:#888'>{row.get('sector','')}</div>"
                                f"<div style='font-size:2rem;font-weight:bold;color:{sc};margin:4px 0'>{score:.0f}</div>"
                                f"<div style='font-size:0.7rem;color:#aaa'>评分</div>"
                                f"<div style='font-size:1.1rem;font-weight:600;color:{sc}'>{grade}</div>"
                                f"<div style='font-size:0.75rem;color:#ccc;margin-top:4px'>"
                                f"胜率: {(row.get('win_rate') or 0):.0f}% | PF: {(row.get('profit_factor') or 0):.1f}</div>"
                                f"<div style='font-size:0.75rem;color:#ccc'>"
                                f"夏普: {(row.get('sharpe') or 0):.2f} | 仓位: {(row.get('kelly_standard') or 0):.1f}%</div>"
                                f"<div style='font-size:0.7rem;color:{sc};margin-top:4px'>{row.get('strategy_name','')}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    st.divider()
                    st.subheader("详细数据")
                    _render_pro_table(top_df)

                # 7条门槛逐条检查（展开）
                if st.session_state.bt_raw_results and worthy_df.empty:
                    st.divider()
                    st.subheader("为何没有策略通过？（7条门槛逐条检查）")
                    best5 = display_df.dropna(subset=["score"]).nlargest(5,"score")
                    for _, row in best5.iterrows():
                        details = row.get("verdict_detail", [])
                        if isinstance(details, list) and details:
                            with st.expander(f"{row['ticker']} · {row.get('strategy_name','')} (评分 {row.get('score',0):.0f})"):
                                for d in details:
                                    st.markdown(f"- {d}")

            # ════════════════════════════════════════════════════════════════
            # VIEW 1: 完整输出表
            # ════════════════════════════════════════════════════════════════
            elif view == "📋 完整输出表":
                st.subheader("完整回测输出表（所有字段）")
                st.caption("包含：动态门槛 · PF · Sortino · Kelly仓位 · 训练/验证期胜率 · 持仓分析 · 7条门槛评级")

                # 样本不足警告
                insuf = display_df[display_df["sample_label"].str.contains("⚠️",na=False)] if "sample_label" in display_df else pd.DataFrame()
                if not insuf.empty:
                    st.info(f"⚠️ {len(insuf)} 个策略样本不足（已显示但可信度低）")

                top20 = display_df.dropna(subset=["win_rate"]).nlargest(20,"win_rate")
                if top20.empty:
                    st.warning("暂无有效信号")
                else:
                    labels = top20["ticker"] + " · " + top20["strategy_name"]
                    colors = top20["score"].apply(
                        lambda s: "#00CC96" if (s or 0) >= 75 else ("#FFA500" if (s or 0) >= 60 else "#EF553B")
                    )
                    fig_top = go.Figure(go.Bar(
                        x=top20["win_rate"], y=labels, orientation="h",
                        marker=dict(color=colors),
                        text=top20.apply(lambda r: f"{r['win_rate']:.0f}% ({r['trade_count']}笔)", axis=1),
                        textposition="outside",
                        customdata=top20[["trade_count","sharpe","profit_factor","score","grade"]].values,
                        hovertemplate=(
                            "<b>%{y}</b><br>胜率: %{x:.1f}%<br>"
                            "笔数: %{customdata[0]}<br>夏普: %{customdata[1]:.2f}<br>"
                            "PF: %{customdata[2]:.2f}<br>评分: %{customdata[3]:.1f} (%{customdata[4]})<extra></extra>"
                        ),
                    ))
                    fig_top.add_vline(x=55, line_dash="dash", line_color="gray", annotation_text="55%")
                    fig_top.update_layout(
                        title="胜率 Top20 (颜色 = 评分)",
                        height=max(400, len(top20)*30), showlegend=False, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

                st.divider()
                st.subheader("完整数据表")
                _render_pro_table(display_df.sort_values("score", ascending=False))

            # ════════════════════════════════════════════════════════════════
            # VIEW 2: 板块对比
            # ════════════════════════════════════════════════════════════════
            elif view == "📊 板块对比":
                st.subheader("各板块综合指标对比")
                sec_stats = get_sector_stats(display_df)

                if sec_stats.empty:
                    st.warning("数据不足")
                else:
                    col_l, col_r = st.columns(2)
                    with col_l:
                        fig_sec = px.bar(
                            sec_stats, x="avg_win_rate", y="sector", orientation="h",
                            color="avg_win_rate", color_continuous_scale="RdYlGn",
                            text="avg_win_rate", title="板块平均胜率",
                        )
                        fig_sec.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        fig_sec.add_vline(x=50, line_dash="dash", line_color="gray")
                        fig_sec.update_layout(height=max(380, len(sec_stats)*30),
                                               showlegend=False, **CHART_LAYOUT)
                        st.plotly_chart(fig_sec, use_container_width=True)

                    with col_r:
                        fig_pf = px.bar(
                            sec_stats.sort_values("avg_profit_factor", ascending=True),
                            x="avg_profit_factor", y="sector", orientation="h",
                            color="avg_profit_factor", color_continuous_scale="RdYlGn",
                            text="avg_profit_factor", title="板块平均 Profit Factor",
                        )
                        fig_pf.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        fig_pf.add_vline(x=1.5, line_dash="dash", line_color="#00CC96",
                                          annotation_text="PF=1.5")
                        fig_pf.update_layout(height=max(380, len(sec_stats)*30),
                                              showlegend=False, **CHART_LAYOUT)
                        st.plotly_chart(fig_pf, use_container_width=True)

                    sec_display = sec_stats.rename(columns={
                        "sector": "板块", "avg_win_rate": "平均胜率%",
                        "avg_trade_count": "平均交易次数", "avg_sharpe": "平均夏普",
                        "avg_profit_factor": "平均PF", "avg_score": "平均评分", "stock_count": "股票数",
                    })
                    fmt_sd = {k: v for k, v in {
                        "平均胜率%": "{:.1f}%", "平均交易次数": "{:.1f}",
                        "平均夏普": "{:.2f}", "平均PF": "{:.2f}", "平均评分": "{:.1f}",
                    }.items() if k in sec_display.columns}
                    st.dataframe(sec_display.style.format(fmt_sd, na_rep="N/A"),
                                  use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("各板块 Top 5（按评分）")
                    sec5_df = get_sector_top5(display_df, min_trades=3)
                    if not sec5_df.empty:
                        for sector_name, grp in sec5_df.groupby("sector", sort=False):
                            with st.expander(f"**{sector_name}**（{len(grp)}只）", expanded=False):
                                _render_pro_table(grp.drop(columns=["sector"]))

            # ════════════════════════════════════════════════════════════════
            # VIEW 3: 策略比较
            # ════════════════════════════════════════════════════════════════
            elif view == "📈 策略比较":
                st.subheader("4 种短线策略横向对比")
                strat_stats = get_strategy_stats(display_df)

                if strat_stats.empty:
                    st.warning("数据不足")
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        fig_s1 = px.bar(strat_stats, x="strategy_name", y="avg_win_rate",
                                        color="avg_win_rate", color_continuous_scale="RdYlGn",
                                        text="avg_win_rate", title="策略平均胜率%")
                        fig_s1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        fig_s1.add_hline(y=55, line_dash="dash", line_color="#00CC96", annotation_text="55%")
                        fig_s1.update_layout(height=340, showlegend=False, **CHART_LAYOUT)
                        st.plotly_chart(fig_s1, use_container_width=True)
                    with c2:
                        fig_s2 = px.bar(strat_stats, x="strategy_name", y="avg_profit_factor",
                                        color="avg_profit_factor", color_continuous_scale="RdYlGn",
                                        text="avg_profit_factor", title="策略平均 Profit Factor")
                        fig_s2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        fig_s2.add_hline(y=1.5, line_dash="dash", line_color="#00CC96", annotation_text="PF=1.5")
                        fig_s2.update_layout(height=340, showlegend=False, **CHART_LAYOUT)
                        st.plotly_chart(fig_s2, use_container_width=True)
                    with c3:
                        fig_s3 = px.bar(strat_stats, x="strategy_name", y="avg_sharpe",
                                        color="avg_sharpe", color_continuous_scale="RdYlGn",
                                        text="avg_sharpe", title="策略平均夏普比率")
                        fig_s3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        fig_s3.add_hline(y=1.0, line_dash="dash", line_color="#FFA500", annotation_text="Sharpe=1.0")
                        fig_s3.update_layout(height=340, showlegend=False, **CHART_LAYOUT)
                        st.plotly_chart(fig_s3, use_container_width=True)

                    st.dataframe(
                        strat_stats.rename(columns={
                            "strategy_name":"策略","avg_win_rate":"平均胜率%",
                            "avg_trade_count":"平均笔数","avg_sharpe":"平均夏普",
                            "avg_profit_factor":"平均PF","avg_score":"平均评分",
                            "total_signals":"总信号数",
                        }).style.format({
                            "平均胜率%":"{:.1f}%","平均笔数":"{:.1f}",
                            "平均夏普":"{:.2f}","平均PF":"{:.2f}","平均评分":"{:.1f}",
                        }, na_rep="N/A"),
                        use_container_width=True, hide_index=True,
                    )

            # ════════════════════════════════════════════════════════════════
            # VIEW 4: 验证期分析
            # ════════════════════════════════════════════════════════════════
            elif view == "🔬 验证期分析":
                st.subheader("训练期 vs 验证期一致性分析")
                st.caption("训练期 2022-2023 | 验证期 2024-2025 — 验证过拟合风险")

                oos_df = display_df.dropna(subset=["train_win_rate","oos_win_rate"]).copy()
                if oos_df.empty:
                    st.info("无验证期数据（需要2024-2025有交易信号）")
                else:
                    oos_df["差值%"] = oos_df["oos_win_rate"] - oos_df["train_win_rate"]
                    oos_df["过拟合风险"] = oos_df["差值%"].apply(
                        lambda d: "✅稳定" if abs(d) < 10 else ("⚠️轻微退化" if d > -20 else "❌严重过拟合")
                    )

                    fig_oos = go.Figure()
                    fig_oos.add_trace(go.Scatter(
                        x=oos_df["train_win_rate"], y=oos_df["oos_win_rate"],
                        mode="markers+text",
                        text=oos_df["ticker"] + "·" + oos_df.get("strategy_name","").fillna(""),
                        textposition="top center",
                        marker=dict(size=10,
                            color=oos_df["差值%"],
                            colorscale="RdYlGn", colorbar=dict(title="验证-训练%"),
                        ),
                        hovertemplate="<b>%{text}</b><br>训练: %{x:.1f}%<br>验证: %{y:.1f}%<extra></extra>",
                    ))
                    max_val = max(oos_df[["train_win_rate","oos_win_rate"]].max().max(), 80)
                    fig_oos.add_shape(type="line", x0=30, y0=30, x1=max_val, y1=max_val,
                                      line=dict(color="gray", dash="dash"))
                    fig_oos.update_layout(
                        title="训练期胜率 vs 验证期胜率 (对角线=完美一致)",
                        xaxis_title="训练期胜率%", yaxis_title="验证期胜率%",
                        height=500, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_oos, use_container_width=True)

                    st.dataframe(
                        oos_df[["ticker","sector","strategy_name","train_win_rate","oos_win_rate",
                                "差值%","score","过拟合风险"]].rename(columns={
                            "ticker":"股票","sector":"板块","strategy_name":"策略",
                            "train_win_rate":"训练胜率%","oos_win_rate":"验证胜率%","score":"评分",
                        }).sort_values("差值%").style.format({
                            "训练胜率%":"{:.1f}%","验证胜率%":"{:.1f}%",
                            "差值%":"{:+.1f}%","评分":"{:.1f}",
                        }, na_rep="N/A"),
                        use_container_width=True, hide_index=True,
                    )

            # ════════════════════════════════════════════════════════════════
            # VIEW 5: 压力测试
            # ════════════════════════════════════════════════════════════════
            elif view == "⚡ 压力测试":
                st.subheader("3大极端场景压力测试")
                st.caption("A. 2022全年熊市 | B. 2020疫情崩盘(2020-02~05) | C. 2023银行危机(2023-03~05)")

                from modules.backtest_pro import STRESS_PERIODS as SP
                stress_rows = []
                for _, row in display_df.iterrows():
                    for pk in SP.keys():
                        wr = row.get(f"stress_{pk}")
                        stress_rows.append({
                            "股票":  row.get("ticker",""),
                            "策略":  row.get("strategy_name",""),
                            "场景":  pk,
                            "胜率%": wr,
                            "状态":  ("✅通过" if (wr or 0) > 50 else ("⚠️无数据" if wr is None else "❌不达标")),
                        })
                if stress_rows:
                    stress_df = pd.DataFrame(stress_rows)
                    pivot = stress_df.pivot_table(
                        index=["股票","策略"], columns="场景", values="胜率%", aggfunc="first"
                    ).reset_index()
                    pivot.columns.name = None
                    pass_cnt = (
                        stress_df[stress_df["状态"]=="✅通过"].groupby(["股票","策略"])
                        .size().reset_index(name="通过场景数")
                    )
                    pivot = pivot.merge(pass_cnt, on=["股票","策略"], how="left").fillna({"通过场景数":0})
                    pivot["通过场景数"] = pivot["通过场景数"].astype(int)

                    stress_col_fmt = {pk: "{:.1f}%" for pk in SP.keys() if pk in pivot.columns}
                    st.dataframe(
                        pivot.style.format(stress_col_fmt, na_rep="无数据").map(
                            lambda v: "background-color:rgba(0,204,150,0.2)" if isinstance(v,float) and v > 50
                                      else ("background-color:rgba(239,85,59,0.15)" if isinstance(v,float) and v <= 50 else ""),
                            subset=[pk for pk in SP.keys() if pk in pivot.columns],
                        ),
                        use_container_width=True, hide_index=True,
                    )

                # 评分体系说明
                st.divider()
                with st.expander("📖 评分体系说明 (100分制)"):
                    st.markdown("""
| 维度 | 权重 | 满分条件 |
|------|------|----------|
| 样本量充足度 | 20% | 达到动态门槛(高波动25/中15/低10) |
| 夏普比率 | 25% | Sharpe ≥ 2.0 |
| 熊市表现 | 20% | 2022熊市胜率 ≥ 80% |
| 样本外一致性 | 25% | 验证期胜率≥70% 且与训练期差<10% |
| 最大回撤控制 | 10% | 最大回撤 = 0% |

**评级：** A+(90-100) / A(75-89) / B(60-74) / C(40-59) / D(<40)

**成本：** 每笔0.15%往返（0.10%佣金+0.05%滑点）
                    """)

        with st.expander("ℹ️ 策略说明 & 成本"):
            st.markdown("""
**4种短线策略**
| 策略 | 买入条件 | 卖出条件 | 止损 |
|------|----------|----------|------|
| RSI超卖反弹 | RSI(5)<25 | RSI(5)>60 | -8% |
| 布林带下轨反弹 | 价格<布林带下轨(20,2) | 回到中轨 | -8% |
| RSI+布林带复合 | RSI(5)<30 AND 价格<下轨 | 中轨 OR +8% | -6% |
| 急跌反弹 | 3日跌幅>10% AND RSI(5)<30 | +10% | -8% |

**第二层执行逻辑（市场状态过滤）**
- 🟢 牛市(SPY>200MA+5%)：4种策略全开，标准仓位
- 🟡 震荡(±5%)：仅策略3，仓位×50%
- 🔴 熊市(SPY<200MA-5%)：仅观察，仓位0%
            """)


    # ════════════════════════════════════════════════════════════════════════════
    # TAB 8  实时雷达
    # ════════════════════════════════════════════════════════════════════════════
    with tab8:
        st.header("📻 实时信号雷达")
        st.caption(
            f"扫描 {len(ALL_RADAR_TICKERS)} 只核心AI+科技股 · "
            "同时满足2条以上触发提醒 · 信号含止损/目标/仓位建议"
        )

        # ── 控制栏 ────────────────────────────────────────────────────────────
        rc1, rc2, rc3, rc4 = st.columns([2, 2, 1, 1])
        with rc1:
            radar_sector = st.selectbox(
                "筛选板块",
                ["全部"] + list(RADAR_TICKERS_BY_SECTOR.keys()),
                key="radar_sector",
            )
        with rc2:
            show_levels = st.multiselect(
                "显示信号等级",
                ["A+", "A", "B", "C"],
                default=["A+", "A", "B", "C"],
                key="radar_level_filter",
            )
        with rc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            scan_btn = st.button("🔍 立即扫描", type="primary", use_container_width=True, key="radar_scan")
        with rc4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            auto_30 = st.toggle("⏰ 30秒刷新", value=False, key="radar_auto")

        # ── 自动刷新逻辑 ─────────────────────────────────────────────────────
        import time as _time
        if auto_30:
            last_t = st.session_state.get("radar_last_scan_ts", 0)
            now_t  = _time.time()
            if now_t - last_t >= 30:
                scan_btn = True
            else:
                remaining = int(30 - (now_t - last_t))
                st.caption(f"⏳ 下次刷新：{remaining}秒后 — 请启用浏览器刷新或手动点击扫描")

        # ── 执行扫描 ─────────────────────────────────────────────────────────
        if scan_btn:
            _wl_tickers = wl["ticker"].tolist()
            full_pool = list(dict.fromkeys(ALL_RADAR_TICKERS + _wl_tickers))
            scan_tickers = (
                full_pool if radar_sector == "全部"
                else RADAR_TICKERS_BY_SECTOR.get(radar_sector, full_pool)
            )
            bt_grades: dict = {}
            if st.session_state.get("bt_flat_df") is not None:
                flat_df_tmp = st.session_state.bt_flat_df
                if flat_df_tmp is not None and not flat_df_tmp.empty and "grade" in flat_df_tmp.columns:
                    for _, row in flat_df_tmp.iterrows():
                        bt_grades[f"{row['ticker']}_{row['strategy']}"] = str(row.get("grade","N/A"))

            with st.spinner("📡 实时扫描中..."):
                sigs, mkt, vix, spy_s, qqq_s = scan_signals(
                    scan_tickers, min_conditions=1, backtest_grades=bt_grades
                )

            # Store spy/qqq as lists so session_state can serialize them
            st.session_state["radar_signals"]      = sigs
            st.session_state["radar_mkt"]          = mkt
            st.session_state["radar_vix"]          = vix
            st.session_state["scan_spy_close"]     = spy_s.tolist() if spy_s is not None else None
            st.session_state["scan_qqq_close"]     = qqq_s.tolist() if qqq_s is not None else None
            st.session_state["radar_last_scan_ts"] = _time.time()
            st.session_state["radar_scan_time"]    = datetime.now().strftime("%H:%M:%S")
            st.rerun()

        # ── 显示扫描结果 ─────────────────────────────────────────────────────
        sigs     = st.session_state.get("radar_signals", [])
        mkt_info = st.session_state.get("radar_mkt", {})
        vix_info = st.session_state.get("radar_vix")
        scan_ts  = st.session_state.get("radar_scan_time","")

        if not sigs and not scan_ts:
            st.markdown(
                "<div style='text-align:center;padding:60px 0;color:#888'>"
                "<div style='font-size:3rem'>📻</div>"
                "<div style='font-size:1.2rem;margin-top:12px'>点击「🔍 立即扫描」开始实时扫描</div>"
                "<div style='font-size:0.85rem;margin-top:8px;color:#aaa'>"
                "RSI&lt;35 / 布林带距下轨5% / 3日跌&gt;5% / 放量&gt;1.5x &nbsp;·&nbsp; "
                "🔴强(3-4条) 🟡中(2条) 🟢弱(1条)</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            mkt_emoji = mkt_info.get("emoji","⚪")
            mkt_label = mkt_info.get("label","N/A")
            vix_str   = f"{vix_info:.1f}" if vix_info else "N/A"
            vix_c     = "#EF553B" if (vix_info or 0) > 30 else ("#FFA500" if (vix_info or 0) > 20 else "#00CC96")

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("今日信号总数", f"{len(sigs)} 个")
            sm2.metric("市场状态", f"{mkt_emoji} {mkt_label}")
            sm3.metric("VIX恐慌指数", vix_str)
            sm4.metric("扫描时间", scan_ts)

            if (vix_info or 0) > 30:
                st.warning(f"⚡ VIX={vix_str} > 30，市场极端恐慌，所有信号标注高风险⚡")

            st.divider()

            if not sigs:
                st.info("✅ 当前无股票满足信号条件（已扫描所选板块全部标的）")
            else:
                # ── 先加载预期指数并对所有信号分级 ────────────────────────────
                _exp_key_all = tuple(sorted({s["ticker"] for s in sigs}))
                _exp_scores  = c_expectation(_exp_key_all) if _exp_key_all else {}

                for _sig in sigs:
                    _es_val = _exp_scores.get(_sig["ticker"], {}).get("score")
                    _sig["signal_level"] = grade_signal(_sig, _es_val)
                    _sig["tradable"] = _sig["signal_level"] == "A+"

                # 更新顶部状态栏各级计数
                st.session_state["radar_signals"] = sigs

                # ── 信号汇总表（含信号等级列）────────────────────────────────
                def _level_badge(lv: str) -> str:
                    return {"A+": "🔴 A+", "A": "🟠 A", "B": "🟡 B", "C": "⚪ C"}.get(lv, lv)

                sig_df = pd.DataFrame([{
                    "信号等级": _level_badge(s.get("signal_level", "C")),
                    "股票":    s["ticker"],
                    "板块":    s["sector"],
                    "价格":    s["price"],
                    "1日涨跌%": s["change_1d_pct"],
                    "信号强度": s["score"],
                    "条件数":  len(s["triggered"]),
                    "触发条件": "·".join(s["triggered"]),
                    "策略":    s["strategy"],
                    "止损%":   s["stop_pct"],
                    "目标%":   s["target_pct"],
                    "R:R":     s["rr_ratio"],
                    "评级":    s["grade_info"],
                    "极端风险": "⚡" if s["extreme_flags"] else "✅",
                } for s in sigs])

                st.subheader(f"📊 信号汇总表（{len(sigs)} 只）")
                _LEVEL_COLORS = {"🔴 A+": "#00CC96", "🟠 A": "#FFA500",
                                 "🟡 B": "#FFD700",  "⚪ C": "#888888"}

                def _level_color(val: str):
                    c = _LEVEL_COLORS.get(val, "")
                    return f"color:{c};font-weight:700" if c else ""

                fmt_sdf = {
                    "价格":"${:.2f}","1日涨跌%":"{:+.2f}%",
                    "信号强度":"{:.0f}","止损%":"{:.1f}%","目标%":"{:.1f}%","R:R":"{:.2f}",
                }
                styled_sdf = (
                    sig_df.style
                    .format({k:v for k,v in fmt_sdf.items() if k in sig_df.columns}, na_rep="N/A")
                    .map(sign_color, subset=["1日涨跌%"] if "1日涨跌%" in sig_df.columns else [])
                    .map(_level_color, subset=["信号等级"] if "信号等级" in sig_df.columns else [])
                )
                st.dataframe(styled_sdf, use_container_width=True, hide_index=True)

                # 气泡图
                try:
                    sdf_chart = pd.DataFrame([{
                        "股票": s["ticker"], "RSI(5)": s["rsi5"] or 30,
                        "3日跌幅%": s["drop3_pct"] or 0,
                        "信号强度": s["score"], "板块": s["sector"],
                    } for s in sigs if s.get("rsi5")])
                    if len(sdf_chart) >= 2:
                        fig_bubble = px.scatter(
                            sdf_chart, x="RSI(5)", y="3日跌幅%", size="信号强度",
                            color="板块", text="股票", title="信号空间：RSI vs 3日跌幅",
                            size_max=40,
                        )
                        fig_bubble.add_vline(x=35, line_dash="dash", line_color="#FFA500", annotation_text="RSI=35")
                        fig_bubble.add_hline(y=5,  line_dash="dash", line_color="#EF553B", annotation_text="3日跌5%")
                        fig_bubble.update_traces(textposition="top center")
                        fig_bubble.update_layout(height=400, **CHART_LAYOUT)
                        st.plotly_chart(fig_bubble, use_container_width=True)
                except Exception:
                    pass

                # ── 按用户选择的信号等级过滤 ──────────────────────────────────
                _allowed_levels = set(show_levels) if show_levels else {"A+", "A", "B", "C"}
                _filtered_sigs  = [s for s in sigs if s.get("signal_level", "C") in _allowed_levels]

                _aplus_sigs = [s for s in _filtered_sigs if s.get("signal_level") == "A+"]
                _a_sigs     = [s for s in _filtered_sigs if s.get("signal_level") == "A"]
                _b_sigs     = [s for s in _filtered_sigs if s.get("signal_level") == "B"]
                _c_sigs     = [s for s in _filtered_sigs if s.get("signal_level") == "C"]

                st.divider()
                _LEVEL_HDR = {
                    "A+": ("#00CC96", "⭐", "A+ 严选信号（可交易）"),
                    "A":  ("#FFA500", "🟠", "A级信号（候选，仅观察）"),
                    "B":  ("#FFD700", "🟡", "B级信号（参考）"),
                    "C":  ("#555566", "⚪", "C级信号（仅观察，不建议操作）"),
                }
                for _lv, _lv_sigs in [("A+", _aplus_sigs), ("A", _a_sigs),
                                       ("B", _b_sigs), ("C", _c_sigs)]:
                    if not _lv_sigs:
                        continue
                    _hdr_c, _emoji, _desc = _LEVEL_HDR[_lv]
                    st.markdown(
                        f"<div style='background:#12122a;border-left:4px solid {_hdr_c};"
                        f"padding:6px 12px;margin:8px 0;border-radius:0 6px 6px 0'>"
                        f"<span style='color:{_hdr_c};font-weight:700;font-size:1.05rem'>"
                        f"{_emoji} {_desc} — {len(_lv_sigs)} 只</span></div>",
                        unsafe_allow_html=True,
                    )
                    for i in range(0, len(_lv_sigs), 3):
                        card_cols = st.columns(3)
                        for j, sig in enumerate(_lv_sigs[i:i+3]):
                            with card_cols[j]:
                                st.markdown(build_signal_card_html(sig), unsafe_allow_html=True)
                                _escore = _exp_scores.get(sig["ticker"], {}).get("score")
                                _sl     = sig.get("signal_level", "")
                                _sl_c   = {"A+": "#00CC96", "A": "#FFA500",
                                           "B": "#FFD700", "C": "#888"}.get(_sl, "#888")
                                _tradable_tag = (
                                    "<span style='color:#00CC96'>✅ 可交易</span>"
                                    if sig.get("tradable") else
                                    "<span style='color:#FFA500'>👁️ 仅观察</span>"
                                )
                                _detail_lines = []
                                if _escore is not None:
                                    _ee = "🟢" if _escore >= 60 else "🟡" if _escore >= 40 else "🔴"
                                    _detail_lines.append(f"预期指数：{_ee} <b>{_escore}/100</b>")
                                st.markdown(
                                    f"<div style='font-size:0.80rem;color:{_sl_c};"
                                    f"padding:4px 8px;border-top:1px solid #2a2a3e;margin-top:4px'>"
                                    f"<b>[{_sl}]</b> {_tradable_tag}"
                                    + (f"<br>{'<br>'.join(_detail_lines)}" if _detail_lines else "")
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )

        with st.expander("📖 扫描条件说明"):
            st.markdown("""
**触发条件（满足1条即显示，按强度分级）**
| 条件 | 阈值 | 说明 |
|------|------|------|
| RSI超卖 | RSI(5) < 35 | 5日RSI超卖（原<25，已放宽） |
| 布林带下轨 | 收盘 < 下轨×1.05 | 距布林下轨5%以内（含跌破） |
| 3日急跌 | 3日跌幅 > 5% | 短期快速下跌（原>8%，已放宽） |
| 放量 | 量 > 20日均量×1.5 | 异常放量（原×2，已放宽） |

**信号分级**
| 级别 | 触发条件数 | 含义 |
|------|-----------|------|
| 🔴 强信号 | 3-4条 | 原标准，高置信度 |
| 🟡 中信号 | 2条 | 降低门槛，值得关注 |
| 🟢 弱信号 | 1条 | 观察列表，谨慎参考 |

**极端风险标记⚡**（标注但不屏蔽信号）
- VIX > 30（市场极端恐慌）
- 单日波动 > 8%
- 疑似财报前后（近3日出现>5%波动）
            """)

        # ── 第二处：预期排行榜视图 ────────────────────────────────────────────
        st.divider()
        st.subheader("📊 预期排行榜")
        _lb_c1, _lb_c2 = st.columns([1, 4])
        with _lb_c1:
            _run_lb = st.button(
                "🔄 刷新排行榜", key="run_lb_btn", use_container_width=True,
                help="首次计算约需1-2分钟，之后使用6小时磁盘缓存",
            )
        with _lb_c2:
            st.caption(
                "所有股票按预期指数从高到低排列 · "
                "评分=EPS修正+目标价+评级+Guidance+CapEx(共100分) · "
                "首次需1-2分钟，之后缓存6小时"
            )
        if _run_lb:
            _lb_key = tuple(sorted(set(ALL_RADAR_TICKERS)))
            with st.spinner(f"计算 {len(_lb_key)} 只股票预期指数，请稍候..."):
                _lb_res = get_expectation_scores(list(_lb_key))
            c_expectation.clear()
            st.session_state["lb_scores"] = _lb_res
            st.rerun()

        _lb_cached = st.session_state.get("lb_scores")
        if _lb_cached:
            _lb_sec_map = {t: s for s, tks in RADAR_TICKERS_BY_SECTOR.items() for t in tks}
            _lb_rows = []
            for _lt, _ld in sorted(
                _lb_cached.items(), key=lambda x: x[1].get("score", 0), reverse=True
            ):
                _wl_row = wl[wl["ticker"] == _lt]
                _sec = (
                    _wl_row["sector"].iloc[0] if not _wl_row.empty
                    else _lb_sec_map.get(_lt, "其他")
                )
                _lb_rows.append({
                    "代码":     _lt,
                    "板块":     _sec,
                    "预期指数": _ld.get("score", 50),
                    "主要触发": "; ".join(_ld.get("reasons", [])[:2]) or "无显著信号",
                })
            if _lb_rows:
                _lb_df = pd.DataFrame(_lb_rows)

                def _lb_score_bg(v):
                    if pd.isna(v): return ""
                    fv = float(v)
                    if fv >= 70: return "background-color:rgba(0,204,150,0.2)"
                    if fv >= 50: return "background-color:rgba(255,200,0,0.12)"
                    if fv <  35: return "background-color:rgba(239,85,59,0.15)"
                    return ""

                st.dataframe(
                    _lb_df.style
                          .map(_lb_score_bg, subset=["预期指数"])
                          .format({"预期指数": "{:.0f}"}),
                    use_container_width=True,
                    hide_index=False,
                )
                st.caption(
                    f"共 {len(_lb_rows)} 只股票 · "
                    f"更新时间: {datetime.now().strftime('%H:%M')} · "
                    "🟢≥70 优 | 🟡50-69 中 | 🔴<35 弱"
                )
        else:
            st.info(
                "点击「🔄 刷新排行榜」计算全部预期指数。\n\n"
                "首次约需 1-2 分钟（yfinance + Yahoo 新闻 RSS），"
                "之后磁盘缓存 6 小时，信号卡片也会自动显示预期指数。"
            )

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 9  历史验证（严选模式）
    # ════════════════════════════════════════════════════════════════════════════
    with tab9:
        st.header("🎯 历史验证 — 严选模式 / High-Conviction Mode")
        st.markdown(
            "<div style='background:#12122a;border:1px solid #4a9eff;border-radius:8px;"
            "padding:10px 16px;margin-bottom:12px'>"
            "<b style='color:#4a9eff'>交易模式：严选模式</b><br>"
            "<span style='color:#aaa;font-size:0.88rem'>"
            "本模式目标是提高信号质量、降低回撤和强化纪律，不保证盈利。"
            "</span></div>",
            unsafe_allow_html=True,
        )

        # ── 严选模式交易规则说明 ───────────────────────────────────────────────
        with st.expander("📋 严选模式规则总览", expanded=False):
            st.markdown("""
**仓位与止损**
| 参数 | 数值 |
|------|------|
| 仓位 | 1.5% – 2% |
| 硬止损 | -2% |
| 止盈一 | +3%，卖一半 |
| 止盈二 | +5%，大部分或全部 |
| 每日最多开仓 | 1笔 |
| 同时最多持仓 | 2只 |

**时间止损（T日为买入日）**
- T+1 收盘：未有效上涨 → 标记 `weak_follow_through`
- T+2 收盘：最高收盘价从未超过买入价+1% → T+2 收盘前平仓
- T+1/T+2 开盘低于硬止损价：硬止损优先，按开盘价平仓

**退出优先级**：重大利空 > 硬止损 > 跌破买入日最低点 > 时间止损 > 止盈

**历史验证样本要求**
| 级别 | 样本要求 |
|------|----------|
| 个股级 | 同股票同类A+样本 ≥ 30 |
| 板块级 | 同板块同类A+样本 ≥ 80 |
| 全市场级 | 全股票池同类A+样本 ≥ 150 |

**验证指标要求**：胜率 > 65% · 平均盈亏比 > 1.5 · 期望值 > 0 · 最大连亏 ≤ 3

**自动关闭触发**
- 最近20笔胜率 < 60% → 进入仅观察模式
- 最近20笔期望值 ≤ 0 → 进入仅观察模式
- 月度回撤 > 3% → 关闭短线交易系统
            """)

        # ── 控制区 ────────────────────────────────────────────────────────────
        hc9_c1, hc9_c2, hc9_c3 = st.columns([2, 2, 2])
        with hc9_c1:
            _hc_ticker_opts = ["全部股票池"] + ALL_RADAR_TICKERS
            _hc_sel_ticker = st.selectbox("选择个股验证", _hc_ticker_opts, key="hc9_ticker")
        with hc9_c2:
            _hc_sel_sector = st.selectbox(
                "选择板块验证",
                ["全部板块"] + list(RADAR_TICKERS_BY_SECTOR.keys()),
                key="hc9_sector",
            )
        with hc9_c3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            _hc_run = st.button(
                "🔬 运行历史验证", type="primary", use_container_width=True, key="hc9_run",
                help="扫描2年历史数据，计算A+信号样本统计。首次约需2-5分钟。",
            )
            _hc_force = st.checkbox("强制刷新缓存", key="hc9_force", value=False)

        # ── 自动关闭检查 ──────────────────────────────────────────────────────
        _hc_market_cached = st.session_state.get("hc_market_stats")
        if _hc_market_cached:
            _shutdown = check_auto_shutdown(_hc_market_cached)
            if _shutdown["shutdown"]:
                st.error("⛔ 严选模式已自动关闭\n\n" + "\n\n".join(_shutdown["reasons"]))

        # ── 执行验证 ──────────────────────────────────────────────────────────
        if _hc_run:
            _hc_tickers_pool = (
                ALL_RADAR_TICKERS if _hc_sel_ticker == "全部股票池"
                else [_hc_sel_ticker]
            )
            _hc_sector_pool = (
                list(RADAR_TICKERS_BY_SECTOR.get(_hc_sel_sector, {}).keys())
                if _hc_sel_sector != "全部板块"
                else ALL_RADAR_TICKERS
            )
            _hc_sector_tickers = (
                RADAR_TICKERS_BY_SECTOR.get(_hc_sel_sector, [])
                if _hc_sel_sector != "全部板块"
                else ALL_RADAR_TICKERS
            )

            with st.spinner("🔬 扫描历史A+信号中，首次约需2-5分钟..."):
                # Market stats
                _hc_mkt = get_market_hc_stats(ALL_RADAR_TICKERS, force=_hc_force)
                st.session_state["hc_market_stats"] = _hc_mkt

                # Stock stats (for selected ticker or all)
                _hc_stock_results = {}
                if _hc_sel_ticker != "全部股票池":
                    _hc_stock_results[_hc_sel_ticker] = get_stock_stats(_hc_sel_ticker, force=_hc_force)
                else:
                    for _t in ALL_RADAR_TICKERS:
                        _hc_stock_results[_t] = get_stock_stats(_t, force=_hc_force)
                st.session_state["hc_stock_results"] = _hc_stock_results

                # Sector stats
                _hc_sec_stats = {}
                for _sec, _tks in RADAR_TICKERS_BY_SECTOR.items():
                    _hc_sec_stats[_sec] = get_sector_hc_stats(_tks, _sec, force=_hc_force)
                st.session_state["hc_sector_stats"] = _hc_sec_stats

                # Classify each ticker and store in validation cache
                _ticker_sec_map = {
                    _t: _sec
                    for _sec, _tks in RADAR_TICKERS_BY_SECTOR.items()
                    for _t in _tks
                }
                _hc_val_cache = {}
                for _t, _st_stats in _hc_stock_results.items():
                    _t_sector = _ticker_sec_map.get(_t, "")
                    _t_sec_stats = st.session_state["hc_sector_stats"].get(_t_sector, {})
                    _t_cls = classify_aplus(_t, _t_sector, _st_stats, _t_sec_stats, _hc_mkt)
                    _hc_val_cache[_t] = _t_cls
                st.session_state["hc_validation_cache"] = _hc_val_cache

            st.success(f"✅ 验证完成 · 全市场样本: {_hc_mkt.get('n', 0)} 笔")
            st.rerun()

        # ── 显示验证结果 ──────────────────────────────────────────────────────
        _hc_mkt_stats   = st.session_state.get("hc_market_stats")
        _hc_stk_results = st.session_state.get("hc_stock_results", {})
        _hc_sec_results = st.session_state.get("hc_sector_stats", {})
        _hc_val_cache   = st.session_state.get("hc_validation_cache", {})

        if not _hc_mkt_stats:
            st.info("点击「🔬 运行历史验证」扫描历史A+信号数据（2年回溯）。")
        else:
            # ── 全市场汇总 ────────────────────────────────────────────────────
            st.subheader("🌏 全市场样本汇总")
            _m = _hc_mkt_stats
            _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
            _mc1.metric("全市场样本数", f"{_m.get('n',0)} 笔",
                        help=f"要求≥{MIN_MARKET_SAMPLES}笔才可用于全市场级验证")
            _mc2.metric("胜率", f"{_m.get('win_rate',0)*100:.1f}%",
                        delta="通过" if _m.get('win_rate',0) > 0.65 else "未达标")
            _mc3.metric("平均盈亏比", f"{_m.get('rr_ratio',0):.2f}",
                        delta="通过" if _m.get('rr_ratio',0) > 1.5 else "未达标")
            _mc4.metric("期望值(%)", f"{_m.get('ev',0):.2f}%",
                        delta="正期望" if _m.get('ev',0) > 0 else "负期望")
            _mc5.metric("最大连亏", f"{_m.get('max_consec_losses',0)} 笔",
                        delta="通过" if _m.get('max_consec_losses',0) <= 3 else "超标")

            _mc6, _mc7, _mc8 = st.columns(3)
            _mc6.metric("平均盈利(%)", f"{_m.get('avg_win',0):.2f}%")
            _mc7.metric("平均亏损(%)", f"{_m.get('avg_loss',0):.2f}%")
            _mc8.metric("最近20笔胜率", f"{_m.get('recent_20_wr',0)*100:.1f}%",
                        delta="正常" if _m.get('recent_20_wr',0) >= 0.60 else "⚠️ 低于60%，触发关闭")

            # 自动关闭检查
            _sd = check_auto_shutdown(_m)
            if _sd["shutdown"]:
                st.error("⛔ **严选模式自动关闭** — " + " | ".join(_sd["reasons"]))
            else:
                st.success("✅ 严选模式运行正常，未触发自动关闭条件")

            st.divider()

            # ── 板块汇总表 ────────────────────────────────────────────────────
            if _hc_sec_results:
                st.subheader("📊 板块级A+验证汇总")
                _sec_rows = []
                for _sn, _ss in _hc_sec_results.items():
                    _sec_ok = _ss.get("n", 0) >= MIN_SECTOR_SAMPLES
                    _sec_rows.append({
                        "板块":       _sn,
                        "样本数":     _ss.get("n", 0),
                        "胜率%":      round(_ss.get("win_rate", 0) * 100, 1),
                        "盈亏比":     _ss.get("rr_ratio", 0),
                        "期望值%":    _ss.get("ev", 0),
                        "最大连亏":   _ss.get("max_consec_losses", 0),
                        "最近20笔%":  round(_ss.get("recent_20_wr", 0) * 100, 1),
                        "验证状态":   "✅ 达标" if (
                            _sec_ok and _ss.get("win_rate",0) > 0.65
                            and _ss.get("rr_ratio",0) > 1.5 and _ss.get("ev",0) > 0
                        ) else ("⚠️ 样本不足" if not _sec_ok else "❌ 未达标"),
                    })
                _sec_df = pd.DataFrame(_sec_rows)

                def _hc_sec_color(v):
                    if "✅" in str(v): return "color:#00CC96;font-weight:600"
                    if "⚠️" in str(v): return "color:#FFA500"
                    if "❌" in str(v): return "color:#EF553B"
                    return ""

                st.dataframe(
                    _sec_df.style.map(_hc_sec_color, subset=["验证状态"]),
                    use_container_width=True, hide_index=True,
                )

            st.divider()

            # ── 个股明细表 ────────────────────────────────────────────────────
            if _hc_stk_results:
                st.subheader("📈 个股级A+验证明细")
                _stk_rows = []
                for _t, _ss in sorted(_hc_stk_results.items(),
                                      key=lambda x: x[1].get("n", 0), reverse=True):
                    _cls = _hc_val_cache.get(_t, {})
                    _stk_rows.append({
                        "代码":       _t,
                        "个股样本":   _ss.get("n", 0),
                        "板块样本":   _cls.get("sector_n", 0),
                        "全市场样本": _cls.get("market_n", 0),
                        "验证级别":   _cls.get("validation_level", "未计算"),
                        "胜率%":      round(_ss.get("win_rate", 0) * 100, 1),
                        "盈亏比":     _ss.get("rr_ratio", 0),
                        "期望值%":    _ss.get("ev", 0),
                        "最大连亏":   _ss.get("max_consec_losses", 0),
                        "信号级别":   _cls.get("signal_level", "待计算"),
                        "可交易":     "✅" if _cls.get("tradable") else "❌",
                        "说明":       _cls.get("reason", ""),
                    })
                _stk_df = pd.DataFrame(_stk_rows)

                def _hc_level_color(v):
                    if str(v) == "A+": return "color:#00CC96;font-weight:700"
                    if "A候选" in str(v): return "color:#FFA500;font-weight:600"
                    return "color:#888"

                def _hc_tradable_color(v):
                    return "color:#00CC96" if "✅" in str(v) else "color:#EF553B"

                st.dataframe(
                    _stk_df.style
                           .map(_hc_level_color, subset=["信号级别"])
                           .map(_hc_tradable_color, subset=["可交易"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    f"共 {len(_stk_rows)} 只股票 · "
                    f"A+可交易: {sum(1 for r in _stk_rows if r['可交易']=='✅')} 只 · "
                    f"A候选: {sum(1 for r in _stk_rows if 'A候选' in r['信号级别'])} 只"
                )

            st.divider()

            # ── A+完整规则清单 ────────────────────────────────────────────────
            with st.expander("📋 A+完整条件清单（19条）"):
                st.markdown("""
**自动核验条件**
| # | 条件 | 阈值 |
|---|------|------|
| 1 | 预期指数 | > 85 |
| 2 | RSI(5) | < 28 |
| 3 | 布林带A+形态 | Low < BB下轨 且 Close ≥ Low×1.03 且 Close ≤ BB下轨×1.015 |
| 4 | 3日跌幅 | > 6% |
| 5 | 成交量 | > 20日均量×1.5 |
| 6 | 均线 | 未同时跌破20/50/200日均线 |
| 7 | SPY | > 200MA |
| 8 | QQQ | > 50MA |
| 9 | VIX | < 25 |
| 10 | SPY | 最近5天无连续大阴线 |

**需人工确认**
| # | 条件 |
|---|------|
| 11 | 无重大利空 |
| 12 | 未来5个交易日内没有财报 |
| 13 | 当天不是CPI/FOMC/非农等重大宏观事件日 |

**历史验证（自动）**
| # | 条件 | 要求 |
|---|------|------|
| 14 | 样本数 | 个股≥30 或 板块≥80 或 全市场≥150 |
| 15 | 胜率 | > 65% |
| 16 | 平均盈亏比 | > 1.5 |
| 17 | 期望值 | > 0 |
| 18 | 最大连续亏损 | ≤ 3 |

> A+ 布林带含义：盘中曾恐慌跌破下轨，收盘有承接反弹，但没有完全涨飞
                """)


    # ════════════════════════════════════════════════════════════════════════════
    # TAB 10  单股查询 + 巴菲特价值评分卡
    # ════════════════════════════════════════════════════════════════════════════
    with tab10:
        import yfinance as yf

        st.header("🔎 单股深度查询")
        st.caption("输入任意美股代码，实时拉取行情、估值、财报、技术指标、巴菲特价值评分 — 数据源 yfinance")

        # ── 输入区 ────────────────────────────────────────────────────────────
        sq_col1, sq_col2, sq_col3 = st.columns([2, 1, 1])
        with sq_col1:
            sq_ticker_raw = st.text_input(
                "股票代码",
                placeholder="例：KULR / WMT / AAPL / 2330.TW",
                key="sq_ticker",
            ).strip().upper()
        with sq_col2:
            sq_period = st.selectbox(
                "历史区间", ["1mo", "3mo", "6mo", "1y", "2y"], index=3, key="sq_period"
            )
        with sq_col3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            sq_run = st.button("🔍 查询", type="primary", use_container_width=True, key="sq_run")

        # 快捷按钮
        st.caption("快捷选择：")
        sq_shortcuts = ["KULR","WMT","AMZN","COST","KO","NKE","NVDA","MU","AAPL","GOOGL","BRK-B","JNJ"]
        sq_btn_cols  = st.columns(len(sq_shortcuts))
        for _i, _tk in enumerate(sq_shortcuts):
            with sq_btn_cols[_i]:
                if st.button(_tk, key=f"sq_short_{_tk}", use_container_width=True):
                    st.session_state["sq_ticker"]      = _tk
                    st.session_state["sq_run_trigger"] = _tk
                    st.rerun()

        _sq_trigger = st.session_state.pop("sq_run_trigger", None)
        if _sq_trigger:
            sq_ticker_raw = _sq_trigger
            sq_run = True

        # ── 巴菲特评分函数 ────────────────────────────────────────────────────
        def calc_buffett_score(info: dict, hist_earnings: list) -> dict:
            items = []
            total = 0

            # 行业分类辅助
            _sector = info.get("sector", "")
            _is_tech     = any(k in _sector for k in ("Technology", "Communication"))
            _is_consumer = any(k in _sector for k in ("Consumer",))
            _is_finance  = any(k in _sector for k in ("Financial", "Bank"))
            _is_health   = any(k in _sector for k in ("Healthcare", "Health"))
            _is_industry = any(k in _sector for k in ("Industrial", "Utilities", "Energy", "Materials"))

            # 1. 护城河：毛利率（权重15，基准按行业调整）
            if _is_finance:
                # 金融业用净利率替代毛利率
                gm = (info.get("profitMargins") or 0) * 100
                _gm_label = "净利率"
                _gm_bench_hi, _gm_bench_mid, _gm_bench_lo = 25, 15, 8
            elif _is_consumer:
                gm = (info.get("grossMargins") or 0) * 100
                _gm_label = "毛利率"
                _gm_bench_hi, _gm_bench_mid, _gm_bench_lo = 30, 20, 10
            elif _is_tech:
                gm = (info.get("grossMargins") or 0) * 100
                _gm_label = "毛利率"
                _gm_bench_hi, _gm_bench_mid, _gm_bench_lo = 50, 40, 25
            elif _is_health or _is_industry:
                gm = (info.get("grossMargins") or 0) * 100
                _gm_label = "毛利率"
                _gm_bench_hi, _gm_bench_mid, _gm_bench_lo = 45, 30, 15
            else:
                gm = (info.get("grossMargins") or 0) * 100
                _gm_label = "毛利率"
                _gm_bench_hi, _gm_bench_mid, _gm_bench_lo = 45, 35, 20

            if gm >= _gm_bench_hi:
                pts, icon, note = 15, "✅", f"{_gm_label} {gm:.1f}% ≥ {_gm_bench_hi}%，定价权强"
            elif gm >= _gm_bench_mid:
                pts, icon, note = 10, "✅", f"{_gm_label} {gm:.1f}% ≥ {_gm_bench_mid}%，护城河合格"
            elif gm >= _gm_bench_lo:
                pts, icon, note = 5,  "⚠️", f"{_gm_label} {gm:.1f}%，偏低，护城河一般（行业基准{_gm_bench_mid}%）"
            else:
                pts, icon, note = 0,  "❌", f"{_gm_label} {gm:.1f}%，无明显定价权"
            items.append(("🏰 护城河（毛利率）", pts, 15, icon, note))
            total += pts

            # 2. 盈利能力：ROE > 15%（权重15）
            roe = (info.get("returnOnEquity") or 0) * 100
            if roe >= 25:
                pts, icon, note = 15, "✅", f"ROE {roe:.1f}% ≥ 25%，盈利能力优秀"
            elif roe >= 15:
                pts, icon, note = 10, "✅", f"ROE {roe:.1f}% ≥ 15%，盈利能力良好"
            elif roe >= 8:
                pts, icon, note = 5,  "⚠️", f"ROE {roe:.1f}%，盈利能力一般"
            else:
                pts, icon, note = 0,  "❌", f"ROE {roe:.1f}%，资本回报率低"
            items.append(("💰 盈利能力（ROE）", pts, 15, icon, note))
            total += pts

            # 3. 现金转化：自由现金流 / 净利润 > 80%（权重15）
            fcf     = info.get("freeCashflow") or 0
            net_inc = info.get("netIncomeToCommon") or 0
            if net_inc > 0 and fcf > 0:
                fcf_ratio = fcf / net_inc * 100
                if fcf_ratio >= 90:
                    pts, icon, note = 15, "✅", f"FCF/净利润 {fcf_ratio:.0f}%，现金质量极高"
                elif fcf_ratio >= 70:
                    pts, icon, note = 10, "✅", f"FCF/净利润 {fcf_ratio:.0f}%，现金质量良好"
                elif fcf_ratio >= 50:
                    pts, icon, note = 5,  "⚠️", f"FCF/净利润 {fcf_ratio:.0f}%，现金转化一般"
                else:
                    pts, icon, note = 0,  "❌", f"FCF/净利润 {fcf_ratio:.0f}%，利润含金量低"
            elif fcf > 0:
                pts, icon, note = 8, "⚠️", "有正向自由现金流，但净利润数据缺失"
            else:
                pts, icon, note = 0, "❌", "自由现金流为负，烧钱阶段"
            items.append(("💵 现金转化（FCF质量）", pts, 15, icon, note))
            total += pts

            # 4. 负债控制：D/E < 0.5（权重15）
            de = info.get("debtToEquity") or 0
            de_ratio = de / 100  # yfinance 返回的是百分比形式
            if de_ratio <= 0.3:
                pts, icon, note = 15, "✅", f"D/E {de_ratio:.2f}，几乎无杠杆，财务稳健"
            elif de_ratio <= 0.7:
                pts, icon, note = 10, "✅", f"D/E {de_ratio:.2f}，负债合理"
            elif de_ratio <= 1.5:
                pts, icon, note = 5,  "⚠️", f"D/E {de_ratio:.2f}，负债偏高"
            else:
                pts, icon, note = 0,  "❌", f"D/E {de_ratio:.2f}，高杠杆，风险较大"
            items.append(("🛡️ 负债控制（D/E）", pts, 15, icon, note))
            total += pts

            # 5. 增长可持续：EPS增速 > 8%（权重15）
            eps_g = (info.get("earningsGrowth") or 0) * 100
            rev_g = (info.get("revenueGrowth")  or 0) * 100
            if eps_g >= 15 and rev_g >= 10:
                pts, icon, note = 15, "✅", f"EPS增速 {eps_g:.1f}% + 营收增速 {rev_g:.1f}%，双轮驱动"
            elif eps_g >= 8:
                pts, icon, note = 10, "✅", f"EPS增速 {eps_g:.1f}%，增长稳健"
            elif eps_g >= 0:
                pts, icon, note = 5,  "⚠️", f"EPS增速 {eps_g:.1f}%，增长较慢"
            else:
                pts, icon, note = 0,  "❌", f"EPS增速 {eps_g:.1f}%，盈利在下滑"
            items.append(("📈 增长可持续（EPS增速）", pts, 15, icon, note))
            total += pts

            # 6. 估值合理：Forward PE（权重15，基准按行业调整）
            fpe = info.get("forwardPE") or 0
            if _is_tech:       pe_bench = 25
            elif _is_consumer: pe_bench = 22
            elif _is_health:   pe_bench = 20
            elif _is_finance:  pe_bench = 15
            elif _is_industry: pe_bench = 18
            else:              pe_bench = 18
            if 0 < fpe <= pe_bench * 0.8:
                pts, icon, note = 15, "✅", f"Forward PE {fpe:.1f}x，明显低于同类基准 {pe_bench}x，有安全边际"
            elif 0 < fpe <= pe_bench * 1.2:
                pts, icon, note = 10, "✅", f"Forward PE {fpe:.1f}x，估值合理"
            elif 0 < fpe <= pe_bench * 1.8:
                pts, icon, note = 5,  "⚠️", f"Forward PE {fpe:.1f}x，估值偏贵"
            elif fpe > pe_bench * 1.8:
                pts, icon, note = 0,  "❌", f"Forward PE {fpe:.1f}x，估值过高，安全边际不足"
            else:
                pts, icon, note = 5,  "⚠️", "Forward PE 数据缺失，无法判断估值"
            items.append(("🎯 估值合理（Forward PE）", pts, 15, icon, note))
            total += pts

            # 7. 股东友好：回购 + 股息（权重10）
            div_yield = (info.get("dividendYield") or 0) * 100
            payout    = (info.get("payoutRatio")   or 0) * 100
            if div_yield >= 1.5 and payout <= 60:
                pts, icon, note = 10, "✅", f"股息率 {div_yield:.1f}%，派息比例 {payout:.0f}%，股东友好"
            elif div_yield >= 0.5:
                pts, icon, note = 7,  "✅", f"股息率 {div_yield:.1f}%，有分红"
            else:
                pts, icon, note = 4,  "⚠️", "无股息，需确认是否有持续回购计划"
            items.append(("🎁 股东回报（股息/回购）", pts, 10, icon, note))
            total += pts

            if total >= 85:   grade, color = "A+  巴菲特最爱", "#00CC96"
            elif total >= 70: grade, color = "A   高质量标的", "#82C8FF"
            elif total >= 55: grade, color = "B   中等质量",   "#FFA500"
            elif total >= 40: grade, color = "C   质量偏弱",   "#FF8C00"
            else:             grade, color = "D   不符合标准", "#EF553B"

            return {"score": total, "grade": grade, "color": color, "items": items}

        def calc_atr_stoploss(hist_df, atr_period=14, atr_multiplier=2.5):
            """计算ATR动态止损：止损价 = 当前价 - ATR × multiplier"""
            if hist_df.empty or len(hist_df) < atr_period + 1:
                return None, None, None
            high  = hist_df["High"]
            low   = hist_df["Low"]
            close = hist_df["Close"]
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(atr_period).mean().iloc[-1]
            current_price = float(close.iloc[-1])
            stop_price    = current_price - atr_multiplier * atr
            stop_pct      = (stop_price / current_price - 1) * 100
            return round(float(atr), 4), round(stop_price, 2), round(stop_pct, 2)

        # ── 执行查询 ──────────────────────────────────────────────────────────
        if sq_run and sq_ticker_raw:
            sq_ticker = sq_ticker_raw
            st.session_state["sq_last_ticker"] = sq_ticker
            st.session_state["sq_last_period"]  = sq_period

            with st.spinner(f"正在拉取 {sq_ticker} 数据..."):
                try:
                    tkobj   = yf.Ticker(sq_ticker)
                    sq_info = tkobj.info or {}
                    sq_hist = tkobj.history(period=sq_period, auto_adjust=True)
                    sq_fins = tkobj.earnings_history if hasattr(tkobj, "earnings_history") else []
                except Exception as _e:
                    st.error(f"数据拉取失败：{_e}")
                    sq_info = {}
                    sq_hist = pd.DataFrame()
                    sq_fins = []

            if not sq_info and sq_hist.empty:
                st.warning(f"未找到 {sq_ticker} 的数据，请检查代码是否正确")
            else:
                # ── 基本信息横幅 ──────────────────────────────────────────────
                _sq_name    = sq_info.get("longName") or sq_info.get("shortName") or sq_ticker
                _sq_sector  = sq_info.get("sector", "N/A")
                _sq_indust  = sq_info.get("industry", "N/A")
                _sq_country = sq_info.get("country", "N/A")
                _sq_website = sq_info.get("website", "")
                _sq_price   = sq_info.get("currentPrice") or sq_info.get("regularMarketPrice")
                _sq_prev    = sq_info.get("previousClose")
                _sq_chg_pct = ((_sq_price - _sq_prev) / _sq_prev * 100) if (_sq_price and _sq_prev) else None
                _chg_color  = "#00CC96" if (_sq_chg_pct or 0) >= 0 else "#EF553B"
                _chg_str    = f"{_sq_chg_pct:+.2f}%" if _sq_chg_pct is not None else "N/A"
                _price_disp = f"${_sq_price:.2f}" if _sq_price else "N/A"

                st.markdown(
                    f"<div style='background:#12122a;border:1px solid #4a9eff;"
                    f"border-radius:10px;padding:14px 20px;margin-bottom:16px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>"
                    f"<div>"
                    f"<span style='font-size:1.6rem;font-weight:700;color:#fff'>{sq_ticker}</span>"
                    f"<span style='color:#aaa;font-size:1rem;margin-left:12px'>{_sq_name}</span>"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<span style='font-size:1.8rem;font-weight:700;color:#fff'>{_price_disp}</span>"
                    f"<span style='font-size:1.1rem;color:{_chg_color};margin-left:8px'>{_chg_str}</span>"
                    f"</div></div>"
                    f"<div style='color:#888;font-size:0.85rem;margin-top:6px'>"
                    f"{_sq_sector} · {_sq_indust} · {_sq_country}"
                    + (f" · <a href='{_sq_website}' target='_blank' style='color:#4a9eff'>{_sq_website}</a>" if _sq_website else "")
                    + "</div></div>",
                    unsafe_allow_html=True,
                )

                # ── 核心指标卡片（两行）──────────────────────────────────────
                st.subheader("📊 核心指标")
                _mkt_cap = sq_info.get("marketCap")
                _mkt_cap_str = (
                    f"${_mkt_cap/1e12:.2f}T" if _mkt_cap and _mkt_cap >= 1e12
                    else (f"${_mkt_cap/1e9:.2f}B" if _mkt_cap and _mkt_cap >= 1e9
                          else (f"${_mkt_cap/1e6:.0f}M" if _mkt_cap else "N/A"))
                )
                _r1 = st.columns(6)
                for _ci, (_lbl, _val) in enumerate({
                    "市值":        _mkt_cap_str,
                    "Trailing PE": f"{sq_info['trailingPE']:.1f}"  if sq_info.get("trailingPE") else "N/A",
                    "Forward PE":  f"{sq_info['forwardPE']:.1f}"   if sq_info.get("forwardPE")  else "N/A",
                    "PEG":         f"{sq_info['pegRatio']:.2f}"    if sq_info.get("pegRatio")   else "N/A",
                    "P/S":         f"{sq_info['priceToSalesTrailing12Months']:.2f}" if sq_info.get("priceToSalesTrailing12Months") else "N/A",
                    "P/B":         f"{sq_info['priceToBook']:.2f}" if sq_info.get("priceToBook") else "N/A",
                }.items()):
                    _r1[_ci].metric(_lbl, _val)

                _r2 = st.columns(6)
                for _ci, (_lbl, _val) in enumerate({
                    "52W最高":      f"${sq_info['fiftyTwoWeekHigh']:.2f}"     if sq_info.get("fiftyTwoWeekHigh") else "N/A",
                    "52W最低":      f"${sq_info['fiftyTwoWeekLow']:.2f}"      if sq_info.get("fiftyTwoWeekLow")  else "N/A",
                    "50日均线":     f"${sq_info['fiftyDayAverage']:.2f}"      if sq_info.get("fiftyDayAverage")  else "N/A",
                    "200日均线":    f"${sq_info['twoHundredDayAverage']:.2f}" if sq_info.get("twoHundredDayAverage") else "N/A",
                    "分析师目标价": f"${sq_info['targetMeanPrice']:.2f}"      if sq_info.get("targetMeanPrice")  else "N/A",
                    "上行空间":     f"{(sq_info['targetMeanPrice']/_sq_price-1)*100:+.1f}%" if (sq_info.get("targetMeanPrice") and _sq_price) else "N/A",
                }.items()):
                    _r2[_ci].metric(_lbl, _val)

                st.divider()

                # ════════════════════════════════════════════════════════════
                # 🏆 巴菲特价值评分卡
                # ════════════════════════════════════════════════════════════
                st.subheader("🏆 巴菲特价值评分")

                _bf = calc_buffett_score(sq_info, sq_fins)
                _score = _bf["score"]
                _grade = _bf["grade"]
                _color = _bf["color"]
                _bar_w = int(_score)

                st.markdown(
                    f"<div style='background:#0d0d1a;border:2px solid {_color};"
                    f"border-radius:12px;padding:20px 24px;margin-bottom:16px'>"
                    f"<div style='display:flex;align-items:center;gap:24px;flex-wrap:wrap'>"
                    f"<div style='text-align:center;min-width:100px'>"
                    f"<div style='font-size:3.5rem;font-weight:900;color:{_color};line-height:1'>{_score}</div>"
                    f"<div style='color:#888;font-size:0.8rem;margin-top:2px'>满分100</div>"
                    f"</div>"
                    f"<div style='flex:1'>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:{_color};margin-bottom:8px'>{_grade}</div>"
                    f"<div style='background:#1e1e2e;border-radius:6px;height:12px;overflow:hidden'>"
                    f"<div style='background:{_color};width:{_bar_w}%;height:100%;border-radius:6px;"
                    f"transition:width 0.5s'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;color:#555;font-size:0.7rem;margin-top:3px'>"
                    f"<span>0</span><span>40</span><span>55</span><span>70</span><span>85</span><span>100</span>"
                    f"</div>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )

                _bf_cols = st.columns(2)
                for _idx, (_dim, _pts, _max, _icon, _note) in enumerate(_bf["items"]):
                    with _bf_cols[_idx % 2]:
                        _item_color = "#00CC96" if _icon == "✅" else ("#FFA500" if _icon == "⚠️" else "#EF553B")
                        _fill_pct   = int(_pts / _max * 100)
                        st.markdown(
                            f"<div style='background:#12122a;border:1px solid #2a2a3e;"
                            f"border-radius:8px;padding:12px 14px;margin-bottom:8px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                            f"<span style='color:#ccc;font-size:0.88rem;font-weight:600'>{_dim}</span>"
                            f"<span style='color:{_item_color};font-weight:700;font-size:1rem'>"
                            f"{_icon} {_pts}/{_max}</span>"
                            f"</div>"
                            f"<div style='background:#1e1e2e;border-radius:4px;height:6px;margin:6px 0;overflow:hidden'>"
                            f"<div style='background:{_item_color};width:{_fill_pct}%;height:100%;border-radius:4px'></div>"
                            f"</div>"
                            f"<div style='color:#888;font-size:0.78rem'>{_note}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                _summary_lines = []
                if _score >= 70:
                    _summary_lines.append("✅ **具备巴菲特选股基本特征**，适合中长期持有")
                else:
                    _summary_lines.append("⚠️ **不完全符合价值投资标准**，建议等待更好价格或寻找更强护城河标的")
                _weak = [i[0] for i in _bf["items"] if i[3] == "❌"]
                if _weak:
                    _summary_lines.append(f"🔴 **主要弱项**：{'、'.join(_weak)}")
                _strong = [i[0] for i in _bf["items"] if i[3] == "✅"]
                if _strong:
                    _summary_lines.append(f"🟢 **核心优势**：{'、'.join(_strong)}")
                st.info("\n\n".join(_summary_lines))

                st.divider()

                # ── 价格走势 + 技术指标 ───────────────────────────────────────
                sq_left, sq_right = st.columns([3, 1])

                with sq_left:
                    st.subheader("📈 价格走势 & 技术指标")
                    if not sq_hist.empty:
                        sq_close = sq_hist["Close"]
                        _ma20    = sq_close.rolling(20).mean()
                        _ma50    = sq_close.rolling(50).mean()
                        _ma200s  = sq_close.rolling(200).mean()
                        _bb_mid  = sq_close.rolling(20).mean()
                        _bb_std  = sq_close.rolling(20).std()
                        _bb_up   = _bb_mid + 2 * _bb_std
                        _bb_dn   = _bb_mid - 2 * _bb_std

                        fig_sq = go.Figure()
                        fig_sq.add_trace(go.Scatter(x=sq_hist.index, y=sq_close,
                            name="收盘价", line=dict(color="#4a9eff", width=2)))
                        fig_sq.add_trace(go.Scatter(x=sq_hist.index, y=_ma20,
                            name="MA20", line=dict(color="#FFA500", width=1, dash="dot")))
                        fig_sq.add_trace(go.Scatter(x=sq_hist.index, y=_ma50,
                            name="MA50", line=dict(color="#00CC96", width=1, dash="dot")))
                        if len(sq_close) >= 200:
                            fig_sq.add_trace(go.Scatter(x=sq_hist.index, y=_ma200s,
                                name="MA200", line=dict(color="#EF553B", width=1.5, dash="dash")))
                        fig_sq.add_trace(go.Scatter(x=sq_hist.index, y=_bb_up,
                            name="布林上轨", line=dict(color="rgba(150,150,150,0.4)", width=1)))
                        fig_sq.add_trace(go.Scatter(x=sq_hist.index, y=_bb_dn,
                            name="布林下轨", line=dict(color="rgba(150,150,150,0.4)", width=1),
                            fill="tonexty", fillcolor="rgba(150,150,150,0.05)"))
                        fig_sq.update_layout(
                            title=f"{sq_ticker} — {_sq_name} ({sq_period})",
                            height=430, hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            **CHART_LAYOUT,
                        )
                        st.plotly_chart(fig_sq, use_container_width=True)

                        # RSI 副图
                        _delta  = sq_close.diff()
                        _gain   = _delta.clip(lower=0)
                        _loss   = (-_delta.clip(upper=0))
                        _rsi14  = 100 - 100 / (1 + _gain.rolling(14).mean() / _loss.rolling(14).mean().replace(0, float("nan")))
                        _rsi5   = 100 - 100 / (1 + _gain.rolling(5).mean()  / _loss.rolling(5).mean().replace(0,  float("nan")))

                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=sq_hist.index, y=_rsi14,
                            name="RSI(14)", line=dict(color="#4a9eff", width=1.5)))
                        fig_rsi.add_trace(go.Scatter(x=sq_hist.index, y=_rsi5,
                            name="RSI(5)", line=dict(color="#FFA500", width=1.5)))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF553B", annotation_text="超买70")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00CC96", annotation_text="超卖30")
                        fig_rsi.add_hline(y=50, line_dash="dot",  line_color="gray")
                        fig_rsi.update_layout(
                            title="RSI", height=200,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
                            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", range=[0,100]),
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    else:
                        st.warning("历史行情数据暂不可用")

                with sq_right:
                    st.subheader("🔢 技术读数")
                    if not sq_hist.empty and len(sq_hist) >= 5:
                        _latest = sq_hist["Close"].iloc[-1]
                        _p5     = sq_hist["Close"].iloc[-6]  if len(sq_hist) >= 6  else sq_hist["Close"].iloc[0]
                        _p20    = sq_hist["Close"].iloc[-21] if len(sq_hist) >= 21 else sq_hist["Close"].iloc[0]
                        _c5     = (_latest/_p5  - 1)*100
                        _c20    = (_latest/_p20 - 1)*100
                        _d2     = sq_hist["Close"].diff()
                        _rv14   = (100 - 100/(1+_d2.clip(lower=0).rolling(14).mean()/_d2.clip(upper=0).abs().rolling(14).mean())).iloc[-1]
                        _rv5    = (100 - 100/(1+_d2.clip(lower=0).rolling(5).mean() /_d2.clip(upper=0).abs().rolling(5).mean())).iloc[-1]
                        _bbm    = sq_hist["Close"].rolling(20).mean().iloc[-1]
                        _bbs    = sq_hist["Close"].rolling(20).std().iloc[-1]
                        _bbpct  = (_latest - (_bbm-2*_bbs))/(4*_bbs)*100 if _bbs else 50

                        for _lbl, _val, _col in [
                            ("RSI(14)", f"{_rv14:.1f}", "#EF553B" if _rv14>70 else ("#00CC96" if _rv14<30 else "#fff")),
                            ("RSI(5)",  f"{_rv5:.1f}",  "#EF553B" if _rv5>70  else ("#00CC96" if _rv5<30  else "#fff")),
                            ("5日涨跌",  f"{_c5:+.1f}%",  "#00CC96" if _c5>=0  else "#EF553B"),
                            ("20日涨跌", f"{_c20:+.1f}%", "#00CC96" if _c20>=0 else "#EF553B"),
                            ("BB位置%",  f"{_bbpct:.0f}%","#EF553B" if _bbpct>80 else ("#00CC96" if _bbpct<20 else "#fff")),
                        ]:
                            st.markdown(
                                f"<div style='display:flex;justify-content:space-between;"
                                f"padding:6px 0;border-bottom:1px solid #1e1e2e'>"
                                f"<span style='color:#888;font-size:0.85rem'>{_lbl}</span>"
                                f"<span style='color:{_col};font-weight:600'>{_val}</span></div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
                        st.markdown("**短线信号**")
                        _slist = []
                        if _rv5 < 30:   _slist.append(("🟢 RSI超卖", "#00CC96"))
                        if _rv5 > 70:   _slist.append(("🔴 RSI超买", "#EF553B"))
                        if _bbpct < 15: _slist.append(("🟢 近布林下轨", "#00CC96"))
                        if _bbpct > 85: _slist.append(("🔴 近布林上轨", "#EF553B"))
                        if _c5 < -8:    _slist.append(("⚡ 5日急跌>8%", "#FFA500"))
                        if not _slist:  _slist.append(("⚪ 无明显信号", "#888"))
                        for _s, _c in _slist:
                            st.markdown(f"<div style='color:{_c};font-size:0.88rem;padding:3px 0'>{_s}</div>",
                                        unsafe_allow_html=True)

                        # ATR 动态止损
                        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
                        st.markdown("**动态止损参考**")
                        _atr_val, _atr_stop, _atr_pct = calc_atr_stoploss(sq_hist)
                        if _atr_val is not None:
                            for _lbl, _val, _col in [
                                ("ATR(14)", f"{_atr_val:.2f}", "#aaa"),
                                ("建议止损价", f"${_atr_stop:.2f}", "#EF553B"),
                                ("止损幅度", f"{_atr_pct:.1f}%", "#EF553B"),
                            ]:
                                st.markdown(
                                    f"<div style='display:flex;justify-content:space-between;"
                                    f"padding:5px 0;border-bottom:1px solid #1e1e2e'>"
                                    f"<span style='color:#888;font-size:0.85rem'>{_lbl}</span>"
                                    f"<span style='color:{_col};font-weight:600'>{_val}</span></div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(
                                f"<div style='color:#555;font-size:0.72rem;margin-top:4px'>"
                                f"ATR×2.5 · 当前价 ${_latest:.2f}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("ATR数据不足")

                st.divider()

                # ── 财务基本面 3 列 ────────────────────────────────────────────
                st.subheader("📋 财务基本面")
                _fa1, _fa2, _fa3 = st.columns(3)

                def _fa_row(label, value):
                    return (
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:4px 0;border-bottom:1px solid #1e1e2e;font-size:0.88rem'>"
                        f"<span style='color:#888'>{label}</span>"
                        f"<span style='color:#fff'>{value}</span></div>"
                    )

                with _fa1:
                    st.markdown("**收入 & 增长**")
                    _rev   = sq_info.get("totalRevenue")
                    _rev_s = (f"${_rev/1e9:.2f}B" if _rev and _rev>=1e9 else (f"${_rev/1e6:.0f}M" if _rev else "N/A"))
                    _edate = "N/A"
                    if sq_info.get("earningsDate"):
                        try: _edate = str(sq_info["earningsDate"][0])
                        except: pass
                    for _k,_v in [
                        ("TTM营收",      _rev_s),
                        ("营收增速YoY",  f"{sq_info.get('revenueGrowth',0)*100:+.1f}%"  if sq_info.get("revenueGrowth") else "N/A"),
                        ("EPS(TTM)",     f"${sq_info['trailingEps']:.2f}"  if sq_info.get("trailingEps") else "N/A"),
                        ("EPS(Forward)", f"${sq_info['forwardEps']:.2f}"   if sq_info.get("forwardEps")  else "N/A"),
                        ("EPS增速",      f"{sq_info.get('earningsGrowth',0)*100:+.1f}%" if sq_info.get("earningsGrowth") else "N/A"),
                        ("下次财报日",   _edate),
                    ]: st.markdown(_fa_row(_k,_v), unsafe_allow_html=True)

                with _fa2:
                    st.markdown("**利润率 & 回报**")
                    for _k,_v in [
                        ("毛利率",     f"{sq_info.get('grossMargins',0)*100:.1f}%"     if sq_info.get("grossMargins")    else "N/A"),
                        ("营业利润率", f"{sq_info.get('operatingMargins',0)*100:.1f}%" if sq_info.get("operatingMargins") else "N/A"),
                        ("净利率",     f"{sq_info.get('profitMargins',0)*100:.1f}%"    if sq_info.get("profitMargins")   else "N/A"),
                        ("ROE",        f"{sq_info.get('returnOnEquity',0)*100:.1f}%"   if sq_info.get("returnOnEquity")  else "N/A"),
                        ("ROA",        f"{sq_info.get('returnOnAssets',0)*100:.1f}%"   if sq_info.get("returnOnAssets")  else "N/A"),
                        ("自由现金流", f"${sq_info.get('freeCashflow',0)/1e9:.2f}B"    if sq_info.get("freeCashflow")    else "N/A"),
                    ]: st.markdown(_fa_row(_k,_v), unsafe_allow_html=True)

                with _fa3:
                    st.markdown("**分析师 & 机构**")
                    _rec   = sq_info.get("recommendationKey","N/A").upper()
                    _rc    = {"STRONG_BUY":"#00CC96","BUY":"#82C8FF","HOLD":"#FFA500",
                              "SELL":"#EF553B","STRONG_SELL":"#CC0000"}.get(_rec,"#aaa")
                    st.markdown(
                        f"<div style='text-align:center;padding:10px;border:1px solid {_rc};"
                        f"border-radius:8px;margin-bottom:8px'>"
                        f"<div style='font-size:1.2rem;font-weight:700;color:{_rc}'>{_rec}</div>"
                        f"<div style='color:#888;font-size:0.78rem'>分析师共识</div></div>",
                        unsafe_allow_html=True,
                    )
                    for _k,_v in [
                        ("分析师数量",  str(sq_info.get("numberOfAnalystOpinions","N/A"))),
                        ("目标价均值",  f"${sq_info['targetMeanPrice']:.2f}" if sq_info.get("targetMeanPrice") else "N/A"),
                        ("目标价最高",  f"${sq_info['targetHighPrice']:.2f}" if sq_info.get("targetHighPrice") else "N/A"),
                        ("目标价最低",  f"${sq_info['targetLowPrice']:.2f}"  if sq_info.get("targetLowPrice")  else "N/A"),
                        ("机构持仓%",   f"{sq_info.get('heldPercentInstitutions',0)*100:.1f}%" if sq_info.get("heldPercentInstitutions") else "N/A"),
                        ("做空比例%",   f"{sq_info.get('shortPercentOfFloat',0)*100:.1f}%"    if sq_info.get("shortPercentOfFloat")      else "N/A"),
                    ]: st.markdown(_fa_row(_k,_v), unsafe_allow_html=True)

                _summary_text = sq_info.get("longBusinessSummary","")
                if _summary_text:
                    st.divider()
                    with st.expander("📖 公司简介"):
                        st.write(_summary_text)

        elif sq_run and not sq_ticker_raw:
            st.warning("请输入股票代码")
        else:
            if not st.session_state.get("sq_last_ticker"):
                st.markdown(
                    "<div style='text-align:center;padding:60px 0;color:#888'>"
                    "<div style='font-size:3rem'>🔎</div>"
                    "<div style='font-size:1.2rem;margin-top:12px'>输入任意美股代码，查询完整数据 + 巴菲特评分</div>"
                    "<div style='font-size:0.85rem;margin-top:8px;color:#aaa'>"
                    "支持：普通股 / ETF / 港股(.HK) / 台股(.TW)</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
