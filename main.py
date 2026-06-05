"""CLI entry point — generates a daily research report and saves it to reports/."""
import os
import argparse
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from modules.market_report import get_daily_summary, get_sox_beta
from modules.valuation import get_valuation_metrics
from modules.supply_chain import get_adr_premium, get_hyperscaler_capex
from modules.earnings import get_earnings_summary
from modules.news_monitor import fetch_news_feed


def load_watchlist(path: str = "watchlist.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _section(title: str, content: str = ""):
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")
    if content:
        print(content)


def _df_str(df: pd.DataFrame, cols: list = None) -> str:
    if df is None or df.empty:
        return "  (无数据)"
    sub = df[cols] if cols else df
    return sub.to_string(index=False)


def run_report(watchlist: pd.DataFrame, save: bool = True) -> str:
    now = datetime.now()
    lines = []

    def p(text: str = ""):
        lines.append(text)
        print(text)

    p(f"\n{'#' * 64}")
    p(f"  AI 半导体研究日报  {now.strftime('%Y-%m-%d %H:%M')}")
    p(f"{'#' * 64}")

    all_tickers = watchlist["ticker"].tolist()
    core = watchlist[watchlist["priority"] == "high"]["ticker"].tolist()
    stocks = [t for t in all_tickers if not t.startswith("^") and not t.endswith(".TW")]

    # 1. Market
    p("\n[1/5] 每日价格变动 ...")
    daily_df = get_daily_summary(all_tickers)
    _section("每日价格变动")
    s = _df_str(daily_df, ["Ticker", "Price", "Change%", "Vol/AvgVol", "% from High"])
    p(s)

    betas = get_sox_beta(stocks)
    if betas:
        beta_df = pd.DataFrame(list(betas.items()), columns=["Ticker", "Beta vs SOX"])
        _section("SOX Beta (3个月)")
        p(_df_str(beta_df))

    # 2. Valuation
    p("\n[2/5] 估值分析 ...")
    val_df = get_valuation_metrics(core)
    _section("估值分析 (核心标的)")
    cols = [c for c in ["Ticker", "Trailing PE", "Forward PE", "PEG", "Market Cap (B)", "Upside%"] if c in val_df.columns]
    p(_df_str(val_df, cols))

    # 3. Earnings
    p("\n[3/5] 财报摘要 ...")
    earn_df = get_earnings_summary(core)
    _section("财报摘要")
    cols = [c for c in ["Ticker", "Revenue TTM (B)", "Revenue Growth YoY%", "Gross Margin%", "EPS Beat%", "Next Earnings"] if c in earn_df.columns]
    p(_df_str(earn_df, cols))

    # 4. Supply chain
    p("\n[4/5] 供应链追踪 ...")
    adr_df = get_adr_premium()
    if not adr_df.empty:
        latest = adr_df["ADR_Premium%"].iloc[-1]
        avg = adr_df["ADR_Premium%"].mean()
        _section("TSM ADR 溢价监控")
        p(f"  最新溢价: {latest:+.2f}%  |  均值: {avg:+.2f}%")
        p(f"  (正=ADR溢价，负=ADR折价)")

    capex_df = get_hyperscaler_capex()
    if not capex_df.empty:
        _section("超大规模云厂 CapEx (B$, 季度)")
        p(_df_str(capex_df.head(12)))

    # 5. News
    p("\n[5/5] 重要新闻 ...")
    news_df = fetch_news_feed(max_items_per_feed=5)
    high_news = news_df[news_df["Impact"].str.contains("High")] if not news_df.empty else pd.DataFrame()
    _section(f"高影响新闻 ({len(high_news)}/{len(news_df)}条)")
    if not high_news.empty:
        for _, row in high_news.head(10).iterrows():
            p(f"  {row['Sentiment']} [{row['Source']}] {row['Title']}")
    else:
        p("  暂无高影响新闻")

    report_text = "\n".join(lines)

    if save:
        os.makedirs("reports", exist_ok=True)
        path = f"reports/daily_{now.strftime('%Y%m%d_%H%M')}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n✅ 报告已保存: {path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="AI 半导体股票研究系统 CLI")
    parser.add_argument("--watchlist", default="watchlist.csv", help="Watchlist CSV 路径")
    parser.add_argument("--no-save", action="store_true", help="不保存报告文件")
    args = parser.parse_args()

    wl = load_watchlist(args.watchlist)
    run_report(wl, save=not args.no_save)


if __name__ == "__main__":
    main()
