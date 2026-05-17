"""Day 2 — 포트폴리오 수익 추적 및 시각화."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

INITIAL_CAPITAL = 10_000_000
FEE_RATE = 0.0005
INVESTMENT_DAYS = 90

PORTFOLIO = {
    "KRW-BTC": {"weight": 0.4, "amount": 4_000_000},
    "KRW-ETH": {"weight": 0.3, "amount": 3_000_000},
    "KRW-SOL": {"weight": 0.2, "amount": 2_000_000},
    "KRW-XRP": {"weight": 0.1, "amount": 1_000_000},
}


def calc_mdd(series: pd.Series) -> float:
    """최대 낙폭(MDD)을 퍼센트로 계산합니다."""
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown.min() * 100


def build_portfolio_series(
    data: dict[str, pd.DataFrame],
    portfolio=None,
    days: int = INVESTMENT_DAYS,
    fee_rate: float = FEE_RATE,
) -> tuple[pd.DataFrame, dict]:
    """
    투자 시작일 종가 기준 매수 후 일별 포트폴리오 가치를 계산합니다.
    반환: (일별 요약 DataFrame, 종목별 상세 dict)
    """
    portfolio = portfolio or PORTFOLIO
    tickers = [t for t in portfolio if t in data]
    if not tickers:
        raise ValueError("포트폴리오 종목 데이터가 없습니다.")

    closes = pd.DataFrame({t: data[t]["close"] for t in tickers}).dropna()
    closes = closes.tail(days + 1)
    if len(closes) < 2:
        raise ValueError("분석 기간 데이터가 부족합니다.")

    start_date = closes.index[0]
    start_prices = closes.loc[start_date]

    holdings = {}
    for ticker in tickers:
        amount = portfolio[ticker]["amount"]
        invest = amount * (1 - fee_rate)
        holdings[ticker] = invest / start_prices[ticker]

    value_df = pd.DataFrame(index=closes.index)
    for ticker in tickers:
        value_df[ticker] = holdings[ticker] * closes[ticker]

    value_df["total"] = value_df[tickers].sum(axis=1)
    value_df["daily_return"] = value_df["total"].pct_change() * 100

    details = {
        "holdings": holdings,
        "start_date": start_date,
        "end_date": closes.index[-1],
        "start_prices": start_prices,
        "tickers": tickers,
    }
    return value_df, details


def calc_contribution(
    data: dict[str, pd.DataFrame],
    portfolio: dict,
    details: dict,
) -> dict[str, dict]:
    """종목별 수익률 및 포트폴리오 기여도(%p)를 계산합니다."""
    start = details["start_date"]
    end = details["end_date"]
    contributions = {}
    for ticker in details["tickers"]:
        prices = data[ticker]["close"]
        start_p = prices.loc[start]
        end_p = prices.loc[end]
        ret = (end_p / start_p - 1) * 100
        weight = portfolio[ticker]["weight"]
        contrib = ret * weight
        contributions[ticker] = {"return_pct": ret, "contribution": contrib}
    return contributions


def print_portfolio_summary(
    value_df: pd.DataFrame,
    details: dict,
    contributions: dict,
    initial_capital: float = INITIAL_CAPITAL,
) -> None:
    """포트폴리오 성과 요약을 출력합니다."""
    start = details["start_date"].strftime("%Y-%m-%d")
    end = details["end_date"].strftime("%Y-%m-%d")
    days = (details["end_date"] - details["start_date"]).days
    initial = value_df["total"].iloc[0]
    final = value_df["total"].iloc[-1]
    total_ret = (final / initial - 1) * 100
    mdd = calc_mdd(value_df["total"])

    print("=== 포트폴리오 성과 요약 ===")
    print(f"투자 기간    : {start} ~ {end} ({days}일)")
    print(f"초기 자산    : {initial:,.0f} 원")
    print(f"현재 자산    : {final:,.0f} 원")
    print(f"총 수익률    : {total_ret:+.2f}%")
    print(f"MDD         : {mdd:.2f}%")
    print("\n종목별 기여도:")
    for ticker, c in contributions.items():
        print(
            f"  {ticker}  수익률 {c['return_pct']:+.2f}%  "
            f"기여 {c['contribution']:+.2f}%p"
        )


def make_portfolio_chart(
    value_df: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    details: dict,
):
    """포트폴리오 자산 곡선·누적 수익률·일별 수익률 3단 차트를 생성합니다."""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "포트폴리오 총 자산",
            "종목별 누적 수익률 (시작=100)",
            "일별 수익률",
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.35, 0.25],
    )

    fig.add_trace(
        go.Scatter(
            x=value_df.index,
            y=value_df["total"],
            line=dict(color="#2563eb", width=2),
            name="총 자산",
        ),
        row=1,
        col=1,
    )

    start = details["start_date"]
    for ticker in details["tickers"]:
        prices = data[ticker]["close"].loc[value_df.index]
        normalized = prices / prices.loc[start] * 100
        fig.add_trace(
            go.Scatter(x=value_df.index, y=normalized, name=ticker, mode="lines"),
            row=2,
            col=1,
        )

    colors = ["#ef4444" if r >= 0 else "#3b82f6" for r in value_df["daily_return"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=value_df.index,
            y=value_df["daily_return"],
            marker_color=colors,
            name="일별 수익률",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(height=900, template="plotly_white", title="포트폴리오 분석")
    fig.update_yaxes(title_text="원", row=1, col=1)
    fig.update_yaxes(title_text="지수 (100=시작)", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    return fig


def make_correlation_heatmap(value_df: pd.DataFrame, data: dict, details: dict):
    """4종목 일별 수익률 상관관계 히트맵을 matplotlib Figure로 반환합니다."""
    plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    daily_rets = pd.DataFrame()
    for ticker in details["tickers"]:
        daily_rets[ticker] = data[ticker]["close"].pct_change()
    daily_rets = daily_rets.loc[value_df.index].dropna()
    corr = daily_rets.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("종목별 일별 수익률 상관관계")
    plt.tight_layout()
    return fig, corr


def get_portfolio_metrics(value_df: pd.DataFrame, details: dict) -> dict:
    """대시보드용 포트폴리오 핵심 지표를 반환합니다."""
    initial = value_df["total"].iloc[0]
    final = value_df["total"].iloc[-1]
    return {
        "initial": initial,
        "final": final,
        "return_pct": (final / initial - 1) * 100,
        "mdd": calc_mdd(value_df["total"]),
        "start": details["start_date"].strftime("%Y-%m-%d"),
        "end": details["end_date"].strftime("%Y-%m-%d"),
    }
