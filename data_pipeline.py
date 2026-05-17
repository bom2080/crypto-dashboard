"""Day 1 — 데이터 수집, 지표 계산, 시각화."""

import time
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyupbit
from plotly.subplots import make_subplots

TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP"]
OHLCV_COUNT = 200
TRADING_DAYS = 180

_cache: dict = {}


def call_upbit_api(ticker: str) -> Optional[pd.DataFrame]:
    """업비트 API를 한 번 호출해 OHLCV 데이터를 반환합니다."""
    return pyupbit.get_ohlcv(ticker, count=OHLCV_COUNT)


def fetch_with_retry(ticker: str, max_retries: int = 3, delay: float = 1.0) -> Optional[pd.DataFrame]:
    """API 호출 실패 시 최대 max_retries번 재시도합니다."""
    for attempt in range(max_retries):
        try:
            result = call_upbit_api(ticker)
            if result is not None:
                return result
        except Exception as e:
            print(f"  [{ticker}] 시도 {attempt + 1}/{max_retries} 실패: {e}")
        if attempt < max_retries - 1:
            time.sleep(delay)
    return None


def get_data(ticker: str) -> Optional[pd.DataFrame]:
    """당일 캐시가 있으면 재호출 없이 반환하고, 없으면 API로 수집합니다."""
    today = str(date.today())
    if ticker in _cache and _cache[ticker]["date"] == today:
        return _cache[ticker]["df"]
    df = fetch_with_retry(ticker)
    if df is not None:
        _cache[ticker] = {"date": today, "df": df}
    return df


def collect_all_data(tickers: Optional[list] = None) -> dict:
    """여러 종목의 최근 180일 OHLCV를 수집합니다."""
    tickers = tickers or TICKERS
    result = {}
    for ticker in tickers:
        df = get_data(ticker)
        if df is not None:
            result[ticker] = df.tail(TRADING_DAYS).copy()
    return result


def calc_returns(df: pd.DataFrame) -> pd.DataFrame:
    """일별·7일·30일 수익률 컬럼을 추가합니다."""
    df["return_1d"] = df["close"].pct_change(1) * 100
    df["return_7d"] = df["close"].pct_change(7) * 100
    df["return_30d"] = df["close"].pct_change(30) * 100
    return df


def calc_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """MA5, MA20, MA60 이동평균 컬럼을 추가합니다."""
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()
    return df


def calc_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """볼린저 밴드 상·하단 컬럼을 추가합니다 (MA20 필요)."""
    std_20 = df["close"].rolling(20).std()
    df["upper_band"] = df["MA20"] + (std_20 * 2)
    df["lower_band"] = df["MA20"] - (std_20 * 2)
    return df


def calc_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """20일 수익률 표준편차를 √252로 연환산한 volatility_20d를 추가합니다."""
    daily_return = df["close"].pct_change(1)
    std_20d = daily_return.rolling(20).std()
    df["volatility_20d"] = std_20d * np.sqrt(252) * 100
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """수익률·이동평균·볼린저·변동성 지표를 한 번에 추가합니다."""
    df = df.copy()
    df = calc_returns(df)
    df = calc_moving_averages(df)
    df = calc_bollinger_bands(df)
    df = calc_volatility(df)
    return df


def prepare_market_data(tickers: Optional[list] = None) -> dict:
    """수집 후 모든 지표를 계산한 종목별 DataFrame 딕셔너리를 반환합니다."""
    data = collect_all_data(tickers)
    return {t: add_indicators(df) for t, df in data.items()}


def format_price(price: float) -> str:
    """가격을 '89,250,000 원' 형식으로 포맷합니다."""
    return f"{price:,.0f} 원"


def format_pct(value: float) -> str:
    """수익률을 '+5.23%' 형식으로 포맷합니다."""
    if pd.isna(value):
        return "N/A"
    return f"{value:+.2f}%"


def format_vol(value: float) -> str:
    """변동성을 '72.3%' 형식으로 포맷합니다."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def print_summary(data: dict[str, pd.DataFrame]) -> None:
    """전체 종목 현황 요약 표를 출력합니다."""
    today = str(date.today())
    print(f"=== 암호화폐 현황 요약 (기준일: {today}) ===\n")
    print(f"{'종목':<12} {'현재가':>18} {'7일 수익률':>12} {'30일 수익률':>12} {'연환산 변동성':>13}")
    print("-" * 74)
    for ticker in data:
        last = data[ticker].iloc[-1]
        print(
            f"{ticker:<12} {format_price(last['close']):>18} "
            f"{format_pct(last['return_7d']):>12} {format_pct(last['return_30d']):>12} "
            f"{format_vol(last['volatility_20d']):>13}"
        )


def make_chart_title(ticker: str, df: pd.DataFrame) -> str:
    """차트 제목 문자열을 생성합니다."""
    last = df.iloc[-1]
    return f"{ticker}  |  {last['close']:,.0f} 원  ({format_pct(last['return_7d'])} 7d)"


def add_bollinger_fill(fig, df: pd.DataFrame, row: int, col: int, show_legend: bool) -> None:
    """볼린저 밴드 상·하단 사이를 반투명으로 채웁니다."""
    x_fill = list(df.index) + list(df.index[::-1])
    y_fill = list(df["upper_band"]) + list(df["lower_band"][::-1])
    fig.add_trace(
        go.Scatter(
            x=x_fill,
            y=y_fill,
            fill="toself",
            fillcolor="rgba(0,100,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="볼린저 밴드",
            showlegend=show_legend,
            legendgroup="band",
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def add_band_lines(fig, df: pd.DataFrame, row: int, col: int) -> None:
    """볼린저 밴드 상·하단 점선을 추가합니다."""
    for col_name, label in [("upper_band", "상단"), ("lower_band", "하단")]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_name],
                line=dict(color="royalblue", width=0.8, dash="dot"),
                showlegend=False,
                name=f"{label} 밴드",
            ),
            row=row,
            col=col,
        )


def add_price_and_ma_lines(fig, df: pd.DataFrame, row: int, col: int, show_legend: bool) -> None:
    """종가 및 MA5/MA20/MA60 선을 추가합니다."""
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            line=dict(color="black", width=1.5),
            name="종가",
            showlegend=show_legend,
            legendgroup="close",
        ),
        row=row,
        col=col,
    )
    for col_name, color in [("MA5", "orange"), ("MA20", "blue"), ("MA60", "purple")]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_name],
                line=dict(color=color, width=1.2),
                name=col_name,
                showlegend=show_legend,
                legendgroup=col_name,
            ),
            row=row,
            col=col,
        )


def make_bollinger_chart(data: dict[str, pd.DataFrame]):
    """4종목 볼린저 밴드 2×2 차트를 생성합니다."""
    titles = [make_chart_title(t, data[t]) for t in data]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i, ticker in enumerate(data):
        df = data[ticker]
        row, col = positions[i]
        show_legend = i == 0
        add_bollinger_fill(fig, df, row, col, show_legend)
        add_band_lines(fig, df, row, col)
        add_price_and_ma_lines(fig, df, row, col, show_legend)
    fig.update_layout(
        title="암호화폐 4종목 — 볼린저 밴드 차트 (최근 180일)",
        height=800,
        template="plotly_white",
    )
    return fig
