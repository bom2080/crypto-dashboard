"""Day 3 — 골든크로스/데드크로스 백테스팅 엔진."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_pipeline import add_indicators, fetch_with_retry

BACKTEST_TICKER = "KRW-BTC"
BACKTEST_DAYS = 365
INITIAL_CAPITAL = 1_000_000
FEE_RATE = 0.0005


def load_backtest_data(ticker: str = BACKTEST_TICKER, days: int = BACKTEST_DAYS) -> pd.DataFrame:
    """백테스팅용 OHLCV 데이터를 수집하고 지표를 계산합니다."""
    df = fetch_with_retry(ticker, max_retries=3, delay=1.0)
    if df is None:
        raise RuntimeError(f"{ticker} 데이터 수집 실패")
    df = df.tail(days + 60).copy()
    df = add_indicators(df)
    return df.tail(days).copy()


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """골든크로스(매수)·데드크로스(매도) 신호 컬럼을 추가합니다."""
    df = df.copy()
    prev_ma5 = df["MA5"].shift(1)
    prev_ma20 = df["MA20"].shift(1)
    df["golden_cross"] = (prev_ma5 < prev_ma20) & (df["MA5"] >= df["MA20"])
    df["dead_cross"] = (prev_ma5 > prev_ma20) & (df["MA5"] <= df["MA20"])
    return df


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
) -> dict:
    """
    MA5/MA20 크로스 전략 백테스트를 실행합니다.
    반환: trades, equity_curve, buy_hold_curve, metrics
    """
    df = generate_signals(df)
    cash = initial_capital
    position = 0.0
    in_position = False
    trades = []
    equity = []
    buy_hold_units = 0.0

    first_price = df["close"].iloc[0]
    buy_hold_units = (initial_capital * (1 - fee_rate)) / first_price

    for dt, row in df.iterrows():
        price = row["close"]

        if row["golden_cross"] and not in_position:
            invest = cash * (1 - fee_rate)
            position = invest / price
            cash = 0.0
            in_position = True
            trades.append(
                {
                    "type": "매수",
                    "date": dt,
                    "price": price,
                    "qty": position,
                    "amount": invest,
                    "pnl_pct": None,
                }
            )
        elif row["dead_cross"] and in_position:
            amount = position * price * (1 - fee_rate)
            buy_amount = trades[-1]["amount"]
            pnl_pct = (amount / buy_amount - 1) * 100
            trades.append(
                {
                    "type": "매도",
                    "date": dt,
                    "price": price,
                    "qty": position,
                    "amount": amount,
                    "pnl_pct": pnl_pct,
                }
            )
            cash = amount
            position = 0.0
            in_position = False

        total = cash + position * price
        equity.append({"date": dt, "strategy": total, "buy_hold": buy_hold_units * price})

    last_dt = df.index[-1]
    last_price = df["close"].iloc[-1]
    if in_position:
        amount = position * last_price * (1 - fee_rate)
        buy_amount = trades[-1]["amount"]
        pnl_pct = (amount / buy_amount - 1) * 100
        trades.append(
            {
                "type": "매도",
                "date": last_dt,
                "price": last_price,
                "qty": position,
                "amount": amount,
                "pnl_pct": pnl_pct,
                "forced": True,
            }
        )
        cash = amount
        position = 0.0
        equity[-1]["strategy"] = cash

    equity_df = pd.DataFrame(equity).set_index("date")
    metrics = calc_backtest_metrics(equity_df, trades, initial_capital, first_price, last_price)
    return {"trades": trades, "equity": equity_df, "df": df, "metrics": metrics}


def calc_backtest_metrics(
    equity_df: pd.DataFrame,
    trades: list,
    initial_capital: float,
    first_price: float,
    last_price: float,
) -> dict:
    """백테스팅 성과 지표를 계산합니다."""
    final_strategy = equity_df["strategy"].iloc[-1]
    final_bh = equity_df["buy_hold"].iloc[-1]
    total_ret = (final_strategy / initial_capital - 1) * 100
    bh_ret = (final_bh / initial_capital - 1) * 100

    rolling_max = equity_df["strategy"].cummax()
    mdd = ((equity_df["strategy"] - rolling_max) / rolling_max).min() * 100

    sells = [t for t in trades if t["type"] == "매도"]
    wins = [t for t in sells if t.get("pnl_pct") and t["pnl_pct"] > 0]
    losses = [t for t in sells if t.get("pnl_pct") and t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0.0
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0.0

    buys = [t for t in trades if t["type"] == "매수"]
    return {
        "initial": initial_capital,
        "final": final_strategy,
        "total_return": total_ret,
        "mdd": mdd,
        "n_trades": len(trades),
        "n_buys": len(buys),
        "n_sells": len(sells),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "buy_hold_return": bh_ret,
        "excess_return": total_ret - bh_ret,
        "days": len(equity_df),
    }


def print_trade_history(trades: list) -> None:
    """거래 내역 표를 출력합니다."""
    print("=== 거래 내역 ===")
    print(f"{'#':>3}  {'유형':<4}  {'날짜':<12}  {'단가':>16}  {'수량':>12}  {'금액':>14}")
    print("-" * 72)
    for i, t in enumerate(trades, 1):
        dt = t["date"].strftime("%Y-%m-%d") if hasattr(t["date"], "strftime") else str(t["date"])[:10]
        qty_label = f"{t['qty']:.5f}"
        if "BTC" in BACKTEST_TICKER:
            qty_label += " BTC"
        line = (
            f"{i:>3}  {t['type']:<4}  {dt:<12}  "
            f"{t['price']:>14,.0f} 원  {qty_label:>12}  {t['amount']:>12,.0f} 원"
        )
        if t["type"] == "매도" and t.get("pnl_pct") is not None:
            line += f"  ({t['pnl_pct']:+.2f}%)"
        print(line)
    print(f"총 {len(trades)}건 거래")


def print_backtest_summary(metrics: dict) -> None:
    """백테스팅 결과 요약을 출력합니다."""
    print("\n=== 백테스팅 결과 ===")
    print(f"기간             : {metrics['days']}일")
    print(f"초기 자본        : {metrics['initial']:,.0f} 원")
    print(f"최종 자산        : {metrics['final']:,.0f} 원")
    print(f"총 수익률        : {metrics['total_return']:+.2f}%")
    print(f"MDD             : {metrics['mdd']:.2f}%")
    print(f"총 거래          : {metrics['n_trades']}회 (매수 {metrics['n_buys']} / 매도 {metrics['n_sells']})")
    print(f"승률            : {metrics['win_rate']:.1f}%")
    print(f"평균 수익 거래   : {metrics['avg_win']:+.2f}%")
    print(f"평균 손실 거래   : {metrics['avg_loss']:.2f}%")
    print("─" * 28)
    print(f"Buy & Hold      : {metrics['buy_hold_return']:+.2f}%")
    print(f"전략 초과 수익   : {metrics['excess_return']:+.2f}%p")


def make_backtest_chart(result: dict):
    """매매 마커 차트와 전략 vs B&H 자산 곡선을 생성합니다."""
    df = result["df"]
    trades = result["trades"]
    equity = result["equity"]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("BTC 가격 + MA5/MA20 + 매매 신호", "전략 vs Buy & Hold"),
        vertical_spacing=0.1,
        row_heights=[0.55, 0.45],
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["close"], name="종가", line=dict(color="black", width=1.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA5"], name="MA5", line=dict(color="orange", width=1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="blue", width=1)),
        row=1,
        col=1,
    )

    buy_trades = [t for t in trades if t["type"] == "매수"]
    sell_trades = [t for t in trades if t["type"] == "매도"]
    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in buy_trades],
                y=[t["price"] for t in buy_trades],
                mode="markers",
                marker=dict(symbol="triangle-up", color="green", size=12),
                name="매수",
            ),
            row=1,
            col=1,
        )
    if sell_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in sell_trades],
                y=[t["price"] for t in sell_trades],
                mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=12),
                name="매도",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["strategy"],
            name="전략",
            line=dict(color="#16a34a", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["buy_hold"],
            name="Buy & Hold",
            line=dict(color="#6b7280", width=2, dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=800, template="plotly_white", title="KRW-BTC 골든크로스 백테스팅")
    return fig


def run_parameter_optimization(df: pd.DataFrame, param_grid=None) -> pd.DataFrame:
    """여러 MA 조합을 테스트하고 결과 테이블을 반환합니다."""
    if param_grid is None:
        param_grid = [(5, 20), (5, 30), (10, 20), (10, 30), (20, 60), (5, 60), (10, 60)]

    rows = []
    for fast, slow in param_grid:
        test_df = df.copy()
        test_df["MA5"] = test_df["close"].rolling(fast).mean()
        test_df["MA20"] = test_df["close"].rolling(slow).mean()
        test_df = test_df.dropna(subset=["MA5", "MA20"])
        if len(test_df) < 30:
            continue
        result = run_backtest(test_df)
        m = result["metrics"]
        rows.append(
            {
                "MA_fast": fast,
                "MA_slow": slow,
                "총 수익률(%)": m["total_return"],
                "MDD(%)": m["mdd"],
                "승률(%)": m["win_rate"],
                "거래수": m["n_trades"],
                "B&H(%)": m["buy_hold_return"],
            }
        )
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        best_idx = result_df["총 수익률(%)"].idxmax()
        result_df["최고"] = False
        result_df.loc[best_idx, "최고"] = True
    return result_df
