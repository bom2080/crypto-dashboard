"""Day 4 — HTML 대시보드 조립."""

import base64
import io
from datetime import datetime

import matplotlib.pyplot as plt

from backtesting import load_backtest_data, make_backtest_chart, run_backtest
from data_pipeline import (
    format_pct,
    format_price,
    make_bollinger_chart,
    prepare_market_data,
)
from portfolio import (
    PORTFOLIO,
    build_portfolio_series,
    get_portfolio_metrics,
    make_correlation_heatmap,
    make_portfolio_chart,
)

STYLES = """
body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 24px; margin: 0; }
h1 { margin-bottom: 4px; }
.subtitle { color: #888; margin-top: 0; }
.section { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.section h2 { margin-top: 0; font-size: 1.25rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.card { background: #f8f9fa; border-radius: 8px; padding: 16px 24px; text-align: center; min-width: 160px; flex: 1; }
.card-label { font-size: 12px; color: #888; margin-bottom: 6px; }
.card-value { font-size: 22px; font-weight: 700; }
.card-sub { font-size: 14px; margin-top: 4px; }
.positive { color: #dc2626; }
.negative { color: #2563eb; }
.metrics-table { width: 100%; border-collapse: collapse; margin: 12px 0; }
.metrics-table th, .metrics-table td { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }
.metrics-table th { background: #f3f4f6; }
"""


def _fig_to_html(fig, include_plotlyjs):
    """plotly Figure를 HTML 조각으로 변환합니다."""
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _matplotlib_to_img(fig):
    """matplotlib Figure를 base64 img 태그로 변환합니다."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        f'style="max-width:480px;margin:16px auto;display:block;" alt="상관관계 히트맵"/>'
    )


def _price_cards_html(data):
    """시세 현황 카드 4개 HTML."""
    parts = []
    for ticker in data:
        last = data[ticker].iloc[-1]
        ret = last["return_7d"]
        cls = "positive" if ret >= 0 else "negative"
        parts.append(
            f'<motionless class="card">'
            f'<motionless class="card-label">{ticker}</motionless>'
            f'<motionless class="card-value">{format_price(last["close"])}</motionless>'
            f'<motionless class="card-sub {cls}">7일 {format_pct(ret)}</motionless>'
            f"</motionless>"
        )
    return "".join(parts).replace("motionless", "div")


def _portfolio_cards_html(metrics):
    """포트폴리오 핵심 지표 카드 HTML."""
    cls = "positive" if metrics["return_pct"] >= 0 else "negative"
    html = f"""
    <motionless class="cards">
        <motionless class="card">
            <motionless class="card-label">총 자산</motionless>
            <motionless class="card-value">{metrics['final']:,.0f}원</motionless>
        </motionless>
        <motionless class="card">
            <motionless class="card-label">총 수익률</motionless>
            <motionless class="card-value {cls}">{metrics['return_pct']:+.2f}%</motionless>
        </motionless>
        <motionless class="card">
            <motionless class="card-label">MDD</motionless>
            <motionless class="card-value negative">{metrics['mdd']:.2f}%</motionless>
        </motionless>
        <motionless class="card">
            <motionless class="card-label">투자 기간</motionless>
            <motionless class="card-value" style="font-size:14px">{metrics['start']} ~ {metrics['end']}</motionless>
        </motionless>
    </motionless>
    """
    return html.replace("motionless", "div")


def _backtest_table_html(metrics):
    """백테스팅 성과 요약 테이블 HTML."""
    rows = [
        ("기간", f"{metrics['days']}일"),
        ("초기 자본", f"{metrics['initial']:,.0f} 원"),
        ("최종 자산", f"{metrics['final']:,.0f} 원"),
        ("총 수익률", f"{metrics['total_return']:+.2f}%"),
        ("MDD", f"{metrics['mdd']:.2f}%"),
        ("총 거래", f"{metrics['n_trades']}회 (매수 {metrics['n_buys']} / 매도 {metrics['n_sells']})"),
        ("승률", f"{metrics['win_rate']:.1f}%"),
        ("Buy & Hold", f"{metrics['buy_hold_return']:+.2f}%"),
        ("전략 초과 수익", f"{metrics['excess_return']:+.2f}%p"),
    ]
    trs = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="metrics-table"><tbody>{trs}</tbody></table>'


def build_dashboard(tickers=None, portfolio_config=None, backtest_config=None):
    """
    전체 대시보드 HTML을 조립합니다.
    Day 1~3 함수를 재사용하고 HTML로 통합합니다.
    """
    tickers = tickers or list(PORTFOLIO.keys())
    portfolio_config = portfolio_config or PORTFOLIO
    backtest_config = backtest_config or {}

    data = prepare_market_data(tickers)
    value_df, details = build_portfolio_series(data, portfolio_config)
    pf_metrics = get_portfolio_metrics(value_df, details)

    bt_df = load_backtest_data(
        ticker=backtest_config.get("ticker", "KRW-BTC"),
        days=backtest_config.get("days", 365),
    )
    bt_result = run_backtest(bt_df)
    bt_metrics = bt_result["metrics"]

    chart_html = [
        _fig_to_html(make_bollinger_chart(data), "cdn"),
        _fig_to_html(make_portfolio_chart(value_df, data, details), False),
        _matplotlib_to_img(make_correlation_heatmap(value_df, data, details)[0]),
        _fig_to_html(make_backtest_chart(bt_result), False),
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    section = "motionless"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>암호화폐 포트폴리오 대시보드</title>
    <style>{STYLES}</style>
</head>
<body>
    <h1>🪙 암호화폐 포트폴리오 대시보드</h1>
    <p class="subtitle">생성: {now}</p>

    <{section} class="section">
        <h2>[섹션 1] 시세 현황</h2>
        <{section} class="cards">{_price_cards_html(data)}</{section}>
        {chart_html[0]}
    </{section}>

    <{section} class="section">
        <h2>[섹션 2] 포트폴리오</h2>
        {_portfolio_cards_html(pf_metrics)}
        {chart_html[1]}
        {chart_html[2]}
    </{section}>

    <{section} class="section">
        <h2>[섹션 3] 백테스팅 (KRW-BTC 골든크로스)</h2>
        {_backtest_table_html(bt_metrics)}
        {chart_html[3]}
    </{section}>
</body>
</html>"""
    return html.replace(section, "div")


def save_dashboard(path="dashboard.html", **kwargs):
    """대시보드 HTML을 파일로 저장합니다."""
    html = build_dashboard(**kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
