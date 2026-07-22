import lib.bootstrap  # noqa: F401  (must run before importing e* packages)

import pandas as pd
import streamlit as st

from lib import config_store

st.set_page_config(page_title="信号与回测", page_icon="🧮", layout="wide")
st.title("🧮 信号与回测")

mkt_data = st.session_state.get("mkt_data")
if mkt_data is None:
    st.info("请先在「股票池与因子」页面拉取行情并计算因子。")
    st.stop()

import ebacktestcraft as ebc

BASE_COLS = {"date", "code", "name", "open", "high", "low", "close", "adjusted", "volume"}
factor_cols = sorted(set(mkt_data.columns) - BASE_COLS)

if not factor_cols:
    st.warning("当前数据还没有任何因子列，请先在「股票池与因子」页面计算至少一个因子。")
    st.stop()

# ── 1. 信号 ──────────────────────────────────────────────────────────────────
st.subheader("1. 交易信号")

factor_col = st.selectbox("打分/排序因子列", options=factor_cols)
signal_type = st.selectbox("信号类型", options=["rank", "threshold", "quantile"])

signal_kwargs = {"indicator_cols": [factor_col], "signal_type": signal_type}
if signal_type == "rank":
    c1, c2 = st.columns(2)
    signal_kwargs["top_n"] = c1.number_input("选取前 N 只", value=3, min_value=1, step=1)
    signal_kwargs["ascending"] = c2.checkbox("升序（选最小值）", value=False)
elif signal_type == "threshold":
    c1, c2 = st.columns(2)
    signal_kwargs["compare_op"] = c1.selectbox("比较符", options=[">", "<", ">=", "<=", "==", "!="])
    signal_kwargs["threshold"] = c2.number_input("阈值", value=0.0)
elif signal_type == "quantile":
    c1, c2 = st.columns(2)
    signal_kwargs["q"] = c1.number_input("分位数 q", value=0.2, min_value=0.01, max_value=0.99)
    signal_kwargs["select"] = c2.selectbox("选取方向", options=["top", "bottom"])

# ── 2. 权重 ──────────────────────────────────────────────────────────────────
st.subheader("2. 权重方案")
weight_scheme = st.selectbox("权重方案", options=["equal", "norm"])
norm_method = "linear"
if weight_scheme == "norm":
    norm_method = st.selectbox("归一化方式", options=["linear", "softmax"])

# ── 3. 回测参数 ──────────────────────────────────────────────────────────────
st.subheader("3. 回测参数")

c1, c2, c3 = st.columns(3)
init_capital = c1.number_input("初始资金", value=100_000.0, step=10_000.0)
rebalance_mode = c2.selectbox("调仓模式", options=["calendar", "weight_shift", "hybrid"])
rebalance_cycle = c3.text_input("调仓周期（monthly/quarterly/整数交易日）", value="monthly")

c1, c2, c3 = st.columns(3)
exec_price_col = c1.selectbox("成交价列", options=["open", "close", "adjusted"], index=0)
eval_price_col = c2.selectbox("估值价列", options=["adjusted", "close", "open"], index=0)
lot_size = c3.number_input("最小交易单位（股）", value=100, step=100)

c1, c2, c3 = st.columns(3)
fee_rate = c1.number_input("手续费率", value=0.0003, format="%.5f")
stamp_tax = c2.number_input("印花税率", value=0.001, format="%.5f")
slippage_rate = c3.number_input("滑点率", value=0.001, format="%.5f")

c1, c2 = st.columns(2)
single_max_weight = c1.number_input("单标的最大权重", value=0.95, min_value=0.01, max_value=1.0)
global_max_hold_pct = c2.number_input("总仓位上限", value=1.0, min_value=0.01, max_value=1.0)

st.markdown("**止损/止盈（可选）**")
c1, c2 = st.columns(2)
with c1:
    enable_component_sl = st.checkbox("启用个股止损（固定比例）")
    component_sl_ratio = st.number_input("个股止损比例", value=0.1, min_value=0.01, max_value=0.9, disabled=not enable_component_sl)
with c2:
    enable_portfolio_sl = st.checkbox("启用组合止损（固定比例）")
    portfolio_sl_ratio = st.number_input("组合止损比例", value=0.1, min_value=0.01, max_value=0.9, disabled=not enable_portfolio_sl)

st.divider()

if st.button("运行回测", type="primary"):
    df = mkt_data.copy()

    before_cols = set(df.columns)
    df = ebc.signal(df, **signal_kwargs)
    signal_col = next(iter(set(df.columns) - before_cols))

    before_cols = set(df.columns)
    if weight_scheme == "equal":
        df = ebc.equal_weight(df, signal_col=signal_col)
    else:
        df = ebc.norm_weight(df, weight_col=factor_col, signal_col=signal_col, norm_method=norm_method)
    weight_col = next(iter(set(df.columns) - before_cols))

    cfg = ebc.Config(
        weight_col=weight_col,
        init_capital=init_capital,
        rebalance_mode=rebalance_mode,
        rebalance_cycle=rebalance_cycle,
        exec_price_col=exec_price_col,
        eval_price_col=eval_price_col,
        fee_rate=fee_rate,
        stamp_tax=stamp_tax,
        slippage_rate=slippage_rate,
        lot_size=int(lot_size),
        single_max_weight=single_max_weight,
        global_max_hold_pct=global_max_hold_pct,
        enable_component_stop_loss=enable_component_sl,
        fixed_component_sl_ratio=component_sl_ratio,
        enable_portfolio_stop_loss=enable_portfolio_sl,
        fixed_portfolio_sl_ratio=portfolio_sl_ratio,
    )

    with st.spinner("正在运行回测..."):
        try:
            result = ebc.run(df, config=cfg, weight_col=weight_col)
        except Exception as e:
            st.error(f"回测运行失败：{e}")
            result = None

    if result is not None:
        st.session_state["backtest_result"] = result
        st.session_state["signal_config"] = {"factor_col": factor_col, "signal_type": signal_type, **{
            k: v for k, v in signal_kwargs.items() if k != "indicator_cols"
        }}
        st.session_state["weight_config"] = {"scheme": weight_scheme, "norm_method": norm_method}
        st.session_state["backtest_config"] = cfg.to_dict()
        st.success("回测完成。")

# ── 4. 结果展示 ──────────────────────────────────────────────────────────────
result = st.session_state.get("backtest_result")
if result is not None:
    st.subheader("4. 回测结果")

    s = result.summary
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("年化收益率", f"{s['annual_return_pct']:.2f}%")
    m2.metric("最大回撤", f"{s['max_drawdown_pct']:.2f}%")
    m3.metric("夏普比率", f"{s['sharpe_ratio']:.2f}")
    m4.metric("总交易次数", f"{s['n_trades']}")
    m5.metric("期末净值", f"{s['final_nav']:,.0f}")

    if ebc._has_plot:
        tabs = st.tabs(["净值曲线", "回撤", "月度收益", "收益分布"])
        with tabs[0]:
            st.pyplot(ebc.plot_equity_curve(result.equity_curve))
        with tabs[1]:
            st.pyplot(ebc.plot_drawdown(result.equity_curve))
        with tabs[2]:
            st.pyplot(ebc.plot_monthly_return(result.equity_curve))
        with tabs[3]:
            st.pyplot(ebc.plot_return_dist(result.equity_curve))
    else:
        st.caption("matplotlib/seaborn 未安装，跳过图表渲染。")

    st.markdown("**交易记录**")
    st.dataframe(result.transactions, use_container_width=True)
    st.download_button(
        "下载净值曲线 CSV",
        result.equity_curve.to_csv(index=False).encode("utf-8"),
        file_name="equity_curve.csv",
    )
    st.download_button(
        "下载交易记录 CSV",
        result.transactions.to_csv(index=False).encode("utf-8"),
        file_name="transactions.csv",
    )

    st.divider()
    st.subheader("5. 保存策略")
    strategy_name = st.text_input("策略名称", value=st.session_state.get("loaded_strategy_name", ""))
    if st.button("保存当前策略配置"):
        if not strategy_name:
            st.error("请填写策略名称。")
        else:
            payload = {
                "data_source": st.session_state.get("data_source"),
                "universe": st.session_state.get("universe").to_dict(orient="records"),
                "start_date": st.session_state.get("start_date"),
                "end_date": st.session_state.get("end_date"),
                "selected_factors": st.session_state.get("selected_factors"),
                "signal_config": st.session_state.get("signal_config"),
                "weight_config": st.session_state.get("weight_config"),
                "backtest_config": st.session_state.get("backtest_config"),
            }
            config_store.save_strategy(strategy_name, payload)
            st.success(f"策略「{strategy_name}」已保存到 ~/.equant/strategies/{strategy_name}.json")
