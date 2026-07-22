import lib.bootstrap  # noqa: F401  (must run before importing e* packages)

import pandas as pd
import streamlit as st

from lib import config_store

st.set_page_config(page_title="eQuant 量化回测工作台", page_icon="📈", layout="wide")

st.title("📈 eQuant 量化回测工作台")

st.markdown(
    """
本地量化研究工作台，基于你已有的 `eFactorCraft` / `eClassic` / `eTTR` / `eAlpha101` /
`eBacktestCraft` 工具链搭建。

**使用流程**（左侧导航栏）：

1. **数据源配置** — 选择 Tushare / AKShare / baostock / Yahoo，Tushare 需要填 token
2. **股票池与因子** — 录入股票代码，拉取行情，勾选因子并计算
3. **信号与回测** — 生成交易信号、配置权重与回测参数，运行回测并查看绩效图表

第一阶段范围：数据源配置 + 因子计算 + 回测。每日盯盘监控是下一阶段。
"""
)

st.divider()
st.subheader("已保存的策略")

strategies = config_store.list_strategies()
if not strategies:
    st.caption("还没有保存过策略。在「信号与回测」页面跑完回测后可以保存当前配置。")
else:
    for name in strategies:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{name}**")
        with col2:
            if st.button("加载", key=f"load_{name}"):
                data = config_store.load_strategy(name)
                if data.get("universe") is not None:
                    data["universe"] = pd.DataFrame(data["universe"])
                for k, v in data.items():
                    st.session_state[k] = v
                st.session_state["loaded_strategy_name"] = name
                st.success(f"已加载策略「{name}」，请前往「股票池与因子」页面继续（需要重新拉取行情和计算因子）。")
