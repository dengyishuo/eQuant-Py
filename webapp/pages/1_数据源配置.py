import datetime as dt

import lib.bootstrap  # noqa: F401  (must run before importing e* packages)

import streamlit as st

from lib import config_store

st.set_page_config(page_title="数据源配置", page_icon="📡", layout="wide")
st.title("📡 数据源配置")

_cfg = config_store.load_config()

SOURCE_LABELS = {
    "yahoo": "Yahoo Finance（美股/全球，免 token）",
    "tushare": "Tushare（A 股，需要 token）",
    "akshare": "AKShare（A 股，免 token）",
    "baostock": "baostock（A 股，免 token）",
}

source = st.radio(
    "选择数据源",
    options=list(SOURCE_LABELS.keys()),
    format_func=lambda k: SOURCE_LABELS[k],
    index=list(SOURCE_LABELS.keys()).index(_cfg.get("source", "yahoo")),
)

token = _cfg.get("tushare_token", "")
if source == "tushare":
    token = st.text_input("Tushare token", value=token, type="password")

st.divider()

if st.button("测试连接", type="primary"):
    recent_end = dt.date.today()
    recent_start = recent_end - dt.timedelta(days=10)

    with st.spinner("正在测试数据源连接..."):
        try:
            if source == "yahoo":
                import yfinance as yf

                hist = yf.Ticker("AAPL").history(start=str(recent_start), end=str(recent_end))
                ok = not hist.empty
            elif source == "tushare":
                from efactorcraft.providers import set_token, tushare

                if not token:
                    raise ValueError("请先填写 Tushare token")
                set_token(token)
                df = tushare.get_daily("600000", str(recent_start), str(recent_end))
                ok = not df.empty
            elif source == "akshare":
                from efactorcraft.providers import akshare

                df = akshare.get_daily("600000", str(recent_start), str(recent_end))
                ok = not df.empty
            elif source == "baostock":
                from efactorcraft.providers import baostock

                df = baostock.get_daily("600000", str(recent_start), str(recent_end))
                ok = not df.empty
        except Exception as e:
            st.error(f"连接失败：{e}")
        else:
            if ok:
                st.success("连接成功，已获取到示例数据。")
            else:
                st.warning("连接没有报错，但没有返回数据（可能是日期区间内没有交易日）。")

if st.button("保存配置"):
    config_store.save_config({"source": source, "tushare_token": token})
    st.session_state["data_source"] = source
    if source == "tushare" and token:
        from efactorcraft.providers import set_token

        set_token(token)
    st.success("已保存到 ~/.equant/config.json，并在当前会话生效。")
