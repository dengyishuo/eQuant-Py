import datetime as dt

import lib.bootstrap  # noqa: F401  (must run before importing e* packages)

import pandas as pd
import streamlit as st

from lib import config_store, data_cache
from lib.factor_catalog import available_factors

st.set_page_config(page_title="股票池与因子", page_icon="📊", layout="wide")
st.title("📊 股票池与因子")

source = st.session_state.get("data_source", config_store.load_config().get("source", "yahoo"))
st.caption(f"当前数据源：**{source}**（在「数据源配置」页面可以修改）")

# ── 1. 股票池 ────────────────────────────────────────────────────────────────
st.subheader("1. 股票池")

default_universe = st.session_state.get(
    "universe",
    pd.DataFrame({"code": ["600000" if source != "yahoo" else "AAPL"], "name": ["示例标的"]}),
)
universe = st.data_editor(default_universe, num_rows="dynamic", key="universe_editor")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "开始日期", value=pd.to_datetime(st.session_state.get("start_date", "2022-01-01"))
    )
with col2:
    end_date = st.date_input("结束日期", value=pd.to_datetime(st.session_state.get("end_date", dt.date.today())))

if st.button("拉取行情", type="primary"):
    universe_clean = universe.dropna(subset=["code"]).reset_index(drop=True)
    if universe_clean.empty:
        st.error("股票池不能为空。")
    else:
        with st.spinner(f"正在从 {source} 拉取行情..."):
            cached = data_cache.get_cached(source, universe_clean, str(start_date), str(end_date))
            if cached is not None:
                mkt_data = cached
                st.info("命中本地缓存，未重新请求数据源。")
            else:
                try:
                    import efactorcraft as efc

                    mkt_data = efc.get_data(universe_clean, str(start_date), str(end_date), source=source)
                    data_cache.set_cached(source, universe_clean, str(start_date), str(end_date), mkt_data)
                except Exception as e:
                    st.error(f"拉取行情失败：{e}")
                    mkt_data = None

        if mkt_data is not None:
            st.session_state["universe"] = universe_clean
            st.session_state["start_date"] = str(start_date)
            st.session_state["end_date"] = str(end_date)
            st.session_state["mkt_data"] = mkt_data
            st.success(f"拉取完成，共 {len(mkt_data)} 行。")

mkt_data = st.session_state.get("mkt_data")
if mkt_data is not None:
    st.dataframe(mkt_data.head(20), use_container_width=True)

st.divider()

# ── 2. 因子选择 ──────────────────────────────────────────────────────────────
st.subheader("2. 因子选择")

if mkt_data is None:
    st.info("请先拉取行情数据。")
else:
    catalog = available_factors(set(mkt_data.columns))
    tabs = st.tabs(["eClassic（经典因子）", "eTTR（技术指标）", "eAlpha101（WorldQuant Alpha）"])

    selected: list = st.session_state.get("selected_factors", [])
    selected_names = {(s["package"], s["name"]) for s in selected}

    for tab, package in zip(tabs, ["eclassic", "ettr", "ealpha101"]):
        with tab:
            entries = catalog[package]
            names = [e.name for e in entries]
            picked = st.multiselect(
                f"勾选 {package} 因子",
                options=names,
                default=[n for (p, n) in selected_names if p == package],
                key=f"pick_{package}",
            )
            for name in picked:
                spec = next(e for e in entries if e.name == name)
                with st.expander(f"{package}.{name} 参数", expanded=False):
                    kwargs = {}
                    for p in spec.params:
                        widget_key = f"{package}_{name}_{p.name}"
                        if p.kind == "bool":
                            kwargs[p.name] = st.checkbox(p.name, value=bool(p.default), key=widget_key)
                        elif p.kind == "int":
                            kwargs[p.name] = st.number_input(
                                p.name, value=int(p.default), step=1, key=widget_key
                            )
                        elif p.kind == "float":
                            kwargs[p.name] = st.number_input(p.name, value=float(p.default), key=widget_key)
                        elif p.kind == "int_list":
                            raw = st.text_input(
                                f"{p.name}（逗号分隔）", value=",".join(str(x) for x in p.default), key=widget_key
                            )
                            kwargs[p.name] = [int(x.strip()) for x in raw.split(",") if x.strip()]
                        else:  # str, possibly optional (default None)
                            raw = st.text_input(
                                p.name + ("（必填）" if p.required else "（留空则自动检测）"),
                                value="" if p.default is None else str(p.default),
                                key=widget_key,
                            )
                            kwargs[p.name] = raw if raw else None

                # Update the running selection list
                already = next((s for s in selected if s["package"] == package and s["name"] == name), None)
                if already is None:
                    selected.append({"package": package, "name": name, "params": kwargs})
                else:
                    already["params"] = kwargs

            # Drop unpicked factors for this package from the selection
            selected = [s for s in selected if s["package"] != package or s["name"] in picked]

    st.session_state["selected_factors"] = selected

    if selected:
        st.write("已选因子：", ", ".join(f"{s['package']}.{s['name']}" for s in selected))

    if st.button("计算因子", type="primary", disabled=not selected):
        catalog_flat = {(e.package, e.name): e.fn for pkg in catalog.values() for e in pkg}
        df = mkt_data.copy()
        errors = []
        for s in selected:
            fn = catalog_flat.get((s["package"], s["name"]))
            if fn is None:
                continue
            try:
                df = fn(df, **s["params"], append=True)
            except Exception as e:
                errors.append(f"{s['package']}.{s['name']}: {e}")

        st.session_state["mkt_data"] = df
        if errors:
            st.warning("部分因子计算失败：\n" + "\n".join(errors))
        st.success("因子计算完成，新增列：" + ", ".join(sorted(set(df.columns) - set(mkt_data.columns))))
        st.dataframe(df.head(20), use_container_width=True)
