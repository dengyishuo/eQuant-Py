"""
ealpha101.catalog
=================
Machine-readable metadata for all 101 alpha functions.

Usage (for AI or programmatic access)
--------------------------------------
    from ealpha101.catalog import ALPHAS, summary, required_cols

    # print a compact reference sheet
    print(summary())

    # look up one alpha
    print(ALPHAS["add_alpha001"])

    # find all alphas that need the 'cap' column
    print(required_cols("cap"))
"""

from __future__ import annotations

# ── Compact metadata for every alpha ─────────────────────────────────────────
# Each entry:
#   formula   : canonical formula (from Kakushadze 2016)
#   params    : {param_name: default_value}  — mkt_data and append omitted
#   required  : columns that MUST exist in mkt_data (with default param names)
#   notes     : special handling (e.g. pre-neutralised columns)

ALPHAS: dict[str, dict] = {
  "add_alpha001": {
    "formula": "rank(ts_argmax(signedpower(if(ret<0, stddev(ret,20), close), 2), 5)) - 0.5",
    "params":  {"close_col": "close", "returns_col": "returns"},
    "required": ["date", "code", "name", "close", "returns"],
    "notes": "",
  },
  "add_alpha002": {
    "formula": "-1 * corr(rank(delta(log(vol),2)), rank((close-open)/open), 6)",
    "params":  {"open_col": "open", "close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "close", "volume"],
    "notes": "",
  },
  "add_alpha003": {
    "formula": "-1 * corr(rank(open), rank(volume), 10)",
    "params":  {"open_col": "open", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "volume"],
    "notes": "",
  },
  "add_alpha004": {
    "formula": "-1 * ts_rank(rank(low), 9)",
    "params":  {"low_col": "low"},
    "required": ["date", "code", "name", "low"],
    "notes": "",
  },
  "add_alpha005": {
    "formula": "rank(open - ts_mean(vwap,10)) * -1 * abs(rank(close - vwap))",
    "params":  {"open_col": "open", "close_col": "close", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "open", "close", "vwap"],
    "notes": "",
  },
  "add_alpha006": {
    "formula": "-1 * corr(open, volume, 10)",
    "params":  {"open_col": "open", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "volume"],
    "notes": "",
  },
  "add_alpha007": {
    "formula": "if adv20 < volume: -1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7)) else -1",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha008": {
    "formula": "-1 * rank(sum(open,5)*sum(ret,5) - delay(sum(open,5)*sum(ret,5),10))",
    "params":  {"open_col": "open", "returns_col": "returns"},
    "required": ["date", "code", "name", "open", "returns"],
    "notes": "",
  },
  "add_alpha009": {
    "formula": "if ts_min(delta(close,1),5)>0: delta(close,1); elif ts_max<0: delta; else -1*delta",
    "params":  {"close_col": "close"},
    "required": ["date", "code", "name", "close"],
    "notes": "",
  },
  "add_alpha010": {
    "formula": "rank(same as 009 but window=4)",
    "params":  {"close_col": "close"},
    "required": ["date", "code", "name", "close"],
    "notes": "",
  },
  "add_alpha011": {
    "formula": "(rank(ts_max(vwap-close,3))+rank(ts_min(vwap-close,3)))*rank(delta(vol,3))",
    "params":  {"vwap_col": "vwap", "close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "close", "volume"],
    "notes": "",
  },
  "add_alpha012": {
    "formula": "sign(delta(vol,1)) * -1 * delta(close,1)",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha013": {
    "formula": "-1 * rank(cov(rank(close), rank(vol), 5))",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha014": {
    "formula": "-1 * rank(delta(ret,3)) * corr(open, vol, 10)",
    "params":  {"open_col": "open", "returns_col": "returns", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "returns", "volume"],
    "notes": "",
  },
  "add_alpha015": {
    "formula": "-1 * sum(rank(corr(rank(high), rank(vol), 3)), 3)",
    "params":  {"high_col": "high", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "volume"],
    "notes": "",
  },
  "add_alpha016": {
    "formula": "-1 * rank(cov(rank(high), rank(vol), 5))",
    "params":  {"high_col": "high", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "volume"],
    "notes": "",
  },
  "add_alpha017": {
    "formula": "-1*rank(ts_rank(close,10))*rank(delta(delta(close,1),1))*rank(ts_rank(vol/adv20,5))",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha018": {
    "formula": "-1*rank(stddev(abs(close-open),5)+(close-open)+corr(close,open,10))",
    "params":  {"close_col": "close", "open_col": "open"},
    "required": ["date", "code", "name", "close", "open"],
    "notes": "",
  },
  "add_alpha019": {
    "formula": "-1*sign(close-delay(close,7)+delta(close,7))*(1+rank(1+sum(ret,250)))",
    "params":  {"close_col": "close", "returns_col": "returns"},
    "required": ["date", "code", "name", "close", "returns"],
    "notes": "",
  },
  "add_alpha020": {
    "formula": "-1*rank(open-delay(high,1))*rank(open-delay(close,1))*rank(open-delay(low,1))",
    "params":  {"open_col": "open", "high_col": "high", "low_col": "low", "close_col": "close"},
    "required": ["date", "code", "name", "open", "high", "low", "close"],
    "notes": "",
  },
  "add_alpha021": {
    "formula": "if mean(c,8)+std(c,8)<mean(c,2): 1; elif mean(c,2)<mean(c,8)-std(c,8): -1; else vol/adv20>=1?1:-1",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha022": {
    "formula": "-1*delta(corr(high,vol,5),5)*rank(stddev(close,20))",
    "params":  {"high_col": "high", "close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "close", "volume"],
    "notes": "",
  },
  "add_alpha023": {
    "formula": "if mean(high,20)<high: -1*delta(high,2) else 0",
    "params":  {"high_col": "high"},
    "required": ["date", "code", "name", "high"],
    "notes": "",
  },
  "add_alpha024": {
    "formula": "if delta(mean(close,100),100)/delay(close,100)<=0.05: -(close-ts_min(close,100)) else -delta(close,3)",
    "params":  {"close_col": "close"},
    "required": ["date", "code", "name", "close"],
    "notes": "",
  },
  "add_alpha025": {
    "formula": "rank(-ret*adv20*vwap*(high-close))",
    "params":  {"returns_col": "returns", "volume_col": "volume", "vwap_col": "vwap", "high_col": "high", "close_col": "close"},
    "required": ["date", "code", "name", "returns", "volume", "vwap", "high", "close"],
    "notes": "",
  },
  "add_alpha026": {
    "formula": "-1*ts_max(corr(ts_rank(vol,5),ts_rank(high,5),5),3)",
    "params":  {"volume_col": "volume", "high_col": "high"},
    "required": ["date", "code", "name", "volume", "high"],
    "notes": "",
  },
  "add_alpha027": {
    "formula": "if rank(mean(corr(rank(vol),rank(vwap),6),2)/0.5)>0.5: -1 else 1",
    "params":  {"volume_col": "volume", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "volume", "vwap"],
    "notes": "",
  },
  "add_alpha028": {
    "formula": "scale(corr(adv20,low,5)+(high+low)/2-close)",
    "params":  {"volume_col": "volume", "low_col": "low", "high_col": "high", "close_col": "close"},
    "required": ["date", "code", "name", "volume", "low", "high", "close"],
    "notes": "",
  },
  "add_alpha029": {
    "formula": "ts_min(ts_product(rank(rank(scale(log(sum(ts_min(rank(...)),2),1))))),1),5)+ts_rank(delay(-ret,6),5)",
    "params":  {"close_col": "close", "returns_col": "returns"},
    "required": ["date", "code", "name", "close", "returns"],
    "notes": "",
  },
  "add_alpha030": {
    "formula": "(1-rank(sign(delta(close,1))+sign(delta(delay(close,1),1))+sign(delta(delay(close,2),1))))*sum(vol,5)/sum(vol,20)",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha031": {
    "formula": "rank(decay_linear(-rank(rank(delta(close,10))),10))+rank(-delta(close,3))+sign(scale(corr(adv20,low,12)))",
    "params":  {"close_col": "close", "volume_col": "volume", "low_col": "low"},
    "required": ["date", "code", "name", "close", "volume", "low"],
    "notes": "",
  },
  "add_alpha032": {
    "formula": "scale(corr(vwap, delay(close,1), 253) / delay(close,1))",
    "params":  {"vwap_col": "vwap", "close_col": "close"},
    "required": ["date", "code", "name", "vwap", "close"],
    "notes": "",
  },
  "add_alpha033": {
    "formula": "rank(-1*(1-open/close))",
    "params":  {"open_col": "open", "close_col": "close"},
    "required": ["date", "code", "name", "open", "close"],
    "notes": "",
  },
  "add_alpha034": {
    "formula": "rank(rank(stddev(ret,2))/stddev(ret,5))+rank(-delta(close,1))",
    "params":  {"returns_col": "returns", "close_col": "close"},
    "required": ["date", "code", "name", "returns", "close"],
    "notes": "",
  },
  "add_alpha035": {
    "formula": "ts_rank(vol,32)*(1-ts_rank(close+high-low,16))*(1-ts_rank(ret,32))",
    "params":  {"volume_col": "volume", "close_col": "close", "high_col": "high", "low_col": "low", "returns_col": "returns"},
    "required": ["date", "code", "name", "volume", "close", "high", "low", "returns"],
    "notes": "",
  },
  "add_alpha036": {
    "formula": "2.21*rank(corr(close-open,delay(vol,1),15))+0.7*rank(open-close)+0.73*rank(ts_rank(delay(-ret,6),5))+rank(abs(corr(vwap,adv20,6)))+0.6*rank((mean(close,200)-open)/close)",
    "params":  {"close_col": "close", "open_col": "open", "volume_col": "volume", "vwap_col": "vwap", "returns_col": "returns"},
    "required": ["date", "code", "name", "close", "open", "volume", "vwap", "returns"],
    "notes": "",
  },
  "add_alpha037": {
    "formula": "rank(corr(delay(open-close,1),close,200))+rank(open-close)",
    "params":  {"open_col": "open", "close_col": "close"},
    "required": ["date", "code", "name", "open", "close"],
    "notes": "",
  },
  "add_alpha038": {
    "formula": "-1*rank(ts_rank(close,10))*rank(close/open)",
    "params":  {"close_col": "close", "open_col": "open"},
    "required": ["date", "code", "name", "close", "open"],
    "notes": "",
  },
  "add_alpha039": {
    "formula": "-1*rank(delta(close,7)*(1-rank(decay_linear(vol/adv20,9))))*(1+rank(sum(ret,250)))",
    "params":  {"close_col": "close", "volume_col": "volume", "returns_col": "returns"},
    "required": ["date", "code", "name", "close", "volume", "returns"],
    "notes": "",
  },
  "add_alpha040": {
    "formula": "-1*rank(stddev(high,10))*corr(high,vol,10)",
    "params":  {"high_col": "high", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "volume"],
    "notes": "",
  },
  "add_alpha041": {
    "formula": "(high*low)^0.5 - vwap",
    "params":  {"high_col": "high", "low_col": "low", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "high", "low", "vwap"],
    "notes": "",
  },
  "add_alpha042": {
    "formula": "rank(vwap-close)/rank(vwap+close)",
    "params":  {"vwap_col": "vwap", "close_col": "close"},
    "required": ["date", "code", "name", "vwap", "close"],
    "notes": "",
  },
  "add_alpha043": {
    "formula": "ts_rank(vol/adv20,20)*ts_rank(-delta(close,7),8)",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha044": {
    "formula": "-1*corr(high,rank(vol),5)",
    "params":  {"high_col": "high", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "volume"],
    "notes": "",
  },
  "add_alpha045": {
    "formula": "-1*rank(mean(delay(close,5),20))*corr(close,vol,2)*rank(corr(sum(close,5),sum(close,20),2))",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha046": {
    "formula": "if (delay(c,20)-delay(c,10))/10-(delay(c,10)-c)/10>0.25: -1; elif <0: 1; else -delta(close,1)",
    "params":  {"close_col": "close"},
    "required": ["date", "code", "name", "close"],
    "notes": "",
  },
  "add_alpha047": {
    "formula": "rank(1/close)*vol/adv20*high*rank(high-close)/rank(vwap-delay(vwap,5)+vwap-close)",
    "params":  {"close_col": "close", "high_col": "high", "volume_col": "volume", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "close", "high", "volume", "vwap"],
    "notes": "",
  },
  "add_alpha048": {
    "formula": "IndNeutralize(delta(close,1)/delay(close,1)) - IndNeutralize(delta(vwap,1)/delay(vwap,1))",
    "params":  {"neut_close_ret_col": "neut_close_ret", "neut_vwap_ret_col": "neut_vwap_ret"},
    "required": ["date", "code", "name", "neut_close_ret", "neut_vwap_ret"],
    "notes": "NEUTRALIZED: pass pre-computed industry-neutralised columns",
  },
  "add_alpha049": {
    "formula": "if (delay(c,20)-delay(c,10))/10-(delay(c,10)-c)/10 >= -0.1: 1 else -delta(close,1)",
    "params":  {"close_col": "close"},
    "required": ["date", "code", "name", "close"],
    "notes": "",
  },
  "add_alpha050": {
    "formula": "-1*ts_max(rank(corr(rank(vol),rank(vwap),5)),5)",
    "params":  {"volume_col": "volume", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "volume", "vwap"],
    "notes": "",
  },
  "add_alpha051": {
    "formula": "if (delay(c,20)-delay(c,10))/10-(delay(c,10)-c)/10 >= -0.05: 1 else -delta(close,1)",
    "params":  {"close_col": "close"},
    "required": ["date", "code", "name", "close"],
    "notes": "",
  },
  "add_alpha052": {
    "formula": "(-ts_min(low,5)+delay(ts_min(low,5),5))*rank((sum(ret,240)-sum(ret,20))/220)*ts_rank(vol,5)",
    "params":  {"low_col": "low", "returns_col": "returns", "volume_col": "volume"},
    "required": ["date", "code", "name", "low", "returns", "volume"],
    "notes": "",
  },
  "add_alpha053": {
    "formula": "-1*delta((high-close)/(close-low+1e-10), 9)",
    "params":  {"high_col": "high", "low_col": "low", "close_col": "close"},
    "required": ["date", "code", "name", "high", "low", "close"],
    "notes": "",
  },
  "add_alpha054": {
    "formula": "-1*(low-close)*(open^5)/((close-high)*(close^5))",
    "params":  {"low_col": "low", "close_col": "close", "open_col": "open", "high_col": "high"},
    "required": ["date", "code", "name", "low", "close", "open", "high"],
    "notes": "",
  },
  "add_alpha055": {
    "formula": "-1*corr(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(vol), 6)",
    "params":  {"close_col": "close", "low_col": "low", "high_col": "high", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "low", "high", "volume"],
    "notes": "",
  },
  "add_alpha056": {
    "formula": "-1*(rank(sum(ret,10)) < rank(ret*cap))",
    "params":  {"returns_col": "returns", "cap_col": "cap"},
    "required": ["date", "code", "name", "returns", "cap"],
    "notes": "CAP: requires market capitalisation column",
  },
  "add_alpha057": {
    "formula": "(close-vwap)/decay_linear(rank(ts_argmax(close,30)),2)",
    "params":  {"close_col": "close", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "close", "vwap"],
    "notes": "",
  },
  "add_alpha058": {
    "formula": "-1*ts_rank(decay_linear(corr(IndNeutralize(vwap),vol,4),8),6)",
    "params":  {"neut_vwap_col": "neut_vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "neut_vwap", "volume"],
    "notes": "NEUTRALIZED: neut_vwap = IndNeutralize(vwap, industry)",
  },
  "add_alpha059": {
    "formula": "-1*ts_rank(decay_linear(corr(IndNeutralize(vwap*0.728317+vwap*0.271683),vol,4),16),8)",
    "params":  {"neut_vwap2_col": "neut_vwap2", "volume_col": "volume"},
    "required": ["date", "code", "name", "neut_vwap2", "volume"],
    "notes": "NEUTRALIZED: neut_vwap2 = IndNeutralize(weighted vwap, industry)",
  },
  "add_alpha060": {
    "formula": "-(scale(rank(((close-low-(high-close))/(high-low))*vol))-scale(rank(ts_argmax(close,10))))",
    "params":  {"close_col": "close", "low_col": "low", "high_col": "high", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "low", "high", "volume"],
    "notes": "",
  },
  "add_alpha061": {
    "formula": "rank(vwap-ts_min(vwap,16)) < rank(corr(vwap,adv180,17))  → 1/0",
    "params":  {"vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha062": {
    "formula": "(rank(corr(vwap,sum(adv20,22),9)) < rank((rank(open)+rank(open))<(rank((h+l)/2)+rank(h)))) * -1",
    "params":  {"vwap_col": "vwap", "volume_col": "volume", "open_col": "open", "high_col": "high", "low_col": "low"},
    "required": ["date", "code", "name", "vwap", "volume", "open", "high", "low"],
    "notes": "",
  },
  "add_alpha063": {
    "formula": "ts_max(rank(decay_linear(delta(IndNeutralize(close),2),8)),6) - rank(decay_linear(corr(vwap*0.318+open*0.682, sum(adv180,37),13),12))",
    "params":  {"neut_close_col": "neut_close", "vwap_col": "vwap", "open_col": "open", "volume_col": "volume"},
    "required": ["date", "code", "name", "neut_close", "vwap", "open", "volume"],
    "notes": "NEUTRALIZED: neut_close = IndNeutralize(close, industry)",
  },
  "add_alpha064": {
    "formula": "rank(corr(sum(open*0.178+low*0.822,12), sum(adv120,12),16)) < rank(delta((h+l)/2*0.178+vwap*0.822,3)) * -1",
    "params":  {"open_col": "open", "low_col": "low", "high_col": "high", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "low", "high", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha065": {
    "formula": "rank(corr(open*0.008+vwap*0.992, sum(adv60,8),6)) < rank(open-ts_min(open,13)) * -1",
    "params":  {"open_col": "open", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha066": {
    "formula": "rank(decay_linear(delta(vwap,3),7)) + ts_rank(decay_linear((close-open)/vwap,11),7)",
    "params":  {"vwap_col": "vwap", "close_col": "close", "open_col": "open"},
    "required": ["date", "code", "name", "vwap", "close", "open"],
    "notes": "",
  },
  "add_alpha067": {
    "formula": "rank(high-ts_min(high,2))^0.5 * sqrt(IndNeutralize(rank(corr(IndNeutralize(vwap),IndNeutralize(adv20),6))))",
    "params":  {"high_col": "high", "neut_vwap_col": "neut_vwap", "neut_adv20_col": "neut_adv20", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "neut_vwap", "neut_adv20", "volume"],
    "notes": "NEUTRALIZED: neut_vwap and neut_adv20 pre-neutralised by sector/subindustry",
  },
  "add_alpha068": {
    "formula": "(ts_rank(corr(rank(high),rank(adv15),8),13) < rank(delta(close*0.518+low*0.482,1))) * -1",
    "params":  {"high_col": "high", "close_col": "close", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "close", "low", "volume"],
    "notes": "",
  },
  "add_alpha069": {
    "formula": "rank(ts_max(delta(IndNeutralize(vwap),2),4))^0.65 - rank(decay_linear(corr((h+l)/2,adv20,8),7))",
    "params":  {"neut_vwap_col": "neut_vwap", "high_col": "high", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "neut_vwap", "high", "low", "volume"],
    "notes": "NEUTRALIZED: neut_vwap = IndNeutralize(vwap, industry)",
  },
  "add_alpha070": {
    "formula": "rank(delta(vwap,1))^0.1 * ts_rank(corr(IndNeutralize(close),adv50,18),18)",
    "params":  {"vwap_col": "vwap", "neut_close_col": "neut_close", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "neut_close", "volume"],
    "notes": "NEUTRALIZED: neut_close = IndNeutralize(close, industry)",
  },
  "add_alpha071": {
    "formula": "max(ts_rank(decay_linear(corr(ts_rank(close,3),ts_rank(adv180,12),18),4),16), ts_rank(decay_linear(rank(low+open-vwap*2)^2,16),4))",
    "params":  {"close_col": "close", "low_col": "low", "open_col": "open", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "low", "open", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha072": {
    "formula": "rank(decay_linear(corr((h+l)/2,adv40,8),10)) / rank(decay_linear(corr(ts_rank(vwap,3),ts_rank(vol,18),6),2))",
    "params":  {"high_col": "high", "low_col": "low", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "low", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha073": {
    "formula": "max(rank(decay_linear(delta(vwap,4),2)), ts_rank(decay_linear(-delta(close*0.147+open*0.853,2)/price+vwap,3)*-1,16)) * -1",
    "params":  {"vwap_col": "vwap", "close_col": "close", "open_col": "open"},
    "required": ["date", "code", "name", "vwap", "close", "open"],
    "notes": "",
  },
  "add_alpha074": {
    "formula": "(rank(corr(close,sum(adv30,37),15)) < rank(corr(rank(high*0.026+vwap*0.974),rank(vol),11))) * -1",
    "params":  {"close_col": "close", "high_col": "high", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "high", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha075": {
    "formula": "rank(corr(vwap,vol,4)) < rank(corr(rank(low),rank(adv50),12))  → 1/0",
    "params":  {"vwap_col": "vwap", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "low", "volume"],
    "notes": "",
  },
  "add_alpha076": {
    "formula": "max(rank(decay_linear(delta(vwap,1),11)), ts_rank(decay_linear(ts_rank(corr(IndNeutralize(low),adv81,8),19),17),19)) * -1",
    "params":  {"vwap_col": "vwap", "neut_low_col": "neut_low", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "neut_low", "volume"],
    "notes": "NEUTRALIZED: neut_low = IndNeutralize(low, sector)",
  },
  "add_alpha077": {
    "formula": "min(rank(decay_linear((h+l)/2+high-(vwap+high),20)), rank(decay_linear(corr((h+l)/2,adv40,3),6)))",
    "params":  {"high_col": "high", "low_col": "low", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "low", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha078": {
    "formula": "rank(corr(sum(low*0.352+vwap*0.648,19),sum(adv40,19),6)) / rank(corr(rank(vwap),rank(vol),5))",
    "params":  {"low_col": "low", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "low", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha079": {
    "formula": "rank(delta(IndNeutralize(open*0.607+close*0.393),1)) < rank(corr(ts_rank(vwap,3),ts_rank(adv150,9),14))  → 1/0",
    "params":  {"neut_price_col": "neut_price79", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "neut_price79", "vwap", "volume"],
    "notes": "NEUTRALIZED: neut_price79 = IndNeutralize(open*0.607+close*0.393, sector)",
  },
  "add_alpha080": {
    "formula": "(rank(sign(delta(IndNeutralize(open*0.868+high*0.132),4)))^2)*(rank(corr(high,adv10,5))^1)",
    "params":  {"neut_price_col": "neut_price80", "high_col": "high", "volume_col": "volume"},
    "required": ["date", "code", "name", "neut_price80", "high", "volume"],
    "notes": "NEUTRALIZED: neut_price80 = IndNeutralize(open*0.868+high*0.132, industry)",
  },
  "add_alpha081": {
    "formula": "rank(log(product(rank(rank(corr(vwap,sum(adv10,49),8))^4),14)) - rank(corr(rank(vwap),rank(vol),5)))",
    "params":  {"vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha082": {
    "formula": "min(rank(decay_linear(delta(open,1),14)), ts_rank(decay_linear(corr(IndNeutralize(vol),open,17),6),6)) * -1",
    "params":  {"open_col": "open", "neut_vol_col": "neut_volume"},
    "required": ["date", "code", "name", "open", "neut_volume"],
    "notes": "NEUTRALIZED: neut_volume = IndNeutralize(volume, sector)",
  },
  "add_alpha083": {
    "formula": "(rank(delay((h-l)/(sum(close,5)/5),2))*rank(rank(vol))) / ((h-l)/(sum(close,5)/5)/(vwap-close))",
    "params":  {"high_col": "high", "low_col": "low", "close_col": "close", "volume_col": "volume", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "high", "low", "close", "volume", "vwap"],
    "notes": "",
  },
  "add_alpha084": {
    "formula": "signedpower(ts_rank(vwap-ts_max(vwap,15),20), delta(close,4))",
    "params":  {"vwap_col": "vwap", "close_col": "close"},
    "required": ["date", "code", "name", "vwap", "close"],
    "notes": "",
  },
  "add_alpha085": {
    "formula": "rank(corr(high*0.877+close*0.123,adv30,9))^rank(corr(ts_rank((h+l)/2,3),ts_rank(vol,10),7))",
    "params":  {"high_col": "high", "close_col": "close", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "close", "low", "volume"],
    "notes": "",
  },
  "add_alpha086": {
    "formula": "(ts_rank(corr(close,sum(adv20,14),6),20) < rank(5*rank(rank(close-ts_min(close,14))/rank(ts_max(close,14)-ts_min(close,14))))) * -1",
    "params":  {"close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "volume"],
    "notes": "",
  },
  "add_alpha087": {
    "formula": "max(rank(decay_linear(delta(close*0.37+vwap*0.63,1),11)), ts_rank(decay_linear(ts_rank(corr(ts_rank(low,7),ts_rank(adv10,11),6),4),14),8)) * -1",
    "params":  {"close_col": "close", "vwap_col": "vwap", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "vwap", "low", "volume"],
    "notes": "",
  },
  "add_alpha088": {
    "formula": "min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)),8)), ts_rank(decay_linear(corr(ts_rank(close,8),ts_rank(adv60,20),8),6),2))",
    "params":  {"open_col": "open", "low_col": "low", "high_col": "high", "close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "low", "high", "close", "volume"],
    "notes": "",
  },
  "add_alpha089": {
    "formula": "ts_rank(decay_linear(corr(low,adv10,6),2),6) - ts_rank(decay_linear(delta(IndNeutralize(vwap),3),13),10)",
    "params":  {"low_col": "low", "volume_col": "volume", "neut_vwap_col": "neut_vwap"},
    "required": ["date", "code", "name", "low", "volume", "neut_vwap"],
    "notes": "NEUTRALIZED: neut_vwap = IndNeutralize(vwap, industry)",
  },
  "add_alpha090": {
    "formula": "rank(close-ts_max(close,4))^rank(corr(adv5,low,5))^-1",
    "params":  {"close_col": "close", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "close", "low", "volume"],
    "notes": "",
  },
  "add_alpha091": {
    "formula": "ts_rank(decay_linear(decay_linear(corr(IndNeutralize(close),vol,9),6),9),13) - rank(decay_linear(corr(vwap,adv30,4),5))",
    "params":  {"neut_close_col": "neut_close", "volume_col": "volume", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "neut_close", "volume", "vwap"],
    "notes": "NEUTRALIZED: neut_close = IndNeutralize(close, industry)",
  },
  "add_alpha092": {
    "formula": "min(ts_rank(decay_linear((h+l)/2+close<low+open)*-1,14),18), ts_rank(decay_linear(rank(corr(rank(low),rank(adv30),3)+rank(close-open)),7),6))",
    "params":  {"high_col": "high", "low_col": "low", "close_col": "close", "open_col": "open", "vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "low", "close", "open", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha093": {
    "formula": "ts_rank(decay_linear(corr(IndNeutralize(vwap),adv81,17),19),7) / rank(decay_linear(delta(close*0.524+vwap*0.476,3),6))",
    "params":  {"neut_vwap_col": "neut_vwap", "volume_col": "volume", "close_col": "close", "vwap_col": "vwap"},
    "required": ["date", "code", "name", "neut_vwap", "volume", "close", "vwap"],
    "notes": "NEUTRALIZED: neut_vwap = IndNeutralize(vwap, industry)",
  },
  "add_alpha094": {
    "formula": "rank(vwap-ts_min(vwap,11))^ts_rank(corr(ts_rank(vwap,19),ts_rank(adv60,4),18),2)",
    "params":  {"vwap_col": "vwap", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "volume"],
    "notes": "",
  },
  "add_alpha095": {
    "formula": "rank(open-ts_min(open,12)) < ts_rank(rank(corr(sum((h+l)/2,19),sum(adv40,19),12)),11)  → 1/0",
    "params":  {"open_col": "open", "high_col": "high", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "open", "high", "low", "volume"],
    "notes": "",
  },
  "add_alpha096": {
    "formula": "max(ts_rank(decay_linear(corr(rank(vwap),rank(vol),3),4),8), ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7),ts_rank(adv60,4),4),12),14),13)) * -1",
    "params":  {"vwap_col": "vwap", "close_col": "close", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "close", "volume"],
    "notes": "",
  },
  "add_alpha097": {
    "formula": "rank(decay_linear(delta(IndNeutralize(low*0.721+vwap*0.279),3),8)) + ts_rank(decay_linear(ts_rank(corr(ts_rank(low,7),ts_rank(adv60,4),6),4),8),6)",
    "params":  {"low_col": "low", "vwap_col": "vwap", "volume_col": "volume", "neut_price_col": "neut_price97"},
    "required": ["date", "code", "name", "low", "vwap", "volume", "neut_price97"],
    "notes": "NEUTRALIZED: neut_price97 = IndNeutralize(low*0.721+vwap*0.279, industry)",
  },
  "add_alpha098": {
    "formula": "rank(decay_linear(corr(vwap,sum(adv5,26),4),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open),rank(adv15),20),8),6),4))",
    "params":  {"vwap_col": "vwap", "open_col": "open", "volume_col": "volume"},
    "required": ["date", "code", "name", "vwap", "open", "volume"],
    "notes": "",
  },
  "add_alpha099": {
    "formula": "(rank(corr(sum((h+l)/2,19),sum(adv60,19),8)) < rank(corr(low,vol,6))) * -1",
    "params":  {"high_col": "high", "low_col": "low", "volume_col": "volume"},
    "required": ["date", "code", "name", "high", "low", "volume"],
    "notes": "",
  },
  "add_alpha100": {
    "formula": "-1*rank(1.5*scale(IndNeutralize(IndNeutralize(rank(...)))) - scale(IndNeutralize(corr(close,rank(adv20),5)-rank(ts_argmin(close,5)))))",
    "params":  {"neut_rank_col": "neut_rank100", "neut_diff_col": "neut_diff100"},
    "required": ["date", "code", "name", "neut_rank100", "neut_diff100"],
    "notes": "NEUTRALIZED: both columns are double/single IndNeutralize outputs",
  },
  "add_alpha101": {
    "formula": "(close - open) / (high - low + 0.001)",
    "params":  {"close_col": "close", "open_col": "open", "high_col": "high", "low_col": "low"},
    "required": ["date", "code", "name", "close", "open", "high", "low"],
    "notes": "",
  },
}


# ── Helper functions ─────────────────────────────────────────────────────────

def summary(neutralized: bool | None = None) -> str:
    """
    Return a compact reference sheet for all 101 alphas.

    Parameters
    ----------
    neutralized : bool or None
        None (default) → show all.
        True  → only alphas with pre-neutralised column requirements.
        False → only standard alphas (no neutralisation needed).
    """
    lines = [
        "eAlpha101 — Function Reference",
        "=" * 60,
        f"{'Function':<18} {'Required cols':<40} {'Notes'}",
        "-" * 80,
    ]
    for name, meta in ALPHAS.items():
        if neutralized is True and not meta["notes"].startswith("NEUTRALIZED"):
            continue
        if neutralized is False and meta["notes"].startswith("NEUTRALIZED"):
            continue
        req = ", ".join(c for c in meta["required"] if c not in ("date", "code", "name"))
        lines.append(f"{name:<18} {req:<40} {meta['notes']}")
    lines.append("-" * 80)
    lines.append(f"Total: {len(ALPHAS)} functions")
    return "\n".join(lines)


def required_cols(col: str) -> list[str]:
    """Return names of all alpha functions that require ``col`` in mkt_data."""
    return [name for name, meta in ALPHAS.items() if col in meta["required"]]


def get(alpha_name: str) -> dict:
    """Return full metadata dict for one alpha (e.g. 'add_alpha001')."""
    if alpha_name not in ALPHAS:
        raise KeyError(f"{alpha_name!r} not found. Use 'add_alpha001' … 'add_alpha101'.")
    return ALPHAS[alpha_name]
