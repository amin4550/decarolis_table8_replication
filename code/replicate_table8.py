"""
Python replication of Decarolis (2015), Table 8.

Input:
    data/master_data_file.dta

Outputs:
    output/table8_replication.csv
    output/table8_key_coefficients.csv
    output/table4_panel_c_market_summary.csv

This script ports the relevant Stata logic from supportfile.do and
Do_Files/table7_table8_market_level.do into Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import pinv
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "master_data_file.dta"
OUT_DIR = ROOT / "output"

def stata_round_to(x: pd.Series, base: float) -> pd.Series:
    """Approximate Stata's round(x, base)."""
    return np.round(x / base) * base


def prepare_market_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Create the region-year dataset used in Table 8."""
    df = pd.read_stata(path, convert_categoricals=False)
    df["year"] = df["year"].astype(int)
    df["regionid"] = df["regionid"].astype(int)

    # Identifiers from supportfile.do
    df["regionid_str"] = df["regionid"].astype(str).str.zfill(2)
    df["year_str"] = df["year"].astype(str)
    df["Plan_ID_str"] = df["Plan_ID"].astype(str)
    df["new_id1"] = df["year_str"] + df["regionid_str"]
    df["new_id4"] = df["regionid_str"] + df["Contract_ID"].astype(str) + df["Plan_ID_str"]

    # Basic PDP flags and de minimis adjustment variables
    df["lis_plusminimis"] = df["lis"].astype(float)
    for y, value in [(2007, 2), (2008, 1), (2011, 2), (2012, 2)]:
        mask = (df["year"] == y) & df["lis"].notna()
        df.loc[mask, "lis_plusminimis"] = df.loc[mask, "lis"] + value

    df["basic"] = (~df["Benefit_Type"].astype(str).str.contains("Enhanced", na=False)).astype(int)
    df["pdp"] = (df["Plan_Type"] == "Medicare Prescription Drug Plan").astype(int)
    excluded_nonpdp = ["1876 Cost", "Continuing Care Retirement Community", "National PACE", "National Pace"]
    df["nonpdp"] = ((df["pdp"] == 0) & (~df["Plan_Type"].isin(excluded_nonpdp))).astype(int)
    df["sample"] = ((df["basic"] == 1) & (df["pdp"] == 1)).astype(int)

    # Rounding fixes for the Below Benchmark/De minimis indicator.
    a = df["Part_D_Basic_Premium"] - df["lis"]
    a2 = df["Part_D_Basic_Premium"] - df["lis_plusminimis"]
    simbelow = (a <= 0) & a.notna() & (df["sample"] == 1)
    simbelow = simbelow.where(a.notna() & (df["sample"] == 1), np.nan)
    below = df["Part_D_Basic_P_Below_Bench"].astype(str).str.contains("Below Regional Benchmark", na=False)
    roundmask = (a < 0.1) & (simbelow != below) & pd.notna(simbelow) & below.notna() & (df["sample"] == 1)
    df.loc[roundmask & (a > 0), "Part_D_Basic_Premium"] = df.loc[roundmask & (a > 0), "lis"]
    df.loc[(~roundmask) & (a > 0) & (a2 <= 0), "Part_D_Basic_P_Below_Bench"] = "Below De minimus Amount"
    df.loc[(~roundmask) & (a2 > 0.1) & a2.notna(), "Part_D_Basic_P_Below_Bench"] = ""

    df["sample2"] = ((df["basic"] == 1) & (df["nonpdp"] == 1)).astype(int)
    a = df["Part_D_Basic_Premium"] - df["lis"]
    simbelow = (a <= 0) & a.notna() & (df["sample2"] == 1)
    simbelow = simbelow.where(a.notna() & (df["sample2"] == 1), np.nan)
    below = df["Part_D_Basic_P_Below_Bench"].astype(str).str.contains("Below Regional Benchmark", na=False)
    roundmask = (a < 0.1) & (simbelow != below) & pd.notna(simbelow) & below.notna() & (df["sample2"] == 1)
    df.loc[roundmask & (a > 0) & a.notna(), "Part_D_Basic_Premium"] = df.loc[roundmask & (a > 0) & a.notna(), "lis"]
    df.loc[roundmask & (stata_round_to(a, 0.1) < -2), "Part_D_Basic_P_Below_Bench"] = "Below Regional Benchmark"
    df.loc[(~roundmask) & (stata_round_to(a, 0.1) < 2) & df["year"].isin([2007, 2011, 2012]) & (a > 0), "Part_D_Basic_P_Below_Bench"] = "Below De minimus Amount"
    df.loc[(~roundmask) & (stata_round_to(a, 0.1) < 1) & (df["year"] == 2008) & (a > 0), "Part_D_Basic_P_Below_Bench"] = "Below De minimus Amount"

    # LIPSA weights.
    df["lis_denominator"] = df.groupby("new_id1")["enrol_lis_num"].transform(lambda s: s.fillna(0).sum())
    df["enrol_tot_denominator"] = df.groupby("new_id1")["enrol_tot_num"].transform(lambda s: s.fillna(0).sum())

    df = df.sort_values(["new_id4", "year"]).copy()
    for source, target in [
        ("lis_denominator", "lis_denominator2"),
        ("enrol_tot_denominator", "enrol_tot_denominator2"),
        ("enrol_lis_num", "enrol_lis_num2"),
        ("enrol_tot_num", "enrol_tot_num2"),
    ]:
        df[target] = df.groupby("new_id4")[source].shift(1)
        df.loc[df["year"] == 2006, target] = df.loc[df["year"] == 2006, source]

    df["lis_wcross"] = np.where(df["pdp"] == 1, df["enrol_lis_num_croWlagged"], np.nan)
    df["lis_wcross"] = df["lis_wcross"].where(df["enrol_lis_num_croWlagged"].notna(), df["enrol_lis_num2"])
    df["lis_denominator3"] = df.groupby("new_id1")["lis_wcross"].transform(lambda s: s.fillna(0).sum())
    df["lis_weight"] = (df["lis_wcross"] / df["lis_denominator3"]).astype(float)

    df["tot_wcross"] = np.where(df["pdp"] == 1, df["enrol_tot_num_croWlagged"], np.nan)
    df["tot_wcross"] = df["tot_wcross"].where(pd.notna(df["tot_wcross"]), df["enrol_tot_num2"])
    df["enrol_tot_denominator3"] = df.groupby("new_id1")["tot_wcross"].transform(lambda s: s.fillna(0).sum())
    df["enrol_tot_weight"] = (df["tot_wcross"] / df["enrol_tot_denominator3"]).astype(float)
    df.loc[df["enrol_tot_weight"].isna(), "enrol_tot_weight"] = 0
    df.loc[df["lis_weight"].isna() & df["enrol_lis_num"].notna(), "lis_weight"] = 0

    weight_nonpdp = df["enrol_tot_weight"].where(df["pdp"] == 0, 0).groupby(df["new_id1"]).transform("sum")
    df["weight_nonpdp"] = weight_nonpdp
    df.loc[df["year"] > 2008, "weight_nonpdp"] = np.nan
    df["weight_pdp"] = 1 - df["weight_nonpdp"]
    denom_map = df[df["pdp"] == 1].groupby("new_id1").size()
    df["denom_initial_year"] = df["new_id1"].map(denom_map)

    mask = (df["year"] <= 2007) & df["denom_initial_year"].notna() & (df["Part_D_Basic_Premium"] > 0) & (df["pdp"] == 1)
    df.loc[mask, "lis_weight"] = df.loc[mask, "weight_pdp"] * (1 / df.loc[mask, "denom_initial_year"])
    df.loc[mask, "enrol_tot_weight"] = df.loc[mask, "weight_pdp"] * (1 / df.loc[mask, "denom_initial_year"])
    mask = (df["year"] == 2008) & df["denom_initial_year"].notna() & (df["Part_D_Basic_Premium"] > 0) & (df["pdp"] == 1)
    df.loc[mask, "lis_weight"] = (df.loc[mask, "weight_pdp"] * (1 / df.loc[mask, "denom_initial_year"])) * 0.5 + df.loc[mask, "lis_weight"] * 0.5
    df.loc[mask, "enrol_tot_weight"] = (df.loc[mask, "weight_pdp"] * (1 / df.loc[mask, "denom_initial_year"])) * 0.5 + df.loc[mask, "enrol_tot_weight"] * 0.5

    df["min_pdpd"] = df.groupby("new_id1")["Part_D_Basic_Premium"].transform(
        lambda s: s.where((df.loc[s.index, "Part_D_Basic_Premium"] > 0) & (df.loc[s.index, "pdp"] == 1)).min()
    )
    df["true_weight_cw"] = df["enrol_tot_weight"]
    df.loc[df["year"] > 2008, "true_weight_cw"] = df.loc[df["year"] > 2008, "lis_weight"]
    df["benchmark_cw"] = (df["true_weight_cw"] * df["Part_D_Basic_Premium"]).groupby(df["new_id1"]).transform(lambda s: s.fillna(0).sum())
    mask = (df["min_pdpd"] > df["benchmark_cw"]) & df["min_pdpd"].notna()
    df.loc[mask, "benchmark_cw"] = df.loc[mask, "min_pdpd"]
    check_map = df[df["Part_D_Basic_Premium"].notna()].groupby("new_id1")["true_weight_cw"].sum(min_count=0)
    df["check"] = df["new_id1"].map(check_map)
    df["true_weight_check_cw"] = df["true_weight_cw"] * (1 / df["check"])

    # Market covariates.
    df["tot_enrollment"] = df.groupby("new_id1")["enrol_tot_num"].transform(lambda s: s.fillna(0).sum())
    df["weig_reg_basic_prem"] = df["Part_D_Basic_Premium"] * (df["enrol_tot_num"] / df["tot_enrollment"])
    dem_mask = df["Part_D_Basic_P_Below_Bench"].astype(str) == "Below De minimus Amount"
    mask = dem_mask & df["enrol_tot_num"].notna() & (df["enrol_tot_num"] >= df["enrol_lis_num"])
    df.loc[mask, "weig_reg_basic_prem"] = (1 / df.loc[mask, "tot_enrollment"]) * (
        (df.loc[mask, "Part_D_Basic_Premium"] * (df.loc[mask, "enrol_tot_num"] - df.loc[mask, "enrol_lis_num"]))
        + (df.loc[mask, "lis"] * df.loc[mask, "enrol_lis_num"])
    )
    df["mean_w_reg_basprem"] = df.groupby("new_id1")["weig_reg_basic_prem"].transform(lambda s: s.fillna(0).sum())
    df["mean_weig_reg_drugs"] = (df["perc_drug_frf"] * (df["enrol_tot_num"] / df["tot_enrollment"])).groupby(df["new_id1"]).transform(lambda s: s.fillna(0).sum())
    df["mean_weig_reg_pharmacies"] = (df["in_area_flag"] * (df["enrol_tot_num"] / df["tot_enrollment"])).groupby(df["new_id1"]).transform(lambda s: s.fillna(0).sum())
    vintage = df.groupby("new_id4")["year"].min()
    df["plan_years_alive"] = df["year"] - df["new_id4"].map(vintage)
    df["mean_weig_reg_vintage"] = (df["plan_years_alive"] * (df["enrol_tot_num"] / df["tot_enrollment"])).groupby(df["new_id1"]).transform(lambda s: s.fillna(0).sum())

    enrol_tot_mapd = df[df["pdp"] == 0].groupby("new_id1")["enrol_tot_num"].sum(min_count=0)
    df["mapd_regio_share_temp"] = df["new_id1"].map(enrol_tot_mapd) / df["enrol_tot_denominator"]
    df.loc[df["mapd_regio_share_temp"].isna(), "mapd_regio_share_temp"] = 0
    df["mapd_regio_share"] = df.groupby("new_id1")["mapd_regio_share_temp"].transform("max")
    df["mapd_regio_2006share"] = df.groupby("regionid")["mapd_regio_share"].transform(lambda s: s.where(df.loc[s.index, "year"] == 2006, 0).max())

    trend_map = {2006: 1, 2007: 2, 2008: 3, 2009: 4, 2010: 5, 2011: 6, 2012: 7}
    df["ttrend"] = df["year"].map(trend_map)
    for r in range(1, 35):
        df[f"ttrend{r}"] = np.where(df["regionid"] == r, df["ttrend"], 0)

    # Main Table 8 regressor: sum of four largest LIPSA weights among basic PDPs in a region-year.
    df["true_weight_check_cw2"] = np.where((df["pdp"] == 1) & (df["basic"] == 1), df["true_weight_check_cw"], np.nan)
    c4 = df.groupby("new_id1")["true_weight_check_cw2"].apply(lambda s: s.dropna().sort_values(ascending=False).head(4).sum())
    df["conc_weig_basicpdpd_c4"] = df["new_id1"].map(c4)

    # Herfindahl index by parent organization.
    df["enrol_tot_share"] = 0.0
    mask = (df["enrol_tot_num"] != 0) & df["enrol_tot_num"].notna() & (df["enrol_tot_denominator"] != 0) & df["enrol_tot_denominator"].notna()
    df.loc[mask, "enrol_tot_share"] = df.loc[mask, "enrol_tot_num"] / df.loc[mask, "enrol_tot_denominator"]
    firm_share = df.groupby(["new_id1", "Parent_Organization"])["enrol_tot_share"].sum().reset_index()
    hhi = firm_share.assign(sq=firm_share["enrol_tot_share"] ** 2).groupby("new_id1")["sq"].sum().to_frame("raw_hhi")
    hhi["num_firms"] = firm_share.groupby("new_id1").size()
    hhi["hershf_firm_tot"] = (hhi["raw_hhi"] - (1 / hhi["num_firms"])) / (1 - (1 / hhi["num_firms"]))
    df = df.merge(hhi[["hershf_firm_tot"]], left_on="new_id1", right_index=True, how="left")

    # Region-year dataset.
    market = df.sort_values(["new_id1", "Parent_Organization"]).drop_duplicates(["regionid", "year"], keep="first").copy()
    market = market.sort_values(["regionid", "year"]).copy()
    market["logmean_w_reg_bpr"] = np.log(market["mean_w_reg_basprem"])
    market["logmean_w_reg_bpr_ch"] = market.groupby("regionid")["logmean_w_reg_bpr"].diff()
    market.loc[market["year"] <= 2006, "logmean_w_reg_bpr_ch"] = np.nan
    market["laghershf_firm_tot"] = market.groupby("regionid")["hershf_firm_tot"].shift(1)
    market["lagunemploymentrate"] = market.groupby("regionid")["unemploymentrate"].shift(1)
    market = market[market["year"] != 2006].copy()
    market["year2009"] = (market["year"] > 2008).astype(int)
    market["regio_2006share2"] = market["mapd_regio_2006share"] * market["year2009"]
    return market


def build_x(data: pd.DataFrame, controls: List[str]) -> pd.DataFrame:
    parts = []
    if controls:
        parts.append(data[controls].astype(float))
    parts.append(pd.get_dummies(data["year"].astype(int), prefix="year", drop_first=True, dtype=float))
    parts.append(pd.get_dummies(data["regionid"].astype(int), prefix="region", drop_first=True, dtype=float))
    x = pd.concat(parts, axis=1)
    return sm.add_constant(x, has_constant="add").astype(float)


def ols_cluster(y: pd.Series, x: pd.DataFrame, clusters: pd.Series):
    return sm.OLS(y.astype(float), x.astype(float), missing="drop").fit(
        cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True}, use_t=True
    )


def iv2sls_cluster(y: pd.Series, exog: pd.DataFrame, endog: pd.DataFrame, instr: pd.DataFrame, clusters: pd.Series):
    """2SLS with cluster-robust standard errors using a Stata-like small-sample correction."""
    yv = y.to_numpy(float).reshape(-1, 1)
    w = pd.concat([exog, endog], axis=1)
    z = pd.concat([exog, instr], axis=1)
    wm = w.to_numpy(float)
    zm = z.to_numpy(float)
    ztz_inv = pinv(zm.T @ zm)
    a = wm.T @ zm @ ztz_inv @ zm.T @ wm
    b = wm.T @ zm @ ztz_inv @ zm.T @ yv
    beta = pinv(a) @ b
    u = yv - wm @ beta

    cluster_values = pd.Series(clusters).astype(int).to_numpy()
    s = np.zeros((zm.shape[1], zm.shape[1]))
    for g in np.unique(cluster_values):
        idx = cluster_values == g
        zgu = zm[idx, :].T @ u[idx, :]
        s += zgu @ zgu.T
    middle = wm.T @ zm @ ztz_inv @ s @ ztz_inv @ zm.T @ wm
    v = pinv(a) @ middle @ pinv(a)
    n = wm.shape[0]
    g = len(np.unique(cluster_values))
    k = np.linalg.matrix_rank(wm)
    v *= (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(v)).reshape(-1, 1)
    beta_s = pd.Series(beta.ravel(), index=w.columns)
    se_s = pd.Series(se.ravel(), index=w.columns)
    yhat = (wm @ beta).ravel()
    r2 = 1 - np.sum((y.to_numpy(float) - yhat) ** 2) / np.sum((y.to_numpy(float) - y.mean()) ** 2)
    return beta_s, se_s, r2


def stars(coef: float, se: float, df: int = 33) -> str:
    if not np.isfinite(se) or se == 0:
        return ""
    p = 2 * (1 - stats.t.cdf(abs(coef / se), df=df))
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt(coef: float, se: float | None = None, scale: float = 1.0) -> str:
    value = coef * scale
    if abs(value) < 0.01 and value != 0:
        out = f"{value:.4f}"
    elif abs(value) < 1:
        out = f"{value:.3f}"
    else:
        out = f"{value:.3f}"
    if se is not None:
        out += stars(coef, se)
    return out


def make_table8(market: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trend_vars = [f"ttrend{i}" for i in range(1, 35)]
    ols_controls: Dict[int, List[str]] = {
        1: ["conc_weig_basicpdpd_c4"],
        2: ["conc_weig_basicpdpd_c4"] + trend_vars,
        3: ["conc_weig_basicpdpd_c4", "laghershf_firm_tot", "lagunemploymentrate"],
        4: ["conc_weig_basicpdpd_c4", "laghershf_firm_tot", "lagunemploymentrate"] + trend_vars,
        5: ["conc_weig_basicpdpd_c4", "laghershf_firm_tot", "lagunemploymentrate", "mean_weig_reg_vintage", "mean_weig_reg_pharmacies", "mean_weig_reg_drugs"],
        6: ["conc_weig_basicpdpd_c4", "laghershf_firm_tot", "lagunemploymentrate", "mean_weig_reg_vintage", "mean_weig_reg_pharmacies", "mean_weig_reg_drugs"] + trend_vars,
    }
    iv_controls: Dict[int, List[str]] = {
        1: [],
        2: trend_vars,
        3: ["laghershf_firm_tot", "lagunemploymentrate"],
        4: ["laghershf_firm_tot", "lagunemploymentrate"] + trend_vars,
        5: ["laghershf_firm_tot", "lagunemploymentrate", "mean_weig_reg_vintage", "mean_weig_reg_pharmacies", "mean_weig_reg_drugs"],
        6: ["laghershf_firm_tot", "lagunemploymentrate", "mean_weig_reg_vintage", "mean_weig_reg_pharmacies", "mean_weig_reg_drugs"] + trend_vars,
    }

    rows = []
    detail = []
    for spec in range(1, 7):
        d = market[["logmean_w_reg_bpr_ch", "regionid", "year"] + ols_controls[spec]].dropna().copy()
        y = d["logmean_w_reg_bpr_ch"]
        x = build_x(d, ols_controls[spec])
        res = ols_cluster(y, x, d["regionid"])
        for name in res.params.index:
            detail.append({"panel": "OLS", "spec": spec, "term": name, "coef": res.params[name], "se": res.bse[name], "r2": res.rsquared, "n": int(res.nobs)})

        d = market[["logmean_w_reg_bpr_ch", "regionid", "year", "conc_weig_basicpdpd_c4", "regio_2006share2"] + iv_controls[spec]].dropna().copy()
        y = d["logmean_w_reg_bpr_ch"]
        exog = build_x(d, iv_controls[spec])
        beta, se, r2 = iv2sls_cluster(y, exog, d[["conc_weig_basicpdpd_c4"]].astype(float), d[["regio_2006share2"]].astype(float), d["regionid"])
        for name in beta.index:
            detail.append({"panel": "2SLS", "spec": spec, "term": name, "coef": beta[name], "se": se[name], "r2": r2, "n": len(d)})

    detail_df = pd.DataFrame(detail)

    display_rows = [
        ("wLIS4", "conc_weig_basicpdpd_c4", 1.0),
        ("HHI", "laghershf_firm_tot", 1.0),
        ("Unemployment", "lagunemploymentrate", 1.0),
        ("Plan age", "mean_weig_reg_vintage", 1.0),
        ("Pharmacies", "mean_weig_reg_pharmacies", 10000.0),
        ("Drugs", "mean_weig_reg_drugs", 1.0),
        ("Constant", "const", 1.0),
    ]
    final_rows = []
    for panel in ["OLS", "2SLS"]:
        final_rows.append([f"Panel {'A' if panel == 'OLS' else 'B'}. {panel}"] + [""] * 6)
        panel_data = detail_df[detail_df["panel"] == panel]
        for label, term, scale in display_rows:
            coef_line = [label]
            se_line = [""]
            any_term = False
            for spec in range(1, 7):
                row = panel_data[(panel_data["spec"] == spec) & (panel_data["term"] == term)]
                if row.empty:
                    coef_line.append("")
                    se_line.append("")
                else:
                    any_term = True
                    coef = float(row["coef"].iloc[0])
                    se_value = float(row["se"].iloc[0])
                    coef_line.append(fmt(coef, se_value, scale=scale))
                    se_line.append(f"[{fmt(se_value, None, scale=scale)}]")
            if any_term:
                final_rows.append(coef_line)
                final_rows.append(se_line)
        final_rows.append(["Region time trends"] + ["Yes" if spec in [2, 4, 6] else "No" for spec in range(1, 7)])
        r2_line = ["R-squared"]
        obs_line = ["Observations"]
        for spec in range(1, 7):
            r2_value = panel_data[panel_data["spec"] == spec]["r2"].iloc[0]
            n_value = panel_data[panel_data["spec"] == spec]["n"].iloc[0]
            r2_line.append(f"{r2_value:.3f}")
            obs_line.append(str(int(n_value)))
        final_rows.append(r2_line)
        final_rows.append(obs_line)
        final_rows.append([""] + [""] * 6)

    table = pd.DataFrame(final_rows, columns=["Variable", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"])
    key = detail_df[detail_df["term"] == "conc_weig_basicpdpd_c4"].copy()
    return table, key


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    market = prepare_market_data(DATA_PATH)
    summary = market[[
        "mean_w_reg_basprem", "logmean_w_reg_bpr_ch", "conc_weig_basicpdpd_c4",
        "laghershf_firm_tot", "mapd_regio_2006share", "MA_2005_penetration",
        "MA_2004_penetration", "MA_2003_penetration"
    ]].describe().T[["mean", "std", "50%", "count"]]
    summary.to_csv(OUT_DIR / "table4_panel_c_market_summary.csv")

    table, key = make_table8(market)
    table.to_csv(OUT_DIR / "table8_replication.csv", index=False)
    key.to_csv(OUT_DIR / "table8_key_coefficients.csv", index=False)
    print(table.to_string(index=False))
    print("\nSaved outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
