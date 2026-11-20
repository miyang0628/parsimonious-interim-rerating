"""
Robustness of the H3 asymmetry: does 'deterioration predicts default, improvement does not'
survive (a) a symmetric link (probit vs logit), (b) a continuous base PD instead of 7 buckets,
(c) placebo (non-solvency) indicator pairs, (d) alternative grade-peer-mean construction?

Model (per manuscript eq. h3):
  link P(D=1) = a + b*G + g+ * d+ + g- * d-   with d = S - mean_S(peer group), d+=max(d,0), d-=min(d,0)
Asymmetry holds if |g+| is significant and |g-| is not, and a Wald test rejects |g+|=|g-|.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
RNG = 0
np.random.seed(RNG)
sys.path.insert(0, "/home/claude/analysis")
from multi_dataset_robustness import oof_pd

import os
POL = os.environ.get("POL", "data/polish")  # repo-relative default; override with POL=... if needed
def load(n):
    d,_ = arff.loadarff(f"{POL}/{n}year.arff"); df = pd.DataFrame(d)
    df.columns = [f"Attr{i+1}" for i in range(len(df.columns)-1)] + ["target"]
    df["target"] = df["target"].apply(lambda v: int(v.decode()) if isinstance(v,bytes) else int(v))
    for c in df.columns[:-1]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["horizon"] = n
    return df

def build(df, two, discretize=True, peer="grade"):
    """Return y, base term(s), and the signed deviation hinge terms d+/d-."""
    y = df["target"].values
    feats = [c for c in df.columns if c.startswith("Attr")]
    others = [c for c in feats if c not in two]
    D = df[feats].copy()
    for c in feats: D[c] = D[c].fillna(D[c].median())
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    base = oof_pd(D[others], y, cv)
    if discretize:
        G = pd.qcut(pd.Series(base).rank(method="first"), 7, labels=False).values.astype(float)
        grp = G
    else:
        G = np.log(np.clip(base,1e-6,1-1e-6)/(1-np.clip(base,1e-6,1-1e-6)))  # continuous base logit
        grp = pd.qcut(pd.Series(base).rank(method="first"), 10, labels=False).values  # peer groups by decile
    S = oof_pd(D[two], y, cv)
    # peer mean of S
    if peer == "grade":
        peer_mean = pd.Series(S).groupby(pd.Series(grp)).transform("mean").values
    else:  # global mean
        peer_mean = np.full_like(S, S.mean())
    d = S - peer_mean
    dp = np.maximum(d, 0.0); dm = np.minimum(d, 0.0)
    return y, G, dp, dm

def fit_asym(y, G, dp, dm, link="logit"):
    X = sm.add_constant(np.column_stack([G, dp, dm]))
    if link == "logit":
        m = sm.Logit(y, X).fit(disp=0)
    else:
        m = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Probit())).fit()
    # params: const, G, d+, d-
    gp, gm = m.params[2], m.params[3]
    zp, zm = m.tvalues[2], m.tvalues[3]
    # ------------------------------------------------------------------
    # CORRECT asymmetry null (fixed after Reviewer 4):
    #   The hypothesis "deterioration is MORE informative than improvement"
    #   is an EQUAL-SLOPES contrast: H0 is gamma+ - gamma- = 0.
    #   A previous version tested gamma+ + gamma- = 0, which merely restates
    #   H1 ("the signal predicts on average") and is NOT a test of asymmetry.
    #   See h3_correct_null.json. The wrong contrast is kept below, commented,
    #   only so the error is transparent and reproducible.
    # ------------------------------------------------------------------
    R_correct = np.zeros((1, len(m.params))); R_correct[0,2] = 1; R_correct[0,3] = -1   # g+ - g- = 0
    wald_correct = m.wald_test(R_correct, scalar=True)
    # R_wrong = np.zeros((1, len(m.params))); R_wrong[0,2] = 1; R_wrong[0,3] = 1        # g+ + g- = 0  (INVALID)
    return {"link": link, "gamma_plus": round(float(gp),3), "z_plus": round(float(zp),2),
            "gamma_minus": round(float(gm),3), "z_minus": round(float(zm),2),
            "ratio_absgp_absgm": round(float(abs(gp)/max(abs(gm),1e-9)),2),
            "asymmetry_null_gp_minus_gm_eq0_p": float(f"{float(wald_correct.pvalue):.3g}"),
            "asymmetry_established": bool(float(wald_correct.pvalue) < 0.05)}

def main():
    pool = pd.concat([load(n) for n in range(1,6)], ignore_index=True)
    solv = ["Attr1","Attr2"]          # profitability + leverage (solvency)
    plac = ["Attr4","Attr33"]         # a liquidity/activity (non-solvency) placebo pair
    out = {}

    # (baseline) discretized grade, grade-peer mean, logit
    y,G,dp,dm = build(pool, solv, discretize=True, peer="grade")
    out["baseline_logit_discretized_solvency"] = fit_asym(y,G,dp,dm,"logit")

    # (a) symmetric link: probit
    out["a_probit_link"] = fit_asym(y,G,dp,dm,"probit")

    # (b) continuous base PD
    y2,G2,dp2,dm2 = build(pool, solv, discretize=False, peer="grade")
    out["b_continuous_base_PD"] = fit_asym(y2,G2,dp2,dm2,"logit")

    # (c) placebo (non-solvency) pair -- asymmetry should be WEAKER/absent if it's about solvency content
    yp,Gp,dpp,dmp = build(pool, plac, discretize=True, peer="grade")
    out["c_placebo_nonsolvency_pair"] = fit_asym(yp,Gp,dpp,dmp,"logit")

    # (d) global-mean deviation instead of grade-peer mean
    yg,Gg,dpg,dmg = build(pool, solv, discretize=True, peer="global")
    out["d_global_mean_deviation"] = fit_asym(yg,Gg,dpg,dmg,"logit")

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
