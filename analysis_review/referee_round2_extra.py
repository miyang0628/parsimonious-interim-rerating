"""
Three additional referee-requested analyses:
 (A) Continuous base-PD benchmark (no 7-bucket discretization): does the 2-indicator
     increment survive against a CONTINUOUS 62-ratio PD?  (Polish pooled + single horizon)
 (B) Multiple-testing correction on the 66 indicator pairs (Benjamini-Hochberg FDR + Bonferroni).
 (C) Study 2 band difference (within-one-grade recovery, middle vs extremes) with bootstrap CI.
"""
import sys, json, warnings, itertools
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.io import arff
import statsmodels.api as sm
warnings.filterwarnings("ignore")
RNG = 0
np.random.seed(RNG)

sys.path.insert(0, "/home/claude/analysis")
from multi_dataset_robustness import delong, oof_pd

import os
POL = os.environ.get("POL", "data/polish")  # repo-relative default; override with POL=... if needed
def load_pol(n):
    d,_ = arff.loadarff(f"{POL}/{n}year.arff"); df = pd.DataFrame(d)
    df.columns = [f"Attr{i+1}" for i in range(len(df.columns)-1)] + ["target"]
    df["target"] = df["target"].apply(lambda v: int(v.decode()) if isinstance(v,bytes) else int(v))
    for c in df.columns[:-1]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def continuous_benchmark(df, label):
    """Increment of 2 proxies over a CONTINUOUS base PD (no discretization)."""
    y = df["target"].values
    feats = [c for c in df.columns if c.startswith("Attr")]
    two = ["Attr1","Attr2"]; others = [c for c in feats if c not in two]
    D = df[feats].copy()
    for c in feats: D[c] = D[c].fillna(D[c].median())
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    # continuous base PD (logit of the fitted prob) as the single base regressor
    base_pd = oof_pd(D[others], y, cv)
    Gc = np.log(np.clip(base_pd,1e-6,1-1e-6)/(1-np.clip(base_pd,1e-6,1-1e-6)))  # base logit, continuous
    S  = oof_pd(D[two], y, cv)
    p0 = cross_val_predict(LogisticRegression(max_iter=3000), Gc.reshape(-1,1), y, cv=cv, method="predict_proba")[:,1]
    p1 = cross_val_predict(LogisticRegression(max_iter=3000), np.column_stack([Gc,S]), y, cv=cv, method="predict_proba")[:,1]
    a1,a0,z,pz = delong(y, p1, p0)
    m0 = sm.Logit(y, sm.add_constant(Gc.reshape(-1,1))).fit(disp=0)
    m1 = sm.Logit(y, sm.add_constant(np.column_stack([Gc,S]))).fit(disp=0)
    LR = 2*(m1.llf - m0.llf)
    return {"spec": label, "base":"continuous PD (no buckets)", "n": int(len(y)),
            "AUC_base": round(a0,4), "AUC_overlay": round(a1,4), "dAUC": round(a1-a0,4),
            "DeLong_z": round(float(z),3), "DeLong_p": float(f"{pz:.2e}"),
            "LR": round(float(LR),2)}

def pairs_fdr(df):
    """66 pairs from 12 ratios; incremental LR + per-pair p, then BH-FDR and Bonferroni."""
    y = df["target"].values
    feats = [c for c in df.columns if c.startswith("Attr")]
    D = df[feats].copy()
    for c in feats: D[c] = D[c].fillna(D[c].median())
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    # fixed full-model base grade (7 buckets) as in paper
    base = oof_pd(D[feats], y, cv)
    G = pd.qcut(pd.Series(base).rank(method="first"), 7, labels=False).values.astype(float)
    m0 = sm.Logit(y, sm.add_constant(G.reshape(-1,1))).fit(disp=0)
    # 12 ratios spanning leverage/profitability/liquidity/activity (indices chosen to mirror paper)
    twelve = ["Attr2","Attr10","Attr17","Attr8",     # leverage family
              "Attr1","Attr6","Attr11","Attr22",     # profitability family
              "Attr4","Attr46","Attr3","Attr33"]     # liquidity/activity family
    twelve = [c for c in twelve if c in feats][:12]
    rows = []
    for a,b in itertools.combinations(twelve, 2):
        Sp = oof_pd(D[[a,b]], y, cv)
        mp = sm.Logit(y, sm.add_constant(np.column_stack([G,Sp]))).fit(disp=0)
        LR = 2*(mp.llf - m0.llf); p = stats.chi2.sf(max(LR,0), 1)
        rows.append((f"{a}+{b}", LR, p))
    R = pd.DataFrame(rows, columns=["pair","LR","p"]).sort_values("LR", ascending=False)
    m = len(R)
    # Bonferroni
    R["sig_bonf_0.05"] = R["p"] < 0.05/m
    # Benjamini-Hochberg
    Rs = R.sort_values("p").reset_index(drop=True)
    Rs["rank"] = np.arange(1,len(Rs)+1)
    Rs["bh_thresh"] = 0.05*Rs["rank"]/m
    passed = Rs["p"] <= Rs["bh_thresh"]
    kmax = np.where(passed.values)[0].max()+1 if passed.any() else 0
    Rs["sig_bh_0.05"] = Rs["rank"] <= kmax
    n_bonf = int(R["sig_bonf_0.05"].sum()); n_bh = int(Rs["sig_bh_0.05"].sum())
    return {"n_pairs": m, "n_sig_uncorrected": int((R["p"]<0.05).sum()),
            "n_sig_bonferroni": n_bonf, "n_sig_BH_FDR": n_bh,
            "top5": R.head(5)[["pair","LR"]].values.tolist(),
            "bottom5": R.tail(5)[["pair","LR"]].values.tolist()}

def study2_band_ci(rating_csv):
    """Within-one-grade recovery, middle vs extreme bands, 2-proxy model; bootstrap CI of the gap."""
    r = pd.read_csv(rating_csv)
    # collapse to 7-grade ordinal
    order = ["AAA","AA","A","BBB","BB","B","CCC","CC","C","D"]
    r = r[r["Rating"].isin(order)].copy()
    r["Rating"] = r["Rating"].replace({"CC":"CCC","C":"CCC","D":"CCC"})
    grades = ["AAA","AA","A","BBB","BB","B","CCC"]
    gi = {g:i for i,g in enumerate(grades)}
    y = r["Rating"].map(gi).values
    mid_mask = r["Rating"].isin(["BBB","BB","B"]).values
    two = ["debtRatio","returnOnAssets"]
    feats = [c for c in r.columns if r[c].dtype!='object' and c not in ["Rating"]]
    X = r[two].copy()
    for c in two: X[c] = X[c].fillna(X[c].median())
    cv = KFold(5, shuffle=True, random_state=RNG)
    pred = np.zeros(len(y), int)
    Xv = StandardScaler().fit_transform(X.values)
    for tr,te in cv.split(Xv):
        rf = RandomForestClassifier(300, max_features="sqrt", random_state=RNG, n_jobs=-1)
        rf.fit(Xv[tr], y[tr]); pred[te] = rf.predict(Xv[te])
    within1 = (np.abs(pred - y) <= 1).astype(float)
    acc_mid = within1[mid_mask].mean(); acc_ext = within1[~mid_mask].mean()
    gap = acc_mid - acc_ext
    # bootstrap CI of the gap
    rng = np.random.default_rng(RNG); gaps=[]
    idx = np.arange(len(y))
    for _ in range(2000):
        bs = rng.choice(idx, len(idx), replace=True)
        w = within1[bs]; mm = mid_mask[bs]
        if mm.sum()>0 and (~mm).sum()>0:
            gaps.append(w[mm].mean() - w[~mm].mean())
    lo,hi = np.percentile(gaps, [2.5,97.5])
    return {"within1_middle": round(float(acc_mid),4), "within1_extreme": round(float(acc_ext),4),
            "gap_pp": round(float(gap)*100,2), "gap_CI95_pp": [round(lo*100,2), round(hi*100,2)],
            "excludes_zero": bool(lo>0 or hi<0)}

def main():
    out = {}
    # (A) continuous benchmark
    cont = [continuous_benchmark(pd.concat([load_pol(n) for n in range(1,6)],ignore_index=True), "pooled")]
    for n in [1,5]:
        cont.append(continuous_benchmark(load_pol(n), f"horizon {n}"))
    out["A_continuous_benchmark"] = cont
    # (B) FDR on 66 pairs (pooled)
    out["B_pairs_multiple_testing"] = pairs_fdr(pd.concat([load_pol(n) for n in range(1,6)],ignore_index=True))
    # (C) Study 2 band CI
    out["C_study2_band_CI"] = study2_band_ci(sys.argv[1])
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
