"""
Multi-dataset robustness for the parsimony overlay claim.
For each dataset: build a base grade from ALL other (clean) features via out-of-fold
logistic regression, discretize to 7 quantile buckets, then test whether a 2-indicator
overlay (a solvency-type pair) adds out-of-fold discrimination beyond the grade.
Also contrast the solvency pair against random 2-feature 'placebo' pairs (identity vs count).

Every dataset here is a row-independent cross-section (no repeated-firm panel), which is the
clean-independence check the referee asked for: if the increment is a pooling/clustering
artifact, it should NOT reappear here.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
warnings.filterwarnings("ignore")

RNG = 0
np.random.seed(RNG)

# ---------- DeLong test for two correlated AUCs ----------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong(y, p1, p0):
    y = np.asarray(y); order = (-y).argsort(kind="mergesort")
    label = y[order]; m = int(label.sum()); n = len(label)-m
    def structural(p):
        p = np.asarray(p)[order]
        pos, neg = p[:m], p[m:]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.r_[pos,neg])
        aucv = (tz[:m].sum() - m*(m+1)/2)/(m*n)
        v01 = (tz[:m]-tx)/n
        v10 = 1 - (tz[m:]-ty)/m
        return aucv, v01, v10
    a1,v01_1,v10_1 = structural(p1)
    a0,v01_0,v10_0 = structural(p0)
    sx = np.cov(np.vstack([v01_1,v01_0]))
    sy = np.cov(np.vstack([v10_1,v10_0]))
    S = sx/m + sy/n
    var = S[0,0]+S[1,1]-2*S[0,1]
    if var <= 0: return a1,a0,np.nan,np.nan
    z = (a1-a0)/np.sqrt(var)
    pval = 2*stats.norm.sf(abs(z))
    return a1,a0,z,pval

def oof_pd(X, y, cv):
    X = StandardScaler().fit_transform(np.asarray(X, float))
    return cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                             X, y, cv=cv, method="predict_proba")[:,1]

def lr_test(y, p_full, p_base):
    # likelihood-ratio via logistic deviance using the oof probabilities as single regressors
    import statsmodels.api as sm
    def dev(p):
        Xr = sm.add_constant(np.clip(p,1e-6,1-1e-6).reshape(-1,1))
        return sm.Logit(y, Xr).fit(disp=0).llf
    # proper nested LR: base = logit(G); full = logit(G,S)
    return None

def run_dataset(name, df, target, two_pair, drop_cols):
    y = df[target].astype(int).values
    feats = [c for c in df.columns if c not in drop_cols + [target]]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    # the two overlay indicators must exist & be numeric
    two = [c for c in two_pair if c in feats]
    others = [c for c in feats if c not in two]
    D = df[feats].copy()
    for c in feats:  # median impute
        D[c] = D[c].fillna(D[c].median())
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)

    # base grade from OTHER features, discretized to 7 quantile buckets
    base_pd = oof_pd(D[others], y, cv)
    G = pd.qcut(pd.Series(base_pd).rank(method="first"), 7, labels=False).values.astype(float)
    # overlay signal from the two indicators alone
    S = oof_pd(D[two], y, cv)

    # nested logits via statsmodels for LR + coef on S
    import statsmodels.api as sm
    Hc = sm.add_constant(G.reshape(-1,1))
    m0 = sm.Logit(y, Hc).fit(disp=0)
    Xf = sm.add_constant(np.column_stack([G, S]))
    m1 = sm.Logit(y, Xf).fit(disp=0)
    LR = 2*(m1.llf - m0.llf)
    LR_p = stats.chi2.sf(LR, 1)
    gamma = m1.params[-1]; gamma_z = m1.tvalues[-1]

    # out-of-fold ROC for M0 (G only) and M1 (G+S): use oof predicted prob from nested CV
    p0 = cross_val_predict(LogisticRegression(max_iter=3000), G.reshape(-1,1), y, cv=cv, method="predict_proba")[:,1]
    p1 = cross_val_predict(LogisticRegression(max_iter=3000), np.column_stack([G,S]), y, cv=cv, method="predict_proba")[:,1]
    a1,a0,z,pz = delong(y, p1, p0)

    # placebo: 30 random 2-feature pairs from 'others', incremental LR distribution
    placebo_LR = []
    rng = np.random.default_rng(RNG)
    pool = others
    for _ in range(min(30, len(pool)*(len(pool)-1)//2)):
        pr = list(rng.choice(pool, 2, replace=False))
        Sp = oof_pd(D[pr], y, cv)
        mp = sm.Logit(y, sm.add_constant(np.column_stack([G,Sp]))).fit(disp=0)
        placebo_LR.append(2*(mp.llf - m0.llf))
    placebo_LR = np.array(placebo_LR)

    return {
        "dataset": name, "n": int(len(y)), "event_rate": round(float(y.mean()),4),
        "two_indicators": two, "n_other_features": len(others),
        "AUC_base": round(a0,4), "AUC_overlay": round(a1,4), "dAUC": round(a1-a0,4),
        "DeLong_z": round(float(z),3), "DeLong_p": float(f"{pz:.2e}"),
        "LR_stat": round(float(LR),2), "LR_p": float(f"{LR_p:.2e}"),
        "gamma_S": round(float(gamma),3), "gamma_z": round(float(gamma_z),2),
        "placebo_LR_median": round(float(np.median(placebo_LR)),2),
        "placebo_LR_max": round(float(placebo_LR.max()),2),
        "solvency_vs_placebo_ratio": round(float(LR/max(np.median(placebo_LR),1e-9)),1),
    }

def main():
    results = []

    # 1) credit_risk_dataset: loan-level consumer credit
    cr = pd.read_csv(sys.argv[1])
    # encode a couple of obvious numerics; pick leverage~loan_percent_income, profitability~ -loan_int_rate (price of risk) ... but keep numeric-only others
    cr_num = cr.select_dtypes(include=[np.number]).copy()
    # leverage-type = loan_percent_income (debt burden); profitability-type proxy = person_income (ability) -> use income + loan_percent_income
    results.append(run_dataset("credit_risk(consumer)", cr_num, "loan_status",
                               two_pair=["loan_percent_income","person_income"],
                               drop_cols=[]))

    # 2) gmsc
    g = pd.read_parquet(sys.argv[2])
    results.append(run_dataset("gmsc(consumer)", g, "DEFAULT",
                               two_pair=["DEBT_RATIO","INCOME"],
                               drop_cols=["VIP","AGE_YEARS"]))

    # 3) taiwan
    t = pd.read_parquet(sys.argv[3])
    results.append(run_dataset("taiwan(card)", t, "DEFAULT",
                               two_pair=["LIMIT_BAL","PAY_0"],
                               drop_cols=["VIP","AGE_YEARS","SEX_STD"]))

    # 4) adult (non-credit control)
    a = pd.read_parquet(sys.argv[4])
    a_num = a.select_dtypes(include=[np.number]).copy()
    results.append(run_dataset("adult(non-credit control)", a_num, "HIGH_INCOME",
                               two_pair=["education-num","hours-per-week"],
                               drop_cols=["VIP","AGE_YEARS","SEX_STD","EDU_STD"]))

    # 5) Polish single-horizon (1-year file) -- clean independence, no pooling
    from scipy.io import arff
    data,_ = arff.loadarff(sys.argv[5])
    pol = pd.DataFrame(data)
    pol.columns = [f"Attr{i+1}" for i in range(len(pol.columns)-1)] + ["target"]
    pol["target"] = pol["target"].apply(lambda v: int(v) if not isinstance(v,bytes) else int(v.decode()))
    for c in pol.columns[:-1]:
        pol[c] = pd.to_numeric(pol[c], errors="coerce")
    # proxies: Attr35 (profit/assets family) & Attr? leverage. Use the paper's: profitability net profit/total assets ~ Attr1; leverage total liab/total assets ~ Attr2
    results.append(run_dataset("polish(1yr, single-horizon)", pol, "target",
                               two_pair=["Attr1","Attr2"], drop_cols=[]))

    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
