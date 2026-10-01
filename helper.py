import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def split_columns(df, target):
    """Return (numeric_columns, categorical_columns), both WITHOUT the target."""
    num = df.select_dtypes(include="number").columns.drop(target, errors="ignore").tolist()
    cat = df.select_dtypes(exclude=["number", "datetime"]).columns.drop(target, errors="ignore").tolist()
    return num, cat

def column_report(df):
    rep = pd.DataFrame({
        "dytpe": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
        "unique_%": (df.nunique(dropna=True) / len(df) * 100).round(2),
    })
    num = df.select_dtypes(include="number")
    rep["zeros"] = (num == 0).sum().reindex(rep.index)
    rep["zeros_%"] = (rep["zeros"] / len(df) * 100).round(2)
    return rep

def plot_missing(df, figsize=(11, 4)):
    plt.figure(figsize=figsize)
    sns.heatmap(df.isna(), cbar=False, yticklabels=True, cmap="viridis")
    plt.title("Missing values:") # Yellow = missing
    plt.tight_layout(); plt.show()
    
def skew_kurt_table(df, cols):
    t = pd.DataFrame({"skew": df[cols].skew(), "excess_kurtosis": df[cols].kurt()})
    t["verdict"] = np.select(
        [t["skew"] > 1, t["skew"] < -1, t["skew"].abs() > 0.5],
        ["highly right skewed", "highly left skewed", "moderately skewed"], 
        "roughly symmetric")
    return t.sort_values("skew", key=abs, ascending=False).round(3)

def hist_grid(df, cols, ncols=3, bins=30):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    for ax, c in zip(np.array(axes).ravel(), cols):
        sns.histplot(df[c].dropna(), bins=bins, kde=True, ax=ax)
        ax.set_title(f"{c} (skew={df[c].skew():.2f})")
    for ax in np.array(axes).ravel()[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
def qq_grid(df, cols, ncols=3):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    for ax, c in zip(np.array(axes).ravel(), cols):
        stats.probplot(df[c].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot: {c}")
    for ax in np.array(axes).ravel()[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
def cardinality_table(df, cols):
    t = pd.DataFrame({"unique": df[cols].nunique()})
    t["unique_%_of_rows"] = (t["unique"] / len(df) * 100).round(2)
    t["top_value_share_%"] = [round(df[c].value_counts(normalize=True).iloc[0] * 100, 1) for c in t.index]
    t["rare_levels(<1%)"] = [int((df[c].value_counts(normalize=True) < 0.01).sum()) for c in t.index]
    return t.sort_values("unique", ascending=False)

def vif_table(df, cols):

    X = df[cols].copy()
    X = X.loc[:, X.std() > 0] # constant columns break the maths
    X = X.fillna(X.median())
    Z = ((X - X.mean()) / X.std()).to_numpy()
    out = {}
    for i, c in enumerate(X.columns):
        y = Z[:, i]
        others = np.delete(Z, i, axis=1)
        others = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        r2 = 1 - resid.var() / y.var()
        out[c] = np.inf if r2 > 0.999999 else 1 / (1 - r2)
    return pd.DataFrame({"VIF": out}).sort_values("VIF", ascending=False).round(2)


def high_corr_pairs(df, cols, threshold=0.8):
    c = df[cols].corr().abs()
    iu = np.triu_indices_from(c, k=1)
    pairs = pd.DataFrame({"feature_1": c.index[iu[0]], "feature_2": c.columns[iu[1]], "abs_corr": c.values[iu]})
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)


def _cramers_v(a, b):
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1)) # bias correction
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1))))


def association_matrix(df, interval_cols=None, max_rows=10000, random_state=42):
    d = df.sample(min(max_rows, len(df)), random_state=random_state) if len(df) > max_rows else df
    try:
        import phik  # noqa: F401
        return d.phik_matrix(interval_cols=interval_cols), "Phi-K"
    except Exception:
        binned = d.copy()
        for c in binned.columns:
            if pd.api.types.is_numeric_dtype(binned[c]) and binned[c].nunique() > 10:
                binned[c] = pd.qcut(binned[c], 10, duplicates="drop")
        cols = list(binned.columns)
        m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                m.loc[a, b] = m.loc[b, a] = _cramers_v(binned[a], binned[b])
        return m, "Cramer's V (fallback, phik not installed)"
    
def iqr_outliers(df, cols, k=1.5):
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([.25, .75]); i = q3 - q1
        lo, hi = q1 - k * i, q3 + k * i
        n = int(((s < lo) | (s > hi)).sum())
        rows.append((c, n, round(n / len(s) * 100, 2), round(lo, 2), round(hi, 2), s.min(), s.max()))
    return pd.DataFrame(rows, columns=["column", "n_outliers", "pct", "lower_fence", "upper_fence", "min", "max"]).set_index("column")


def zscore_outliers(df, cols, thr=3):
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    return pd.DataFrame({"n_|z|>3": (z.abs() > thr).sum(), "pct_z": ((z.abs() > thr).mean() * 100).round(2)})


def mad_outliers(df, cols, thr=3.5):
    rows = {}
    for c in cols:
        s = df[c].dropna()
        mad = np.median(np.abs(s - s.median()))
        if mad == 0:
            rows[c] = (np.nan, np.nan); continue
        mz = 0.6745 * (s - s.median()) / mad
        rows[c] = (int((mz.abs() > thr).sum()), round((mz.abs() > thr).mean() * 100, 2))
    return pd.DataFrame(rows, index=["n_MAD_outliers", "pct_MAD"]).T


def isolation_forest_flags(df, cols, contamination=0.02, random_state=42):
    from sklearn.ensemble import IsolationForest
    X = df[cols].fillna(df[cols].median())
    return pd.Series(IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
                     .fit_predict(X) == -1, index=df.index, name="isolation_forest_outlier")


def elliptic_flags(df, cols, contamination=0.02, random_state=42, max_rows=20000):
    from sklearn.covariance import EllipticEnvelope
    X = df[cols].fillna(df[cols].median())
    X = (X - X.mean()) / X.std().replace(0, 1)
    sub = X.sample(min(max_rows, len(X)), random_state=random_state)
    ee = EllipticEnvelope(contamination=contamination, random_state=random_state).fit(sub)
    return pd.Series(ee.predict(X) == -1, index=df.index, name="elliptic_outlier")

def make_model_matrix(df, target, drop=(), max_levels=40):
    d = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    y = d.pop(target)
    d = d.select_dtypes(exclude="datetime")
    num = d.select_dtypes(include="number")
    cat = d.select_dtypes(exclude="number")
    too_many = [c for c in cat.columns if cat[c].nunique() > max_levels]
    cat = cat.drop(columns=too_many)
    X = pd.concat([num.fillna(num.median()), pd.get_dummies(cat, dtype=float)], axis=1)
    return X, y, too_many
