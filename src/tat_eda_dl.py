"""
EDA, Visualization and Deep Learning on Edge-Server Task Turnaround Times (TAT)
Course: U21ADP05 - Exploratory Data Analysis and Visualization (Assignment II)

Run:  python src/tat_eda_dl.py        (from the project root)
Outputs: figures/*.png, outputs/*.json|csv, outputs/interactive_tat.html
"""
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score, log_loss,
                             precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures"); OUT = os.path.join(ROOT, "outputs")
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
R = {}  # results collected for the report

sns.set_theme(style="whitegrid", context="notebook")
PAL = {"MacBookPro1": "#1f77b4", "MacBookPro2": "#2ca02c", "VM": "#ff7f0e", "RasberryPi": "#d62728"}
ORDER = list(PAL)

def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=170); plt.close()

# ------------------------------------------------------------------ 1. ACQUIRE / PARSE
frames = []
for dev in ORDER:
    d = pd.read_csv(os.path.join(DATA, f"{dev}.csv"))
    d["device"] = dev
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
raw = raw.rename(columns={"Time": "time_str", "Execution Time": "tat"})
raw["time"] = pd.to_datetime(raw["time_str"], format="%a %b %d %H:%M:%S %Y")
R["raw_shape"] = list(raw.shape)
R["raw_missing"] = int(raw[["time_str", "tat"]].isna().sum().sum())
R["raw_dup_rows"] = int(raw.duplicated(["device", "time_str", "tat"]).sum())
R["raw_dup_ts"] = int(raw.duplicated(["device", "time"]).sum())
R["devices"] = {dv: dict(start=str(g.time.min()), end=str(g.time.max()), n=int(len(g)))
                for dv, g in raw.groupby("device")}

# ------------------------------------------------------------------ 2. FILTER / CLEAN
df = raw.drop_duplicates(["device", "time"]).sort_values(["device", "time"]).reset_index(drop=True)
# IQR outlier flag per device (kept: genuine network / system spikes, not errors)
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s > q3 + 1.5 * i) | (s < q1 - 1.5 * i)
df["iqr_outlier"] = df.groupby("device")["tat"].transform(iqr_flag)
R["outliers"] = df.groupby("device")["iqr_outlier"].agg(["sum", "mean"]).round(4).to_dict("index")
R["raw_stats"] = df.groupby("device")["tat"].agg(["count", "mean", "std", "min", "median", "max"]).round(4).to_dict("index")
R["skew_kurt"] = {dv: dict(skew=float(stats.skew(g.tat)), kurt=float(stats.kurtosis(g.tat)))
                  for dv, g in df.groupby("device")}

# ------------------------------------------------------------------ 3. FEATURE ENGINEERING (>= 15 fields)
g = df.groupby("device", group_keys=False)
df["log_tat"] = np.log(df["tat"])
df["gap_s"] = g["time"].diff().dt.total_seconds().fillna(1)           # seconds since previous task
df["elapsed_s"] = g["time"].transform(lambda s: (s - s.min()).dt.total_seconds())
df["hour"] = df.time.dt.hour; df["minute"] = df.time.dt.minute; df["second"] = df.time.dt.second
df["task_index"] = g.cumcount()
for k in (1, 2, 3):
    df[f"lag{k}"] = g["tat"].shift(k)
df["diff1"] = df["lag1"] - df["lag2"]
sh = g["tat"].shift(1)                                                  # shifted -> no leakage
df["roll_mean5"] = sh.groupby(df["device"]).transform(lambda s: s.rolling(5, min_periods=3).mean())
df["roll_std5"] = sh.groupby(df["device"]).transform(lambda s: s.rolling(5, min_periods=3).std())
df["roll_max5"] = sh.groupby(df["device"]).transform(lambda s: s.rolling(5, min_periods=3).max())
df["roll_mean20"] = sh.groupby(df["device"]).transform(lambda s: s.rolling(20, min_periods=5).mean())
df["prev_gap"] = g["gap_s"].shift(1)
df["lag1_ratio"] = df["lag1"] / df["roll_mean20"]
# target: "slow task" = TAT above the device's own 90th percentile (training portion only, below)
df = df.dropna().reset_index(drop=True)
R["clean_shape"] = list(df.shape)

# split chronologically *within each device* 70/15/15 (avoids look-ahead leakage)
df["split"] = "train"
for dv, idx in df.groupby("device").groups.items():
    idx = list(idx); n = len(idx)
    df.loc[idx[int(.70 * n):int(.85 * n)], "split"] = "val"
    df.loc[idx[int(.85 * n):], "split"] = "test"
thr = df[df.split == "train"].groupby("device")["tat"].quantile(.90)
df["is_slow"] = (df["tat"] > df["device"].map(thr)).astype(int)
R["thresholds"] = thr.round(4).to_dict()
R["split_counts"] = df.groupby("split").size().to_dict()
R["slow_rate"] = df.groupby("split")["is_slow"].mean().round(4).to_dict()
df.to_csv(os.path.join(OUT, "tat_engineered_dataset.csv"), index=False)

NUM = ["tat", "log_tat", "gap_s", "elapsed_s", "hour", "minute", "second", "task_index",
       "lag1", "lag2", "lag3", "diff1", "roll_mean5", "roll_std5", "roll_max5", "roll_mean20",
       "prev_gap", "lag1_ratio"]
R["n_fields"] = len(df.columns)

# ------------------------------------------------------------------ 4. MINE : univariate / distributions
# --- Fig1 histogram + KDE per device (log x)
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for a, dv in zip(ax.ravel(), ORDER):
    s = df[df.device == dv]["tat"]
    hi_ = s.quantile(.995); sns.histplot(s[s <= hi_], bins=40, kde=True, color=PAL[dv], ax=a, stat="density")
    a.text(.98, .55, f"max = {s.max():.2f}s\n(values > {hi_:.2f}s\nnot shown)", transform=a.transAxes, ha="right", fontsize=8, color="dimgray")
    a.axvline(s.mean(), c="k", ls="--", lw=1, label=f"mean {s.mean():.3f}s")
    a.axvline(s.median(), c="grey", ls=":", lw=1.4, label=f"median {s.median():.3f}s")
    a.set_title(dv); a.set_xlabel("Turnaround time (s)"); a.set_ylabel("Density"); a.legend(fontsize=8)
fig.suptitle("Fig. 1  Distribution of task turnaround time per edge server", fontweight="bold")
save("fig1_histograms.png")

# --- Fig2 boxplot (log scale)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="device", y="tat", order=ORDER, palette=PAL, fliersize=2.5)
plt.yscale("log"); plt.xlabel("Edge server"); plt.ylabel("Turnaround time (s, log scale)")
plt.title("Fig. 2  Box plots of TAT by device (outliers shown as points)", fontweight="bold")
save("fig2_boxplot.png")

# --- Fig3 time series
fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=False)
for a, dv in zip(ax, ORDER):
    s = df[df.device == dv]
    a.plot(s.task_index, s.tat, color=PAL[dv], lw=.7)
    o = s[s.iqr_outlier]; a.scatter(o.task_index, o.tat, c="k", s=9, zorder=3, label="IQR outlier")
    a.set_ylabel("TAT (s)"); a.set_title(dv, loc="left", fontsize=10); a.legend(loc="upper right", fontsize=8)
ax[-1].set_xlabel("Task sequence number")
fig.suptitle("Fig. 3  TAT over the course of each experiment", fontweight="bold")
save("fig3_timeseries.png")

# --- Fig4 correlation heat-map of engineered numeric features
cm = df[["tat", "gap_s", "prev_gap", "lag1", "lag2", "lag3", "diff1", "roll_mean5", "roll_std5",
         "roll_max5", "roll_mean20", "lag1_ratio", "task_index"]].corr(method="spearman")
plt.figure(figsize=(9, 7.3))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": "Spearman rho"}, annot_kws={"size": 7})
plt.title("Fig. 4  Spearman correlation heat-map of engineered features", fontweight="bold")
save("fig4_heatmap.png")
R["corr_tat"] = cm["tat"].drop("tat").round(3).to_dict()

# --- Fig5 lag scatter (autocorrelation) per device
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
R["lag1_spearman"] = {}
for a, dv in zip(ax.ravel(), ORDER):
    s = df[df.device == dv]
    a.scatter(s.lag1, s.tat, s=8, alpha=.5, color=PAL[dv])
    rho = stats.spearmanr(s.lag1, s.tat)[0]; R["lag1_spearman"][dv] = round(float(rho), 3)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_title(f"{dv}  (Spearman rho = {rho:.2f})"); a.set_xlabel("Previous TAT (s)"); a.set_ylabel("Current TAT (s)")
fig.suptitle("Fig. 5  Scatter plot: current TAT vs previous TAT", fontweight="bold")
save("fig5_lagscatter.png")

# --- Fig6 pairplot of key variables
pp = df.assign(gap_s=df.gap_s.clip(upper=5))[["log_tat", "gap_s", "roll_mean5", "roll_std5", "lag1_ratio", "device"]].copy()
pp.columns = ["log TAT", "gap (s, capped at 5)", "roll mean5", "roll std5", "lag1/mean20", "device"]
pp["roll std5"] = np.log10(pp["roll std5"] + 1e-4); pp["lag1/mean20"] = np.log10(pp["lag1/mean20"])
pp = pp.rename(columns={"roll std5": "log10 roll std5", "lag1/mean20": "log10 lag1/mean20"})
pp["roll mean5"] = np.log10(pp["roll mean5"])
pp = pp.rename(columns={"roll mean5": "log10 roll mean5"})
gp = sns.pairplot(pp.sample(1500, random_state=RNG), hue="device", palette=PAL, hue_order=ORDER,
                  plot_kws=dict(s=8, alpha=.5), diag_kind="kde", height=1.7)
gp.fig.suptitle("Fig. 6  Pair plot of selected engineered features", y=1.02, fontweight="bold")
gp.fig.savefig(os.path.join(FIG, "fig6_pairplot.png"), dpi=140, bbox_inches="tight"); plt.close("all")

# --- Fig7 ECDF
plt.figure(figsize=(8, 5))
for dv in ORDER:
    sns.ecdfplot(df[df.device == dv].tat, label=dv, color=PAL[dv])
plt.xscale("log"); plt.xlabel("TAT (s, log scale)"); plt.ylabel("Cumulative probability")
plt.axhline(.9, c="grey", ls=":"); plt.legend(); plt.title("Fig. 7  Empirical CDF of TAT", fontweight="bold")
save("fig7_ecdf.png")

# --- Fig8 continuous distribution fitting (lognormal / gamma / normal) + QQ
cands = {"norm": stats.norm, "lognorm": stats.lognorm, "gamma": stats.gamma}
fit_rows, fig = [], plt.figure(figsize=(12, 8))
for i, dv in enumerate(ORDER):
    s_all = df[df.device == dv]["tat"].values
    s = s_all[s_all <= np.percentile(s_all, 99)]          # fit the 'body' (central 99%); tail analysed separately
    ax = fig.add_subplot(2, 4, i + 1)
    xs = np.linspace(np.percentile(s, .5), np.percentile(s, 99), 300)
    ax.hist(s, bins=35, density=True, color=PAL[dv], alpha=.35)
    best = None
    for nm, dist in cands.items():
        p = dist.fit(s) if nm == "norm" else dist.fit(s, floc=0)
        ll = dist.logpdf(s, *p).sum(); k = 2; aic = 2 * k - 2 * ll
        ks = stats.kstest(s, dist(*p).cdf)
        fit_rows.append(dict(device=dv, dist=nm, aic=aic, ks_stat=ks.statistic, ks_p=ks.pvalue))
        ax.plot(xs, dist.pdf(xs, *p), label=nm, lw=1.6)
        if best is None or aic < best[1]: best = (nm, aic, p)
    ax.set_title(dv, fontsize=9); ax.legend(fontsize=7); ax.set_xlabel("TAT (s)")
    ax2 = fig.add_subplot(2, 4, i + 5)
    stats.probplot(np.log(s_all), dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markersize(3); ax2.set_title(f"QQ of log(TAT) - {dv}", fontsize=8)
    ax2.set_xlabel("Theoretical quantiles"); ax2.set_ylabel("Ordered log TAT")
fig.suptitle("Fig. 8  Continuous distribution fitting (top) and log-normal Q-Q plots (bottom)", fontweight="bold")
save("fig8_distfit.png")
fit = pd.DataFrame(fit_rows); fit.to_csv(os.path.join(OUT, "distribution_fits.csv"), index=False)
R["fit"] = fit.round(4).to_dict("records")
R["best_fit"] = fit.loc[fit.groupby("device")["aic"].idxmin()][["device", "dist"]].set_index("device")["dist"].to_dict()

# --- Fig9 discrete distribution: slow-task counts per block of 50 tasks vs Binomial / Poisson
df["block"] = df.groupby("device").cumcount() // 50
blk = df.groupby(["device", "block"]).agg(slow=("iqr_outlier", "sum"), n=("tat", "size")).reset_index()
blk = blk[blk.n == 50]
cnt = blk["slow"].values; p_hat = cnt.sum() / (50 * len(cnt)); lam = cnt.mean()
ks = np.arange(0, cnt.max() + 2)
obs = np.array([(cnt == k).mean() for k in ks])
plt.figure(figsize=(8, 5))
plt.bar(ks - .2, obs, .4, label="Observed", color="#4c72b0")
plt.bar(ks + .2, stats.binom.pmf(ks, 50, p_hat), .4, label=f"Binomial(50, p={p_hat:.3f})", color="#dd8452")
plt.plot(ks, stats.poisson.pmf(ks, lam), "k-o", ms=4, label=f"Poisson(lambda={lam:.2f})")
plt.xlabel("IQR outliers per block of 50 tasks"); plt.ylabel("Probability")
plt.title("Fig. 9  Discrete distribution of outlier counts per block", fontweight="bold"); plt.legend()
save("fig9_discrete.png")
R["discrete"] = dict(blocks=int(len(cnt)), p_hat=float(p_hat), lam=float(lam), var=float(cnt.var(ddof=1)),
                     dispersion=float(cnt.var(ddof=1) / cnt.mean()))
gaps = df["gap_s"].astype(int).value_counts().sort_index()
R["gap_dist"] = {int(k): int(v) for k, v in gaps.items()}

# --- Fig10 gap (missed seconds) by device
gp = df.groupby(["device", "gap_s"]).size().unstack(fill_value=0).reindex(ORDER)
gp = gp.div(gp.sum(1), axis=0)
gp.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="viridis")
plt.ylabel("Proportion of tasks"); plt.xlabel("Edge server"); plt.xticks(rotation=0)
plt.legend(title="Gap since previous\ntask (s)", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.title("Fig. 10  Inter-task gap by device", fontweight="bold")
save("fig10_gap.png")

# --- statistical tests
groups = [df[df.device == d].tat.values for d in ORDER]
R["kruskal"] = dict(zip(["H", "p"], map(float, stats.kruskal(*groups))))
R["levene"] = dict(zip(["W", "p"], map(float, stats.levene(*[np.log(x) for x in groups]))))
sl = df[df.is_slow == 1]; ns = df[df.is_slow == 0]
R["gap_vs_slow"] = dict(
    mean_prev_gap_slow=float(sl.prev_gap.mean()), mean_prev_gap_normal=float(ns.prev_gap.mean()),
    mw_p=float(stats.mannwhitneyu(sl.prev_gap, ns.prev_gap)[1]))
# does the previous task being an outlier predict an outlier next?
df["prev_out"] = df.groupby("device")["iqr_outlier"].shift(1)
ct = pd.crosstab(df.prev_out, df.iqr_outlier)
R["chi2_prev_outlier"] = dict(table=ct.values.tolist(), p=float(stats.chi2_contingency(ct)[1]))
# 95% CI of mean
R["ci"] = {dv: [float(x) for x in stats.t.interval(.95, len(s) - 1, s.mean(), stats.sem(s))]
           for dv, s in zip(ORDER, groups)}
R["log_mean_sd"] = {dv: [float(np.log(s).mean()), float(np.log(s).std())] for dv, s in zip(ORDER, groups)}

# ------------------------------------------------------------------ 5. PRE-PROCESSING FOR MODEL
# NOTE: current gap_s and elapsed_s are deliberately EXCLUDED: the log timestamp is written when the
# result returns, so gap_s of the current task contains the current TAT itself (target leakage).
feat_num = ["prev_gap", "lag1", "lag2", "lag3", "diff1", "roll_mean5", "roll_std5",
            "roll_max5", "roll_mean20", "lag1_ratio", "second"]
X = df[feat_num].copy()
tr = df.split == "train"
# winsorise at train 99.5th percentile (taming the extreme 20.8 s VM spike), then log1p for skew
for c in feat_num:
    if c in ("diff1",): continue
    hi = X.loc[tr, c].quantile(.995); X[c] = X[c].clip(upper=hi)
for c in ["lag1", "lag2", "lag3", "roll_mean5", "roll_std5", "roll_max5", "roll_mean20", "lag1_ratio", "prev_gap"]:
    X[c] = np.log1p(X[c])
X["diff1"] = np.sign(X["diff1"]) * np.log1p(np.abs(X["diff1"]))
dev_oh = pd.get_dummies(df.device, prefix="dev").astype(float)
X = pd.concat([X, dev_oh], axis=1)
mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
Xs = ((X - mu) / sd).values.astype(np.float32)
names = list(X.columns)
y = df.is_slow.values
itr, iva, ite = [np.where(df.split == s)[0] for s in ("train", "val", "test")]
R["n_model_features"] = len(names)
R["feature_names"] = names


# ------------------------------------------------------------------ 6a. DEEP LEARNING (PRIMARY): MLP regression of log(TAT)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import copy
yr = df.log_tat.values; tat_s = df.tat.values
def train_mlpr(hidden, lr, alpha, epochs=80, batch=64, seed=RNG):
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam", learning_rate_init=lr,
                     alpha=alpha, batch_size=batch, random_state=seed)
    rs = np.random.RandomState(seed); h = dict(tl=[], vl=[], tr2=[], vr2=[]); best, bm, wait = 1e9, None, 0
    for ep in range(epochs):
        o = rs.permutation(itr)
        for b0 in range(0, len(o), batch):
            ix = o[b0:b0 + batch]; m.partial_fit(Xs[ix], yr[ix])
        ptr, pva = m.predict(Xs[itr]), m.predict(Xs[iva])
        h["tl"].append(mean_squared_error(yr[itr], ptr)); h["vl"].append(mean_squared_error(yr[iva], pva))
        h["tr2"].append(r2_score(yr[itr], ptr)); h["vr2"].append(r2_score(yr[iva], pva))
        if h["vl"][-1] < best - 1e-5: best, wait, bm = h["vl"][-1], 0, copy.deepcopy(m)
        else:
            wait += 1
            if wait >= 12: break
    return bm, h
rows = []
for hd in [(16,), (32, 16), (64, 32, 16)]:
    for lr_ in (1e-3, 1e-2):
        for a_ in (1e-4, 1e-2):
            m_, h_ = train_mlpr(hd, lr_, a_)
            pv = np.exp(m_.predict(Xs[iva]))
            rows.append(dict(hidden=str(hd), lr=lr_, alpha=a_, epochs=len(h_["tl"]), val_mse_log=min(h_["vl"]),
                             val_mae_s=mean_absolute_error(tat_s[iva], pv)))
gr = pd.DataFrame(rows).sort_values("val_mse_log"); gr.to_csv(os.path.join(OUT, "hyperparameter_search_regression.csv"), index=False)
R["grid_reg"] = gr.round(5).to_dict("records")
bh = gr.iloc[0]; hdr = eval(bh.hidden)
mr, hr = train_mlpr(hdr, float(bh.lr), float(bh.alpha))
R["best_hp_reg"] = dict(hidden=hdr, lr=float(bh.lr), alpha=float(bh.alpha), epochs_run=len(hr["tl"]), best_epoch=int(np.argmin(hr["vl"]) + 1))
R["n_params_reg"] = int(sum(w.size for w in mr.coefs_) + sum(b_.size for b_ in mr.intercepts_))
# baselines (all fitted on train only)
med = df[df.split == "train"].groupby("device")["tat"].median()
base = {
 "Device median (train)": df.device.map(med).values[ite],
 "Persistence (previous TAT)": df.lag1.values[ite],
 "Rolling mean (last 5)": df.roll_mean5.values[ite],
 "Ridge regression": np.exp(Ridge(alpha=1.0).fit(Xs[itr], yr[itr]).predict(Xs[ite])),
 "MLP (proposed)": np.exp(mr.predict(Xs[ite])),
}
yt = tat_s[ite]; dte = df.device.values[ite]
def reg_metrics(p):
    d = dict(mae=mean_absolute_error(yt, p), rmse=float(np.sqrt(mean_squared_error(yt, p))),
             medae=float(np.median(np.abs(yt - p))), r2_log=r2_score(np.log(yt), np.log(p)), r2=r2_score(yt, p))
    d["within_device_r2_log"] = float(np.mean([r2_score(np.log(yt[dte == v]), np.log(p[dte == v])) for v in ORDER]))
    return d
R["reg_metrics"] = {k: reg_metrics(v) for k, v in base.items()}
R["reg_per_device_mae"] = {v: float(mean_absolute_error(yt[dte == v], base["MLP (proposed)"][dte == v])) for v in ORDER}
R["reg_per_device_mean"] = {v: float(yt[dte == v].mean()) for v in ORDER}
R["reg_per_device_rel"] = {v: float(np.median(np.abs(yt[dte == v] - base["MLP (proposed)"][dte == v]) / yt[dte == v])) for v in ORDER}
pm = base["MLP (proposed)"]; res = yt - pm
R["res_stats"] = dict(mean=float(res.mean()), median=float(np.median(res)), std=float(res.std()),
                      p5=float(np.percentile(res, 5)), p95=float(np.percentile(res, 95)))
ep = np.arange(1, len(hr["tl"]) + 1)
plt.figure(figsize=(8, 5)); plt.plot(ep, hr["tl"], label="Train loss"); plt.plot(ep, hr["vl"], label="Validation loss")
plt.axvline(R["best_hp_reg"]["best_epoch"], c="grey", ls=":", label="Best epoch"); plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("MSE of log(TAT) (log scale)"); plt.legend(); plt.title("Fig. 11  MLP regression: loss vs epoch", fontweight="bold")
save("fig11_loss.png")
plt.figure(figsize=(8, 5)); plt.plot(ep, hr["tr2"], label="Train R$^2$"); plt.plot(ep, hr["vr2"], label="Validation R$^2$")
plt.xlabel("Epoch"); plt.ylabel("R$^2$ on log(TAT)"); plt.legend(); plt.title("Fig. 12  MLP regression: R$^2$ (accuracy) vs epoch", fontweight="bold")
save("fig12_acc.png")
plt.figure(figsize=(7, 6))
for v in ORDER:
    mk = dte == v; plt.scatter(yt[mk], pm[mk], s=10, alpha=.55, color=PAL[v], label=v)
lim = [yt.min() * .9, yt.max() * 1.1]; plt.plot(lim, lim, "k--", lw=1)
plt.xscale("log"); plt.yscale("log"); plt.xlabel("Actual TAT (s)"); plt.ylabel("Predicted TAT (s)"); plt.legend()
plt.title("Fig. 13  Predicted vs actual TAT (test set)", fontweight="bold")
save("fig13_predactual.png")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
sns.histplot(res, bins=60, kde=True, ax=ax[0], color="#4c72b0"); ax[0].set_xlim(-.5, .5); ax[0].axvline(0, c="k", ls="--")
ax[0].set_xlabel("Residual = actual - predicted (s)"); ax[0].set_title("Overall (x-axis clipped to +/-0.5 s)")
for v in ORDER:
    sns.kdeplot(res[dte == v] / yt[dte == v], ax=ax[1], color=PAL[v], label=v, clip=(-1, 1.5))
ax[1].set_xlabel("Relative error (actual - predicted) / actual"); ax[1].legend(); ax[1].set_title("Relative error by device")
fig.suptitle("Fig. 14  Error distribution of the MLP regressor (test set)", fontweight="bold")
save("fig14_errors.png")
pir = permutation_importance(mr, Xs[np.r_[iva, ite]], yr[np.r_[iva, ite]], scoring="neg_mean_absolute_error", n_repeats=15, random_state=RNG)
imr = pd.DataFrame(dict(feature=names, imp=pir.importances_mean, sd=pir.importances_std)).sort_values("imp")
imr.to_csv(os.path.join(OUT, "feature_importance_regression.csv"), index=False)
R["importance_reg"] = imr.sort_values("imp", ascending=False).head(8).round(4).to_dict("records")
plt.figure(figsize=(8, 6)); plt.barh(imr.feature, imr.imp, xerr=imr.sd, color="#55a868")
plt.xlabel("Increase in MAE of log(TAT) when feature is shuffled"); plt.title("Fig. 15  Permutation feature importance (MLP regressor)", fontweight="bold")
save("fig15_importance.png")
# feature-group ablation with Ridge (fast, comparable): device identity only vs history only vs all
def ridge_mae(cols):
    ix = [names.index(c) for c in cols]; mm = Ridge(alpha=1.0).fit(Xs[itr][:, ix], yr[itr])
    return float(mean_absolute_error(yt, np.exp(mm.predict(Xs[ite][:, ix]))))
devcols = [c for c in names if c.startswith("dev_")]
R["ablation_mae"] = {"Device identity only": ridge_mae(devcols), "History features only (no device)": ridge_mae([c for c in names if c not in devcols]),
                     "All features": ridge_mae(names)}

# ------------------------------------------------------------------ 6b. DEEP LEARNING (SECONDARY): MLP classifier for 'slow task'
def train_mlp(hidden, lr, alpha, epochs=60, batch=64, seed=RNG, verbose=False):
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        learning_rate_init=lr, alpha=alpha, batch_size=batch, random_state=seed)
    rs = np.random.RandomState(seed); hist = dict(tl=[], vl=[], ta=[], va=[], vauc=[])
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        order = rs.permutation(itr)
        for b in range(0, len(order), batch):
            ix = order[b:b + batch]
            clf.partial_fit(Xs[ix], y[ix], classes=[0, 1])
        ptr, pva = clf.predict_proba(Xs[itr])[:, 1], clf.predict_proba(Xs[iva])[:, 1]
        hist["tl"].append(log_loss(y[itr], ptr)); hist["vl"].append(log_loss(y[iva], pva))
        hist["ta"].append(accuracy_score(y[itr], ptr > .5)); hist["va"].append(accuracy_score(y[iva], pva > .5))
        hist["vauc"].append(roc_auc_score(y[iva], pva))
        if hist["vl"][-1] < best - 1e-4:
            best, wait = hist["vl"][-1], 0
            import copy; best_state = copy.deepcopy(clf)
        else:
            wait += 1
            if wait >= 12: break          # early stopping on validation loss
    return best_state, hist

# basic hyper-parameter grid search on validation AUC
grid = [(h, lr, a) for h in [(16,), (32, 16), (64, 32, 16)] for lr in (1e-3, 1e-2) for a in (1e-4, 1e-2)]
rows = []
for h, lr, a in grid:
    m, hi = train_mlp(h, lr, a)
    rows.append(dict(hidden=str(h), lr=lr, alpha=a, epochs=len(hi["tl"]),
                     val_loss=min(hi["vl"]), val_auc=roc_auc_score(y[iva], m.predict_proba(Xs[iva])[:, 1])))
gs = pd.DataFrame(rows).sort_values("val_auc", ascending=False); gs.to_csv(os.path.join(OUT, "hyperparameter_search.csv"), index=False)
R["grid"] = gs.round(4).to_dict("records")
b = gs.iloc[0]
best_h = eval(b.hidden); best_lr, best_a = float(b.lr), float(b.alpha)
model, hist = train_mlp(best_h, best_lr, best_a)
R["best_hp"] = dict(hidden=best_h, lr=best_lr, alpha=best_a, epochs_run=len(hist["tl"]),
                    best_epoch=int(np.argmin(hist["vl"]) + 1))
n_params = sum(w.size for w in model.coefs_) + sum(b_.size for b_ in model.intercepts_)
R["n_params"] = int(n_params)

# baselines
lr_m = LogisticRegression(max_iter=2000, class_weight=None).fit(Xs[itr], y[itr])
rf = RandomForestClassifier(300, random_state=RNG, n_jobs=-1, min_samples_leaf=3).fit(Xs[itr], y[itr])

# threshold chosen on validation (maximise F1) -> applied to test
pva = model.predict_proba(Xs[iva])[:, 1]
pr, rc, th = precision_recall_curve(y[iva], pva)
f1s = 2 * pr * rc / (pr + rc + 1e-9); thr_opt = float(th[np.argmax(f1s[:-1])])
pte = model.predict_proba(Xs[ite])[:, 1]; yte = y[ite]; pred = (pte > thr_opt).astype(int)
def metrics(p, thr_):
    return dict(accuracy=accuracy_score(yte, p > thr_), precision=precision_score(yte, p > thr_, zero_division=0),
                recall=recall_score(yte, p > thr_), f1=f1_score(yte, p > thr_),
                auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p))
R["thr_opt"] = thr_opt
R["metrics"] = {"MLP": metrics(pte, thr_opt),
                "Logistic Regression": metrics(lr_m.predict_proba(Xs[ite])[:, 1], .5),
                "Random Forest": metrics(rf.predict_proba(Xs[ite])[:, 1], .5)}
R["majority_acc"] = float(1 - yte.mean())
R["test_pos"] = int(yte.sum()); R["test_n"] = int(len(yte))
R["cm"] = confusion_matrix(yte, pred).tolist()
R["per_device_auc"] = {}
dte = df.device.values[ite]
for dv in ORDER:
    mk = dte == dv
    R["per_device_auc"][dv] = float(roc_auc_score(yte[mk], pte[mk])) if yte[mk].min() != yte[mk].max() else None

# ------------------------------------------------------------------ 7. RESULT VISUALS
ep = np.arange(1, len(hist["tl"]) + 1)
plt.figure(figsize=(8, 5)); plt.plot(ep, hist["tl"], label="Train loss"); plt.plot(ep, hist["vl"], label="Validation loss")
plt.axvline(R["best_hp"]["best_epoch"], c="grey", ls=":", label="Best epoch")
plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy"); plt.legend(); plt.title("Fig. 16  Classifier: loss vs epoch", fontweight="bold")
save("fig16_cls_loss.png")
plt.figure(figsize=(8, 5)); plt.plot(ep, hist["ta"], label="Train accuracy"); plt.plot(ep, hist["va"], label="Validation accuracy")
plt.plot(ep, hist["vauc"], label="Validation AUC", ls="--")
plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.title("Fig. 17  Classifier: accuracy / AUC vs epoch", fontweight="bold")
save("fig17_cls_acc.png")
cmx = np.array(R["cm"])
plt.figure(figsize=(5.5, 4.8))
sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Slow"], yticklabels=["Normal", "Slow"], cbar=False, annot_kws={"size": 14})
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Fig. 18  Confusion matrix (test, thr={thr_opt:.2f})", fontweight="bold")
save("fig18_cm.png")
plt.figure(figsize=(6.5, 5.5))
for nm, p, c in [("MLP", pte, "#d62728"), ("Logistic Regression", lr_m.predict_proba(Xs[ite])[:, 1], "#1f77b4"),
                 ("Random Forest", rf.predict_proba(Xs[ite])[:, 1], "#2ca02c")]:
    f_, t_, _ = roc_curve(yte, p); plt.plot(f_, t_, label=f"{nm} (AUC={roc_auc_score(yte, p):.3f})", color=c)
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.legend(loc="lower right"); plt.title("Fig. 19  ROC curves (test set)", fontweight="bold")
save("fig19_roc.png")

# permutation feature importance on validation + test (AUC drop)
pi = permutation_importance(model, Xs[np.r_[iva, ite]], y[np.r_[iva, ite]], scoring="roc_auc",
                            n_repeats=15, random_state=RNG)
imp = pd.DataFrame(dict(feature=names, imp=pi.importances_mean, sd=pi.importances_std)).sort_values("imp")
imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
R["importance"] = imp.sort_values("imp", ascending=False).head(8).round(4).to_dict("records")
plt.figure(figsize=(8, 6)); plt.barh(imp.feature, imp.imp, xerr=imp.sd, color="#4c72b0")
plt.xlabel("Decrease in ROC-AUC when feature is shuffled"); plt.title("Fig. 20  Permutation feature importance (classifier)", fontweight="bold")
save("fig20_cls_importance.png")

# score distribution
plt.figure(figsize=(7.5, 4.8))
sns.histplot(x=pte, hue=np.where(yte == 1, "Slow", "Normal"), bins=30, stat="probability", common_norm=False, element="step")
plt.axvline(thr_opt, c="k", ls="--"); plt.xlabel("Predicted probability of slow task"); plt.title("Fig. 21  Predicted-probability distribution by true class", fontweight="bold")
save("fig21_scores.png")

# ------------------------------------------------------------------ 8. INTERACTIVE (stage 7 of visualization): self-contained HTML
pts = {dv: [round(v, 4) for v in df[df.device == dv].tat.tolist()] for dv in ORDER}
html = """<!doctype html><meta charset=utf-8><title>Interactive TAT explorer</title>
<style>body{font-family:system-ui;margin:20px}canvas{border:1px solid #ccc;width:100%;max-width:1000px}label{margin-right:14px}</style>
<h3>Interactive TAT explorer (toggle devices, change scale, move the window)</h3>
<div id=ctl></div><div>Window start: <input type=range id=s min=0 max=900 value=0 style=width:300px> size:
<select id=w><option>100<option>250<option selected>1000</select> <label><input type=checkbox id=lg checked> log scale</label></div>
<canvas id=c width=1000 height=420></canvas><div id=tip></div>
<script>const D=__D__,C=__C__;const ctl=document.getElementById('ctl'),cv=document.getElementById('c'),g=cv.getContext('2d');
const on={};for(const k in D){on[k]=true;ctl.insertAdjacentHTML('beforeend',`<label style="color:${C[k]}"><input type=checkbox checked onchange="on['${k}']=this.checked;draw()"> ${k}</label>`)}
['s','w','lg'].forEach(i=>document.getElementById(i).oninput=draw);
function draw(){const st=+s.value,w=+document.getElementById('w').value,L=lg.checked;g.clearRect(0,0,1000,420);
let lo=1e9,hi=0;for(const k in D)if(on[k])D[k].slice(st,st+w).forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v)});
if(hi==0)return;const f=v=>L?Math.log(v):v,a=f(lo),b=f(hi),Y=v=>400-(f(v)-a)/(b-a||1)*380;
g.strokeStyle='#ddd';g.fillStyle='#555';for(let i=0;i<=4;i++){const y=20+i*95;g.beginPath();g.moveTo(50,y);g.lineTo(1000,y);g.stroke();
const val=L?Math.exp(b-(b-a)*i/4):hi-(hi-lo)*i/4;g.fillText(val.toFixed(3)+'s',2,y+4)}
for(const k in D)if(on[k]){g.strokeStyle=C[k];g.beginPath();D[k].slice(st,st+w).forEach((v,i)=>{const x=50+i/Math.max(1,Math.min(w,D[k].length-st)-1)*940;i?g.lineTo(x,Y(v)):g.moveTo(x,Y(v))});g.stroke()}
g.fillStyle='#000';g.fillText('task index '+st+' - '+(st+w),450,415)}draw();</script>""".replace("__D__", json.dumps(pts)).replace("__C__", json.dumps(PAL))
open(os.path.join(OUT, "interactive_tat.html"), "w").write(html)

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["clean_shape", "n_fields", "thresholds", "split_counts", "slow_rate", "best_hp", "metrics", "cm", "per_device_auc", "best_fit", "discrete", "kruskal", "importance", "lag1_spearman", "majority_acc"]}, indent=1, default=str))
