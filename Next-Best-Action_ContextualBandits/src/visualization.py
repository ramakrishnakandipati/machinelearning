"""300-DPI Seaborn/Matplotlib figures for the NBA study."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ACTION_SHORT, ARCHETYPES, CHANNELS, PLOTS_DIR, TIERS

DPI = 300
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
PALETTE = sns.color_palette("tab10")


def _save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_data_distributions(df):
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    sns.histplot(df.lead_time, bins=40, ax=ax[0, 0], color=PALETTE[0]).set(title="Lead time (days)")
    sns.countplot(x="los", data=df, ax=ax[0, 1], color=PALETTE[1]).set(title="Length of stay (nights)")
    sns.histplot(np.log10(df.hist_spend), bins=40, ax=ax[0, 2], color=PALETTE[2]).set(title="Historical spend (log10 $)")
    sns.countplot(x="loyalty_tier", data=df, order=TIERS, ax=ax[1, 0], hue="booking_channel",
                  hue_order=CHANNELS).set(title="Loyalty tier x booking channel")
    sns.countplot(x="guest_archetype", data=df, order=ARCHETYPES, ax=ax[1, 1], color=PALETTE[4]).set(title="Guest archetype")
    sns.histplot(df.ancillary_rate, bins=30, ax=ax[1, 2], color=PALETTE[5]).set(title="Past ancillary attachment rate")

    for a_, (col, order, title) in zip(ax[2], [("guest_archetype", ARCHETYPES, "archetype"),
                                               ("loyalty_tier", TIERS, "loyalty tier"),
                                               ("booking_channel", CHANNELS, "channel")]):
        m = np.stack([df[df[col] == g][[f"log_p_a{k}" for k in range(6)]].mean().values for g in order])
        sns.heatmap(pd.DataFrame(m, index=order, columns=ACTION_SHORT), annot=True, fmt=".2f", cmap="rocket_r",
                    ax=a_, cbar=False)
        a_.set_title(f"Logging-policy propensity by {title}\n(confounded: depends on context)")
        a_.tick_params(axis="x", rotation=30)
    fig.suptitle("Context feature distributions & logging-policy action propensity bias", fontsize=16, y=1.0)
    fig.tight_layout()
    return _save(fig, "01_data_distributions.png")


def plot_correlations_importance(df, X, feature_names, importance):
    names = list(feature_names)
    Xd = pd.DataFrame(X, columns=names)
    for k in range(6):
        Xd[f"E[margin|a={k}]"] = df[f"true_r_a{k}"].values
    corr = Xd.corr().loc[names, [f"E[margin|a={k}]" for k in range(6)]]
    corr.columns = ACTION_SHORT
    fig, ax = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [1.1, 1]})
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax[0])
    ax[0].set_title("Context feature vs true expected net margin per action (Pearson r)")
    imp = pd.Series(importance).sort_values()
    imp = imp.tail(20)
    ax[1].barh(imp.index, imp.values / imp.values.sum(), color=PALETTE[0])
    ax[1].set_title("XGBoost reward-model feature importance (gain share, top 20)")
    ax[1].set_xlabel("share of total gain")
    fig.tight_layout()
    return _save(fig, "02_feature_correlations_importance.png")


def plot_training_curves(metrics_path):
    rows = [json.loads(l) for l in open(metrics_path, encoding="utf-8") if l.strip()]
    ep = pd.DataFrame([r for r in rows if r.get("event") == "epoch"])
    if ep.empty:
        return None
    if "tag" not in ep:
        ep["tag"] = ""
    ep["tag"] = ep["tag"].fillna("")
    models = list(ep["model"].unique())
    n = len(models)
    cols = 4
    nr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(5 * cols, 3.8 * nr), squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        d = ep[ep.model == m]
        tags = [t for t in d.tag.unique() if t in ("", "policy", "member0", "reward_model_fold0")]
        for ti, tg in enumerate(tags):
            g = d[d.tag == tg].copy()
            g["fit_id"] = (g["epoch"].diff().fillna(1) < 0).cumsum()
            g = g[g.fit_id == 0]  # first (offline) fit only
            lab = f" [{tg}]" if tg else ""
            ax.plot(g.epoch, g.train_loss, label="train" + lab, color=PALETTE[ti * 2 % 10])
            if "val_loss" in g and g.val_loss.notna().any():
                ax.plot(g.epoch, g.val_loss, ls="--", label="validation" + lab, color=PALETTE[(ti * 2 + 1) % 10])
        unit = "boosting round" if (d.get("kind") == "boost_round").any() else "epoch"
        extra = " (policy loss = -DR value)" if m == "DR-PolicyGradient" else (" (RMSE, scaled $)" if "XGBoost" in m else " (scaled MSE)")
        ax.set_title(m + extra, fontsize=9)
        ax.set_xlabel(unit)
        ax.set_ylabel("loss")
        ax.legend(fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("GPU training loss & convergence curves", fontsize=15)
    fig.tight_layout()
    return _save(fig, "03_training_loss_curves.png")


def plot_model_comparison(results):
    names = [k for k, v in results.items() if v["type"] == "bandit"]
    base = [k for k, v in results.items() if v["type"] == "baseline"]
    est = ["DM", "IPS", "SNIPS", "DR"]
    fig, ax = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
    w = 0.16
    x = np.arange(len(names))
    for i, e in enumerate(est):
        vals = np.array([results[n][e] for n in names])
        lo = np.array([results[n]["ci95"][e][0] for n in names])
        hi = np.array([results[n]["ci95"][e][1] for n in names])
        ax[0].bar(x + (i - 2) * w, vals, w, yerr=[vals - lo, hi - vals], label=e, color=PALETTE[i], capsize=2)
    ax[0].bar(x + 2 * w, [results[n]["TRUE"] for n in names], w, label="True value (simulator)", color="black")
    lo_b = results["Logging policy"]["DR"]
    ax[0].axhline(lo_b, ls="--", color="gray", label="Logging policy (DR)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=35, ha="right")
    ax[0].set_ylabel("Estimated policy value: net margin per session ($)")
    lo_y = min(results[n][e] for n in names for e in est + ["TRUE"])
    ax[0].set_ylim(max(0, lo_y * 0.85), None)
    ax[0].set_title("Off-policy estimators per bandit (error bars: 95% bootstrap CI)")
    ax[0].legend(ncol=3, fontsize=8)
    allp = {k: results[k]["TRUE"] for k in results}
    ser = pd.Series(allp).sort_values()
    colors = ["#999999" if results[k]["type"] == "baseline" else PALETTE[0] for k in ser.index]
    ax[1].barh(ser.index, ser.values, color=colors)
    for i, v in enumerate(ser.values):
        ax[1].text(v, i, f" ${v:,.0f}", va="center", fontsize=8)
    ax[1].set_xlim(ser.min() * 0.8, ser.max() * 1.1)
    ax[1].set_title("True net contribution margin / session (grey = baselines)")
    fig.suptitle("Off-policy evaluation: DM, IPS, SNIPS, DR and net margin", fontsize=16)
    fig.tight_layout()
    return _save(fig, "04_model_comparison_ope.png")


def plot_replay(replays):
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    first = next(iter(replays.values()))
    n = len(first["cum_regret"])
    t = np.arange(1, n + 1)
    for i, (name, rp) in enumerate(replays.items()):
        ax[0].plot(t, rp["cum_regret"], label=name, color=sns.color_palette("tab20", len(replays))[i])
    for k, c in zip(["Logging policy", "Uniform random", "Always control"], ["black", "gray", "brown"]):
        ax[0].plot(t, first["ref_cum_regret"][k], ls="--", color=c, label=k)
    ax[0].set(title="Cumulative regret vs oracle (online replay)", xlabel="sessions replayed", ylabel="cumulative regret ($)")
    ax[0].legend(fontsize=7, ncol=2)

    log = first["ref_cum_margin"]["Logging policy"]
    for i, (name, rp) in enumerate(replays.items()):
        ax[1].plot(t, rp["cum_margin"] - log, label=name, color=sns.color_palette("tab20", len(replays))[i])
    ax[1].axhline(0, color="black", ls="--")
    ax[1].set(title="Cumulative net-margin lift vs logging policy", xlabel="sessions replayed", ylabel="cumulative lift ($)")

    rnd = np.array_split(np.arange(n), 20)
    for i, (name, rp) in enumerate(replays.items()):
        cm = rp["cum_margin"]
        rm = [(cm[c[-1]] - (cm[c[0] - 1] if c[0] > 0 else 0)) / len(c) for c in rnd]
        ax[2].plot(np.arange(1, 21), rm, marker="o", ms=3, label=name, color=sns.color_palette("tab20", len(replays))[i])
    ax[2].axhline(first["ref_cum_margin"]["Logging policy"][-1] / n, color="black", ls="--", label="Logging policy")
    ax[2].axhline(first["ref_cum_margin"]["Oracle (true best)"][-1] / n, color="green", ls=":", label="Oracle")
    ax[2].set(title="Per-round mean net margin as policies learn", xlabel="replay round", ylabel="$ / session")
    ax[2].legend(fontsize=7, ncol=2)
    fig.suptitle("Online policy replay: cumulative regret & reward lift over time", fontsize=16)
    fig.tight_layout()
    return _save(fig, "05_cumulative_regret_replay.png")


def plot_segment_heatmap(df, X_actions, best_name, oracle_actions):
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    seg_cols = [("guest_archetype", ARCHETYPES), ("loyalty_tier", TIERS), ("booking_channel", CHANNELS)]
    for j, (col, order) in enumerate(seg_cols):
        m = np.stack([np.bincount(X_actions[(df[col] == g).values], minlength=6) / (df[col] == g).sum() for g in order])
        sns.heatmap(pd.DataFrame(m, index=order, columns=ACTION_SHORT), annot=True, fmt=".2f", cmap="YlGnBu", vmin=0,
                    vmax=1, ax=ax[0, j], cbar=False)
        ax[0, j].set_title(f"{best_name}: action share by {col.replace('_', ' ')}")
        ax[0, j].tick_params(axis="x", rotation=30)
        mo = np.stack([np.bincount(oracle_actions[(df[col] == g).values], minlength=6) / (df[col] == g).sum() for g in order])
        sns.heatmap(pd.DataFrame(mo, index=order, columns=ACTION_SHORT), annot=True, fmt=".2f", cmap="YlOrBr", vmin=0,
                    vmax=1, ax=ax[1, j], cbar=False)
        ax[1, j].set_title(f"Oracle: action share by {col.replace('_', ' ')}")
        ax[1, j].tick_params(axis="x", rotation=30)
    # archetype x tier -> dominant action of the learned policy
    seg = df.assign(act=X_actions)
    dom = seg.groupby(["guest_archetype", "loyalty_tier"])["act"].agg(lambda s: np.bincount(s, minlength=6).argmax())
    grid = dom.unstack().reindex(index=ARCHETYPES, columns=TIERS)
    sns.heatmap(grid.astype(float), annot=grid.map(lambda v: ACTION_SHORT[int(v)]).values, fmt="", cmap="tab10",
                vmin=0, vmax=9, cbar=False, ax=ax[0, 3], annot_kws={"size": 8})
    ax[0, 3].set_title(f"{best_name}: dominant NBA per archetype x tier")
    dom_o = seg.assign(act=oracle_actions).groupby(["guest_archetype", "loyalty_tier"])["act"].agg(
        lambda s: np.bincount(s, minlength=6).argmax())
    grid_o = dom_o.unstack().reindex(index=ARCHETYPES, columns=TIERS)
    sns.heatmap(grid_o.astype(float), annot=grid_o.map(lambda v: ACTION_SHORT[int(v)]).values, fmt="", cmap="tab10",
                vmin=0, vmax=9, cbar=False, ax=ax[1, 3], annot_kws={"size": 8})
    ax[1, 3].set_title("Oracle: dominant NBA per archetype x tier")
    fig.suptitle("Learned Next-Best-Action allocation across guest segments", fontsize=16)
    fig.tight_layout()
    return _save(fig, "06_segment_action_heatmap.png")
