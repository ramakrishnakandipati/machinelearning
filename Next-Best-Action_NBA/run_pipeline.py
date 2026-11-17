"""End-to-end Retail NBA pipeline: synthetic data -> diagnostics -> uplift models
-> guardrailed policy -> counterfactual validation -> saved artifacts.

Usage:  python run_pipeline.py [--n 30000] [--seed 42]
Populates every folder under ``outputs/`` with real PNGs and model files.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from nba.artifact_manager import ArtifactManager
from nba.config import ACTION_COSTS, DEFAULT_LAMBDA, SEED, UNSUB_RATE_CAP
from nba import data_plots, eval_plots
from nba.evaluation import (
    deterministic_probs, evaluate_policy, fit_naive_propensity_model, lambda_sweep, legacy_policy_probs,
    naive_propensity_actions, select_lambda, unconstrained_uplift_actions,
)
from nba.policy import decide
from nba.schemas import validate_log_frame
from nba.synthetic_data import make_dataset
from nba.uplift_model import fit_uplift_bundle, save_bundle, update_manifest
from nba.config import FEATURE_COLUMNS


def main(n_customers: int = 30000, seed: int = SEED) -> None:
    t0 = time.time()
    am = ArtifactManager()
    print(f"[1/6] Artifact directories ready under {am.root}")

    # ---- Module 2: data, schema validation, diagnostics -------------------
    ds = make_dataset(n_customers, seed)
    n_ok = validate_log_frame(ds.log.assign(**ds.x.reset_index(drop=True)), FEATURE_COLUMNS)
    print(f"[2/6] {len(ds.log):,} logged decisions ({n_ok} schema-validated); mature rewards: "
          f"{ds.log['reward_mature'].mean():.1%}; cold-start: {ds.x['is_cold_start'].mean():.1%}; "
          f"train/holdout: {len(ds.train_idx):,}/{len(ds.test_idx):,}")
    m_idx = np.sort(np.concatenate([ds.train_idx, ds.test_idx]))
    xm, lm = ds.x.iloc[m_idx].reset_index(drop=True), ds.log.iloc[m_idx].reset_index(drop=True)
    is_test = np.isin(m_idx, ds.test_idx)
    action_m = lm["action_id"].to_numpy()
    data_plots.plot_propensity_overlap(am, xm, action_m)
    data_plots.plot_correlations_and_drift(am, xm, action_m, lm["propensity_score"].to_numpy(), is_test)
    tier = ds.raw_features["discount_tier"].fillna(1).iloc[m_idx].to_numpy()
    data_plots.plot_margin_and_returns(am, lm, tier)
    print("      data plots saved")

    # ---- Module 3: causal uplift models (mature rewards only) ------------
    x_tr, l_tr = ds.x.iloc[ds.train_idx].reset_index(drop=True), ds.log.iloc[ds.train_idx].reset_index(drop=True)
    x_te, l_te = ds.x.iloc[ds.test_idx].reset_index(drop=True), ds.log.iloc[ds.test_idx].reset_index(drop=True)
    bundle = fit_uplift_bundle(x_tr, l_tr, ds.transformer, seed)
    save_bundle(bundle, am, DEFAULT_LAMBDA)
    print(f"[3/6] models saved; CV metrics: { {k: round(v, 3) for k, v in bundle.cv_metrics.items()} }")

    # ---- Module 5: counterfactual validation on the holdout --------------
    tm, tu = bundle.predict_tau(x_te.to_numpy(dtype=np.float32))
    truth = l_te[[f"truth_tau_margin_a{a}" for a in (1, 2, 3, 4)]].to_numpy()
    corr = [float(np.corrcoef(tm[:, j], truth[:, j])[0, 1]) for j in range(4)]
    _, qini = eval_plots.plot_qini(am, l_te, tm)
    _, calib = eval_plots.plot_cate_calibration(am, l_te, tm)
    eval_plots.plot_nuisance_diagnostics(am, x_tr, l_tr, x_te, l_te, seed)
    print(f"[4/6] validation plots saved; corr(pred CATE, true CATE) per action = {np.round(corr, 2).tolist()}")

    sweep = lambda_sweep(bundle, x_te, l_te)
    lam_star = select_lambda(sweep)
    a_log = l_te["action_id"].to_numpy()
    nba = decide(bundle, x_te, l_te["promos_last_7d"].to_numpy(), lam=lam_star)
    naive_model = fit_naive_propensity_model(x_tr, l_tr["purchased_flag"].to_numpy())
    naive = naive_propensity_actions(naive_model, x_te)
    uncon = unconstrained_uplift_actions(bundle, x_te)
    reports = [
        evaluate_policy("(a) Historical\nRule-Based CRM", l_te, legacy_policy_probs(x_te, a_log)),
        evaluate_policy("(b) Naive Propensity\nTargeting", l_te, deterministic_probs(naive, a_log), naive),
        evaluate_policy("(c) Unconstrained\nUplift Policy", l_te, deterministic_probs(uncon, a_log), uncon),
        evaluate_policy("(d) Guardrailed\nMargin-Aware NBA", l_te, deterministic_probs(nba.actions, a_log), nba.actions),
    ]
    eval_plots.plot_policy_comparison(am, reports)
    eval_plots.plot_allocation_by_segment(am, x_te, nba.actions)
    eval_plots.plot_pareto(am, sweep, lam_star)
    print("[5/6] comparison plots saved")
    print(f"      selected lambda = {lam_star:g} (smallest meeting the {UNSUB_RATE_CAP:.1%} unsubscribe cap)")
    for r in reports:
        orc = f" | oracle incremental ${r.oracle_incremental:.2f}" if r.oracle_incremental is not None else ""
        print(f"      {r.name.replace(chr(10), ' '):45s} ${r.value:7.2f}  [{r.ci_low:.2f}, {r.ci_high:.2f}]  ESS={r.ess:,.0f}{orc}")
    print(sweep.round(4).to_string(index=False))

    # ---- persist final config + serving snapshot ---------------------------
    update_manifest(
        am, risk_aversion_lambda=lam_star,
        holdout_metrics={
            **qini, **calib, "cate_truth_corr": dict(zip(["a1", "a2", "a3", "a4"], corr)),
            "policy_snips_net_margin": {r.name.replace("\n", " "): r.value for r in reports},
            "lambda_sweep": sweep.round(5).to_dict(orient="records"),
        },
    )
    snap = ds.raw_features.iloc[:5000].copy()
    snap.insert(0, "customer_id", ds.log["customer_id"].iloc[:5000].to_numpy())
    snap.to_csv(am.dir("saved_models") / "serving_feature_snapshot.csv", index=False)
    print(f"[6/6] done in {time.time() - t0:.0f}s. Outputs in {am.root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="number of synthetic customers")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    main(args.n, args.seed)
