"""End-to-end NBA pipeline: data -> train 10 bandits -> OPE -> online replay -> plots -> screenshots.

Usage:  python run_pipeline.py [--n 30000] [--quick] [--rounds 20] [--skip-screenshots]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src import visualization as viz
from src.config import DATA_DIR, DEVICE, LOGS_DIR, MODELS_DIR, N_ACTIONS, SEED
from src.data_generator import generate_dataset, logging_prob_matrix, true_reward_matrix
from src.evaluation import DMRewardModel, replay, run_ope
from src.models import model_zoo
from src.preprocessing import feature_names, fit_preprocessor, save_preprocessor, transform
from src.telemetry import log_jsonl, training_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--quick", action="store_true", help="lighter models / fewer epochs (smoke run)")
    ap.add_argument("--rounds", type=int, default=20, help="online replay rounds")
    ap.add_argument("--skip-screenshots", action="store_true")
    args = ap.parse_args()
    mode = "quick" if args.quick else "full"
    t_start = time.time()

    metrics_path = LOGS_DIR / "training_metrics.jsonl"
    metrics_path.write_text("")  # fresh telemetry for this run
    tlog = training_logger(metrics_path)
    tlog({"event": "pipeline_start", "device": str(DEVICE), "n": args.n, "mode": mode})
    print(f"[1/7] device={DEVICE}  generating {args.n} sessions")

    # ------------------------------------------------------------------ data
    df = generate_dataset(args.n, SEED)
    df.to_csv(DATA_DIR / "guest_sessions.csv", index=False)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(df))
    n_tr = int(0.7 * len(df))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    dtr, dte = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    pre = fit_preprocessor(dtr)
    save_preprocessor(pre, MODELS_DIR / "preprocessor.joblib")
    fnames = feature_names(pre)
    Xtr, Xte, Xall = transform(pre, dtr), transform(pre, dte), transform(pre, df)
    a_tr, r_tr, p_tr = dtr.action.values, dtr.reward.values, dtr.propensity.values
    a_te, r_te, p_te = dte.action.values, dte.reward.values, dte.propensity.values
    T_te, L_te = true_reward_matrix(dte), logging_prob_matrix(dte)
    print(f"      train={len(dtr)} test={len(dte)} features={Xtr.shape[1]} min propensity={df.propensity.min():.3f}")

    # --------------------------------------------------------------- training
    print("[2/7] training bandits")
    zoo = model_zoo(Xtr.shape[1], N_ACTIONS, mode, logger=tlog)
    models, fit_seconds = {}, {}
    for name, factory in zoo.items():
        t0 = time.time()
        m = factory(SEED)
        m.feature_names = fnames
        m.fit(Xtr, a_tr, r_tr, p_tr)
        m.save(MODELS_DIR)
        models[name] = m
        fit_seconds[name] = time.time() - t0
        print(f"      {name:20s} fit+save {fit_seconds[name]:6.1f}s")

    # -------------------------------------------------------------------- OPE
    print("[3/7] off-policy evaluation")
    dm = DMRewardModel(N_ACTIONS, SEED).fit(Xtr, a_tr, r_tr)
    results = run_ope(models, Xte, a_te, r_te, p_te, T_te, L_te, dm, audit_path=LOGS_DIR / "ope_evaluation_audit.json",
                      meta={"mode": mode, "n_train": int(len(dtr)), "fit_seconds": fit_seconds, "device": str(DEVICE)})
    table = pd.DataFrame({k: {e: v[e] for e in ("DM", "IPS", "SNIPS", "DR", "TRUE")} for k, v in results.items()}).T
    print(table.round(1).to_string())

    # ----------------------------------------------------------------- replay
    print("[4/7] online replay")
    rzoo = model_zoo(Xtr.shape[1], N_ACTIONS, "replay", logger=None)
    T_all, a_all, r_all, p_all = true_reward_matrix(df), df.action.values, df.reward.values, df.propensity.values
    replays = {}
    for name, factory in rzoo.items():
        t0 = time.time()
        rp = replay(factory, Xall, a_all, r_all, p_all, T_all, n_rounds=args.rounds, seed=SEED, logger=tlog)
        rp.pop("final_model")
        replays[name] = rp
        print(f"      {name:20s} regret/session={rp['cum_regret'][-1] / len(a_all):7.1f}  ({time.time() - t0:5.1f}s)")

    # ------------------------------------------------------------------ plots
    print("[5/7] plots")
    best = max((k for k in models), key=lambda k: results[k]["DR"])
    viz.plot_data_distributions(df)
    imp = models["XGBoost-EpsGreedy"].feature_importance()
    viz.plot_correlations_importance(df, Xall, fnames, imp)
    viz.plot_training_curves(metrics_path)
    viz.plot_model_comparison(results)
    viz.plot_replay(replays)
    best_actions = models[best].action_probs(Xte).argmax(1)
    viz.plot_segment_heatmap(dte, best_actions, best, T_te.argmax(1))

    # --------------------------------------------------------------- registry
    registry = {"best_model": best, "models": list(models), "n_features": int(Xtr.shape[1]), "n_actions": N_ACTIONS,
                "feature_names": fnames, "dr_value": {k: results[k]["DR"] for k in models},
                "true_value": {k: results[k]["TRUE"] for k in models}}
    (MODELS_DIR / "model_registry.json").write_text(json.dumps(registry, indent=2))
    log_jsonl(metrics_path, {"event": "pipeline_done", "best_model": best, "seconds": round(time.time() - t_start, 1)})
    print(f"[6/7] best model by DR: {best}  (DR=${results[best]['DR']:.1f}, true=${results[best]['TRUE']:.1f}, "
          f"logging=${results['Logging policy']['TRUE']:.1f}, oracle=${results['Oracle (true best)']['TRUE']:.1f})")

    if not args.skip_screenshots:
        print("[7/7] screenshots")
        from src.screenshots import capture
        capture()
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
