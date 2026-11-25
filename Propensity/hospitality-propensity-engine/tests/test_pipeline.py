"""Pipeline verification: shapes, leakage, calibration bounds, GPU placement, API contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import cuda_info, load_config  # noqa: E402
from src.feature_engineering import META_COLS, Snapshotter, load_tables, snapshot_dates  # noqa: E402

CFG = load_config()
SCENS = list(CFG["scenarios"])
PROC = ROOT / CFG["paths"]["processed"]
MET = ROOT / CFG["paths"]["metrics"]
needs_data = pytest.mark.skipif(not (ROOT / CFG["paths"]["raw"] / "bookings.parquet").exists(), reason="run data_generator first")
needs_models = pytest.mark.skipif(not (MET / "best_models.json").exists(), reason="run trainer first")
needs_cuda = pytest.mark.skipif(not cuda_info()["cuda_available"], reason="CUDA not available")
BANNED = {"y", "treatment", "status", "cancel_date", "label", "target"}


@pytest.fixture(scope="module")
def tables():
    return load_tables(CFG)


# ----------------------------------------------------------------------------- data shapes
@needs_data
def test_raw_table_shapes(tables):
    assert len(tables["guests"]) >= 50_000
    assert len(tables["bookings"]) >= 150_000
    assert {"OTA", "Direct Web", "Mobile App", "Corporate"} == set(tables["bookings"].channel.unique())
    assert {"Refundable", "Non-Refundable"} == set(tables["bookings"].rate_plan.unique())
    assert tables["bookings"].guest_id.nunique() > 40_000
    assert (tables["bookings"].checkout_date > tables["bookings"].checkin_date).all()
    assert (tables["bookings"].booking_date <= tables["bookings"].checkin_date).all()
    for t in ("folios", "web_sessions", "email_monthly", "surveys", "complaints"):
        assert len(tables[t]) > 1000, t


@pytest.mark.parametrize("scen", SCENS)
@needs_data
def test_processed_split_shapes(scen):
    f = PROC / scen
    if not (f / "schema.json").exists():
        pytest.skip("features not built")
    sch = json.loads((f / "schema.json").read_text())
    for split in ("train", "val", "test"):
        df = pd.read_parquet(f / f"{split}.parquet")
        assert len(df) > 2000 and set(df.y.unique()) == {0, 1}
        assert 0.03 < df.y.mean() < 0.8
        assert set(sch["numeric"] + sch["categorical"]) <= set(df.columns)


# ----------------------------------------------------------------------------- leakage
@pytest.mark.parametrize("scen", SCENS)
@needs_data
def test_no_leakage_in_features_or_time(scen):
    f = PROC / scen
    if not (f / "schema.json").exists():
        pytest.skip("features not built")
    sch = json.loads((f / "schema.json").read_text())
    feats = set(sch["numeric"] + sch["categorical"])
    assert not (feats & set(META_COLS)), "meta/label columns leaked into features"
    assert not (feats & BANNED)
    w = sch["window_days"]
    snaps = snapshot_dates(CFG, w)
    val_c, test_c = pd.Timestamp(CFG["data"]["val_cutoff"]), pd.Timestamp(CFG["data"]["test_cutoff"])
    assert max(snaps["train"]) + pd.Timedelta(days=w) <= val_c, "train label window overlaps validation cutoff"
    assert val_c + pd.Timedelta(days=w) <= test_c + pd.Timedelta(days=1), "val label window overlaps test cutoff"
    for split in ("train", "val", "test"):
        df = pd.read_parquet(f / f"{split}.parquet", columns=[c for c in ["snapshot_date", "latest_source_ts", "checkin_date", "session_date"] if c in sch["meta"] + ["snapshot_date"]])
        assert (df.latest_source_ts.dropna() < df.snapshot_date.loc[df.latest_source_ts.notna()]).all(), "feature source event at/after cutoff"
        if "checkin_date" in df:
            assert (df.checkin_date >= df.snapshot_date).all() and (df.checkin_date < df.snapshot_date + pd.Timedelta(days=w)).all()
        if "session_date" in df:
            assert (df.session_date < df.snapshot_date).all()


@needs_data
def test_features_unchanged_when_future_data_removed(tables):
    """Point-in-time proof: deleting every event on/after the cutoff must not change any feature."""
    c = pd.Timestamp(CFG["data"]["test_cutoff"])
    full, _ = Snapshotter(tables).guest_features(c)
    cut = dict(tables)
    cut["bookings"] = tables["bookings"][tables["bookings"].booking_date < c]
    cut["folios"] = tables["folios"][tables["folios"].folio_date < c]
    cut["surveys"] = tables["surveys"][tables["surveys"].survey_date < c]
    cut["complaints"] = tables["complaints"][tables["complaints"].complaint_date < c]
    cut["web_sessions"] = tables["web_sessions"][tables["web_sessions"].session_date < c]
    cm = c.to_period("M").to_timestamp()
    cut["email_monthly"] = tables["email_monthly"][tables["email_monthly"].month < cm]
    trunc, _ = Snapshotter(cut).guest_features(c)
    idx = full.index.intersection(trunc.index).to_series().sample(3000, random_state=0).index
    num = [x for x in full.columns if pd.api.types.is_numeric_dtype(full[x]) and x in trunc]
    a, b = full.loc[idx, num].astype(float), trunc.loc[idx, num].astype(float)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


@needs_data
def test_status_resolution_respects_cutoff(tables):
    c = pd.Timestamp(CFG["data"]["test_cutoff"])
    K = Snapshotter(tables).known(c)
    assert (K.booking_date < c).all()
    assert (K.loc[K.kstat == "cancelled", "cancel_date"] < c).all()
    assert (K.loc[K.kstat == "completed", "checkout_date"] < c).all()
    assert (K.loc[K.kstat == "no_show", "checkin_date"] < c).all()


@needs_data
def test_target_shuffle_sanity():
    """Shuffled labels -> AUC ~ 0.5; real labels -> clearly better than random (no trivial leak, real signal)."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    f = PROC / "cancel_noshow"
    if not (f / "schema.json").exists():
        pytest.skip("features not built")
    sch = json.loads((f / "schema.json").read_text())
    tr, te = pd.read_parquet(f / "train.parquet"), pd.read_parquet(f / "test.parquet")
    num = sch["numeric"]
    xtr, xte = tr[num].astype(float), te[num].astype(float)
    dev = "cuda" if cuda_info()["cuda_available"] else "cpu"
    aucs = {}
    for name, y in (("real", tr.y.values), ("shuffled", np.random.default_rng(0).permutation(tr.y.values))):
        m = xgb.XGBClassifier(n_estimators=120, max_depth=4, tree_method="hist", device=dev, learning_rate=0.1).fit(xtr, y)
        aucs[name] = roc_auc_score(te.y, m.predict_proba(xte)[:, 1])
    assert abs(aucs["shuffled"] - 0.5) < 0.04, aucs
    assert 0.58 < aucs["real"] < 0.97, aucs  # strong but not suspiciously perfect


# ----------------------------------------------------------------------------- models / calibration
@needs_models
def test_calibrated_probabilities_in_unit_interval():
    n = 0
    for p in MET.glob("*/preds_*.parquet"):
        d = pd.read_parquet(p)
        assert d.p_cal.between(0, 1).all() and d.p_raw.between(0, 1).all(), p
        assert np.isfinite(d.p_cal).all()
        n += 1
    assert n >= 8


@needs_models
def test_leaderboard_complete_and_calibration_helps():
    lb = pd.read_csv(MET / "master_leaderboard.csv")
    assert lb.scenario.nunique() == 8
    assert set(lb.model) == {"xgboost", "lightgbm", "catboost", "pytorch_nn"}
    assert len(lb) == 32
    assert np.isfinite(lb[["roc_auc", "pr_auc", "logloss", "brier", "top_decile_lift"]].to_numpy()).all()
    assert (lb.roc_auc > 0.5).all()
    # calibration should not materially hurt log-loss, and usually helps
    assert ((lb.logloss - lb.logloss_raw) < 0.01).all()
    assert (lb.logloss <= lb.logloss_raw + 1e-4).mean() > 0.6


@needs_cuda
def test_gpu_tensor_placement_and_xgboost_device():
    import torch
    from src.models.deep_tabular import TorchTabularModel
    from src.models.gbdt_models import XGBoostModel
    X = pd.DataFrame(np.random.rand(600, 6).astype("float32"), columns=list("abcdef"))
    X["a"] = np.random.randint(0, 3, 600).astype("float32")
    y = (np.random.rand(600) > 0.7).astype(int)
    nn = TorchTabularModel(["a"], [3], use_gpu=True, max_epochs=2).fit(X, y, X, y, {"hidden": 32, "n_blocks": 1})
    assert next(nn.model.parameters()).is_cuda
    xc, xn = nn._split(X)
    assert xc.is_cuda and xn.is_cuda
    assert nn.predict_proba(X).shape == (600,)
    xm = XGBoostModel(["a"], use_gpu=True, max_rounds=5).fit(X, y, X, y, {"max_depth": 3})
    assert xm.model.get_params()["device"] == "cuda"
    assert xm.model.get_booster().save_config().count('"device":"cuda"') >= 1 or "cuda" in xm.model.get_booster().save_config()


# ----------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.app import app
    with TestClient(app) as c:
        yield c


GUEST = {"loyalty_tier": "Gold", "is_loyalty_member": 1, "n_stays": 6, "recency_days": 140, "inter_stay_mean_days": 180,
         "ota_share": 0.35, "n_ota_12m": 1, "nps_mean": 8, "channel": "OTA", "rate_plan": "Refundable", "room_type": "Deluxe",
         "property_type": "Coastal Vacation", "lead_time": 45, "los": 4, "adults": 2, "children": 1, "adr": 340, "checkin_month": 7}


@needs_models
def test_api_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and len(j["models_loaded"]) == 8
    assert j["cuda_available"] == cuda_info()["cuda_available"]


@needs_models
def test_api_predict_guest(client):
    r = client.post("/predict/guest", json={"guest": {**GUEST, "guest_id": "T-1"}, "explain": True})
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j["scores"]) == set(SCENS)
    for s in j["scores"].values():
        assert 0 <= s["propensity"] <= 1 and 1 <= s["decile"] <= 10 and s["next_best_action"]
        assert len(s["top_drivers"]) == 5


@needs_models
def test_api_predict_batch_and_validation(client):
    r = client.post("/predict/batch", json={"guests": [GUEST, {**GUEST, "n_stays": 0}, {}]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n"] == 3 and all(0 <= v["propensity"] <= 1 for x in j["results"] for v in x["scores"].values())
    assert client.post("/predict/guest", json={"guest": {"ota_share": 2.0}}).status_code == 422
    assert client.post("/predict/batch", json={"guests": []}).status_code == 422
    assert client.post("/predict/guest", json={"guest": {"channel": "Carrier Pigeon"}}).status_code == 422
