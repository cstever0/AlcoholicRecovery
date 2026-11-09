"""Export the fitted recovery model and cohort for the dashboard prototype.

Writes src/data/{model,cohort,metrics}.json. The prototype scores patients in the
browser with these exact coefficients, so rerun this whenever the cohort or model
changes:  python design/export_data.py  (from the repo root, with .venv active)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from crip.modeling.artifacts import load_recovery_model_artifacts
from crip.modeling.dataset import (
    DEFAULT_COHORT_CSV,
    build_feature_preprocessor,
    load_modeling_dataset,
)
from crip.modeling.train import (
    compute_binary_classification_metrics,
    train_baseline_models,
)

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "src" / "data"

# Six demo patients, each showing a different state. All come from the validation and test
# splits, so the model never trained on them, and all are 7+ days past their last drink,
# because the plan assumes withdrawal is complete. Within each state the pick has the LVEF
# closest to that group's median, so the demo shows a typical heart for this cohort rather
# than the weakest one.
DEMO_NOTES = {
    "favorable": "Recovery likely: toxic picture, no scar, no TTN variant",
    "substrate": "Recovery unlikely: TTN variant with scar",
    "missing": "Not sure: TTN and LGE missing",
    "fence": "Not sure: complete data, model on the fence",
    "vus": "VUS handled as uncertain, not pathogenic",
    "device": "Severe LV dysfunction, recovery unlikely",
}


def export_model(pipeline, X_train: pd.DataFrame) -> dict:
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    coef = dict(zip(names, clf.coef_[0].tolist()))
    means = dict(zip(names, np.asarray(prep.transform(X_train)).mean(axis=0).tolist()))

    num_cols = next(c for n, _, c in prep.transformers_ if n == "num")
    num = prep.named_transformers_["num"]
    numeric = [
        {
            "f": col,
            "median": float(num.named_steps["imputer"].statistics_[i]),
            "mean": float(num.named_steps["scaler"].mean_[i]),
            "scale": float(num.named_steps["scaler"].scale_[i]),
            "coef": coef[f"num__{col}"],
            "xbar": means[f"num__{col}"],
        }
        for i, col in enumerate(num_cols)
    ]

    cat_cols = next(c for n, _, c in prep.transformers_ if n == "cat")
    cat = prep.named_transformers_["cat"]
    categorical = [
        {
            "f": col,
            "levels": {
                str(level): {
                    "coef": coef[f"cat__{col}_{level}"],
                    "xbar": means[f"cat__{col}_{level}"],
                }
                for level in cat.named_steps["ohe"].categories_[i]
            },
        }
        for i, col in enumerate(cat_cols)
    ]

    bin_cols = next(c for n, _, c in prep.transformers_ if n == "bin")
    binary = [
        {"f": col, "coef": coef[f"bin__{col}"], "xbar": means[f"bin__{col}"]}
        for col in bin_cols
    ]
    return {
        "version": "v1.0-m3b",
        "intercept": float(clf.intercept_[0]),
        "numeric": numeric,
        "categorical": categorical,
        "binary": binary,
    }


def compact_cohort(df: pd.DataFrame) -> dict:
    cols = [
        "patient_id", "age", "biological_sex", "baseline_lvef", "edv_ml", "esv_ml",
        "stroke_volume_ml", "rvedv_ml", "rvesv_ml", "rvef_percent", "myocardial_mass_g",
        "peth_value_ng_ml", "peth_available", "peth_measurement_date",
        "reported_drinks_per_week", "time_since_last_drink_days", "years_heavy_drinking",
        "alcohol_history_available", "ttn_classification", "lge_status", "lge_pattern",
        "lge_extent_percent", "lge_available", "assumed_abstinence_scenario",
        "follow_up_lvef", "delta_lvef", "recovery_category", "adverse_cardiac_events",
    ]
    out = df[cols].copy()
    for c in ("peth_available", "alcohol_history_available", "lge_available"):
        out[c] = out[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(1)
    rows = out.astype(object).where(out.notna(), None).values.tolist()
    return {"columns": cols, "rows": rows}


def sigmoid_probs(pipeline, X):
    return pipeline.predict_proba(X)[:, 1]


def pick_demo(pipeline, df: pd.DataFrame, unseen_ids: list) -> dict:
    pool = df[df.patient_id.astype(str).isin(unseen_ids) & (df.time_since_last_drink_days >= 7)].copy()
    from crip.modeling.dataset import FEATURE_WHITELIST

    X = pool[FEATURE_WHITELIST].copy()
    for c in ("peth_available", "alcohol_history_available", "lge_available"):
        X[c] = X[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    for c in ("biological_sex", "ttn_classification", "lge_status", "lge_pattern"):
        X[c] = X[c].astype(str)
    pool["p"] = sigmoid_probs(pipeline, X)
    complete = (
        pool.peth_available.astype(str).str.lower().isin(["true", "1", "1.0"])
        & pool.lge_available.astype(str).str.lower().isin(["true", "1", "1.0"])
        & (pool.ttn_classification != "NOT_TESTED")
    )
    none_ttn = pool.ttn_classification == "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"

    def typical(group: pd.DataFrame) -> pd.Series:
        gap = (group.baseline_lvef - group.baseline_lvef.median()).abs()
        return group.assign(gap=gap).sort_values(["gap", "patient_id"]).iloc[0]

    picks = {}
    picks["favorable"] = typical(pool[complete & none_ttn & (pool.lge_status == "NEGATIVE") & (pool.peth_value_ng_ml > 150) & (pool.baseline_lvef < 35) & (pool.p > 0.95) & (pool.recovery_binary == 1)])
    picks["substrate"] = typical(pool[complete & (pool.ttn_classification == "PATHOGENIC") & (pool.lge_status == "POSITIVE") & (pool.p < 0.05) & (pool.recovery_binary == 0)])
    no_lge = ~pool.lge_available.astype(str).str.lower().isin(["true", "1", "1.0"])
    picks["missing"] = pool[(pool.ttn_classification == "NOT_TESTED") & no_lge].sort_values("p", ascending=False).iloc[0]
    picks["fence"] = typical(pool[complete & pool.p.between(0.40, 0.62)])
    picks["vus"] = typical(pool[complete & (pool.ttn_classification == "VARIANT_OF_UNCERTAIN_SIGNIFICANCE") & (pool.p > 0.9)])
    used = {str(s.patient_id) for s in picks.values()}
    picks["device"] = typical(pool[complete & ~pool.patient_id.astype(str).isin(used) & (pool.baseline_lvef <= 30) & (pool.lge_status != "NEGATIVE") & (pool.p < 0.1)])
    return {k: {"id": str(v.patient_id), "note": DEMO_NOTES[k], "p": round(float(v.p), 4)} for k, v in picks.items()}


# Same rule as score() in src/model.js: distance from a coin flip, capped by how far the
# missing results could move the answer. Only missing results are replaced.
BEST = {
    "peth": {"peth_value_ng_ml": 450.0, "peth_available": 1.0},
    "ttn": {"ttn_classification": "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"},
    "lge": {"lge_status": "NEGATIVE", "lge_pattern": "NONE", "lge_extent_percent": 0.0, "lge_available": 1.0},
}
WORST = {
    "peth": {"peth_value_ng_ml": 15.0, "peth_available": 1.0},
    "ttn": {"ttn_classification": "PATHOGENIC"},
    "lge": {"lge_status": "POSITIVE", "lge_pattern": "MIDWALL", "lge_extent_percent": 15.0, "lge_available": 1.0},
}


def prototype_confidence(pipeline, X: pd.DataFrame) -> np.ndarray:
    p = sigmoid_probs(pipeline, X)
    missing = {
        "peth": (X.peth_available < 0.5) | X.peth_value_ng_ml.isna(),
        "ttn": X.ttn_classification == "NOT_TESTED",
        "lge": X.lge_available < 0.5,
    }
    best, worst = X.copy(), X.copy()
    for key, mask in missing.items():
        for col, value in BEST[key].items():
            best.loc[mask, col] = value
        for col, value in WORST[key].items():
            worst.loc[mask, col] = value
    spread = sigmoid_probs(pipeline, best) - sigmoid_probs(pipeline, worst)
    return np.minimum(np.maximum(p, 1 - p), 1 - np.where(missing["peth"] | missing["ttn"] | missing["lge"], spread, 0.0))


def export_metrics(models, ds) -> dict:
    lr = models["logistic_regression"]
    names = {
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "dummy_prevalence": "Baseline (always predict the average)",
    }
    table = []
    for key in ("logistic_regression", "random_forest", "dummy_prevalence"):
        m = compute_binary_classification_metrics(ds.y_val, sigmoid_probs(models[key], ds.X_val))
        table.append({
            "name": names[key], "key": key,
            **{k: m[k] for k in ("roc_auc", "pr_auc", "brier_score", "accuracy", "sensitivity", "specificity")},
        })

    # Threshold tradeoff on validation only: the test split stays untouched by tuning.
    pv = sigmoid_probs(lr, ds.X_val)
    yv = np.asarray(ds.y_val)
    conf = prototype_confidence(lr, ds.X_val)
    pred = (pv >= 0.5).astype(int)
    tradeoff = []
    for t in np.round(np.arange(0.50, 0.991, 0.01), 2):
        s = conf >= t
        tradeoff.append({
            "t": float(t),
            "coverage": round(float(s.mean()), 4),
            "accuracy": round(float((pred[s] == yv[s]).mean()), 4) if s.sum() else None,
        })

    # Calibration on validation + test pooled, so each bin has enough patients to read.
    Xp = pd.concat([ds.X_val, ds.X_test])
    yp = np.r_[np.asarray(ds.y_val), np.asarray(ds.y_test)]
    prep = build_feature_preprocessor(ds.numerical_features, ds.categorical_features, ds.binary_features)
    unweighted = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ]).fit(ds.X_train, ds.y_train)

    def curve(p):
        edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (p >= a) & (p < b)
            if s.sum():
                rows.append({"pred": round(float(p[s].mean()), 3), "obs": round(float(yp[s].mean()), 3), "n": int(s.sum())})
        return rows

    return {
        "n_val": int(len(ds.y_val)),
        "n_test": int(len(ds.y_test)),
        "n_train": int(len(ds.y_train)),
        "table": table,
        "tradeoff": tradeoff,
        "calibration": {
            "current": curve(sigmoid_probs(lr, Xp)),
            "unweighted": curve(sigmoid_probs(unweighted, Xp)),
        },
        "unweighted_val": compute_binary_classification_metrics(ds.y_val, sigmoid_probs(unweighted, ds.X_val)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_modeling_dataset(DEFAULT_COHORT_CSV, scenario="SUSTAINED_ABSTINENCE", seed=42)
    models = train_baseline_models(ds, seed=42)
    fitted, _ = load_recovery_model_artifacts()
    df = pd.read_csv(DEFAULT_COHORT_CSV)

    (OUT / "model.json").write_text(json.dumps(export_model(fitted, ds.X_train)))
    cohort = compact_cohort(df)
    cohort["demo"] = pick_demo(fitted, df, ds.val_ids + ds.test_ids)
    (OUT / "cohort.json").write_text(json.dumps(cohort, separators=(",", ":")))
    metrics = export_metrics(models, ds)
    (OUT / "metrics.json").write_text(json.dumps(metrics))
    for k, v in cohort["demo"].items():
        print(f"{k:10s} {v}")
    print({p.name: p.stat().st_size for p in OUT.glob("*.json")})


if __name__ == "__main__":
    main()
