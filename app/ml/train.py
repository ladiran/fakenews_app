"""Train the classifiers.

    python -m app.ml.train --data liar:data/liar/train.tsv --extra isot:data/isot

Two different models vote, so one model's blind spot is less likely to decide the result:
  1. word 1-2 gram TF-IDF + logistic regression      (what words are used)
  2. char 3-5 gram TF-IDF + calibrated linear SVM    (how it is written: caps, punctuation)
Metrics are measured on data the models never saw. --extra datasets are never trained
on at all; they test whether the models generalise to a different source, which is the
number you should trust most.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import data

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "word_logreg": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                                      lowercase=True, strip_accents="unicode")),
            ("clf", LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=seed)),
        ]),
        "char_svm": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                      lowercase=False)),  # keep case: ALL CAPS is a signal
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
                                           cv=3)),
        ]),
    }


def evaluate(y_true, p_fake, threshold: float = 0.5) -> dict:
    pred = (np.asarray(p_fake) >= threshold).astype(int)
    out = {"n": int(len(y_true)),
           "fake_share": round(float(np.mean(y_true)), 3),
           "accuracy": round(accuracy_score(y_true, pred), 4),
           "precision_fake": round(precision_score(y_true, pred, zero_division=0), 4),
           "recall_fake": round(recall_score(y_true, pred, zero_division=0), 4),
           "f1_fake": round(f1_score(y_true, pred, zero_division=0), 4)}
    if len(set(y_true)) == 2:
        out["roc_auc"] = round(roc_auc_score(y_true, p_fake), 4)
    return out


def ensemble_proba(models: dict, X: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
    return np.mean(list(per.values()), axis=0), per


def train(datasets: list[str], extras: list[str], out_dir: Path, seed: int = 42,
          use_body: bool = False) -> dict:
    X, y = [], []
    for spec in datasets:
        kw = {"use_body": use_body} if spec.startswith("isot:") else {}
        xs, ys = data.load(spec, **kw)
        X += xs
        y += ys
    X, y = data.dedupe(X, y)
    if len(set(y)) < 2 or min(np.bincount(y)) < 20:
        raise SystemExit("Need at least 20 examples of each class to train.")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    models = build_models(seed)
    for m in models.values():
        m.fit(X_tr, y_tr)

    # Pick the decision threshold on validation data only, then report on untouched test data.
    p_va, _ = ensemble_proba(models, X_va)
    thresholds = np.linspace(0.3, 0.7, 41)
    best_t = float(max(thresholds, key=lambda t: f1_score(y_va, p_va >= t, zero_division=0)))

    p_te, per_te = ensemble_proba(models, X_te)
    metrics = {"test": {"ensemble": evaluate(y_te, p_te, best_t),
                        **{k: evaluate(y_te, v, 0.5) for k, v in per_te.items()}}}

    for spec in extras:  # out-of-source check, never trained on
        kw = {"use_body": use_body} if spec.startswith("isot:") else {}
        xe, ye = data.dedupe(*data.load(spec, **kw))
        pe, _ = ensemble_proba(models, xe)
        metrics.setdefault("external", {})[spec] = evaluate(ye, pe, best_t)

    # Final models use all non-external data so no labelled example is wasted.
    models = build_models(seed)
    for m in models.values():
        m.fit(X, y)

    card = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": datasets, "external_datasets": extras, "use_body": use_body,
        "n_train_total": len(X), "threshold": round(best_t, 3), "metrics": metrics,
        "sklearn": sklearn.__version__, "python": platform.python_version(),
        "note": "Writing-style classifier. It cannot check facts.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "card": card}, out_dir / "model.joblib", compress=3)
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="training sets, e.g. liar:train.tsv")
    ap.add_argument("--extra", nargs="*", default=[], help="held-out sets from another source")
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    ap.add_argument("--use-body", action="store_true", help="ISOT only: include article body, not just title")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    card = train(a.data, a.extra, a.out, a.seed, a.use_body)
    print(json.dumps(card["metrics"], indent=2))
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    main()
