"""
SoilQ pipeline: classical benchmark + Qiskit variational classifier + Quantum Kernel.

Classifiers
-----------
• Logistic Regression       (sklearn)
• RBF SVM                   (sklearn)
• Random Forest             (sklearn)
• Qiskit Variational (VQC)  (soilq.quantum.QiskitVariationalClassifier)
• Quantum Kernel (QSVM)     (soilq.quantum.QuantumKernelClassifier)

All classifiers are wrapped in sklearn Pipelines with median imputation + scaling.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .data import read_table, sha256_file
from .quantum import QiskitVariationalClassifier, QuantumKernelClassifier


def metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> dict:
    """Calculate comprehensive classification metrics."""
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, proba)) if proba is not None else None
    except ValueError:
        out["roc_auc"] = None

    if proba is not None:
        try:
            out["brier_score"] = float(brier_score_loss(y, proba))
        except Exception:
            out["brier_score"] = None
    return out


def train_benchmark(
    input_path: str | Path,
    target: str,
    features: list[str],
    output_dir: str | Path,
    report_path: str | Path | None = None,
    positive_label: str | None = None,
    test_size: float = 0.25,
    seed: int = 42,
    quantum_maxiter: int = 60,
    include_quantum_kernel: bool = True,
) -> dict:
    """
    Train benchmark suite: Logistic Regression, RBF SVM, Random Forest,
    Qiskit Variational Classifier (VQC), and Quantum Kernel Classifier (QSVM).
    """
    path = Path(input_path)
    df = read_table(path)

    # ── Validate columns ──────────────────────────────────────────────────────
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available: {list(df.columns)}")
    if not features:
        raise ValueError("At least one feature column must be specified.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in data: {missing}")
    if target in features:
        raise ValueError("Target column must not be in the feature list.")

    # ── Prepare X, y ─────────────────────────────────────────────────────────
    data = df[features + [target]].copy().dropna(subset=[target])
    for c in features:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)

    classes = sorted(data[target].dropna().unique().tolist(), key=lambda x: str(x))
    if len(classes) != 2:
        raise ValueError(
            f"Target must have exactly two observed classes; found {len(classes)}: {classes}. "
            "Do not auto-binarize without a documented rationale."
        )

    if positive_label is not None:
        match = [c for c in classes if str(c) == str(positive_label)]
        if not match:
            raise ValueError(f"Positive label '{positive_label}' not found in {classes}")
        pos = match[0]
        neg = next(c for c in classes if c != pos)
    else:
        neg, pos = classes[0], classes[1]

    y = (data[target].astype(str).to_numpy() == str(pos)).astype(int)
    X = data[features].to_numpy(dtype=float)

    valid = ~np.isnan(X).all(axis=1)
    X, y = X[valid], y[valid]

    if len(X) < 12:
        raise ValueError(f"Only {len(X)} usable rows; need at least 12.")
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        raise ValueError(f"Each class needs at least two rows. Counts: {counts.tolist()}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "project": "SoilQ",
        "challenge": "Qiskit Fall Fest Nairobi 2026 — Challenge 08: Quantum AI Classifier",
        "dataset": {
            "source": "Mutuku et al. — Mendeley Data https://data.mendeley.com/datasets/8bx88zpvdp/2",
            "license": "CC BY 4.0",
        },
        "input_file": str(path.resolve()),
        "input_sha256": sha256_file(path),
        "target_column": target,
        "feature_columns": features,
        "label_mapping": {"0": str(neg), "1": str(pos)},
        "rows_used": int(len(X)),
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "split": {
            "strategy": "stratified random holdout",
            "test_size": test_size,
            "seed": seed,
        },
        "models": {},
    }

    # ── 1. Classical classifiers ──────────────────────────────────────────────
    classical_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "RBF_SVM": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=seed,
            min_samples_leaf=2,
        ),
    }

    for name, estimator in classical_models.items():
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        start = time.perf_counter()
        pipe.fit(Xtr, ytr)
        elapsed = time.perf_counter() - start

        pred = pipe.predict(Xte)
        prob = pipe.predict_proba(Xte)[:, 1]
        result = metrics(yte, pred, prob)
        result["fit_seconds"] = round(elapsed, 4)
        report["models"][name] = result
        dump(pipe, outdir / f"{name.lower()}.joblib")

    # ── 2. Quantum Preprocessor (MinMaxScaler to [-1, 1]) ─────────────────────
    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler(feature_range=(-1, 1), clip=True)),
        ]
    )
    qtr = prep.fit_transform(Xtr)
    qte = prep.transform(Xte)

    # ── 3. Qiskit Variational Classifier (VQC) ────────────────────────────────
    n_q = min(6, qtr.shape[1])
    start = time.perf_counter()
    qmodel = QiskitVariationalClassifier(
        n_features=n_q, reps=2, maxiter=quantum_maxiter, seed=seed
    ).fit(qtr, ytr)
    elapsed = time.perf_counter() - start

    result = metrics(yte, qmodel.predict(qte), qmodel.predict_proba(qte)[:, 1])
    result["fit_seconds"] = round(elapsed, 4)
    result["quantum_metadata"] = qmodel.get_metadata()
    result["loss_history"] = qmodel.loss_history_
    result["circuit_diagram"] = qmodel.get_circuit_diagram()
    report["models"]["QiskitVariationalClassifier"] = result

    dump(
        {
            "preprocessor": prep,
            "model": qmodel,
            "features": features[:qmodel.n_qubits],
            "target": target,
            "labels": {"0": str(neg), "1": str(pos)},
        },
        outdir / "qiskit_variational_classifier.joblib",
    )

    # ── 4. Quantum Kernel Classifier (QSVM) ───────────────────────────────────
    if include_quantum_kernel:
        # Subsample for kernel computation if training set is large
        n_k = min(len(qtr), 80)
        rng = np.random.default_rng(seed)
        k_idx = rng.choice(len(qtr), n_k, replace=False)
        qtr_k, ytr_k = qtr[k_idx], ytr[k_idx]

        start = time.perf_counter()
        qkernel_model = QuantumKernelClassifier(
            n_features=n_q, feature_map="zz", C=1.0, seed=seed
        ).fit(qtr_k, ytr_k)
        elapsed = time.perf_counter() - start

        result_k = metrics(yte, qkernel_model.predict(qte), qkernel_model.predict_proba(qte)[:, 1])
        result_k["fit_seconds"] = round(elapsed, 4)
        result_k["quantum_metadata"] = qkernel_model.get_metadata()
        report["models"]["QuantumKernelClassifier"] = result_k

        dump(
            {
                "preprocessor": prep,
                "model": qkernel_model,
                "features": features[:n_q],
                "target": target,
                "labels": {"0": str(neg), "1": str(pos)},
            },
            outdir / "quantum_kernel_classifier.joblib",
        )

    # ── Save metadata bundle ──────────────────────────────────────────────────
    report["holdout_rows"] = int(len(yte))
    report["holdout_class_counts"] = {
        "0": int((yte == 0).sum()),
        "1": int((yte == 1).sum()),
    }

    dest = Path(report_path) if report_path else outdir / "report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(dest.resolve())
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return report


def predict_from_model_dir(
    model_dir: str | Path,
    input_data: Dict[str, float] | pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run inference across all trained models in model_dir for a given feature dictionary.
    Returns predicted classes, probabilities, and confidence scores.
    """
    mdir = Path(model_dir)
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    predictions = {}

    # Retrieve feature column names from report if available
    features = None
    rep_file = mdir / "report.json"
    if rep_file.exists():
        try:
            r = json.loads(rep_file.read_text(encoding="utf-8"))
            features = r.get("feature_columns")
        except Exception:
            pass

    if features is not None:
        for f in features:
            if f not in df.columns:
                df[f] = np.nan
        X_eval = df[features].to_numpy(dtype=float)
    else:
        X_eval = df.to_numpy(dtype=float)

    # Classical pipelines
    for mname in ["logisticregression", "rbf_svm", "randomforest"]:
        mfile = mdir / f"{mname}.joblib"
        if mfile.exists():
            pipe = load(mfile)
            pred = pipe.predict(X_eval)[0]
            prob = float(pipe.predict_proba(X_eval)[0, 1])
            predictions[mname] = {
                "prediction": int(pred),
                "probability_positive": round(prob, 4),
                "confidence": round(abs(prob - 0.5) * 2.0, 4),
            }

    # Quantum VQC bundle
    vqc_file = mdir / "qiskit_variational_classifier.joblib"
    if vqc_file.exists():
        bundle = load(vqc_file)
        prep = bundle["preprocessor"]
        qmodel = bundle["model"]
        feats = bundle["features"]
        X_sub = df[[f for f in feats if f in df.columns]].to_numpy()
        X_scaled = prep.transform(X_sub)
        pred = qmodel.predict(X_scaled)[0]
        prob = float(qmodel.predict_proba(X_scaled)[0, 1])
        predictions["qiskit_variational_classifier"] = {
            "prediction": int(pred),
            "probability_positive": round(prob, 4),
            "confidence": round(abs(prob - 0.5) * 2.0, 4),
            "qubits_used": qmodel.n_qubits,
        }

    # Quantum Kernel bundle
    qkernel_file = mdir / "quantum_kernel_classifier.joblib"
    if qkernel_file.exists():
        bundle = load(qkernel_file)
        prep = bundle["preprocessor"]
        qkernel = bundle["model"]
        feats = bundle["features"]
        X_sub = df[[f for f in feats if f in df.columns]].to_numpy()
        X_scaled = prep.transform(X_sub)
        pred = qkernel.predict(X_scaled)[0]
        prob = float(qkernel.predict_proba(X_scaled)[0, 1])
        predictions["quantum_kernel_classifier"] = {
            "prediction": int(pred),
            "probability_positive": round(prob, 4),
            "confidence": round(abs(prob - 0.5) * 2.0, 4),
        }

    return predictions


def run_cross_validation(
    input_path: str | Path,
    target: str,
    features: List[str],
    cv_type: str = "stratified",
    group_column: Optional[str] = None,
    n_splits: int = 5,
    seed: int = 42,
    include_quantum: bool = True,
) -> Dict[str, Any]:
    """
    Perform rigorous k-fold cross-validation (Stratified K-Fold or Group K-Fold)
    to evaluate generalization and test for spatial/temporal data leakage.
    """
    df = read_table(input_path)
    data = df[features + [target] + ([group_column] if group_column and group_column in df.columns else [])].dropna(subset=[target]).copy()
    for c in features:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)

    classes = sorted(data[target].dropna().unique().tolist(), key=lambda x: str(x))
    pos = classes[1]
    y = (data[target].astype(str).to_numpy() == str(pos)).astype(int)
    X = data[features].to_numpy(dtype=float)

    if cv_type == "group" and group_column and group_column in data.columns:
        from sklearn.model_selection import GroupKFold
        groups = data[group_column].to_numpy()
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        split_gen = splitter.split(X, y, groups)
        strategy_desc = f"GroupKFold on {group_column} ({splitter.n_splits} splits)"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_gen = splitter.split(X, y)
        strategy_desc = f"StratifiedKFold ({n_splits} splits, seed={seed})"

    model_factories = {
        "LogisticRegression": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "RBF_SVM": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]),
        "RandomForest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=seed, min_samples_leaf=2)),
        ]),
    }

    fold_scores = {m: {"accuracy": [], "f1": [], "roc_auc": []} for m in model_factories}
    if include_quantum:
        fold_scores["QuantumKernelClassifier"] = {"accuracy": [], "f1": [], "roc_auc": []}

    for fold_i, (tr_idx, te_idx) in enumerate(split_gen):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # Classical models
        for mname, m_func in model_factories.items():
            clf = m_func()
            clf.fit(Xtr, ytr)
            preds = clf.predict(Xte)
            probs = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else None
            m_res = metrics(yte, preds, probs)
            fold_scores[mname]["accuracy"].append(m_res["accuracy"])
            fold_scores[mname]["f1"].append(m_res["f1"])
            if m_res["roc_auc"] is not None:
                fold_scores[mname]["roc_auc"].append(m_res["roc_auc"])

        # Quantum Kernel Classifier on subsampled training fold
        if include_quantum:
            prep = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler(feature_range=(-1, 1), clip=True)),
            ])
            qtr = prep.fit_transform(Xtr)
            qte = prep.transform(Xte)
            n_sub = min(len(qtr), 60)
            rng_f = np.random.default_rng(seed + fold_i)
            sub_ix = rng_f.choice(len(qtr), n_sub, replace=False)
            qsvm = QuantumKernelClassifier(n_features=min(qtr.shape[1], 5), feature_map="zz", seed=seed)
            qsvm.fit(qtr[sub_ix], ytr[sub_ix])
            q_preds = qsvm.predict(qte)
            q_probs = qsvm.predict_proba(qte)[:, 1]
            q_res = metrics(yte, q_preds, q_probs)
            fold_scores["QuantumKernelClassifier"]["accuracy"].append(q_res["accuracy"])
            fold_scores["QuantumKernelClassifier"]["f1"].append(q_res["f1"])
            if q_res["roc_auc"] is not None:
                fold_scores["QuantumKernelClassifier"]["roc_auc"].append(q_res["roc_auc"])

    summary = {
        "strategy": strategy_desc,
        "n_samples": len(X),
        "features": features,
        "target": target,
        "models": {},
    }

    for mname, m_metrics in fold_scores.items():
        summary["models"][mname] = {
            "accuracy_mean": round(float(np.mean(m_metrics["accuracy"])), 4),
            "accuracy_std": round(float(np.std(m_metrics["accuracy"])), 4),
            "f1_mean": round(float(np.mean(m_metrics["f1"])), 4),
            "f1_std": round(float(np.std(m_metrics["f1"])), 4),
            "roc_auc_mean": round(float(np.mean(m_metrics["roc_auc"])), 4) if m_metrics["roc_auc"] else None,
            "roc_auc_std": round(float(np.std(m_metrics["roc_auc"])), 4) if m_metrics["roc_auc"] else None,
        }

    return summary
