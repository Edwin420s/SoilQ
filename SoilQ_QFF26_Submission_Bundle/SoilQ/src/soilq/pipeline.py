import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from .data import sha256_file, read_table
from .quantum import QiskitVariationalClassifier

def metrics(y, pred, proba=None):
    out = {"accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
           "classification_report": classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)}
    try: out["roc_auc"] = float(roc_auc_score(y, proba)) if proba is not None else None
    except ValueError: out["roc_auc"] = None
    return out

def train_benchmark(input_path, target, features, output_dir, report_path=None, positive_label=None,
                    test_size=0.25, seed=42, quantum_maxiter=60):
    path = Path(input_path); df = read_table(path)
    if target not in df.columns: raise ValueError(f"Target '{target}' missing. Available: {list(df.columns)}")
    if not features: raise ValueError("Specify feature columns.")
    missing = [c for c in features if c not in df.columns]
    if missing: raise ValueError(f"Features not found: {missing}")
    if target in features: raise ValueError("Target must not be included in features.")
    data = df[features + [target]].copy().dropna(subset=[target])
    for c in features: data[c] = pd.to_numeric(data[c], errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    classes = sorted(data[target].dropna().unique().tolist(), key=lambda x: str(x))
    if len(classes) != 2:
        raise ValueError(f"Target must have exactly two observed classes; found {len(classes)}: {classes}. Do not auto-binarize without a documented rationale.")
    if positive_label is not None:
        match = [c for c in classes if str(c) == str(positive_label)]
        if not match: raise ValueError(f"Positive label '{positive_label}' not found in {classes}")
        pos = match[0]; neg = next(c for c in classes if c != pos)
    else: neg, pos = classes[0], classes[1]
    y = (data[target].astype(str).to_numpy() == str(pos)).astype(int)
    X = data[features].to_numpy(dtype=float)
    valid = ~np.isnan(X).all(axis=1); X, y = X[valid], y[valid]
    if len(X) < 12: raise ValueError(f"Only {len(X)} usable rows; need at least 12 for a minimal holdout experiment.")
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2: raise ValueError(f"Each class needs at least two rows. Counts: {counts.tolist()}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    report = {"project": "SoilQ", "input_file": str(path.resolve()), "input_sha256": sha256_file(path),
              "target_column": target, "feature_columns": features,
              "label_mapping": {"0": str(neg), "1": str(pos)}, "rows_used": int(len(X)),
              "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
              "split": {"strategy": "stratified random holdout", "test_size": test_size, "seed": seed},
              "warning": "Random split can leak plot/site/season/time information. Review study design.",
              "models": {}}
    models = {"LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
              "RBF_SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
              "RandomForest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=seed, min_samples_leaf=2)}
    for name, estimator in models.items():
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
        start = time.perf_counter(); pipe.fit(Xtr, ytr); elapsed = time.perf_counter() - start
        pred = pipe.predict(Xte); prob = pipe.predict_proba(Xte)[:, 1]
        result = metrics(yte, pred, prob); result["fit_seconds"] = round(elapsed, 4)
        report["models"][name] = result; dump(pipe, outdir / f"{name.lower()}.joblib")
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler(feature_range=(-1, 1), clip=True))])
    qtr = prep.fit_transform(Xtr); qte = prep.transform(Xte)
    start = time.perf_counter()
    qmodel = QiskitVariationalClassifier(n_features=min(6, qtr.shape[1]), reps=2, maxiter=quantum_maxiter, seed=seed).fit(qtr, ytr)
    elapsed = time.perf_counter() - start
    result = metrics(yte, qmodel.predict(qte), qmodel.predict_proba(qte)[:, 1])
    result["fit_seconds"] = round(elapsed, 4); result["quantum_metadata"] = qmodel.get_metadata()
    result["note"] = "Qiskit Statevector classical simulation, not quantum hardware."
    report["models"]["QiskitVariationalClassifier"] = result
    dump({"preprocessor": prep, "model": qmodel, "features": features[:qmodel.n_qubits],
          "target": target, "labels": {"0": str(neg), "1": str(pos)}}, outdir / "qiskit_variational_classifier.joblib")
    report["holdout_rows"] = int(len(yte))
    report["holdout_class_counts"] = {"0": int((yte == 0).sum()), "1": int((yte == 1).sum())}
    report["classification_note"] = "Single split only; not evidence of generalization or quantum advantage."
    dest = Path(report_path) if report_path else outdir / "report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(dest.resolve())
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
