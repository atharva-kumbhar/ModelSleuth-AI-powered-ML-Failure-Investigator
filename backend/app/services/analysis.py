from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ..schemas import AnalysisRequest
from .datasets import get_dataset

_analyses: dict[str, dict[str, Any]] = {}


def _sample_id(row: pd.Series, id_column: str | None, index: int) -> str:
    value = row.get(id_column) if id_column else None
    return str(value) if value is not None and not pd.isna(value) else f"row-{index + 1}"


def run_analysis(request: AnalysisRequest) -> dict[str, Any]:
    filename, original = get_dataset(request.dataset_id)
    required = [request.true_column, request.predicted_column, *request.group_columns]
    if request.sample_id_column:
        required.append(request.sample_id_column)
    required.extend(request.probability_columns.values())
    unknown = sorted(set(required) - set(map(str, original.columns)))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown column(s): {', '.join(unknown)}")
    if request.true_column == request.predicted_column:
        raise HTTPException(status_code=422, detail="True and predicted labels must be different columns.")
    if request.true_column in request.group_columns or request.predicted_column in request.group_columns:
        raise HTTPException(status_code=422, detail="Metadata groups must be separate from label columns.")

    frame = original.copy()
    initial_rows = len(frame)
    missing_label_mask = frame[[request.true_column, request.predicted_column]].isna().any(axis=1)
    excluded_missing = int(missing_label_mask.sum())
    frame = frame.loc[~missing_label_mask].copy().reset_index(drop=True)
    if frame.empty:
        raise HTTPException(status_code=422, detail="No rows have both true and predicted labels.")
    frame[request.true_column] = frame[request.true_column].astype(str)
    frame[request.predicted_column] = frame[request.predicted_column].astype(str)
    labels = sorted(set(frame[request.true_column]) | set(frame[request.predicted_column]))
    y_true = frame[request.true_column].to_numpy()
    y_pred = frame[request.predicted_column].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    class_metrics = [
        {"class": label, "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i])}
        for i, label in enumerate(labels)
    ]
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    correct = y_true == y_pred
    frame["__correct"] = correct
    frame["__sample_id"] = [_sample_id(row, request.sample_id_column, i) for i, row in frame.iterrows()]
    sample_count = len(frame)

    group_results: list[dict[str, Any]] = []
    for column in request.group_columns:
        for group, indices in frame.groupby(column, dropna=False, sort=True).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            group_results.append({
                "column": column, "group": "(missing)" if pd.isna(group) else str(group),
                "support": int(len(positions)),
                "accuracy": float(accuracy_score(y_true[positions], y_pred[positions])),
                "macro_f1": float(precision_recall_fscore_support(y_true[positions], y_pred[positions], labels=labels, average="macro", zero_division=0)[2]),
            })

    probabilities: np.ndarray | None = None
    probability_error: str | None = None
    if request.probability_columns:
        if set(request.probability_columns) != set(labels):
            probability_error = "Provide one probability column for every observed class, keyed by class label."
        else:
            try:
                probabilities = frame[[request.probability_columns[label] for label in labels]].astype(float).to_numpy()
                sums = probabilities.sum(axis=1)
                if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
                    raise ValueError("Probabilities must be finite values between 0 and 1.")
                if not np.allclose(sums, 1.0, atol=0.035):
                    raise ValueError("Each row's class probabilities must sum to approximately 1.")
                probabilities = probabilities / sums[:, None]
            except (ValueError, TypeError) as exc:
                probability_error = str(exc)
                probabilities = None

    pair_counts: list[dict[str, Any]] = []
    for i, truth in enumerate(labels):
        for j, pred in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pair_counts.append({"true_class": truth, "predicted_class": pred, "count": int(cm[i, j])})
    pair_counts.sort(key=lambda entry: entry["count"], reverse=True)

    findings: list[dict[str, Any]] = []
    for i, item in enumerate(class_metrics):
        if item["recall"] < 0.75:
            affected = frame.loc[(frame[request.true_column] == labels[i]) & ~frame["__correct"]]
            main_confusion = max((p for p in pair_counts if p["true_class"] == labels[i]), key=lambda p: p["count"], default=None)
            findings.append({
                "id": f"class-recall-{i}", "title": f"Low recall for {labels[i]}",
                "category": "Class performance", "priority": "Critical" if item["recall"] < .4 else "High" if item["recall"] < .6 else "Moderate",
                "rationale": "Ranked by lowest per-class recall; this rule exposes missed examples directly.",
                "evidence": {**item, "most_common_confusion": main_confusion}, "affected_count": len(affected),
                "sample_ids": affected["__sample_id"].head(8).tolist(),
                "hypotheses": ([f"Features may overlap between {labels[i]} and {main_confusion['predicted_class']}.", "The class definition or annotations may be ambiguous; inspect examples before changing labels."] if main_confusion else ["The class has limited support; inspect representative examples before drawing conclusions."]),
                "experiment": "Review missed examples, then compare a targeted threshold or class-balanced training run against the same held-out split.",
                "expected_outcome": "Compare recall and macro F1 for this class, and verify whether improvements reduce another class's precision.",
            })
    for pair in pair_counts[:4]:
        if pair["count"] >= max(4, int(sample_count * .015)):
            findings.append({
                "id": f"confusion-{labels.index(pair['true_class'])}-{labels.index(pair['predicted_class'])}",
                "title": f"Repeated confusion: {pair['true_class']} → {pair['predicted_class']}",
                "category": "Class confusion", "priority": "High" if pair["count"] >= sample_count * .08 else "Moderate",
                "rationale": "Ranked by the observed off-diagonal confusion count.", "evidence": pair,
                "affected_count": pair["count"],
                "sample_ids": frame.loc[(frame[request.true_column] == pair["true_class"]) & (frame[request.predicted_column] == pair["predicted_class"]), "__sample_id"].head(8).tolist(),
                "hypotheses": ["The two classes may share predictive features or contain inconsistent annotation boundaries."],
                "experiment": "Inspect the confused examples, then test targeted data augmentation or a revised decision threshold.",
                "expected_outcome": "Reduce this confusion pair while maintaining macro F1 and other class recalls.",
            })
    for column in request.group_columns:
        groups = [g for g in group_results if g["column"] == column and g["support"] >= 10]
        if len(groups) >= 2:
            best = max(g["accuracy"] for g in groups)
            for group in groups:
                if best - group["accuracy"] >= .12:
                    findings.append({
                        "id": f"group-{column}-{group['group']}", "title": f"Performance gap in {group['group']}",
                        "category": "Group performance", "priority": "High" if best - group["accuracy"] >= .22 else "Moderate",
                        "rationale": f"At least 10 samples in group; accuracy trails the best group by {best - group['accuracy']:.1%}.",
                        "evidence": {**group, "best_group_accuracy": best, "accuracy_gap": best - group["accuracy"]},
                        "affected_count": int(((frame[column].fillna("(missing)").astype(str) == group["group"]) & ~frame["__correct"]).sum()),
                        "sample_ids": frame.loc[(frame[column].fillna("(missing)").astype(str) == group["group"]) & ~frame["__correct"], "__sample_id"].head(8).tolist(),
                        "hypotheses": ["The group may differ in data coverage, acquisition conditions, or class mix; those causes have not been tested."],
                        "experiment": f"Compare per-class support and error examples in {group['group']}, then evaluate a targeted data or calibration change on the same split.",
                        "expected_outcome": "Measure group accuracy and per-class recall; check overall macro F1 for regressions.",
                    })

    uncertainty: dict[str, Any] | None = None
    cleanlab_status = "Not run: valid class probability columns were not provided."
    if probabilities is not None:
        confidence = probabilities.max(axis=1)
        uncertainty = {
            "mean_max_probability": float(confidence.mean()),
            "low_confidence_count": int((confidence < .55).sum()),
            "error_mean_confidence": float(confidence[~correct].mean()) if (~correct).any() else None,
            "correct_mean_confidence": float(confidence[correct].mean()) if correct.any() else None,
        }
        low = confidence < .55
        if low.any():
            findings.append({
                "id": "low-confidence", "title": "Low-confidence predictions need review", "category": "Uncertainty",
                "priority": "Moderate", "rationale": "Ranked by the count of samples whose highest supplied class probability is below 55%. This score is not calibrated confidence unless calibration has been measured.",
                "evidence": uncertainty, "affected_count": int(low.sum()),
                "sample_ids": frame.loc[low, "__sample_id"].head(8).tolist(),
                "hypotheses": ["Predictions near class boundaries or atypical inputs may account for lower confidence; inspect the records before acting."],
                "experiment": "Review low-confidence cases and assess calibrated probabilities on an independent validation split.",
                "expected_outcome": "Report calibration and selective error rates at a chosen coverage, alongside existing classification metrics.",
            })
        if not request.predictions_out_of_sample:
            cleanlab_status = "Skipped: confirm these are out-of-sample predictions before running label-issue checks."
        else:
            try:
                from cleanlab.filter import find_label_issues
                label_to_index = {label: index for index, label in enumerate(labels)}
                encoded_labels = np.asarray([label_to_index[str(label)] for label in y_true], dtype=int)
                indices = find_label_issues(labels=encoded_labels, pred_probs=probabilities, return_indices_ranked_by="self_confidence")
                cleanlab_status = "Ran Cleanlab on supplied out-of-sample probabilities. Results are potential label-issue candidates, not confirmed mistakes."
                if len(indices):
                    candidates = frame.iloc[indices]
                    findings.append({
                        "id": "label-review-candidates", "title": "Potential label issues for human review", "category": "Data quality",
                        "priority": "Moderate", "rationale": "Cleanlab ranked candidates from supplied out-of-sample class probabilities; human review is required.",
                        "evidence": {"method": "Cleanlab find_label_issues", "candidate_count": len(indices), "input_rows": sample_count},
                        "affected_count": len(indices), "sample_ids": candidates["__sample_id"].head(8).tolist(),
                        "hypotheses": ["Some annotations may be inconsistent with the model's class probabilities; this signal can also reflect model errors or distribution shift."],
                        "experiment": "Blind-review ranked examples with a domain reviewer, then re-evaluate only after documented label adjudication.",
                        "expected_outcome": "Measure agreement and held-out metrics before and after any verified correction.",
                    })
            except ImportError:
                cleanlab_status = "Skipped: install Cleanlab to enable potential label-issue analysis."
            except Exception as exc:
                cleanlab_status = f"Skipped: Cleanlab could not evaluate these inputs ({type(exc).__name__})."
    elif probability_error:
        cleanlab_status = f"Skipped: {probability_error}"

    findings.sort(key=lambda f: ({"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}[f["priority"]], -f["affected_count"], f["title"]))
    finding_models = findings
    warnings: list[str] = []
    if excluded_missing:
        warnings.append(f"Excluded {excluded_missing} of {initial_rows} rows with missing true or predicted labels.")
    if request.sample_id_column and frame[request.sample_id_column].duplicated().any():
        warnings.append("Sample ID column contains duplicate identifiers.")
    if probability_error:
        warnings.append(probability_error)

    result = {
        "analysis_id": uuid.uuid4().hex, "dataset_id": request.dataset_id, "filename": filename,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "sample_count": sample_count,
        "labels": labels,
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
            "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
            "correct_count": int(correct.sum()), "error_count": int((~correct).sum()),
        },
        "class_metrics": class_metrics, "confusion": {"labels": labels, "matrix": cm.tolist()},
        "confusion_pairs": pair_counts[:10], "groups": group_results,
        "imbalance": {"max_support": int(max(support)), "min_support": int(min(support)),
                      "ratio": float(max(support) / max(1, min(support)))},
        "uncertainty": uncertainty,
        "data_quality": {"cleanlab": cleanlab_status, "excluded_missing_labels": excluded_missing,
                         "missing_by_column": {str(c): int(n) for c, n in original.isna().sum().items() if n},
                         "duplicate_sample_ids": int(frame[request.sample_id_column].duplicated().sum()) if request.sample_id_column else 0},
        "warnings": warnings, "findings": finding_models,
        "mapping": request.model_dump(exclude={"dataset_id"}),
    }
    _analyses[result["analysis_id"]] = {"result": result, "frame": frame, "request": request, "probabilities": probabilities}
    return result


def get_analysis(analysis_id: str) -> dict[str, Any]:
    analysis = _analyses.get(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis session expired. Run it again.")
    return analysis


def filter_samples(analysis_id: str, *, true_class: str | None, predicted_class: str | None,
                   status: str | None, group: str | None, min_confidence: float | None,
                   search: str | None, limit: int, offset: int) -> dict[str, Any]:
    analysis = get_analysis(analysis_id)
    frame = analysis["frame"]
    request: AnalysisRequest = analysis["request"]
    mask = pd.Series(True, index=frame.index)
    if true_class:
        mask &= frame[request.true_column] == true_class
    if predicted_class:
        mask &= frame[request.predicted_column] == predicted_class
    if status == "incorrect":
        mask &= ~frame["__correct"]
    elif status == "correct":
        mask &= frame["__correct"]
    if group and request.group_columns:
        mask &= frame[request.group_columns[0]].fillna("(missing)").astype(str) == group
    probabilities = analysis["probabilities"]
    confidence = probabilities.max(axis=1) if probabilities is not None else None
    if min_confidence is not None and confidence is not None:
        mask &= confidence >= min_confidence
    elif min_confidence is not None:
        raise HTTPException(status_code=422, detail="Confidence filter requires valid probability columns.")
    selected = frame.loc[mask].copy()
    if search:
        text_columns = selected.select_dtypes(include=["object", "string"]).columns
        text_match = selected[text_columns].fillna("").astype(str).apply(lambda col: col.str.contains(search, case=False, regex=False)).any(axis=1)
        selected = selected.loc[text_match]
    total = len(selected)
    page = selected.iloc[offset:offset + limit].drop(columns=["__correct", "__sample_id"], errors="ignore").copy()
    records = page.astype(object).where(pd.notna(page), None).to_dict(orient="records")
    start = offset
    for i, record in enumerate(records):
        record["sample_id"] = str(selected.iloc[start + i]["__sample_id"])
        record["correct"] = bool(selected.iloc[start + i]["__correct"])
        if confidence is not None:
            record["confidence"] = float(confidence[selected.index[start + i]])
    return {"total": total, "limit": limit, "offset": offset, "samples": records}
