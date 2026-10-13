from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .analysis import get_analysis

DB_PATH = Path(os.getenv("MODELSLEUTH_DB", Path(__file__).resolve().parents[2] / "data" / "modelsleuth.sqlite3"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize() -> None:
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            name TEXT NOT NULL,
            condition_note TEXT NOT NULL DEFAULT '',
            is_baseline INTEGER NOT NULL DEFAULT 0,
            dataset TEXT NOT NULL,
            created_at TEXT NOT NULL,
            result_json TEXT NOT NULL
        )""")


def list_experiments() -> list[dict[str, Any]]:
    initialize()
    with connect() as db:
        rows = db.execute("SELECT id,analysis_id,name,condition_note,is_baseline,dataset,created_at,result_json FROM experiments ORDER BY id DESC").fetchall()
    return [{**dict(row), "is_baseline": bool(row["is_baseline"]), "result": json.loads(row["result_json"])} for row in rows]


def save_experiment(analysis_id: str, name: str, condition: str, baseline: bool) -> dict[str, Any]:
    analysis = get_analysis(analysis_id)["result"]
    initialize()
    with connect() as db:
        if baseline:
            db.execute("UPDATE experiments SET is_baseline=0")
        cursor = db.execute(
            "INSERT INTO experiments(analysis_id,name,condition_note,is_baseline,dataset,created_at,result_json) VALUES(?,?,?,?,?,?,?)",
            (analysis_id, name[:100], condition[:300], int(baseline), analysis["filename"], analysis["created_at"], json.dumps(analysis)),
        )
        row_id = cursor.lastrowid
    return next(entry for entry in list_experiments() if entry["id"] == row_id)


def compare_experiments(first_id: int, second_id: int) -> dict[str, Any]:
    entries = {entry["id"]: entry for entry in list_experiments()}
    if first_id not in entries or second_id not in entries:
        raise HTTPException(status_code=404, detail="Choose two saved experiments.")
    if first_id == second_id:
        raise HTTPException(status_code=422, detail="Select two different experiment snapshots.")
    before, after = entries[first_id], entries[second_id]
    a, b = before["result"], after["result"]
    metric_delta = {key: b["metrics"][key] - a["metrics"][key] for key in ("accuracy",)}
    for avg in ("macro", "weighted"):
        for metric in ("precision", "recall", "f1"):
            metric_delta[f"{avg}_{metric}"] = b["metrics"][avg][metric] - a["metrics"][avg][metric]
    before_classes = {c["class"]: c for c in a["class_metrics"]}
    after_classes = {c["class"]: c for c in b["class_metrics"]}
    class_delta = {label: {m: after_classes[label][m] - before_classes[label][m] for m in ("precision", "recall", "f1")}
                   for label in before_classes.keys() & after_classes.keys()}
    def keyed_groups(result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        return {(g["column"], g["group"]): g for g in result["groups"]}
    groups_a, groups_b = keyed_groups(a), keyed_groups(b)
    group_delta = {f"{column}={group}": {
        "accuracy": groups_b[(column, group)]["accuracy"] - groups_a[(column, group)]["accuracy"],
        "macro_f1": groups_b[(column, group)]["macro_f1"] - groups_a[(column, group)]["macro_f1"],
        "support_before": groups_a[(column, group)]["support"],
        "support_after": groups_b[(column, group)]["support"],
    } for column, group in groups_a.keys() & groups_b.keys()}
    confusion_labels = sorted(set(a["confusion"]["labels"]) | set(b["confusion"]["labels"]))
    def align_confusion(result: dict[str, Any]) -> dict[str, list[list[int]]]:
        old_labels = result["confusion"]["labels"]
        old_matrix = result["confusion"]["matrix"]
        return {"matrix": [
            [old_matrix[old_labels.index(label)][old_labels.index(target)] if label in old_labels and target in old_labels else 0
             for target in confusion_labels] for label in confusion_labels
        ]}
    matrix_a, matrix_b = align_confusion(a)["matrix"], align_confusion(b)["matrix"]
    confusion_delta = {"labels": confusion_labels, "matrix": [[matrix_b[i][j] - matrix_a[i][j] for j in range(len(confusion_labels))] for i in range(len(confusion_labels))]}
    regressions = [f"{label} {metric}" for label, deltas in class_delta.items() for metric, delta in deltas.items() if delta < -.02]
    same_conditions = (a["filename"] == b["filename"] and a["sample_count"] == b["sample_count"] and a["labels"] == b["labels"])
    return {
        "first": {"id": before["id"], "name": before["name"], "dataset": before["dataset"], "sample_count": a["sample_count"]},
        "second": {"id": after["id"], "name": after["name"], "dataset": after["dataset"], "sample_count": b["sample_count"]},
        "metric_delta": metric_delta, "class_delta": class_delta,
        "confusion_before": a["confusion"], "confusion_after": b["confusion"], "confusion_delta": confusion_delta,
        "group_before": a["groups"], "group_after": b["groups"], "group_delta": group_delta,
        "regressions": regressions,
        "comparable_signature": same_conditions,
        "warning": "Dataset name, sample count, or class set differs. Confirm evaluation split and conditions before interpreting this comparison." if not same_conditions else "Dataset signature matches. Confirm that evaluation split and preprocessing conditions also match.",
    }
