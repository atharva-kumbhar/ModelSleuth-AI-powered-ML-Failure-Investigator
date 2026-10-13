from __future__ import annotations

import csv
import io
import uuid
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 300_000
_datasets: OrderedDict[str, tuple[str, pd.DataFrame]] = OrderedDict()
_dataset_bytes: dict[str, int] = {}


def remember(name: str, frame: pd.DataFrame) -> str:
    size = int(frame.memory_usage(index=True, deep=True).sum())
    if size > 128 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV expands beyond the 128 MB in-memory dataset limit.")
    while _datasets and (len(_datasets) >= 8 or sum(_dataset_bytes.values()) + size > 384 * 1024 * 1024):
        oldest, _ = _datasets.popitem(last=False)
        _dataset_bytes.pop(oldest, None)
    dataset_id = uuid.uuid4().hex
    _datasets[dataset_id] = (name, frame)
    _dataset_bytes[dataset_id] = size
    return dataset_id


def get_dataset(dataset_id: str) -> tuple[str, pd.DataFrame]:
    value = _datasets.get(dataset_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Dataset session expired. Upload the CSV again.")
    _datasets.move_to_end(dataset_id)
    return value


def demo_frame() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    classes = np.array(["canopy", "water", "urban"])
    rows: list[dict[str, Any]] = []
    for i in range(720):
        group = "sensor-east" if i % 3 else "sensor-west"
        truth = str(rng.choice(classes, p=[0.43, 0.34, 0.23]))
        # Controlled errors: west sensor confuses canopy with urban; a small
        # noise rate models ambiguous labels without asserting they are wrong.
        error_rate = 0.31 if group == "sensor-west" else 0.10
        pred = truth
        if rng.random() < error_rate:
            pred = "urban" if truth == "canopy" else ("canopy" if truth == "urban" else "canopy")
        if rng.random() < 0.025:
            truth = str(rng.choice(classes[classes != truth]))
        probs = rng.dirichlet([5.5, 2.2, 2.2])
        class_index = int(np.where(classes == pred)[0][0])
        probs = probs * 0.35
        probs[class_index] += 0.65
        probs /= probs.sum()
        rows.append({
            "sample_id": f"tile-{i + 1:04d}", "true_label": truth,
            "predicted_label": pred, "sensor_region": group,
            **{f"prob_{label}": float(probs[j]) for j, label in enumerate(classes)},
        })
    return pd.DataFrame(rows)


async def read_upload(file: UploadFile) -> tuple[str, pd.DataFrame]:
    name = (file.filename or "upload.csv").strip()
    if not name.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 20 MB upload limit.")
    try:
        header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise HTTPException(status_code=400, detail="CSV header could not be read as UTF-8.") from exc
    if len(header) != len(set(header)):
        raise HTTPException(status_code=400, detail="CSV contains duplicate column names. Rename them and upload again.")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    if frame.empty or len(frame.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV must have rows and at least two columns.")
    if len(frame) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"CSV exceeds the {MAX_ROWS:,} row limit.")
    return name[:160], frame


def preview_dataset(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    missing = {str(col): int(count) for col, count in frame.isna().sum().items() if count}
    duplicate_columns = [str(c) for c in frame.columns[frame.columns.duplicated()]]
    if duplicate_columns:
        raise HTTPException(status_code=400, detail="CSV contains duplicate column names.")
    candidates = {"true_column": None, "predicted_column": None, "sample_id_column": None}
    for col in frame.columns:
        key = str(col).lower().strip()
        if key in {"true", "truth", "label", "target", "actual", "y_true", "true_label"}:
            candidates["true_column"] = str(col)
        if key in {"prediction", "pred", "predicted", "y_pred", "predicted_label"}:
            candidates["predicted_column"] = str(col)
        if key in {"id", "sample_id", "record_id", "image_id"}:
            candidates["sample_id_column"] = str(col)
    return {
        "dataset_id": remember(name, frame), "filename": name,
        "row_count": len(frame), "column_count": len(frame.columns),
        "columns": [{"name": str(c), "dtype": str(frame[c].dtype), "missing": int(frame[c].isna().sum()),
                     "numeric": bool(pd.api.types.is_numeric_dtype(frame[c])),
                     "categories": sorted(frame[c].dropna().astype(str).unique().tolist()) if frame[c].nunique(dropna=True) <= 12 else []} for c in frame.columns],
        "missing_by_column": missing,
        "suggested_mapping": candidates,
        "preview": frame.head(8).replace({np.nan: None}).to_dict(orient="records"),
        "warnings": ([f"{len(missing)} column(s) contain missing values; label rows with missing labels will be excluded."] if missing else []),
    }
