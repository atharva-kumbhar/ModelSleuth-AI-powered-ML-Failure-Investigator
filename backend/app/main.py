from __future__ import annotations

import io
import json
import base64
from queue import Queue
from threading import Thread
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .llm.provider import chat_reply, explain, provider_status, suggest_mapping
from .schemas import AnalysisRequest, MappingSuggestionRequest
from .services.analysis import filter_samples, get_analysis, run_analysis
from .services.datasets import demo_frame, get_dataset, preview_dataset, read_upload
from .services.experiments import compare_experiments, list_experiments, save_experiment

app = FastAPI(title="ModelSleuth API", version="1.0.0", description="Evidence-first classification failure investigation")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])


class SaveExperiment(BaseModel):
    analysis_id: str
    name: str = Field(min_length=1, max_length=100)
    condition: str = Field(default="", max_length=300)
    baseline: bool = False


class CompareRequest(BaseModel):
    first_id: int
    second_id: int


class ExplainRequest(BaseModel):
    analysis_id: str
    finding_id: str
    summary: dict[str, Any] | None = None
    finding: dict[str, Any] | None = None


class ChatHistoryMessage(BaseModel):
    role: str
    content: str = Field(max_length=3000)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ModelSleuth API", "llm": provider_status()}


@app.get("/api/ready")
def readiness() -> dict[str, Any]:
    from .services.experiments import initialize
    initialize()
    return {"ready": True, "database": "ready", "llm_required": False}


@app.get("/api/settings")
def settings() -> dict[str, Any]:
    return {"backend": "connected", "llm": provider_status(), "privacy": "CSV data stays in local backend memory. AI mapping sends column names/types only when requested. Finding explanations send aggregate evidence only. Chat sends your submitted question and attached files to the configured provider; active-case context contains aggregate metrics only.",
            "upload_limit_mb": 20, "cleanlab": "optional; enabled when installed and valid class probabilities are supplied"}


@app.post("/api/datasets/validate")
async def validate_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    name, frame = await read_upload(file)
    return preview_dataset(name, frame)


@app.post("/api/datasets/suggest-mapping")
def suggest_dataset_mapping(request: MappingSuggestionRequest) -> dict[str, Any]:
    _, frame = get_dataset(request.dataset_id)
    columns = [{
        "name": str(column), "dtype": str(frame[column].dtype),
        "numeric": bool(pd.api.types.is_numeric_dtype(frame[column])),
        "unique_count": int(frame[column].nunique(dropna=True)),
        "missing": int(frame[column].isna().sum()),
    } for column in frame.columns]
    label_candidates = [column for column in columns if not column["numeric"] and 2 <= column["unique_count"] <= 20]
    likely_truth = next((column for column in label_candidates if str(column["name"]).lower().strip() in {
        "true", "truth", "label", "target", "actual", "y_true", "true_label", "ground_truth"
    }), label_candidates[0] if label_candidates else None)
    labels = sorted(frame[likely_truth["name"]].dropna().astype(str).unique().tolist()) if likely_truth else []
    if len(labels) > 20:
        labels = []
    return suggest_mapping(columns, labels)


@app.post("/api/analyze")
def analyze(request: AnalysisRequest) -> dict[str, Any]:
    return run_analysis(request)


@app.get("/api/analysis/{analysis_id}")
def analysis_detail(analysis_id: str) -> dict[str, Any]:
    return get_analysis(analysis_id)["result"]


@app.get("/api/analysis/{analysis_id}/samples")
def samples(analysis_id: str, true_class: str | None = None, predicted_class: str | None = None,
            status: str | None = Query(default=None, pattern="^(correct|incorrect)$"), group: str | None = None,
            min_confidence: float | None = Query(default=None, ge=0, le=1), search: str | None = Query(default=None, max_length=120),
            limit: int = Query(default=60, ge=1, le=200), offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    return filter_samples(analysis_id, true_class=true_class, predicted_class=predicted_class, status=status,
                          group=group, min_confidence=min_confidence, search=search, limit=limit, offset=offset)


@app.post("/api/demo")
def demo() -> dict[str, Any]:
    from .services.datasets import remember
    frame = demo_frame()
    dataset_id = remember("modelsleuth-synthetic-demo.csv", frame)
    return run_analysis(AnalysisRequest(dataset_id=dataset_id, true_column="true_label", predicted_column="predicted_label",
                                        sample_id_column="sample_id", group_columns=["sensor_region"],
                                        probability_columns={label: f"prob_{label}" for label in ("canopy", "water", "urban")},
                                        predictions_out_of_sample=True))


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    return list_experiments()


@app.post("/api/experiments")
def create_experiment(request: SaveExperiment) -> dict[str, Any]:
    return save_experiment(request.analysis_id, request.name.strip(), request.condition.strip(), request.baseline)


@app.post("/api/experiments/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    return compare_experiments(request.first_id, request.second_id)


@app.post("/api/explain")
def explain_finding(request: ExplainRequest) -> dict[str, Any]:
    try:
        result = get_analysis(request.analysis_id)["result"]
    except HTTPException as exc:
        if exc.status_code != 404 or request.summary is None or request.finding is None:
            raise
        if request.finding.get("id") != request.finding_id:
            raise HTTPException(status_code=422, detail="Finding does not match the requested explanation.") from exc
        allowed = {"sample_count", "metrics", "class_metrics", "confusion_pairs", "groups", "uncertainty"}
        summary = {key: value for key, value in request.summary.items() if key in allowed}
        return explain(summary, request.finding)
    finding = next((f for f in result["findings"] if f["id"] == request.finding_id), None)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found in this analysis.")
    summary = {key: result[key] for key in ("sample_count", "metrics", "class_metrics", "confusion_pairs", "groups", "uncertainty")}
    return explain(summary, finding)


@app.post("/api/chat")
async def assistant_chat(message: str = Form(..., min_length=1, max_length=8000),
                         history: str = Form(default="[]", max_length=30000),
                         analysis_id: str = Form(default="", max_length=100),
                         files: list[UploadFile] = File(default=[])) -> dict[str, str]:
    if len(files) > 4:
        raise HTTPException(status_code=413, detail="Attach up to 4 files per message.")
    try:
        raw_history = json.loads(history)
        messages = [ChatHistoryMessage.model_validate(item).model_dump() for item in raw_history[-8:]]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Chat history must be a list of recent messages.") from exc
    context = None
    if analysis_id:
        try:
            result = get_analysis(analysis_id)["result"]
            context = {key: result[key] for key in ("filename", "sample_count", "labels", "metrics", "class_metrics", "confusion_pairs", "groups", "uncertainty", "findings") if key in result}
            context["findings"] = [{k: f[k] for k in ("title", "category", "priority", "rationale", "evidence", "affected_count", "hypotheses", "experiment") if k in f} for f in context.get("findings", [])]
        except HTTPException:
            raise
    attachments: list[dict[str, Any]] = []
    allowed_text = {".txt", ".csv", ".tsv", ".md", ".json", ".log"}
    allowed_images = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        data = await file.read(5 * 1024 * 1024 + 1)
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{file.filename} exceeds the 5 MB per-file limit.")
        if suffix in allowed_text:
            attachments.append({"name": (file.filename or "attachment")[:120], "text": data.decode("utf-8-sig", errors="replace")[:30000]})
        elif suffix in allowed_images:
            attachments.append({"name": (file.filename or "image")[:120], "mime_type": allowed_images[suffix], "data": base64.b64encode(data).decode("ascii")})
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                extracted = "\n".join((page.extract_text() or "") for page in reader.pages[:20])[:30000]
                attachments.append({"name": (file.filename or "document.pdf")[:120], "text": f"PDF text extraction (first 20 pages):\n{extracted or '[No extractable text found; this PDF may be scanned.]'}"})
            except ImportError as exc:
                raise HTTPException(status_code=415, detail="PDF reading support is unavailable; install the backend requirements.") from exc
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Could not read PDF {file.filename}.") from exc
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix or 'unknown'}. Attach images, PDF, TXT, CSV, TSV, Markdown, JSON, or LOG files.")
    try:
        if provider_status()["provider"] in {"openai_compatible", "google", "race"}:
            events: Queue[tuple[str, str] | None] = Queue()

            def run_chat() -> None:
                try:
                    chat_reply(message.strip(), messages, context, attachments,
                               on_token=lambda token: events.put(("token", token)))
                except RuntimeError as exc:
                    events.put(("error", str(exc)))
                finally:
                    events.put(None)

            def stream_events():
                Thread(target=run_chat, daemon=True).start()
                while True:
                    event = events.get()
                    if event is None:
                        break
                    kind, content = event
                    if kind == "error":
                        yield f"event: error\ndata: {json.dumps({'detail': content})}\n\n"
                        return
                    yield f"data: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return chat_reply(message.strip(), messages, context, attachments)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/export/{analysis_id}")
def export_analysis(analysis_id: str) -> StreamingResponse:
    result = get_analysis(analysis_id)["result"]
    return StreamingResponse(io.BytesIO(json.dumps(result, indent=2).encode()), media_type="application/json",
                             headers={"Content-Disposition": f'attachment; filename="modelsleuth-{analysis_id[:8]}.json"'})
