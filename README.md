# ModelSleuth

ModelSleuth is a local-first ML failure investigation website. A React + TypeScript frontend talks to a FastAPI analysis service. Metrics and findings are calculated from uploaded rows; an optional Gemma 4 provider can explain a selected finding from a compact aggregate summary.

The interface is an editorial paper investigation desk. The landing scene uses an interactive Three.js paper stack on desktop. Its generated sheets are explicitly illustrative, not live metrics. Small screens and reduced-motion settings use an interactive CSS paper fallback.

## Run on Windows

Use Python 3.10+ and Node.js 20.19+ (Node 22 is recommended). From the repository root, create the backend environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
npm --prefix frontend install
```

Open two PowerShell windows in the repository root. Start the API in the first:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Start the website in the second:

```powershell
npm --prefix frontend run dev -- --host 127.0.0.1
```

Open the Vite URL printed in the second window (normally `http://localhost:5173`). The API health route is `http://localhost:8000/api/health`; interactive API docs are at `http://localhost:8000/docs`.

## Try the demo

Select **Try interactive demo** from the landing page, or open the workspace and choose **Load synthetic demo**. It generates a reproducible 720-row classification evaluation with controlled confusion, a small amount of label noise, class probabilities, and two sensor groups with different error rates. The same API analysis engine calculates every shown metric and finding. Demo results are marked as synthetic.

## Analyze a CSV

Upload a UTF-8 CSV up to 20 MB and 300,000 rows. Map distinct true-label and predicted-label columns. Sample IDs, a metadata group, and one probability column per observed class are optional. Probabilities must be finite values in `[0, 1]` and sum to approximately one per row. A confirmation is required before Cleanlab will run, because its potential label-issue checks require out-of-sample probabilities. Any flagged rows are candidates for human review, not confirmed label errors.

Missing true/predicted labels are excluded and reported. Other columns remain available in sample review. The service holds at most eight recent dataset sessions and caps aggregate in-memory dataset storage; restarting the API clears uploaded data. The app does not persist uploaded CSV files; the multipart server may briefly buffer a request in the operating system's temporary storage before validation.

## Optional Gemma 4 explanations

The app works without an AI key. Without one, **Ask Gemma** returns a deterministic evidence-based template. To enable hosted Gemma 4, copy `.env.example` to `.env`, set `LLM_API_KEY`, and restart the backend. Chat can race Google AI and NVIDIA NIM using Gemma 4 31B and display the first successful answer. Configure one or both API keys; when both are set, both providers receive the submitted prompt and attachments. Mapping and finding explanations use NVIDIA when its key is configured. The credential stays in the backend environment, is never returned to the UI, and no model weights are downloaded. The application sends only aggregate metrics and the selected finding when the user explicitly requests an explanation; it omits sample IDs and CSV rows.

Gemma is used as an **evidence interpreter, experiment planner, optional column-mapping assistant, and chat-based investigation partner**, not as the classifier. In Analyze Data, **Suggest mapping** sends column names/types and class names only; it does not send row values. Gemma can explain measured class confusions or group gaps, list possible causes as untested hypotheses, suggest up to three ranked experiments, and state how to validate results. Click **Ask Gemma** on a finding to request this explanation, or open the floating **Field Assistant** to ask questions and attach images, PDFs, CSVs, text, Markdown, TSV, JSON, or log files (up to 4 files, 5 MB each). Image attachments are sent to the configured NVIDIA NIM Gemma 4 provider; PDFs are text-extracted locally before their text is sent. Chat content is sent only when submitted. It cannot create the model's predictions, confirm causes, inspect spectrograms, or give clinical advice. Keep the final test split untouched while tuning; measure every proposed change with the deterministic analysis engine.

```dotenv
LLM_PROVIDER=race
LLM_API_KEY=your_nvidia_api_key
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
LLM_MODEL=google/gemma-4-31b-it
GEMINI_API_KEY=your_google_api_key
GEMINI_ENDPOINT=https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
GEMINI_MODEL=gemma-4-31b-it
```

For fastest chat, set `LLM_PROVIDER=race` and configure either or both keys: `LLM_API_KEY` from [NVIDIA Build](https://build.nvidia.com/google/gemma-4-31b-it), and `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey). When both are configured, the chat request goes to both Gemma 4 endpoints concurrently and the first successful answer is shown. Keep credentials in `.env`. Do not commit `.env` or share its credentials.


## Local experiment history

Saved snapshots are persisted in `backend/data/modelsleuth.sqlite3` (override with `MODELSLEUTH_DB`). The database records aggregate results and evaluation context, not uploaded feature rows. Comparisons show measured metric differences, class regressions, group deltas, and aligned confusion-matrix changes. A matching dataset name/count/class signature is only a hint; confirm split and preprocessing conditions yourself.

## Checks

```powershell
python -m unittest discover -s backend\tests -v
npm --prefix frontend run lint
npm --prefix frontend run build
```

The tests cover upload validation, actual metrics, group findings, filters, probability validation, deterministic demo output, fallback explanations, and SQLite comparisons.

## Project map

- `frontend/src/` — React application, workspace pages, paper design system, 3D hero, and typed API client.
- `backend/app/main.py` — FastAPI routes and request validation.
- `backend/app/services/analysis.py` — metrics, confusion/group/uncertainty analysis, and traceable findings.
- `backend/app/services/datasets.py` — CSV validation, bounded in-memory dataset sessions, and synthetic demo generation.
- `backend/app/services/experiments.py` — SQLite snapshot history and measured comparisons.
- `backend/app/llm/provider.py` — Gemma 4 and OpenAI-compatible provider adapter with validated template fallback.
- `backend/tests/` — backend workflow tests.

Potential Cleanlab checks require valid class probabilities and out-of-sample confirmation. If Cleanlab is unavailable, or the user doesn't confirm the required prediction conditions, the rest of the evaluation still runs and the skipped check is explained. The backend memory store is for local development; this app does not include multi-user authentication or remote deployment hardening.

