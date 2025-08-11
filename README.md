# Truth Weaver

Analyze multiple testimony sessions (audio or text), transcribe audio locally, and use an AI model to extract a structured JSON of **revealed truths** and **deception patterns**. A built‑in web frontend lets you upload files and view results.

---

## What changed recently
- **Single root outputs**: results are saved in the project **root** as:
  - `transcribed.txt` (appended transcript log)
  - `PrelimsSubmission.json` (JSON array, upserted by `shadow_id`)
- **Simpler Web UI**: the **Subject ID is auto‑derived** from the first uploaded filename; the Analyze button enables only when files are present.
- **Unified entrypoint**: run the server via `python Prelims_Source_Code/main.py serve`

> Tip: a `.env` file at the project root is loaded automatically for environment variables (e.g., `GEMINI_API_KEY`).

---

## Project structure (key files)

```
.
├── Prelims_Source_Code/
│   └── main.py                 # FastAPI app + CLI; serves ./web and exposes /api/analyze
├── web/
│   ├── index.html              # Frontend (drag & drop uploader; inline script & styles)
│   ├── main.js                 # (Optional) legacy script not required by the new index.html
│   └── styles.css              # (Optional) legacy styles not required by the new index.html
├── requirements.txt
├── .env                        # put GEMINI_API_KEY=... here or in code
├── transcribed.txt             # created after first save (root, single file)
└── PrelimsSubmission.json      # created after first save (root, single file)
```

---

## Requirements

- **Python 3.10+**
- **ffmpeg** installed and on `PATH` (for audio decoding)
- Python packages:

```bash
pip install -r requirements.txt
# If FastAPI pieces are not included in requirements.txt on your machine:
pip install fastapi uvicorn python-multipart
```

### Environment variables
Set your Gemini API key (or place it in `.env` at the project root as `GEMINI_API_KEY=...`).

**Windows (cmd):**
```bat
set GEMINI_API_KEY=YOUR_API_KEY
```

**macOS/Linux (bash):**
```bash
export GEMINI_API_KEY=YOUR_API_KEY
```

---

## Run the Web Server

From the project root:

```bash
python Prelims_Source_Code/main.py serve
```

By default the app binds to **http://127.0.0.1:8000** and serves the frontend from the `web/` folder.

You can override host/port:

```bash
# Windows
set HOST=0.0.0.0
set PORT=8080

# macOS/Linux
export HOST=0.0.0.0
export PORT=8080

python Prelims_Source_Code/main.py serve
```

Open **http://127.0.0.1:8000** in your browser on the same machine.  
(Use `http://<your-ip>:<port>` from other devices if you bind to `0.0.0.0` and your firewall allows it.)

---

## Using the Web UI

1. Open **http://127.0.0.1:8000**.
2. Drag & drop or browse to select audio/text files (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.txt`).
3. (Optional) Check **“Save transcript and truth.json to transcribed.txt & PrelimsSubmission.json”** to persist results.
4. Click **Analyze**.
5. View/copy the JSON result from the right‑hand panel.

**Notes**
- `.txt` files are used verbatim as transcripts (handy for quick tests).
- Subject ID is **derived automatically** from the first file’s base name.
- Long audio and larger Whisper models may increase processing time.

---

## REST API

### `POST /api/analyze`

**Content‑Type**: `multipart/form-data`

**Fields**
- `subject_id` — string (the UI derives this automatically)
- `files` — one or more files (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.txt`)
- `whisper_model` — optional; defaults to `base`
- `save_outputs` — optional; `"true"`/`"false"`

**Response JSON**
```json
{
  "shadow_id": "<subject_id>",
  "revealed_truth": {
    "programming_experience": "string",
    "programming_language": "string",
    "skill_mastery": "string",
    "leadership_claims": "string",
    "team_experience": "string",
    "skills and other keywords": ["string", "..."]
  },
  "deception_patterns": [
    { "lie_type": "string", "contradictory_claims": ["string", "..."] }
  ]
}
```

### `GET /health`
Returns:
```json
{ "status": "ok" }
```

---

## CLI mode (optional)

Batch‑process a folder of sessions without the web UI:

```bash
python Prelims_Source_Code/main.py cli --subject_id <ID> --sessions_dir <path-to-sessions> --out_dir . --whisper_model base
```

---

## Outputs

When **Save** is checked (or `save_outputs=true` in API):
- **Transcript (single file, appended)**: `transcribed.txt`
- **JSON (single file, upserted by shadow_id)**: `PrelimsSubmission.json`

> Legacy per‑subject writers also exist and write `truth_<subject_id>.json` and `transcript_<subject_id>.txt` to the chosen `--out_dir` if you use the lower‑level functions.

---

## Troubleshooting

- **Cannot reach** `http://0.0.0.0:8000` in a browser  
  `0.0.0.0` is a bind address. From the same machine use `http://127.0.0.1:8000`.

- **ImportError: uvicorn / fastapi missing**  
  `pip install fastapi uvicorn python-multipart`

- **Audio decoding errors**  
  Ensure **ffmpeg** is installed and on PATH.

- **No AI output / empty fields**  
  Check `GEMINI_API_KEY` and network access. Use `--offline` to test transcription only.

---

## License

This project may process audio/text that you supply. Ensure you have the right to process that content.
