# --- suppress ctranslate2/pkg_resources deprecation warning ---
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated.*",
    module=r"ctranslate2(\..*)?$",
)
# --------------------------------------------------------------


# ==============================
# Standard libs
# ==============================
import argparse
import base64
import collections
from collections import OrderedDict
import dataclasses
from dataclasses import dataclass
import glob
import io
import json
import os
import re
import sys
import tempfile
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ==============================
# Third-party (server)
# ==============================
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# ==============================
# Audio / ASR
# ==============================
# NOTE: using faster-whisper (CT2 backend) for speed & easy CPU deploy
# Fallback to normal whisper if needed.
try:
    from faster_whisper import WhisperModel as _FWModel
    _HAS_FW = True
except Exception:
    _HAS_FW = False

try:
    import whisper as _WSModel
    _HAS_WS = True
except Exception:
    _HAS_WS = False

# ==============================
# JSON tolerant loader
# ==============================
try:
    import json5
except Exception:
    # minimal tolerant loader using json with a few small cleanups
    class _Json5Compat:
        @staticmethod
        def loads(s: str):
            s2 = s.strip()
            # replace single quotes with double quotes when it appears to be JSON-ish
            if s2 and s2[0] == "{" and "'" in s2 and '"' not in s2:
                s2 = s2.replace("'", '"')
            return json.loads(s2)
    json5 = _Json5Compat()

# ==============================
# Project config
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = PROJECT_ROOT

# ==============================
# Transcriber
# ==============================
@dataclass
class TranscribeResult:
    text: str
    segments: Optional[List[dict]] = None

class Transcriber:
    def __init__(self, model_size: str = "base", device: Optional[str] = None, compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device or ("cpu")
        self.compute_type = compute_type

        self._backend = None
        if _HAS_FW:
            try:
                self._backend = _FWModel(model_size, device=self.device, compute_type=self.compute_type)
            except Exception:
                self._backend = None

        if self._backend is None and _HAS_WS:
            try:
                self._backend = _WSModel.load_model(model_size)
            except Exception:
                self._backend = None

        if self._backend is None:
            raise RuntimeError("No ASR backend available (faster-whisper or whisper).")

    def transcribe(self, path: str) -> str:
        # If a plain .txt was uploaded, pass-through
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        if _HAS_FW and isinstance(self._backend, _FWModel):
            text_chunks: List[str] = []
            try:
                segments, info = self._backend.transcribe(path, beam_size=1, vad_filter=True)
                for seg in segments:
                    text_chunks.append(seg.text)
            except Exception:
                return ""
            return " ".join(x.strip() for x in text_chunks if x and x.strip())

        # whisper fallback
        if _HAS_WS:
            try:
                result = self._backend.transcribe(path)
                return result.get("text", "").strip()
            except Exception:
                pass

        return ""

# ==============================
# Generation config
# ==============================
@dataclass
class GenCfg:
    model_name: str = "gemini-2.5-flash"
    api_key: str = os.getenv("GEMINI_API_KEY", "AIzaSyAdOoOGs7jgrxnCrpYwostMndqCINvNB2E")
    save_ai_debug: Optional[str] = None

# ==============================
# AI Truth Weaver
# ==============================
# Very tiny emotion model for consistency decisions (optional).
# Falls back to a very simple heuristic if Transformers is unavailable.
from math import exp

EMOTION_LABELS = [
    # 27 emotions from GoEmotions student variant + neutral
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

_PIPELINE = None

def _load_pipeline():
    global _PIPELINE
    try:
        from transformers import pipeline as _hf_pipeline
        _PIPELINE = _hf_pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    except Exception:
        _PIPELINE = None

def emotion_distribution(text: str) -> Dict[str, float]:
    if _PIPELINE is None:
        _load_pipeline()
    if _PIPELINE is None:
        # tiny heuristic fallback
        base = {k: 1.0 for k in EMOTION_LABELS}
        words = (text or "").lower().split()
        if any(w in words for w in ["fear","scared","afraid","worried"]):
            base["fear"] += 3.0
        s = sum(base.values()) or 1.0
        return {k: v / s for k, v in base.items()}
    try:
        outs = _PIPELINE(text, truncation=True)
        # aggregate distribution
        dist = collections.defaultdict(float)
        for item in outs[0]:
            dist[item["label"].lower()] += float(item["score"])
        s = sum(dist.values()) or 1.0
        return {k: v / s for k, v in dist.items()}
    except Exception:
        return {k: 1.0/len(EMOTION_LABELS) for k in EMOTION_LABELS}

class AITruthWeaver:
    def __init__(self, cfg: GenCfg, offline: bool = False):
        self.cfg = cfg
        self.offline = offline

    # -----------------------------
    # INNOVA schema coercion helpers
    # -----------------------------
    def _to_string(self, v) -> str:
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x).strip() for x in v if str(x).strip())
        if isinstance(v, dict):
            return ", ".join(f"{k}:{v}" for k, v in v.items())
        return "" if v is None else str(v).strip()

    def _to_list_of_str(self, v) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = "" if v is None else str(v)
        parts: List[str] = []
        for seg in re.split(r"[,\n;/]+", s):
            seg = seg.strip()
            if seg:
                parts.append(seg)
        return parts

    def _collect_numbers(self, text: str) -> list:
        return [int(n) for n in re.findall(r"\b(\d+)\b", str(text or ""))]

    def _mk_years_phrase(self, numbers: list) -> str:
        if not numbers:
            return ""
        nums = sorted(set(numbers))
        if len(nums) >= 2 and nums[0] != nums[-1]:
            return f"{nums[0]}-{nums[-1]} years"
        return f"{nums[0]} years"

    def _normalize_lie_type(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unknown"

    _REQUIRED_RT_KEYS = [
        "programming_experience",
        "programming_language",
        "skill_mastery",
        "leadership_claims",
        "team_experience",
        "skills and other keywords",
    ]

    def _enforce_innova_schema(self, subject_id: str, parsed: dict) -> dict:
        """Enforce exact schema and lightly normalize values."""
        out = {
            "shadow_id": str(subject_id),
            "revealed_truth": {},
            "deception_patterns": [],
        }
        rt = parsed.get("revealed_truth", {}) if isinstance(parsed, dict) else {}

        # Core scalar fields
        core_keys = [
            "programming_experience",
            "programming_language",
            "skill_mastery",
            "leadership_claims",
            "team_experience",
        ]
        for k in core_keys:
            v = self._to_string(rt.get(k, "unknown")) or "unknown"
            # style-only lowercasing (no ontology hardcoding)
            if k in ("programming_language", "skill_mastery", "leadership_claims", "team_experience"):
                v = v.lower()
            out["revealed_truth"][k] = v

        # Keywords list
        out["revealed_truth"]["skills and other keywords"] = self._to_list_of_str(
            rt.get("skills and other keywords", [])
        )

        # Derive compact years phrase if any numbers present anywhere
        nums = []
        for val in list(rt.values()):
            nums += self._collect_numbers(val)
        for item in (parsed.get("deception_patterns") or []):
            if isinstance(item, dict):
                nums += self._collect_numbers(item.get("lie_type", ""))
                for c in self._to_list_of_str(item.get("contradictory_claims", [])):
                    nums += self._collect_numbers(c)
        years_phrase = self._mk_years_phrase(nums)
        if years_phrase:
            out["revealed_truth"]["programming_experience"] = years_phrase or out["revealed_truth"]["programming_experience"]

        # Deception patterns normalized
        dps = []
        for item in (parsed.get("deception_patterns") or []):
            if not isinstance(item, dict):
                continue
            lie = self._normalize_lie_type(self._to_string(item.get("lie_type", "")))
            claims = self._to_list_of_str(item.get("contradictory_claims", []))
            dps.append({"lie_type": lie, "contradictory_claims": claims})
        out["deception_patterns"] = dps
        return out

    # -----------------------------
    # LLM call
    # -----------------------------
    def _gen(self, prompt: str) -> str:
        if self.offline:
            return "offline"

        import requests

        api_key = self.cfg.api_key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.cfg.model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        params = {"key": api_key}

        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        text = "unknown"
        if isinstance(data, dict):
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    text = parts[0]["text"].strip()

        # optional: save raw output for debugging
        if self.cfg.save_ai_debug:
            try:
                with open(self.cfg.save_ai_debug, "w", encoding="utf-8") as f:
                    f.write(text or "")
            except Exception:
                pass

        return text or "unknown"

    # -----------------------------
    # Public API
    # -----------------------------
    def infer(self, subject_id: str, session_texts: Dict[str, str], offline: bool = False) -> Tuple[Dict, list]:
        # Build transcript
        transcript = "\n\n".join([
            f"Session {i+1} ({name}): {session_texts[name]}"
            for i, name in enumerate(sorted(session_texts.keys()))
        ])

        # Prompt (kept semantically aligned with your earlier version)
        prompt = (
            "You are an information-extraction specialist. Read the transcript and produce ONE JSON object that "
            "STRICTLY follows the schema below. Principles: Be concise and evidence-based; prefer short phrases. "
            "If conflicting experience counts appear (e.g., '6 years' vs '3 years'), add a deception pattern with "
            "lie_type 'experience_inflation' and list the minimal conflicting phrases in contradictory_claims. "
            "If mutually incompatible statements appear (e.g., 'led a team' vs 'I mostly work alone'), add lie_type 'contradiction'. "
            "If self-doubt/insecurity is evident, add lie_type 'insecurity'. Always use snake_case for lie_type. "
            "If the transcript does not support a value, use 'unknown' (or [] for arrays). Output only the JSON.\n"
            "SCHEMA:\n"
            "{"
            "  'revealed_truth': {"
            "    'programming_experience': string,"
            "    'programming_language': string,"
            "    'skill_mastery': string,"
            "    'leadership_claims': string,"
            "    'team_experience': string,"
            "    'skills and other keywords': List[string]"
            "  },"
            "  'deception_patterns': [ { 'lie_type': string, 'contradictory_claims': List[string] } ]"
            "}\n"
            f"TRANSCRIPT:\n{transcript}"
        )

        out = self._gen(prompt)

        # Strip code fences if present
        if out.strip().startswith("```"):
            out = re.sub(r"^```[a-zA-Z]*\n|```$", "", out.strip(), flags=re.MULTILINE)

        # Extract JSON blob
        try:
            m = re.search(r"\{.*\}", out, flags=re.S)
            if m:
                matched_json = m.group(0)
                parsed = json5.loads(matched_json)
                normalized = self._enforce_innova_schema(subject_id, parsed)
                revealed = normalized["revealed_truth"]
                deception_patterns = normalized["deception_patterns"]
                return revealed, deception_patterns
        except Exception as e:
            print("Exception in infer():", e)

        # Fallback (guaranteed schema)
        revealed = {
            "programming_experience": "unknown",
            "programming_language": "unknown",
            "skill_mastery": "unknown",
            "leadership_claims": "unknown",
            "team_experience": "unknown",
            "skills and other keywords": [],
        }
        return revealed, []

# ==============================
# ------------ report ----------
# parameters extended with optional out_dir
# ==============================

SINGLE_TRANSCRIPT_NAME = "transcribed.txt"   # shared/append-only transcript (root)
SINGLE_TRUTH_NAME = "PrelimsSubmission.json"            # shared JSON array (UPSERT by shadow_id, root)

def _atomic_write_json(path: str, obj: Any) -> None:
    """
    Atomic write for JSON (avoid partial writes).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

def _ordered_truth_payload(subject_id: str, revealed: Dict, contradictions) -> OrderedDict:
    """
    Build an OrderedDict with the required key ordering.
    """
    rt = revealed or {}
    rt_ordered = OrderedDict([
        ("programming_experience", rt.get("programming_experience", "unknown")),
        ("programming_language", rt.get("programming_language", "unknown")),
        ("skill_mastery", rt.get("skill_mastery", "unknown")),
        ("leadership_claims", rt.get("leadership_claims", "unknown")),
        ("team_experience", rt.get("team_experience", "unknown")),
        ("skills and other keywords", rt.get("skills and other keywords", [])),
    ])
    payload = OrderedDict([
        ("shadow_id", subject_id),
        ("revealed_truth", rt_ordered),
        ("deception_patterns", contradictions or []),
    ])
    return payload

# --- New helper: normalize transcript text to lowercase letters & spaces only
def _normalize_transcript_text(text: str) -> str:
    """
    Keep only lowercase a–z and spaces; collapse multiple spaces.
    """
    import re as _re_norm
    text = (text or "").lower()
    text = _re_norm.sub(r'[^a-z\s]+', ' ', text)
    text = _re_norm.sub(r'\s+', ' ', text).strip()
    return text

# Legacy per-subject writers (root files)
def write_transcript(subject_id: str, session_texts: Dict[str, str], out_dir: Optional[str] = None) -> str:
    """
    Write per-file transcript with original filename + ext as header and normalized text body.
    (Kept for backward compatibility; not used by /api/analyze v2 single-file output.)
    """
    target_dir = out_dir or "."
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"transcript_{subject_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for filename_with_ext, text in session_texts.items():
            norm = _normalize_transcript_text(text)
            f.write(f"== {filename_with_ext} ==\n{norm}\n\n")
    return path

def write_truth(subject_id: str, revealed: Dict, contradictions, out_dir: Optional[str] = None) -> str:
    target_dir = out_dir or "."
    os.makedirs(target_dir, exist_ok=True)
    payload = _ordered_truth_payload(subject_id, revealed, contradictions)
    json_path = os.path.join(target_dir, f"truth_{subject_id}.json")
    _atomic_write_json(json_path, payload)
    return json_path

# New: Single-file writers (root-only by default, but support out_dir for compatibility)
def append_transcript_single(subject_id: str, session_texts: Dict[str, str], out_dir: Optional[str] = None) -> str:
    """
    Append this run's transcripts as:
    <original_filename_with_extension>: <normalized text>
    (one line per file)
    """
    target_dir = out_dir or "."
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, SINGLE_TRANSCRIPT_NAME)
    with open(path, "a", encoding="utf-8") as f:
        for filename_with_ext, text in session_texts.items():
            norm = _normalize_transcript_text(text)
            f.write(f"{filename_with_ext}: {norm}\n")
    return path

def upsert_truth_single(subject_id: str, revealed: Dict, contradictions, out_dir: Optional[str] = None) -> str:
    """
    UPSERT this subject's truth object into ONE JSON array file: truth.json
    - If the file doesn't exist, create: []
    - If an object with the same shadow_id exists, replace it; else append.
    """
    target_dir = out_dir or "."
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, SINGLE_TRUTH_NAME)

    # load existing array if present
    data: List[dict] = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
        except Exception:
            data = []

    payload = _ordered_truth_payload(subject_id, revealed, contradictions)

    # upsert by shadow_id
    idx = next(
        (i for i, obj in enumerate(data)
         if isinstance(obj, dict) and obj.get("shadow_id") == subject_id),
        None
    )
    if idx is None:
        data.append(payload)
    else:
        data[idx] = payload

    _atomic_write_json(path, data)
    return path

# ==========================
# -------------- CLI -----------
# uses Transcriber + AITruthWeaver; collects sessions and writes outputs
# ==============================
def collect_sessions(sessions_dir: str) -> Dict[str, str]:
    """
    Collect audio/text session files; returns { original_filename_with_ext: path }
    """
    exts = ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.txt")
    files = []
    for ext in exts:
        files.extend(sorted(glob.glob(os.path.join(sessions_dir, ext))))
    files.sort()
    if not files:
        return {}
    # Use the ORIGINAL BASENAME WITH EXTENSION as the key (so we can print it verbatim)
    return {os.path.basename(p): p for p in files}  # CHANGED from stem→with extension

def _cli_main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Truth Weaver CLI")
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--sessions_dir", required=True)
    parser.add_argument("--whisper_model", default="base")
    parser.add_argument("--save_outputs", action="store_true")
    ns = parser.parse_args(args)

    sess = collect_sessions(ns.sessions_dir)
    if not sess:
        print("No session files found.")
        sys.exit(2)

    # Read files and transcribe if needed
    tr = Transcriber(model_size=ns.whisper_model, device="cpu", compute_type="int8")
    session_texts: Dict[str, str] = {}
    for filename_with_ext, path in sess.items():
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = tr.transcribe(path)
        session_texts[filename_with_ext] = text

    # Run AI analysis
    weaver = AITruthWeaver(cfg=GenCfg(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY", "")))
    revealed, contradictions = weaver.infer(ns.subject_id, session_texts)

    print(json.dumps({
        "shadow_id": ns.subject_id,
        "revealed_truth": revealed,
        "deception_patterns": contradictions,
    }, indent=2, ensure_ascii=False))

    if ns.save_outputs:
        append_transcript_single(ns.subject_id, session_texts, DEFAULT_OUT_DIR)
        upsert_truth_single(ns.subject_id, revealed, contradictions, DEFAULT_OUT_DIR)

# ==============================
# -------------- Server ----------
# ==============================
def build_app() -> FastAPI:
    app = FastAPI(title="Truth Weaver", version="1.2")

    _transcriber: Optional[Transcriber] = None
    _transcriber_model: Optional[str] = None

    def get_transcriber(model_size: str = "base") -> Transcriber:
        nonlocal _transcriber, _transcriber_model
        if _transcriber is None or _transcriber_model != model_size:
            _transcriber = Transcriber(model_size=model_size, device="cpu", compute_type="int8")
            _transcriber_model = model_size
        return _transcriber

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/analyze")
    async def analyze(
        subject_id: str = Form(...),
        files: List[UploadFile] = File(...),
        whisper_model: str = Form("base"),
        save_outputs: bool = Form(False),
    ):
        if not files:
            from fastapi import HTTPException as _HTTPException
            raise _HTTPException(status_code=400, detail="No files uploaded")

        # Prepare temp files for transcription
        temp_paths: List[str] = []
        session_texts: Dict[str, str] = {}
        try:
            tr = get_transcriber(whisper_model)

            for uf in files:
                filename = uf.filename or "session"  # KEEP EXTENSION
                name, ext = os.path.splitext(filename)
                ext = ext.lower() if ext else ".wav"

                # Persist upload to a temporary file so faster-whisper can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    data = await uf.read()
                    tmp.write(data)
                    tmp_path = tmp.name
                    temp_paths.append(tmp_path)

                # Transcribe (supports .txt passthrough as well)
                text = tr.transcribe(tmp_path)
                session_texts[filename] = text  # key = original filename with extension

            # Run AI analysis
            api_key = os.getenv("GEMINI_API_KEY", GenCfg().api_key if hasattr(GenCfg, "api_key") else "")
            weaver = AITruthWeaver(cfg=GenCfg(model_name="gemini-2.5-flash", api_key=api_key))
            revealed, contradictions = weaver.infer(subject_id, session_texts)

            payload = {
                "shadow_id": subject_id,
                "revealed_truth": revealed,
                "deception_patterns": contradictions,
            }

            # Persist into single files in ROOT when requested
            if save_outputs:
                out_dir = DEFAULT_OUT_DIR # root
                append_transcript_single(subject_id, session_texts, out_dir)
                upsert_truth_single(subject_id, revealed, contradictions, out_dir)

            return JSONResponse(payload)

        finally:
            # Cleanup temp files
            for p in temp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

    # Favicon to avoid 404 in logs
    @app.get("/favicon.ico")
    def favicon():
        # 1x1 transparent PNG
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        )
        return Response(content=base64.b64decode(png_b64), media_type="image/png")

    # Serve ./web if present
    STATIC_DIR = os.path.join(PROJECT_ROOT, "web")
    if os.path.isdir(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app

# ==============================
# Entry points
# ==============================
def _serve():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)

def _router():
    if len(sys.argv) == 1:
        print("\nUsage:\n  python main.py cli --subject_id S1 --sessions_dir ./sessions\n  python main.py serve")
        print("\nExamples:\n  python main.py cli --subject_id S1 --sessions_dir ./sessions\n  python main.py serve")
        sys.exit(0)
    mode = sys.argv[1].lower()
    rest = sys.argv[2:]
    if mode == "cli":
        _cli_main(rest)
    elif mode == "serve":
        _serve()
    else:
        print(f"Unknown mode: {mode}. Use 'cli' or 'serve'.")
        sys.exit(2)

app = build_app()

if __name__ == "__main__":
    _router()

