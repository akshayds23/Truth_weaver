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
import glob
import json
import os
import re
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# === project-aware paths & .env ===
import os
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    _DOTENV_OK = True
except Exception:
    _DOTENV_OK = False

HERE = os.path.dirname(os.path.abspath(__file__))          # .../Prelims_Source_Code
PROJECT_ROOT = os.path.dirname(HERE)                        # project/
DEFAULT_OUT_DIR = PROJECT_ROOT                              # write transcript.txt / truth.json in project root

if _DOTENV_OK:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
# ================================


# ==============================
# ---------- emotions ----------
# ==============================
# Lightweight emotion (multi-label) + sentiment helper for transcripts.
# Uses a small GoEmotions-student DistilBERT to get emotion probabilities.
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
    if _PIPELINE is not None:
        return _PIPELINE
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        _PIPELINE = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, function_to_apply="sigmoid")
    except Exception:
        _PIPELINE = None
    return _PIPELINE

def _softmax(xs: List[float]) -> List[float]:
    m = max(xs) if xs else 0.0
    exps = [exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def analyze_emotions(text: str, top_k: int = 5) -> Dict[str, float]:
    """
    Return a dict of emotion -> score in [0,1].
    Uses HF pipeline if available, else a tiny keyword heuristic.
    """
    text = (text or "").strip()
    if not text:
        return {"neutral": 1.0}

    pipe = _load_pipeline()
    if pipe is None:
        # Heuristic fallback
        t = text.lower()
        score: Dict[str, float] = {}
        def add(lbl, val): score[lbl] = min(1.0, score.get(lbl, 0.0) + val)
        for w in ("maybe","not sure","i think","perhaps","guess","probably","unsure"):
            if w in t: add("nervousness", 0.3); add("confusion", 0.2)
        for w in ("angry","mad","furious","irritated"):
            if w in t: add("anger", 0.7)
        for w in ("scared","afraid","fear","panic"):
            if w in t: add("fear", 0.7)
        if not score:
            score["neutral"] = 0.8
        # normalize
        s = sum(score.values()) or 1.0
        return {k: v/s for k,v in score.items()}

    # HF pipeline path
    out = pipe(text[:4000])  # limit for speed
    # out is list[list[{"label":..., "score":...}]]
    if isinstance(out, list) and out and isinstance(out[0], list):
        scores = {d["label"].lower(): float(d["score"]) for d in out[0] if "label" in d and "score" in d}
        # Keep only known labels; normalize
        filtered = {k: scores.get(k, 0.0) for k in EMOTION_LABELS}
        s = sum(filtered.values()) or 1.0
        norm = {k: v/s for k, v in filtered.items()}
        # take top_k for brevity
        top = dict(sorted(norm.items(), key=lambda kv: kv[1], reverse=True)[:top_k])
        return top
    return {"neutral": 1.0}

def summarize_confidence(emotion_scores: Dict[str, float]) -> float:
    """
    A crude scalar confidence in [0,1] derived from emotions.
    Higher with joy/neutral/relief/pride/admiration; lower with fear/anger/confusion/grief.
    """
    pos = ["neutral","joy","relief","pride","admiration","approval","gratitude","love","optimism"]
    neg = ["fear","anger","disgust","disapproval","sadness","grief","nervousness","confusion","remorse","embarrassment"]
    p = sum(emotion_scores.get(k,0.0) for k in pos)
    n = sum(emotion_scores.get(k,0.0) for k in neg)
    s = p + n
    if s <= 1e-9: return 0.5
    # map to 0..1 with a bit of bias to center
    raw = (p - n) / s  # -1..1
    return 0.5 + 0.5*raw

# ==============================
# ----------- hedges -----------
# ==============================
import re as _re

HEDGES = [
    "maybe","perhaps","i guess","i think","i believe","i feel","likely","unlikely","probably","possibly",
    "seems","appears","around","roughly","sort of","kind of","not sure","unsure","i'm not certain",
    "i cannot recall","i can't recall","i don't remember","don't remember","approximately","about"
]
ASSERTIVES = [
    "definitely","certainly","clearly","for sure","without a doubt","no doubt","exactly","precisely",
    "must","undeniably","always","never"
]

def hedge_score(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    # count occurrences (word-boundary aware where possible)
    def count_any(words):
        c = 0
        for w in words:
            if "\\'" in w:
                # already escaped
                pattern = w
            else:
                pattern = r"\b" + _re.escape(w) + r"\b"
            c += len(_re.findall(pattern, t))
        return c
    h = count_any(HEDGES)
    a = count_any(ASSERTIVES)
    denom = h + a if (h + a) > 0 else 1
    ratio = h / denom
    if ratio < 0.25: bucket = "low"
    elif ratio < 0.6: bucket = "medium"
    else: bucket = "high"
    return {"hedge_ratio": float(ratio), "hedge_bucket": bucket, "hedges": h, "assertives": a}

# ==============================
# ----------- audio ------------
# ==============================
import numpy as np

def trim_silence(y: np.ndarray, top_db: int = 25) -> np.ndarray:
    """
    Remove leading/trailing and internal long silences using energy-based splitting.
    """
    import librosa
    intervals = librosa.effects.split(y, top_db=top_db)  # keeps non-silent intervals
    if len(intervals) == 0:
        return y
    chunks = [y[s:e] for s, e in intervals]
    y_trim = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return y_trim

def load_and_enhance(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load any FFmpeg-readable audio, mono @16k, denoise, trim silence, normalize.
    """
    import librosa, noisereduce as nr
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Noise profile from the first 0.5s (fallback to a small slice if shorter)
    n_samples = int(0.5 * sr)
    noise_profile = y[:n_samples] if len(y) > n_samples else y[: max(1, len(y)//10)]
    try:
        y = nr.reduce_noise(y=y, y_noise=noise_profile, sr=sr, prop_decrease=0.9)
    except Exception:
        pass

    # Trim silences
    try:
        y = trim_silence(y, top_db=25)
    except Exception:
        pass
    

    # Normalize
    max_abs = np.max(np.abs(y)) + 1e-9
    y = 0.98 * (y / max_abs)
    return y

def write_wav_temp(y: np.ndarray, sr: int = 16000) -> str:
    import soundfile as sf
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp_path, y, sr)
    return tmp_path

# ==============================
# --------- transcriber --------
# ==============================
class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        model_size: tiny/base/small/medium/large-v3
        device: "cpu" or "cuda"
        compute_type: "int8" (CPU), "float16" (CUDA), etc.
        """
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, path: str) -> str:
        # Allow .txt stand-ins for testing
        if path.lower().endswith(".txt") and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()

        y = load_and_enhance(path)
        tmp_wav = write_wav_temp(y, 16000)
        try:
            segments, _ = self.model.transcribe(
                tmp_wav,
                beam_size=5,
                vad_filter=False,  # energy-based trimming used above
                temperature=0.0
            )
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            try: os.remove(tmp_wav)
            except OSError: pass

# ==============================
# ---- ai_analyzer_signals -----
# adapted to reference local helpers
# ==============================
import json5

@dataclass
class GenCfg:
    model_name: str = "gemini-2.5-flash"
    api_key: str = "AIzaSyAdOoOGs7jgrxnCrpYwostMndqCINvNB2E"
    save_ai_debug: Optional[str] = None

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

    _REQUIRED_RT_KEYS = [
        "programming_experience",
        "programming_language",
        "skill_mastery",
        "leadership_claims",
        "team_experience",
        "skills and other keywords",
    ]

    def _enforce_innova_schema(self, subject_id: str, parsed: dict) -> dict:
        """
        Enforce EXACT INNOVA 'Submission Scroll' schema & types.
        """
        out = {
            "shadow_id": str(subject_id),
            "revealed_truth": {},
            "deception_patterns": [],
        }

        rt = parsed.get("revealed_truth", {}) if isinstance(parsed, dict) else {}

        # five scalar string fields
        for k in self._REQUIRED_RT_KEYS[:-1]:
            out["revealed_truth"][k] = self._to_string(rt.get(k, "unknown")) or "unknown"

        # the list field
        out["revealed_truth"]["skills and other keywords"] = self._to_list_of_str(
            rt.get("skills and other keywords", [])
        )

        # deception_patterns: list of dicts with exact keys & types
        dps = []
        for item in (parsed.get("deception_patterns") or []):
            if not isinstance(item, dict):
                continue
            lie = self._to_string(item.get("lie_type", "")) or "unknown"
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

        try:
            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=180)
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Gemini API request timed out after 120s") from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            snippet = ""
            if exc.response is not None and exc.response.text:
                snippet = f" (body: {exc.response.text[:200].strip()})"
            raise RuntimeError(f"Gemini API returned HTTP {status}{snippet}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

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
            "You are Truth Weaver, an AI detective. Analyze the following transcript of five testimonies. "
            "Extract the following fields and return a single valid JSON object in this format. "
            "The filed should be filled filled exact keywords not sentences and if multiple keywords are present, return them as a list. "
            "ALWAYS fill every field with your best guess based on the transcript, even if uncertain. Only use 'unknown' or [] if absolutely no information is available. "
            "In decepetion patterns, find all the lie type with separate contracting claims for each in different section. "
            "Never leave any field blank. If you do not comply, your output will be discarded.\n"
            "Follow this JSON format exactly and return the exact format of the values fro each keys as given below:\n"
            "{\n"
            "  'revealed_truth': {\n"
            "    'programming_experience': string,\n"
            "    'programming_language': string,\n"
            "    'skill_mastery': string,\n"
            "    'leadership_claims': string,\n"
            "    'team_experience': string,\n"
            "    'skills and other keywords': List[string]\n"
            "  },\n"
            "  'deception_patterns': [\n"
            "    { 'lie_type': string, 'contradictory_claims': List[string] }, ...\n"
            "  ]\n"
            "}\n"
            ""
            "Return only the JSON object, nothing else. If you cannot infer a field, use 'unknown' or [].\n\n"
            f"Transcript:\n{transcript}"
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
        ("programming_experience", str(rt.get("programming_experience", "unknown"))),
        ("programming_language",   str(rt.get("programming_language", "unknown"))),
        ("skill_mastery",          str(rt.get("skill_mastery", "unknown"))),
        ("leadership_claims",      str(rt.get("leadership_claims", "unknown"))),
        ("team_experience",        str(rt.get("team_experience", "unknown"))),
        ("skills and other keywords", list(rt.get("skills and other keywords", []))),
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
    Legacy behavior adapted to root:
    Writes ./transcript_<subject_id>.txt (or <out_dir>/transcript_<subject_id>.txt)
    """
    target_dir = out_dir or "."
    os.makedirs(target_dir, exist_ok=True)
    transcript_path = os.path.join(target_dir, f"transcript_{subject_id}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        for i, (sid, text) in enumerate(sorted(session_texts.items()), 1):
            f.write(f"=== Session {i}: {sid} ===\n")
            f.write((text or "").strip() + "\n\n")
    return transcript_path

def write_json(subject_id: str, revealed: Dict, contradictions, out_dir: Optional[str] = None) -> str:
    """
    Legacy behavior adapted to root:
    Writes ./truth_<subject_id>.json (or <out_dir>/truth_<subject_id>.json)
    Keeps key order aligned with the Submission Scroll.
    """
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

# ==============================
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
    return {os.path.basename(p): p for p in files}  # CHANGED from stem→with extension:contentReference[oaicite:1]{index=1}

def _cli_main(args: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_id", required=True)
    ap.add_argument("--sessions_dir", required=True)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--whisper_model", default="base")
    ap.add_argument("--ai_model", default="gemini-2.5-flash")
    ap.add_argument("--offline", action="store_true", help="Skip LLM call (for tests)")
    ns = ap.parse_args(args)

    sessions = collect_sessions(ns.sessions_dir)
    if not sessions:
        sys.exit(f"No session files found in {ns.sessions_dir}")

    print(f"Found {len(sessions)} session file(s): {[os.path.basename(p) for p in sessions.values()]}")

    # Transcribe (or read .txt passthrough)
    tr = Transcriber(model_size=ns.whisper_model, device="cpu", compute_type="int8")
    session_texts: Dict[str, str] = {}
    for sid, path in sessions.items():
        session_texts[sid] = tr.transcribe(path)

    # AI truth weaving
    try:
        api_key = os.getenv("GEMINI_API_KEY", GenCfg().api_key if hasattr(GenCfg, "api_key") else "")
        weaver = AITruthWeaver(cfg=GenCfg(model_name=ns.ai_model, api_key=api_key), offline=ns.offline)
        revealed, contradictions = weaver.infer(ns.subject_id, session_texts, offline=ns.offline)
    except Exception as e:
        sys.exit(f"AI analyzer failed: {e}")

    # Write to single files (root by default)
    tpath = append_transcript_single(ns.subject_id, session_texts, ns.out_dir)
    jpath = upsert_truth_single(ns.subject_id, revealed, contradictions, ns.out_dir)

# ==============================
# ----------- web server -------
# ==============================
def _build_fastapi_app():
    try:
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.responses import JSONResponse, Response
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise RuntimeError("FastAPI dependencies missing. Install with: pip install fastapi uvicorn python-multipart") from e

    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".pkg_resources is deprecated.",
        category=UserWarning,
        module="ctranslate2.*",
    )

    app = FastAPI(title="Truth Weaver API")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazily initialized transcriber (reuse model across requests)
    _transcriber = None  # type: Optional[Transcriber]
    _transcriber_model = None  # type: Optional[str]

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
        from fastapi import HTTPException as _HTTPException

        if not files:
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
                session_texts[filename] = text  # CHANGED: key is full original filename with extension:contentReference[oaicite:2]{index=2}

            # Run AI analysis
            api_key = os.getenv("GEMINI_API_KEY", GenCfg().api_key if hasattr(GenCfg, "api_key") else "")
            weaver = AITruthWeaver(cfg=GenCfg(model_name="gemini-2.5-flash", api_key=api_key))
            try:
                revealed, contradictions = weaver.infer(subject_id, session_texts)
            except RuntimeError as exc:
                raise _HTTPException(status_code=504, detail=str(exc)) from exc

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

# Expose FastAPI app as module-level var for uvicorn support
try:
    app = _build_fastapi_app()
except Exception as _e:
    app = None  # Will raise if 'serve' is invoked
    _FASTAPI_IMPORT_ERROR = _e
else:
    _FASTAPI_IMPORT_ERROR = None

# ==============================
# --------- run server ---------
# kept as a function so this file can be used by uvicorn
# ==============================
def _serve():
    if app is None:
        raise _FASTAPI_IMPORT_ERROR or RuntimeError("FastAPI app not initialized")
    try:
        import uvicorn
    except ImportError as e:
        print("Missing dependency: uvicorn (and fastapi, python-multipart).\n"
              "Install with: pip install fastapi uvicorn python-multipart")
        raise
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")

# ==============================
# -------- argument router -----
# ==============================
def _router():
    if len(sys.argv) == 1:
        print(__doc__)
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

if __name__ == "__main__":
    _router()

