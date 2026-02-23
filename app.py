import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, send_file
from faster_whisper import WhisperModel
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path("/tmp") if os.getenv("VERCEL") else BASE_DIR
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

_model = None
_jobs: Dict[str, Dict[str, object]] = {}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def transcribe_words(video_path: Path) -> List[Dict[str, float]]:
    segments, _ = get_model().transcribe(
        str(video_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    words: List[Dict[str, float]] = []
    for segment_idx, segment in enumerate(segments):
        for word in segment.words or []:
            token = (word.word or "").strip()
            if not token:
                continue
            start = float(word.start) if word.start is not None else None
            end = float(word.end) if word.end is not None else None
            if start is None or end is None or end <= start:
                continue
            words.append(
                {
                    "word": token,
                    "start": start,
                    "end": end,
                    "segment": float(segment_idx),
                }
            )

    return words


def chunk_words(words: List[Dict[str, float]], size: int) -> List[List[Dict[str, float]]]:
    return [words[i : i + size] for i in range(0, len(words), size)]


def is_sentence_end(token: str) -> bool:
    return re.search(r'[.!?]["\')\]]*$', token) is not None


def split_phrases(words: List[Dict[str, float]], max_gap: float = 0.5) -> List[List[Dict[str, float]]]:
    if not words:
        return []

    phrases: List[List[Dict[str, float]]] = []
    current: List[Dict[str, float]] = []

    for idx, word in enumerate(words):
        current.append(word)
        if idx == len(words) - 1:
            continue

        next_word = words[idx + 1]
        gap = float(next_word["start"]) - float(word["end"])
        segment_changed = int(next_word.get("segment", 0)) != int(word.get("segment", 0))

        if is_sentence_end(str(word["word"])) or gap >= max_gap or segment_changed:
            phrases.append(current)
            current = []

    if current:
        phrases.append(current)

    return phrases


def ass_escape(text: str) -> str:
    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{").replace("}", r"\}")
    return text


def ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours = total_cs // 360000
    minutes = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    centis = total_cs % 100
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"


def hex_to_ass_bgr(hex_color: str) -> str:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", (hex_color or "").strip())
    if not match:
        return "&H0000D7FF"
    rgb = match.group(1)
    rr = rgb[0:2]
    gg = rgb[2:4]
    bb = rgb[4:6]
    return f"&H00{bb}{gg}{rr}"


def probe_resolution(video_path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload.get("streams", [{}])[0]
    width = int(stream.get("width", 1080))
    height = int(stream.get("height", 1920))
    return width, height


def words_to_editor_text(words: List[Dict[str, float]]) -> str:
    lines = []
    for w in words:
        lines.append(f"{float(w['start']):.2f}\t{float(w['end']):.2f}\t{str(w['word'])}")
    return "\n".join(lines)


def parse_editor_text(text: str) -> List[Dict[str, float]]:
    parsed: List[Dict[str, float]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for idx, line in enumerate(lines, start=1):
        parts = line.split("\t", 2)
        if len(parts) != 3:
            parts = line.split(",", 2)

        if len(parts) != 3:
            raise ValueError(f"Invalid subtitle editor row at line {idx}. Use: start<TAB>end<TAB>word")

        try:
            start = float(parts[0].strip())
            end = float(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid timing at line {idx}.") from exc

        word = parts[2].strip()
        if not word:
            raise ValueError(f"Empty word at line {idx}.")

        if end <= start:
            raise ValueError(f"End must be greater than start at line {idx}.")

        parsed.append({"word": word, "start": start, "end": end, "segment": 0.0})

    if not parsed:
        raise ValueError("No subtitle rows provided.")

    parsed.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    return parsed


def build_ass(
    words: List[Dict[str, float]],
    ass_path: Path,
    video_size: Tuple[int, int],
    offset_pct: float,
    highlight_color: str,
    words_per_caption: int,
    font_scale_pct: float,
) -> None:
    width, height = video_size
    margin_v = max(20, int(height * (offset_pct / 100.0)))
    highlight_ass = hex_to_ass_bgr(highlight_color)
    words_per_caption = min(max(int(words_per_caption), 1), 3)
    font_scale_pct = min(max(float(font_scale_pct), 60.0), 180.0)
    base_font_size = max(30, int(height * 0.05))
    font_size = max(20, int(base_font_size * (font_scale_pct / 100.0)))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,1,0,0,0,100,100,0,0,1,4,0,2,48,48,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]

    for phrase in split_phrases(words):
        for chunk in chunk_words(phrase, words_per_caption):
            for idx, current in enumerate(chunk):
                start = float(current["start"])
                end = float(current["end"])
                if end <= start:
                    end = start + 0.12

                rendered = []
                for j, token in enumerate(chunk):
                    w = ass_escape(str(token["word"]))
                    if j == idx:
                        rendered.append(r"{\c" + highlight_ass + r"\b1}" + w + r"{\c&H00FFFFFF&\b0}")
                    else:
                        rendered.append(w)

                text = " ".join(rendered)
                lines.append(
                    f"Dialogue: 0,{ass_time(start)},{ass_time(end)},TikTok,,0,0,0,,{text}\n"
                )

    ass_path.write_text("".join(lines), encoding="utf-8")


def burn_subtitles(video_path: Path, ass_path: Path, output_path: Path) -> None:
    escaped_ass = ass_path.resolve().as_posix().replace(":", r"\:")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles='{escaped_ass}'",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def render_video(
    video_path: Path,
    output_path: Path,
    words: List[Dict[str, float]],
    offset_pct: float,
    highlight_color: str,
    words_per_caption: int,
    font_scale_pct: float,
) -> None:
    ass_path = output_path.with_suffix(".ass")
    try:
        video_size = probe_resolution(video_path)
        build_ass(
            words,
            ass_path,
            video_size,
            offset_pct,
            highlight_color,
            words_per_caption,
            font_scale_pct,
        )
        burn_subtitles(video_path, ass_path, output_path)
    finally:
        ass_path.unlink(missing_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_upload():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify({"error": "Please upload a video file."}), 400

    safe_name = secure_filename(video.filename)
    if not allowed_file(safe_name):
        return jsonify({"error": "Unsupported video format."}), 400

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"

    try:
        video.save(input_path)
    except Exception as exc:
        return jsonify({"error": f"Failed to save upload: {exc}"}), 500

    try:
        words = transcribe_words(input_path)
    except Exception as exc:
        input_path.unlink(missing_ok=True)
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    if not words:
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "No speech was detected in this video."}), 400

    _jobs[job_id] = {
        "input": input_path,
        "output": None,
        "words": words,
        "render_version": 0,
    }

    return jsonify(
        {
            "job_id": job_id,
            "preview_source_url": f"/source/{job_id}",
            "editor_text": words_to_editor_text(words),
        }
    )

@app.route("/render", methods=["POST"])
def render_job():
    job_id = (request.form.get("job_id") or "").strip()
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found. Please upload the video again."}), 404

    input_path = job.get("input")
    if not isinstance(input_path, Path) or not input_path.exists():
        return jsonify({"error": "Original video is missing. Please upload again."}), 404

    editor_text = request.form.get("editor_text", "")
    try:
        words = parse_editor_text(editor_text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        offset_pct = float(request.form.get("caption_offset_pct", "10"))
    except ValueError:
        offset_pct = 10.0
    offset_pct = min(max(offset_pct, 5.0), 40.0)

    highlight_color = request.form.get("highlight_color", "#ffd700")

    try:
        words_per_caption = int(request.form.get("words_per_caption", "3"))
    except ValueError:
        words_per_caption = 3
    words_per_caption = min(max(words_per_caption, 1), 3)

    try:
        font_scale_pct = float(request.form.get("font_scale_pct", "100"))
    except ValueError:
        font_scale_pct = 100.0
    font_scale_pct = min(max(font_scale_pct, 60.0), 180.0)

    previous_version = int(job.get("render_version", 0))
    render_version = previous_version + 1
    output_path = OUTPUT_DIR / f"{job_id}_{render_version}.mp4"

    try:
        render_video(
            input_path,
            output_path,
            words,
            offset_pct,
            highlight_color,
            words_per_caption,
            font_scale_pct,
        )
    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"FFmpeg error: {exc}"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    _jobs[job_id]["output"] = output_path
    _jobs[job_id]["render_version"] = render_version
    return jsonify(
        {
            "job_id": job_id,
            "download_url": f"/download/{job_id}",
            "preview_render_url": f"/preview/{job_id}?v={render_version}",
        }
    )


@app.route("/source/<job_id>")
def source_video(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    input_path = job.get("input")
    if not isinstance(input_path, Path) or not input_path.exists():
        return jsonify({"error": "File not found."}), 404

    response = send_file(input_path, mimetype="video/mp4")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/preview/<job_id>")
def preview(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    output = job.get("output")
    if not isinstance(output, Path) or not output.exists():
        return jsonify({"error": "Rendered file not found."}), 404

    response = send_file(output, mimetype="video/mp4")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/download/<job_id>")
def download(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    output = job.get("output")
    if not isinstance(output, Path) or not output.exists():
        return jsonify({"error": "Rendered file not found."}), 404

    return send_file(output, as_attachment=True, download_name=f"captioned_{job_id}.mp4")


if __name__ == "__main__":
    app.run(debug=True)




