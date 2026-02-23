# TikTok Caption Video App

Upload a video, generate speech-to-text captions, and burn TikTok-style subtitles into the video.

## Features
- Video upload (`.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`)
- Word-level speech transcription using Faster-Whisper
- Caption style: 3 words displayed at once, current spoken word highlighted (yellow), others white
- Download processed video with burned-in subtitles

## Requirements
- Python 3.10+
- `ffmpeg` and `ffprobe` available in PATH

## Run
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Notes
- First run downloads the Whisper model (`small`), so it may take time.
- Processing speed depends on your CPU/GPU and video length.
