"""
VidDrop Companion Server v1.0
FastAPI backend for the VidDrop YouTube downloader.
Bridges the frontend UI with yt-dlp for downloading.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import requests
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ─── Configuration ─────────────────────────────────────────────
PORT = 7701
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "vidrop_downloads"

# ─── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vidrop")

# ─── FastAPI App ───────────────────────────────────────────────
app = FastAPI(title="VidDrop Companion", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-Memory Job Store ──────────────────────────────────────
jobs: dict = {}

# ─── Content-Type Map ─────────────────────────────────────────
CONTENT_TYPE_MAP = {
    "mp4": "video/mp4",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "webm": "audio/webm",
    "opus": "audio/ogg",
}

# ─── yt-dlp Error Translation Map ────────────────────────────
ERROR_MAP = {
    "Sign in to confirm your age": "Age-restricted video. Cannot download.",
    "Video unavailable": "This video is unavailable or deleted.",
    "Private video": "This video is private.",
    "HTTP Error 403": "Access denied by YouTube.",
    "Unable to extract": "Could not extract video info. Try again.",
}


# ─── Pydantic Models ─────────────────────────────────────────

class DownloadRequest(BaseModel):
    url: str
    format_id: str
    ext: str
    filename: str
    type: str  # "video" or "audio"
    audio_format: str | None = None


# ─── Utility Functions ────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in system PATH."""
    return shutil.which("ffmpeg") is not None


def check_ytdlp() -> str | None:
    """Check if yt-dlp is installed and return its version string."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def sanitize_filename(name: str) -> str:
    """Remove dangerous characters and truncate filename."""
    # Remove path separators and dangerous characters
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # Remove .. sequences
    name = name.replace("..", "")
    # Strip whitespace
    name = name.strip()
    # Truncate to 120 characters
    if len(name) > 120:
        name = name[:120]
    return name


def format_duration(seconds: int) -> str:
    """Format duration in seconds to H:MM:SS or M:SS string."""
    if seconds is None or seconds <= 0:
        return "0:00"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_size(bytes_val) -> str:
    """Format byte count to human-readable ~X MB or ~X GB string."""
    if bytes_val is None or bytes_val == 0:
        return "~? MB"
    bytes_val = float(bytes_val)
    if bytes_val >= 1_073_741_824:  # 1 GB
        return f"~{bytes_val / 1_073_741_824:.1f} GB"
    if bytes_val >= 1_048_576:  # 1 MB
        return f"~{bytes_val / 1_048_576:.0f} MB"
    if bytes_val >= 1024:
        return f"~{bytes_val / 1024:.0f} KB"
    return f"~{bytes_val:.0f} B"


def parse_formats(formats_list: list) -> tuple:
    """
    Parse yt-dlp formats list into (video_formats, audio_formats).
    Video: grouped by height, best per height, format_id overridden.
    Audio: sorted by abr desc, MP3 conversion prepended at index 0.
    """
    video_map: dict = {}
    audio_list: list = []

    for fmt in formats_list:
        vcodec = fmt.get("vcodec", "none")
        acodec = fmt.get("acodec", "none")
        height = fmt.get("height")
        abr = fmt.get("abr", 0) or 0
        filesize = fmt.get("filesize") or fmt.get("filesize_approx") or 0

        # Video formats: vcodec is not "none"
        if vcodec and vcodec != "none" and height:
            # Keep best quality per height (largest filesize)
            if height not in video_map or filesize > (video_map[height].get("_filesize", 0)):
                video_map[height] = {
                    "label": f"{height}p",
                    "format_id": f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
                    "ext": "mp4",
                    "size": format_size(filesize),
                    "has_audio": acodec != "none",
                    "_filesize": filesize,
                    "_height": height,
                }

        # Audio formats: vcodec is "none" and has audio
        elif (vcodec == "none" or not vcodec) and acodec and acodec != "none":
            ext = fmt.get("ext", "webm")
            format_id = fmt.get("format_id", "")
            audio_list.append({
                "label": f"{ext.upper()} {int(abr)}kbps" if abr else f"{ext.upper()}",
                "format_id": format_id,
                "ext": ext,
                "size": format_size(filesize),
                "abr": abr,
                "conversion": False,
            })

    # Build video formats list — ordered by height descending
    target_heights = [2160, 1440, 1080, 720, 480, 360, 240, 144]
    video_formats = []
    for h in target_heights:
        if h in video_map:
            entry = video_map[h]
            video_formats.append({
                "label": entry["label"],
                "format_id": entry["format_id"],
                "ext": entry["ext"],
                "size": entry["size"],
                "has_audio": entry["has_audio"],
            })

    # Sort audio by bitrate descending
    audio_list.sort(key=lambda x: x.get("abr", 0), reverse=True)

    # Prepend MP3 conversion option at index 0
    mp3_option = {
        "label": "MP3 (Best Quality)",
        "format_id": "bestaudio",
        "ext": "mp3",
        "size": "~varies",
        "abr": 0,
        "conversion": True,
    }
    audio_list.insert(0, mp3_option)

    # Cap audio list at 5 items
    audio_formats = audio_list[:5]

    return video_formats, audio_formats


def translate_ytdlp_error(stderr_text: str) -> str:
    """Translate yt-dlp error messages to human-readable form."""
    for pattern, message in ERROR_MAP.items():
        if pattern.lower() in stderr_text.lower():
            return message
    # Fallback: return last non-empty line of stderr
    lines = [line.strip() for line in stderr_text.strip().split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]
        # Strip yt-dlp prefixes like "ERROR: "
        last_line = re.sub(r"^ERROR:\s*", "", last_line)
        return last_line
    return "An unknown error occurred."


def cleanup_old_jobs():
    """Remove jobs older than 1 hour from the in-memory store."""
    now = time.time()
    expired = [
        jid for jid, job in jobs.items()
        if now - job.get("created_at", now) > 3600
    ]
    for jid in expired:
        jobs.pop(jid, None)


def cleanup_old_files():
    """Delete leftover temp files older than 1 hour on startup."""
    if not DOWNLOAD_DIR.exists():
        return
    now = time.time()
    for filepath in DOWNLOAD_DIR.iterdir():
        try:
            if filepath.is_file() and (now - filepath.stat().st_mtime) > 3600:
                filepath.unlink()
                logger.info("Cleaned up old temp file: %s", filepath.name)
        except OSError:
            pass


# ─── URL Validation ───────────────────────────────────────────

URL_PATTERN = re.compile(
    r"(youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/|m\.youtube\.com/watch\?.*v=)"
)


def is_valid_youtube_url(url: str) -> bool:
    """Check if URL matches known YouTube URL patterns."""
    return bool(URL_PATTERN.search(url))


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    """Health check endpoint. Returns yt-dlp version and ffmpeg status."""
    # Clean up old jobs on each ping
    cleanup_old_jobs()

    version = check_ytdlp()
    if version is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "yt-dlp not found. Run: pip install yt-dlp"},
        )
    return {
        "status": "ok",
        "yt_dlp_version": version,
        "ffmpeg": check_ffmpeg(),
        "port": PORT,
    }


@app.get("/info")
async def get_info(url: str = Query(..., description="YouTube video URL")):
    """Fetch video metadata and available format list."""
    # Validate URL
    if not is_valid_youtube_url(url):
        return JSONResponse(status_code=400, content={"error": "Invalid YouTube URL"})

    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            error_msg = translate_ytdlp_error(result.stderr)
            logger.error("yt-dlp error for %s: %s", url, result.stderr[:200])
            return JSONResponse(status_code=400, content={"error": error_msg})

        data = json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=408,
            content={"error": "Request timed out. Check your connection."},
        )
    except json.JSONDecodeError:
        logger.error("Failed to parse yt-dlp JSON output")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to parse video data."},
        )
    except Exception as exc:
        logger.exception("Unexpected error in /info")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to parse video data."},
        )

    # Parse formats
    raw_formats = data.get("formats", [])
    video_formats, audio_formats = parse_formats(raw_formats)

    # Extract video ID
    video_id = data.get("id", "")

    # Build thumbnail URL (use best available)
    thumbnail = data.get("thumbnail", "")
    if not thumbnail and video_id:
        thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    # Build response
    return {
        "video_id": video_id,
        "title": data.get("title", "Unknown Title"),
        "uploader": data.get("uploader", data.get("channel", "Unknown Channel")),
        "duration": format_duration(data.get("duration", 0)),
        "thumbnail": thumbnail,
        "video_formats": video_formats,
        "audio_formats": audio_formats,
    }


def run_download(job_id: str, req: DownloadRequest):
    """Execute yt-dlp download in a background thread. Updates jobs dict."""
    output_template = str(DOWNLOAD_DIR / f"{job_id}.%(ext)s")

    # Build yt-dlp command as a list (never shell=True)
    cmd = ["yt-dlp"]

    if req.type == "video":
        cmd.extend([
            "-f", req.format_id,
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--newline",
            "--progress",
            "--no-playlist",
            req.url,
        ])
    elif req.type == "audio" and req.audio_format and req.audio_format == "mp3":
        # Audio with conversion (MP3)
        cmd.extend([
            "-f", "bestaudio",
            "-x",
            "--audio-format", req.audio_format,
            "--audio-quality", "0",
            "-o", output_template,
            "--newline",
            "--no-playlist",
            req.url,
        ])
    elif req.type == "audio" and req.audio_format and req.audio_format in ("m4a", "opus", "webm"):
        # Audio with conversion for m4a/opus
        if req.audio_format == req.ext and req.format_id != "bestaudio":
            # Native format, no conversion needed
            cmd.extend([
                "-f", req.format_id,
                "-o", output_template,
                "--newline",
                "--no-playlist",
                req.url,
            ])
        else:
            cmd.extend([
                "-f", "bestaudio",
                "-x",
                "--audio-format", req.audio_format,
                "--audio-quality", "0",
                "-o", output_template,
                "--newline",
                "--no-playlist",
                req.url,
            ])
    else:
        # Fallback: native audio, no conversion
        cmd.extend([
            "-f", req.format_id,
            "-o", output_template,
            "--newline",
            "--no-playlist",
            req.url,
        ])

    logger.info("Starting download job %s: %s", job_id, " ".join(cmd[:6]) + "...")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        progress_pattern = re.compile(r"\[download\]\s+(\d+\.?\d*)%")

        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue

            # Parse progress percentage
            match = progress_pattern.search(line)
            if match:
                jobs[job_id]["progress"] = float(match.group(1))

            # Detect status keywords
            if "[Merger]" in line:
                jobs[job_id]["status"] = "merging"
            elif "[ExtractAudio]" in line or "[ffmpeg]" in line:
                jobs[job_id]["status"] = "converting"
            elif "[download]" in line:
                jobs[job_id]["status"] = "downloading"

        process.stdout.close()
        process.wait()

        if process.returncode == 0:
            # Find the output file (extension may differ from requested)
            output_files = list(DOWNLOAD_DIR.glob(f"{job_id}.*"))
            if output_files:
                jobs[job_id]["filename"] = str(output_files[0])
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
            else:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = "Download completed but output file not found."
        else:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Download failed. Please try again."

    except Exception as exc:
        logger.exception("Error in download job %s", job_id)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = "Download failed unexpectedly."
    finally:
        if process and process.stdout and not process.stdout.closed:
            process.stdout.close()
        if process and process.poll() is None:
            process.kill()


@app.post("/download")
async def download(req: DownloadRequest):
    """Start a download job. Returns job_id immediately for progress tracking."""
    job_id = str(uuid.uuid4())

    # Sanitize filename
    safe_filename = sanitize_filename(req.filename) if req.filename else sanitize_filename("download")
    if not safe_filename:
        safe_filename = "download"

    # Initialize job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "starting",
        "progress": 0,
        "filename": "",
        "safe_filename": safe_filename,
        "error": "",
        "created_at": time.time(),
    }

    # Run download in a background thread (non-blocking)
    thread = threading.Thread(target=run_download, args=(job_id, req), daemon=True)
    thread.start()

    # Return job_id immediately so frontend can track progress via SSE
    return JSONResponse(
        content={"job_id": job_id},
        headers={"X-Job-ID": job_id, "Access-Control-Expose-Headers": "X-Job-ID"},
    )


@app.get("/file/{job_id}")
async def get_file(job_id: str):
    """Stream the completed download file to the browser."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found."})

    job = jobs[job_id]
    if job["status"] != "done":
        return JSONResponse(status_code=400, content={"error": "Download not complete yet."})

    filepath = Path(job["filename"]) if job["filename"] else None
    if not filepath or not filepath.exists():
        return JSONResponse(status_code=404, content={"error": "File not found."})

    actual_ext = filepath.suffix.lstrip(".")
    content_type = CONTENT_TYPE_MAP.get(actual_ext, "application/octet-stream")

    safe_filename = job.get("safe_filename", "download")
    download_filename = f"{safe_filename}.{actual_ext}"
    encoded_filename = quote(download_filename)

    def iterfile(path: Path):
        """Stream file in 8KB chunks."""
        try:
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        finally:
            # Delete temp file after streaming
            try:
                if path.exists():
                    path.unlink()
                    logger.info("Cleaned up temp file: %s", path.name)
            except OSError:
                pass

    return StreamingResponse(
        iterfile(filepath),
        media_type=content_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{download_filename}"; '
                f"filename*=UTF-8''{encoded_filename}"
            ),
            "X-Job-ID": job_id,
            "Access-Control-Expose-Headers": "X-Job-ID",
        },
    )


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    """SSE endpoint for real-time download progress updates."""

    async def event_generator():
        # Check if job exists
        if job_id not in jobs:
            error_data = json.dumps({"progress": 0, "status": "error", "eta": ""})
            yield f"data: {error_data}\n\n"
            return

        while jobs[job_id]["status"] not in ("done", "error"):
            data = json.dumps({
                "progress": jobs[job_id]["progress"],
                "status": jobs[job_id]["status"],
                "eta": "",
            })
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.3)

        # Final event
        final_status = jobs[job_id]["status"]
        final_progress = 100 if final_status == "done" else jobs[job_id]["progress"]
        final_data = json.dumps({
            "progress": final_progress,
            "status": final_status,
            "eta": "",
        })
        yield f"data: {final_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/thumbnail")
async def thumbnail(
    video_id: str = Query(..., description="YouTube video ID"),
    quality: str = Query("hq", description="Thumbnail quality: max, hq, or low"),
):
    """Proxy YouTube thumbnail images to avoid CORS issues."""
    # Validate inputs
    if not video_id or len(video_id) > 20:
        return JSONResponse(status_code=400, content={"error": "Invalid video_id or quality"})
    if quality not in ("max", "hq", "low"):
        return JSONResponse(status_code=400, content={"error": "Invalid video_id or quality"})

    # Quality to URL map
    quality_map = {
        "max": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "hq": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "low": f"https://img.youtube.com/vi/{video_id}/default.jpg",
    }

    url = quality_map[quality]

    try:
        resp = requests.get(url, timeout=10, stream=True)

        # Fallback: if max quality returns 404, retry with hq
        if quality == "max" and resp.status_code == 404:
            logger.info("maxresdefault not found for %s, falling back to hqdefault", video_id)
            url = quality_map["hq"]
            resp = requests.get(url, timeout=10, stream=True)

        if resp.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"error": "Could not fetch thumbnail from YouTube"},
            )

        download_name = f"{video_id}_{quality}.jpg"

        return StreamingResponse(
            resp.iter_content(chunk_size=8192),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{download_name}"',
            },
        )

    except requests.RequestException:
        logger.exception("Failed to fetch thumbnail for %s", video_id)
        return JSONResponse(
            status_code=502,
            content={"error": "Could not fetch thumbnail from YouTube"},
        )


# ─── Main Entry Point ─────────────────────────────────────────

if __name__ == "__main__":
    # Startup banner
    print()
    print("  ┌─────────────────────────────┐")
    print("  │   VidDrop Companion v1.0    │")
    print("  └─────────────────────────────┘")
    print()

    # Check yt-dlp
    ytdlp_version = check_ytdlp()
    if ytdlp_version is None:
        print("  ✗ yt-dlp not found!")
        print("    Install it with: pip install yt-dlp")
        sys.exit(1)
    print(f"  ✓ yt-dlp: {ytdlp_version}")

    # Check ffmpeg
    if check_ffmpeg():
        print("  ✓ ffmpeg: found")
    else:
        print("  ⚠ ffmpeg: not found (MP3 conversion disabled)")
        print("    Install ffmpeg to enable audio conversion.")

    # Create temp directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Temp dir: {DOWNLOAD_DIR}")

    # Clean up leftover files from previous sessions
    cleanup_old_files()

    # Print server info
    print()
    print(f"  Server starting at http://localhost:{PORT}")
    print("  Open index.html in your browser")
    print("  API docs at http://localhost:{}/docs".format(PORT))
    print()

    # Start server
    import os
import uvicorn

PORT = int(os.environ.get("PORT", PORT))

uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

