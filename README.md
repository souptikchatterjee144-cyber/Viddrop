# VidDrop

A clean, local YouTube downloader with a beautiful glassmorphism UI. No ads, no tracking, no cloud — runs entirely on your machine.

## What You Need

- **Python 3.8+** — [Download from python.org](https://python.org)
- **ffmpeg** — Required for MP3 conversion (optional but recommended)
- **A modern browser** — Chrome 90+, Firefox 88+, Safari 14+, or Edge 90+

## Quick Start

1. **Download** or clone this folder to your computer

2. **Start the companion server:**
   - **Windows:** Double-click `start.bat`
   - **Mac/Linux:** Open a terminal in this folder and run:
     ```
     chmod +x start.sh
     bash start.sh
     ```
   - **Or manually:**
     ```
     pip install -r requirements.txt
     python companion.py
     ```

3. **Open `index.html`** in your browser — VidDrop will auto-detect the companion server

## How to Use

1. **Paste** a YouTube URL into the input bar (or click the Paste button)
2. **Pick a format** — choose from Video, Audio, or Thumbnail tabs
3. **Click** the format row to start downloading
4. **Done** — the file saves to your browser's Downloads folder

## Install ffmpeg

ffmpeg is needed for MP3 conversion and merging video+audio tracks.

- **Windows:** `winget install ffmpeg`
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

After installing, restart the companion server.

## Supported URLs

- `youtube.com/watch?v=VIDEO_ID`
- `youtu.be/VIDEO_ID`
- `youtube.com/shorts/VIDEO_ID`
- `youtube.com/embed/VIDEO_ID`

## Troubleshooting

1. **"Companion not running" overlay won't go away**
   Make sure `companion.py` is running in a terminal. Run `start.bat` (Windows) or `bash start.sh` (Mac/Linux).

2. **MP3 option is disabled / greyed out**
   Install ffmpeg (see above) and restart the companion server.

3. **"Video unavailable" error**
   The video may be private, deleted, or age-restricted. VidDrop can only download public videos.

4. **Download starts but no file appears**
   Check your browser's Downloads folder. Some browsers save to a default location without prompting.

5. **Fetching video info is slow**
   This is normal on the first run — yt-dlp needs to initialize. Subsequent fetches are faster.

## Credits

Powered by [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [FastAPI](https://fastapi.tiangolo.com).
