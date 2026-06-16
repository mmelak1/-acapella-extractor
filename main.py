import asyncio
import io
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import modal
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

load_dotenv()

LASTFM_API_KEY  = os.getenv("LASTFM_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

COMMON_BPMS = [70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 124, 128, 140]
COMMON_KEYS = [
    'C major', 'G major', 'D major', 'A major', 'F major',
    'A minor', 'E minor', 'D minor', 'G minor', 'C minor',
]

app = FastAPI()

jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)

_modal_cls = modal.Cls.from_name("acapella-extractor", "VocalSeparator")


@app.post("/api/separate")
async def separate(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    safe_name = "".join(
        c if c.isalnum() or c in "._- " else "_" for c in (file.filename or "audio")
    )

    audio_bytes = await file.read()
    content_type = file.content_type or "audio/mpeg"

    jobs[job_id] = {
        "status": "processing",
        "error": None,
        "vocals_bytes": None,
        "vocals_mp3": None,
        "original_bytes": audio_bytes,
        "original_type": content_type,
        "bpm": None,
        "key": None,
        "elapsed": None,
        "started_at": time.time(),
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_modal, job_id, audio_bytes, safe_name)
    loop.run_in_executor(_executor, _analyze_audio, job_id, audio_bytes)

    return {"job_id": job_id}


def _run_modal(job_id: str, audio_bytes: bytes, filename: str):
    try:
        result = _modal_cls().separate_vocals.remote(audio_bytes, filename)
        elapsed = round(time.time() - jobs[job_id]["started_at"], 1)
        jobs[job_id].update({
            "status": "done",
            "error": None,
            "vocals_mp3": result["mp3"],
            "elapsed": elapsed,
        })
    except Exception:
        import traceback
        jobs[job_id].update({"status": "error", "error": traceback.format_exc()})


def _analyze_audio(job_id: str, audio_bytes: bytes):
    try:
        import numpy as np
        import librosa

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, duration=60)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(np.atleast_1d(tempo)[0]), 1)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_score, best_key, best_mode = -2.0, 'C', 'major'
        for i in range(12):
            rotated = np.roll(chroma_mean, -i)
            maj = float(np.corrcoef(rotated, major_profile)[0, 1])
            min_ = float(np.corrcoef(rotated, minor_profile)[0, 1])
            if maj > best_score:
                best_score, best_key, best_mode = maj, key_names[i], 'major'
            if min_ > best_score:
                best_score, best_key, best_mode = min_, key_names[i], 'minor'

        jobs[job_id].update({"bpm": bpm, "key": f"{best_key} {best_mode}"})
    except Exception:
        pass


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "error": job["error"],
        "bpm": job.get("bpm"),
        "key": job.get("key"),
        "elapsed": job.get("elapsed"),
    }


@app.get("/api/original/{job_id}")
async def original(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return Response(content=job["original_bytes"], media_type=job.get("original_type", "audio/mpeg"))


@app.get("/api/preview/{job_id}")
async def preview(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not ready")
    return Response(content=job["vocals_mp3"], media_type="audio/mpeg")


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    import subprocess
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not ready")
    result = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "wav", "pipe:1"],
        input=job["vocals_mp3"], capture_output=True,
    )
    wav_bytes = result.stdout if result.returncode == 0 else job["vocals_mp3"]
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="acapella.wav"'},
    )


@app.get("/api/download-mp3/{job_id}")
async def download_mp3(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done" or not job.get("vocals_mp3"):
        raise HTTPException(status_code=404, detail="MP3 not available")
    return Response(
        content=job["vocals_mp3"],
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="acapella.mp3"'},
    )


@app.get("/api/mix/generate")
async def generate_mix():
    async with httpx.AsyncClient(timeout=15) as client:
        # Get a popular track from Last.fm charts
        page = random.randint(1, 5)
        lfm_res = await client.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "chart.getTopTracks",
                "api_key": LASTFM_API_KEY,
                "format": "json",
                "limit": 50,
                "page": page,
            },
        )
        if lfm_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Last.fm failed: {lfm_res.text}")

        tracks = lfm_res.json().get("tracks", {}).get("track", [])
        if not tracks:
            raise HTTPException(status_code=500, detail="No tracks found from Last.fm")

        track     = random.choice(tracks)
        song_name = track["name"]
        artist    = track["artist"]["name"]

        # YouTube: search for acapella and instrumental of the same song
        # so they share the same BPM, key, and structure
        yt_base = "https://www.googleapis.com/youtube/v3/search"
        acap_res, inst_res = await asyncio.gather(
            client.get(yt_base, params={
                "part": "snippet", "type": "video", "maxResults": 5,
                "q": f"{artist} {song_name} acapella",
                "key": YOUTUBE_API_KEY,
            }),
            client.get(yt_base, params={
                "part": "snippet", "type": "video", "maxResults": 5,
                "q": f"{artist} {song_name} karaoke",
                "key": YOUTUBE_API_KEY,
            }),
        )
        acap_items = acap_res.json().get("items", [])
        inst_items = inst_res.json().get("items", [])
        acapella_id     = next((i["id"]["videoId"] for i in acap_items if i.get("id", {}).get("videoId")), None)
        instrumental_id = next((i["id"]["videoId"] for i in inst_items if i.get("id", {}).get("videoId")), None)

        # Fallback: if karaoke not found, search for official instrumental
        if not instrumental_id:
            fb = await client.get(yt_base, params={
                "part": "snippet", "type": "video", "maxResults": 5,
                "q": f"{artist} {song_name} instrumental",
                "key": YOUTUBE_API_KEY,
            })
            fb_items = fb.json().get("items", [])
            instrumental_id = next((i["id"]["videoId"] for i in fb_items if i.get("id", {}).get("videoId")), None)

        return {
            "song": song_name,
            "artist": artist,
            "acapella_id": acapella_id,
            "instrumental_id": instrumental_id,
        }


# Static frontend — must be mounted after all API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")
