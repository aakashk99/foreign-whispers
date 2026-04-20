"""POST /api/diarize/{video_id} — speaker diarization."""

import json
import logging
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.schemas.diarize import DiarizeResponse
from foreign_whispers.diarization import assign_speakers, diarize_audio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["diarize"])


def _resolve_title(video_id: str) -> str | None:
    """Return the video title for *video_id*, or None if not found."""
    from api.src.core.video_registry import get_all_videos
    for video in get_all_videos():
        if video.id == video_id:
            return video.title
    return None


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a downloaded video's audio track.

    Steps:
        1. Resolve video title from registry (404 if unknown).
        2. Return cached result immediately if it already exists.
        3. Extract a 16 kHz mono WAV from the source MP4 via ffmpeg.
        4. Run pyannote diarization (requires FW_HF_TOKEN in environment).
        5. Merge speaker labels into the existing Whisper transcription JSON.
        6. Cache the diarization result and return it.
    """
    title = _resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found in registry.")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # ── Step 2: return cached result ─────────────────────────────────────
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        logger.info("Diarization cache hit for %s — skipping.", title)
        return DiarizeResponse(
            video_id=video_id,
            speakers=data["speakers"],
            segments=data["segments"],
            skipped=True,
        )

    # ── Step 3: extract audio ─────────────────────────────────────────────
    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Source video not found at {video_path}. Run /api/download first.",
        )

    audio_path = diar_dir / f"{title}.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Audio extracted to %s", audio_path)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg audio extraction failed: {exc.stderr.decode(errors='replace')}",
        )

    # ── Step 4: run diarization ───────────────────────────────────────────
    diar_segments = diarize_audio(str(audio_path), hf_token=settings.hf_token)
    if not diar_segments:
        # diarize_audio already logs a warning; surface it to the caller too
        raise HTTPException(
            status_code=503,
            detail=(
                "Diarization returned no segments. "
                "Check that FW_HF_TOKEN is set and the pyannote model licence is accepted."
            ),
        )

    speakers = sorted({s["speaker"] for s in diar_segments})
    logger.info("Diarization complete: %d speakers, %d intervals.", len(speakers), len(diar_segments))

    # ── Step 5: merge speaker labels into transcription ───────────────────
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
        labeled = assign_speakers(transcript.get("segments", []), diar_segments)
        transcript["segments"] = labeled
        transcript_path.write_text(json.dumps(transcript, ensure_ascii=False))
        logger.info("Speaker labels merged into transcription at %s", transcript_path)
    else:
        logger.warning(
            "Transcription not found at %s — speaker labels not merged. "
            "Run /api/transcribe/%s first if you need labeled segments.",
            transcript_path,
            video_id,
        )

    # ── Step 6: cache and return ──────────────────────────────────────────
    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result, ensure_ascii=False))
    logger.info("Diarization cached at %s", diar_path)

    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
        skipped=False,
    )
