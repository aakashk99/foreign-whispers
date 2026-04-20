"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import logging
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


def _load_speaker_wav_map(
    svc: TTSService,
    video_id: str,
    title: str,
    target_language: str,
) -> dict[str, str] | None:
    """Read the diarized transcription and build a speaker→WAV map if speaker
    labels are present.  Returns ``None`` when diarization was not run.
    """
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if not transcript_path.exists():
        return None

    try:
        transcript = json.loads(transcript_path.read_text())
    except Exception as exc:
        logger.warning("Could not read transcription for %s: %s", video_id, exc)
        return None

    segments = transcript.get("segments", [])
    speakers = sorted({seg["speaker"] for seg in segments if "speaker" in seg})

    if not speakers:
        logger.debug("No speaker labels in transcription for %s — skipping voice map.", video_id)
        return None

    logger.info("Diarized speakers found for %s: %s", video_id, speakers)
    return svc.build_speaker_voice_map(speakers, language=target_language)


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    target_language: str = Query("es"),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    *target_language* is used to select per-speaker reference voices from
    ``pipeline_data/speakers/{target_language}/``.
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = str(trans_dir / f"{title}.json")

    # Build per-speaker voice map if diarization has run
    speaker_wav_map = _load_speaker_wav_map(svc, video_id, title, target_language)
    if speaker_wav_map:
        logger.info(
            "Using per-speaker voices for %s: %s",
            video_id,
            {k: pathlib.Path(v).name for k, v in speaker_wav_map.items()},
        )
    else:
        logger.info("No diarization data for %s — using single default voice.", video_id)

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path,
        str(audio_dir),
        alignment=alignment,
        speaker_wav_map=speaker_wav_map,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
