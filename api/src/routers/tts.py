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
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


def _build_speaker_wav_map(
    title: str,
    target_language: str,
) -> dict[str, str] | None:
    """Build a per-speaker voice map from the diarized transcription.

    Reads ``transcriptions_dir/{title}.json``, extracts unique speaker labels,
    and resolves a reference WAV for each via ``resolve_speaker_wav``.

    Returns ``None`` when:
    - the transcription file does not exist, or
    - no segments carry a ``speaker`` field (diarization was not run).
    """
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if not transcript_path.exists():
        return None

    try:
        transcript = json.loads(transcript_path.read_text())
    except Exception as exc:
        logger.warning("Could not read transcription for %s: %s", title, exc)
        return None

    segments = transcript.get("segments", [])
    speakers = sorted({seg["speaker"] for seg in segments if "speaker" in seg})

    if not speakers:
        logger.debug("No speaker labels in transcription for %s — skipping voice map.", title)
        return None

    voice_map = {
        spk: resolve_speaker_wav(settings.speakers_dir, target_language, spk)
        for spk in speakers
    }
    logger.info(
        "Built per-speaker voice map for %s: %s",
        title,
        {k: v for k, v in voice_map.items()},
    )
    return voice_map


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    target_language: str = Query("es"),
    speaker_wav: str | None = Query(
        None,
        description=(
            "Reference voice WAV relative to pipeline_data/speakers/ "
            "(e.g. 'es/default.wav'). Auto-resolved from target_language "
            "when omitted. Ignored when diarized speaker labels are present."
        ),
    ),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    *target_language* drives both automatic voice resolution and per-speaker
    voice selection from ``pipeline_data/speakers/{target_language}/``.
    *speaker_wav* overrides auto-resolution for single-voice dubbing.

    Voice selection priority (highest → lowest):

    1. **Per-speaker map** — when diarized segments exist, each speaker gets
       its own voice via ``resolve_speaker_wav(speakers_dir, lang, speaker_id)``.
    2. **Explicit speaker_wav** — caller-supplied relative WAV path.
    3. **Auto-resolved default** — ``resolve_speaker_wav(speakers_dir, lang)``
       picks the language default or global fallback.
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(ui_dir=settings.data_dir, tts_engine=None)

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"
    if wav_path.exists():
        return {"video_id": video_id, "audio_path": str(wav_path), "config": config}

    source_path = str(trans_dir / f"{title}.json")

    # ── Priority 1: per-speaker map from diarization ──────────────────────────
    speaker_wav_map = _build_speaker_wav_map(title, target_language)

    # ── Priority 2 & 3: single-voice resolution ───────────────────────────────
    resolved_wav: str | None = None
    if not speaker_wav_map:
        if speaker_wav:
            # Caller-supplied — validate the file actually exists
            candidate = settings.speakers_dir / speaker_wav
            if candidate.exists():
                resolved_wav = speaker_wav
            else:
                logger.warning(
                    "Requested speaker_wav '%s' not found under speakers_dir — "
                    "falling back to auto-resolution.",
                    speaker_wav,
                )
        if resolved_wav is None:
            resolved_wav = resolve_speaker_wav(settings.speakers_dir, target_language)
            logger.info(
                "Auto-resolved speaker_wav for lang=%s: %s", target_language, resolved_wav
            )

    # Log which voice strategy is in use
    if speaker_wav_map:
        logger.info(
            "Per-speaker voices for %s: %s",
            video_id,
            {k: v for k, v in speaker_wav_map.items()},
        )
    else:
        logger.info("Single voice for %s: %s", video_id, resolved_wav)

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path,
        str(audio_dir),
        alignment=alignment,
        speaker_wav=resolved_wav,
        speaker_wav_map=speaker_wav_map,
    )

    return {"video_id": video_id, "audio_path": str(wav_path), "config": config}


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
