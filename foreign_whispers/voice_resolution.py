"""Voice resolution for Chatterbox speaker cloning.

Resolves which reference WAV to use for a given target language and optional
speaker ID. The Chatterbox container expects a filename relative to its
``/app/voices/`` mount point, which corresponds to
``pipeline_data/speakers/`` on the host.
"""

from pathlib import Path


def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
) -> str:
    """Resolve the reference WAV path for voice cloning.

    Walks a three-level fallback chain and returns the first path that
    exists on disk.  The return value is always a **relative** path string
    (e.g. ``"es/SPEAKER_00.wav"``) so that it can be forwarded directly to
    the Chatterbox container which mounts ``pipeline_data/speakers/`` at
    ``/app/voices/``.

    Resolution order:

    1. ``speakers/{lang}/{speaker_id}.wav`` — speaker-specific voice
       (only checked when *speaker_id* is provided)
    2. ``speakers/{lang}/default.wav``      — language-level default
    3. ``speakers/default.wav``             — global fallback

    Args:
        speakers_dir: Absolute path to the speakers root directory
            (``pipeline_data/speakers/`` on the host).
        target_language: BCP-47 language code (e.g. ``"es"``, ``"fr"``).
        speaker_id: Optional pyannote-style speaker label
            (e.g. ``"SPEAKER_00"``).  When ``None`` the speaker-specific
            step is skipped and resolution starts from the language default.

    Returns:
        Relative path string suitable for the Chatterbox ``/app/voices/``
        mount point (e.g. ``"es/default.wav"`` or ``"default.wav"``).
    """
    lang_dir = speakers_dir / target_language

    # Step 1: speaker-specific WAV
    if speaker_id:
        candidate = lang_dir / f"{speaker_id}.wav"
        if candidate.exists():
            return f"{target_language}/{speaker_id}.wav"

    # Step 2: language-level default
    lang_default = lang_dir / "default.wav"
    if lang_default.exists():
        return f"{target_language}/default.wav"

    # Step 3: global fallback
    return "default.wav"
