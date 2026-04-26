"""HTTP-agnostic service wrapping TTS engine functions."""

import logging
import pathlib
from pathlib import Path
from typing import Any

from api.src.services.tts_engine import text_file_to_speech as tts_text_file_to_speech

logger = logging.getLogger(__name__)


class TTSService:
    """Thin wrapper around the TTS pipeline.

    Accepts *ui_dir* and a pre-loaded *tts_engine* via constructor injection.
    """

    def __init__(self, ui_dir: Path, tts_engine: Any) -> None:
        self.ui_dir = ui_dir
        self.tts_engine = tts_engine

    def build_speaker_voice_map(
        self,
        speakers: list[str],
        language: str = "es",
    ) -> dict[str, str]:
        """Map each speaker label to a reference WAV path for voice cloning.

        Resolution order per speaker:
          1. ``speakers/{lang}/SPEAKER_XX.wav``  — exact match by label
          2. Round-robin over sorted WAVs in ``speakers/{lang}/``
          3. ``speakers/{lang}/default.wav``      — language-level default
          4. ``speakers/default.wav``             — global fallback

        Any speaker that cannot be resolved is omitted from the returned dict
        so the caller can decide to use the engine's built-in default voice.

        Args:
            speakers:  Sorted list of unique speaker labels, e.g. ``["SPEAKER_00", "SPEAKER_01"]``.
            language:  Target language code (matches sub-directory name under ``speakers/``).

        Returns:
            ``{speaker_label: wav_path_str}`` for every speaker that could be resolved.
        """
        speakers_root = self.ui_dir / "speakers"
        lang_dir = speakers_root / language

        # Collect available WAV files, sorted for deterministic assignment
        lang_wavs: list[Path] = sorted(lang_dir.glob("*.wav")) if lang_dir.exists() else []
        global_default = speakers_root / "default.wav"
        lang_default = lang_dir / "default.wav"

        # Build a name→path lookup for exact-match resolution
        wav_by_stem: dict[str, Path] = {w.stem: w for w in lang_wavs}

        voice_map: dict[str, str] = {}
        # Round-robin pool excludes the generic "default" file so it isn't
        # consumed as a speaker slot
        rr_pool = [w for w in lang_wavs if w.stem != "default"]

        for idx, speaker in enumerate(sorted(speakers)):
            # 1. Exact label match (e.g. speakers/es/SPEAKER_00.wav)
            if speaker in wav_by_stem:
                voice_map[speaker] = str(wav_by_stem[speaker])
                logger.debug("Speaker %s → exact match %s", speaker, voice_map[speaker])

            # 2. Round-robin over available WAVs
            elif rr_pool:
                wav = rr_pool[idx % len(rr_pool)]
                voice_map[speaker] = str(wav)
                logger.debug("Speaker %s → round-robin %s", speaker, voice_map[speaker])

            # 3. Language-level default
            elif lang_default.exists():
                voice_map[speaker] = str(lang_default)
                logger.debug("Speaker %s → lang default %s", speaker, voice_map[speaker])

            # 4. Global fallback
            elif global_default.exists():
                voice_map[speaker] = str(global_default)
                logger.debug("Speaker %s → global default %s", speaker, voice_map[speaker])

            else:
                logger.warning(
                    "No reference WAV found for speaker %s (lang=%s) — "
                    "engine will use its built-in default voice.",
                    speaker, language,
                )

        logger.info(
            "Built speaker voice map for %d speaker(s): %s",
            len(voice_map),
            {k: Path(v).name for k, v in voice_map.items()},
        )
        return voice_map

    def text_file_to_speech(
        self,
        source_path: str,
        output_path: str,
        *,
        alignment: bool | None = None,
        speaker_wav_map: dict[str, str] | None = None,
    ) -> None:
        """Generate time-aligned TTS audio from a translated JSON transcript.

        Args:
            source_path:      Path to the translated segments JSON.
            output_path:      Directory where the output WAV will be written.
            alignment:        Enable temporal alignment (clamped stretch).
            speaker_wav_map:  Per-speaker ``{speaker_label: relative_wav_path}``
                              mapping built from diarization data.  Takes
                              precedence over *speaker_wav*.
        """
        tts_text_file_to_speech(
            source_path,
            output_path,
            self.tts_engine,
            alignment=alignment,
            speaker_wav_map=speaker_wav_map,
        )

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        """Find a title by scanning *search_dir* for JSON files."""
        for f in search_dir.glob("*.json"):
            return f.stem
        return None

    def compute_alignment(
        self,
        en_transcript: dict,
        es_transcript: dict,
        silence_regions: list[dict],
        max_stretch: float = 1.4,
    ) -> list:
        """Run global alignment over EN and ES transcripts.

        Returns list[AlignedSegment].  Combines compute_segment_metrics and
        global_align into a single facade call for use by the align router.
        """
        from foreign_whispers.alignment import compute_segment_metrics, global_align
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        return global_align(metrics, silence_regions, max_stretch)
