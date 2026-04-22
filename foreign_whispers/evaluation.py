"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


# ── Task 4: Multi-dimensional dubbing quality scorecard ───────────────────────

def _char_ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity between two strings.

    Used as a lightweight semantic fidelity proxy that requires no external
    model.  Trigrams (``n=3``) capture morphological similarity well across
    Romance-language pairs.

    Returns a value in [0, 1] where 1.0 means identical n-gram sets.
    """
    def _ngrams(t: str) -> set[str]:
        t = t.lower().strip()
        return {t[i : i + n] for i in range(len(t) - n + 1)} if len(t) >= n else {t}

    a, b = _ngrams(text_a), _ngrams(text_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _speaking_rate_variance(aligned: list[AlignedSegment]) -> float:
    """Variance of words-per-second across all scheduled segments.

    Low variance indicates natural, consistent delivery pace.  High variance
    means some segments are extremely fast-spoken while others are very slow —
    a sign of poor timing alignment.  Returns 0.0 for fewer than two segments.
    """
    rates = []
    for seg in aligned:
        duration = seg.scheduled_end - seg.scheduled_start
        if duration > 0:
            word_count = max(1, len(seg.text.split()))
            rates.append(word_count / duration)
    return _stats.variance(rates) if len(rates) >= 2 else 0.0


def _intelligibility_score(
    tts_wav_path: str,
    expected_text: str,
    model_size: str = "base",
) -> float | None:
    """Whisper STT round-trip intelligibility score.

    Transcribes the dubbed audio with Whisper, then measures character
    trigram similarity between the round-tripped transcription and the
    expected translated text.  Returns ``None`` if ``whisper`` is not
    installed or the file cannot be transcribed.

    A score near 1.0 means the TTS output is highly intelligible.  Values
    below 0.5 indicate significant pronunciation or timing distortion.

    Args:
        tts_wav_path: Path to the synthesised TTS WAV file.
        expected_text: The translated text the TTS was given to speak.
        model_size: Whisper model size (``"base"`` balances speed/accuracy).

    Returns:
        Float in [0, 1], or ``None`` if whisper is unavailable.
    """
    try:
        import whisper  # type: ignore
        model  = whisper.load_model(model_size)
        result = model.transcribe(tts_wav_path)
        transcribed = result.get("text", "").strip()
        return _char_ngram_similarity(expected_text, transcribed)
    except Exception:
        return None


def dubbing_quality_scorecard(
    metrics:          list[SegmentMetrics],
    aligned:          list[AlignedSegment],
    tts_wav_path:     str | None = None,
    whisper_model:    str        = "base",
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Combines four complementary dimensions into a single ``overall_score``
    in [0, 1] and exposes each dimension individually for diagnostics.

    Dimensions
    ----------
    1. **Timing accuracy** (from existing metrics) — mean absolute error
       between predicted and source durations, and % severe stretches.
       Score = max(0, 1 − mean_error / 2).

    2. **Naturalness** — variance of words-per-second across segments.
       Lower variance → more consistent delivery pace → higher score.
       Score = max(0, 1 − variance / 10).

    3. **Semantic fidelity** — mean character-trigram Jaccard similarity
       between each source and translated segment.  High similarity means
       the translation preserves content rather than paraphrasing heavily.
       Score in [0, 1] directly.

    4. **Intelligibility** — Whisper STT round-trip on the TTS WAV
       (optional; only computed when ``tts_wav_path`` is supplied and
       ``whisper`` is installed).  ``None`` if unavailable.

    The ``overall_score`` averages the three always-available dimensions
    (timing, naturalness, semantic fidelity).  Intelligibility is excluded
    from the composite when not available so the score remains comparable.

    Args:
        metrics: Per-segment metrics from ``compute_segment_metrics``.
        aligned: Aligned segments from ``global_align`` or ``global_align_dp``.
        tts_wav_path: Optional path to the dubbed TTS WAV file for
            intelligibility measurement.
        whisper_model: Whisper model size for the STT round-trip.

    Returns:
        Dict with keys: ``timing_accuracy``, ``naturalness``,
        ``semantic_fidelity``, ``intelligibility`` (may be ``None``),
        ``overall_score``, and detailed sub-metrics.
    """
    if not metrics:
        return {
            "timing_accuracy":   0.0,
            "naturalness":       0.0,
            "semantic_fidelity": 0.0,
            "intelligibility":   None,
            "overall_score":     0.0,
            "detail":            {},
        }

    # ── Dimension 1: Timing accuracy ─────────────────────────────────────────
    base_report   = clip_evaluation_report(metrics, aligned)
    mean_err      = base_report["mean_abs_duration_error_s"]
    pct_severe    = base_report["pct_severe_stretch"]
    timing_score  = max(0.0, 1.0 - mean_err / 2.0)  # error of 2s → score 0

    # ── Dimension 2: Naturalness (speaking rate variance) ────────────────────
    rate_variance  = _speaking_rate_variance(aligned)
    natural_score  = max(0.0, 1.0 - rate_variance / 10.0)

    # ── Dimension 3: Semantic fidelity ────────────────────────────────────────
    fidelity_scores = [
        _char_ngram_similarity(m.source_text, m.translated_text)
        for m in metrics
        if m.source_text and m.translated_text
    ]
    fidelity_score = round(_stats.mean(fidelity_scores), 3) if fidelity_scores else 0.0

    # ── Dimension 4: Intelligibility (optional) ───────────────────────────────
    intelligibility: float | None = None
    if tts_wav_path:
        all_text = " ".join(seg.text for seg in aligned)
        intelligibility = _intelligibility_score(tts_wav_path, all_text, whisper_model)

    # ── Composite overall score ───────────────────────────────────────────────
    core_scores = [timing_score, natural_score, fidelity_score]
    if intelligibility is not None:
        core_scores.append(intelligibility)
    overall = round(_stats.mean(core_scores), 3)

    return {
        "timing_accuracy":   round(timing_score,   3),
        "naturalness":       round(natural_score,   3),
        "semantic_fidelity": fidelity_score,
        "intelligibility":   round(intelligibility, 3) if intelligibility is not None else None,
        "overall_score":     overall,
        "detail": {
            **base_report,
            "speaking_rate_variance_wps2": round(rate_variance, 4),
        },
    }
