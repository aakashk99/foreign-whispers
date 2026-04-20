"""Deterministic failure analysis and translation re-ranking.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The re-ranking function uses an LLM to generate shorter
translation candidates for any target language, with a character-level
truncation fallback if the API is unavailable.
"""

import dataclasses
import json
import logging
import re
import urllib.request

logger = logging.getLogger(__name__)

_CHARS_PER_SECOND: float = 15.0


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def _char_budget(duration_s: float) -> int:
    """Return the maximum character count that fits *duration_s*."""
    return int(duration_s * _CHARS_PER_SECOND)


def _truncation_candidate(text: str, budget_chars: int) -> TranslationCandidate | None:
    """Last-resort: truncate at the last sentence or word boundary within budget."""
    if len(text) <= budget_chars:
        return None
    chunk = text[:budget_chars]
    for punct in (".", "?", "!", ";", ","):
        idx = chunk.rfind(punct)
        if idx > budget_chars // 2:
            truncated = chunk[: idx + 1].strip()
            return TranslationCandidate(
                text=truncated,
                char_count=len(truncated),
                brevity_rationale="truncated at sentence boundary",
            )
    idx = chunk.rfind(" ")
    truncated = (chunk[:idx] if idx > 0 else chunk).strip()
    return TranslationCandidate(
        text=truncated,
        char_count=len(truncated),
        brevity_rationale="truncated at word boundary",
    )


def _llm_candidates(
    source_text: str,
    baseline_translation: str,
    target_duration_s: float,
    budget_chars: int,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Call the Anthropic API to generate condensed translation candidates.

    The LLM infers the target language from *baseline_translation*, so this
    works for any language without any language-specific configuration.

    Returns an empty list on any error so the caller can fall back gracefully.
    """
    context_block = ""
    if context_prev:
        context_block += f'Previous segment: "{context_prev}"\n'
    if context_next:
        context_block += f'Next segment:     "{context_next}"\n'

    prompt = f"""\
You are a professional translator and video dubbing editor.

Task: produce 3 shorter translations of the segment below in the SAME language \
as the baseline translation.
Each must fit within {budget_chars} characters \
(TTS budget: {target_duration_s:.1f}s at 15 chars/s).
Preserve the core meaning. Prefer natural phrasing over word-for-word accuracy.

{context_block}\
Source:   "{source_text}"
Baseline: "{baseline_translation}" ({len(baseline_translation)} chars — too long)

Return ONLY a JSON array of objects with keys "text" and "rationale".
Example: [{{"text": "...", "rationale": "removed filler"}}]
No markdown, no extra keys, no explanation outside the array."""

    try:
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        raw_text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()
        items = json.loads(raw_text)

        return [
            TranslationCandidate(
                text=(t := str(item.get("text", "")).strip()),
                char_count=len(t),
                brevity_rationale="LLM: " + str(item.get("rationale", "condensed")).strip(),
            )
            for item in items
            if (t := str(item.get("text", "")).strip())
            and len(t) < len(baseline_translation)
        ]

    except Exception as exc:
        logger.warning(
            "LLM re-ranking failed (%s: %s); using truncation fallback.",
            type(exc).__name__, exc,
        )
        return []


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    Works for any target language — the LLM infers the language from
    *baseline_es* and rewrites in kind.

    Strategy:
        1. **LLM** — ask the model for 3 condensed alternatives within the
           character budget.  Context segments are included for coherence.
        2. **Truncation fallback** — if the LLM is unavailable or all its
           candidates still exceed the budget, truncate at the nearest
           sentence or word boundary as a last resort.

    Args:
        source_text:       Original source-language segment text.
        baseline_es:       Baseline translation (any target language).
        target_duration_s: TTS time budget in seconds for this segment.
        context_prev:      Preceding segment text (improves LLM coherence).
        context_next:      Following segment text (improves LLM coherence).

    Returns:
        List of ``TranslationCandidate`` objects sorted shortest-first.
        Returns an empty list if the baseline already fits the budget.
    """
    budget_chars = _char_budget(target_duration_s)

    if len(baseline_es) <= budget_chars:
        logger.debug("Baseline already fits budget — returning empty list.")
        return []

    logger.info(
        "get_shorter_translations: budget=%.1fs (%d chars), baseline=%d chars.",
        target_duration_s, budget_chars, len(baseline_es),
    )

    # Stage 1: LLM candidates
    candidates = _llm_candidates(
        source_text=source_text,
        baseline_translation=baseline_es,
        target_duration_s=target_duration_s,
        budget_chars=budget_chars,
        context_prev=context_prev,
        context_next=context_next,
    )

    # Stage 2: truncation fallback if nothing fits yet
    if not any(c.char_count <= budget_chars for c in candidates):
        fallback = _truncation_candidate(baseline_es, budget_chars)
        if fallback:
            candidates.append(fallback)

    # Deduplicate, exclude the unchanged baseline, sort shortest-first
    seen: set[str] = set()
    unique: list[TranslationCandidate] = []
    for c in candidates:
        if c.text and c.text != baseline_es and c.text not in seen:
            seen.add(c.text)
            unique.append(c)

    unique.sort(key=lambda c: c.char_count)

    logger.info(
        "Returning %d candidate(s); shortest=%d chars.",
        len(unique), unique[0].char_count if unique else 0,
    )
    return unique
