"""
Podcast Engine for Fish Speech — Multi-Speaker Long-Form TTS.

This module provides a high-level interface for generating podcast-style
multi-speaker audio using the Fish Audio S2-Pro model.

Speaker assignment
------------------
Each speaker is given a numeric ID (0-based) matching their position in
the ``speakers`` list passed to ``synthesize_podcast``.  These IDs map
directly to the ``<|speaker:N|>`` tokens the model understands.

Reference audio for each speaker is passed in speaker-ID order so the
model can associate each voice with the correct ``<|speaker:N|>`` slot in
the system prompt.

Script format
-------------
Write your podcast script in "Name: text" dialogue form::

    Alice: Welcome to Tech Talk!
    Bob: [excited] Thanks for having me, Alice.
    Alice: Let's dive right in. What do you think about the new models?

Inline emotion/prosody tags are supported: ``[laugh]``, ``[whisper]``,
``[pause]``, ``[excited]``, ``[angry]``, ``[sad]``, ``[sigh]``, etc.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


MAX_SPEAKERS = 4

# Per-speaker accent colours (used in HTML preview and UI badges)
SPEAKER_COLORS = ["#3B82F6", "#F97316", "#22C55E", "#A855F7"]

_DEFAULT_NAMES = ["Alice", "Bob", "Charlie", "Diana"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpeakerConfig:
    """Configuration for a single podcast speaker.

    Attributes
    ----------
    name:
        Human-readable speaker name used in the script (e.g. ``"Alice"``).
    speaker_id:
        Zero-based numeric index that maps to the model's ``<|speaker:N|>``
        token.  Assigned automatically by position in the speakers list.
    reference_audio:
        Raw bytes of the reference audio clip (5–30 s recommended).
        When ``None``, the model generates a generic voice for this speaker.
    reference_text:
        Exact transcription of ``reference_audio``.  Required when
        ``reference_audio`` is set; improves cloning accuracy.
    """

    name: str
    speaker_id: int = 0
    reference_audio: Optional[bytes] = None
    reference_text: str = ""

    @property
    def has_reference(self) -> bool:
        """True when a non-empty reference audio clip has been provided."""
        return bool(self.reference_audio)

    @property
    def color(self) -> str:
        return SPEAKER_COLORS[self.speaker_id % len(SPEAKER_COLORS)]

    def __repr__(self) -> str:
        status = "✓ ref" if self.has_reference else "⚠ no ref"
        return f"SpeakerConfig({self.name!r}, id={self.speaker_id}, {status})"


@dataclass
class PodcastTurn:
    """A single parsed speaker turn in the podcast script."""

    speaker_name: str
    speaker_id: int
    text: str

    def to_tagged(self) -> str:
        """Return the turn formatted as ``<|speaker:N|>text``."""
        return f"<|speaker:{self.speaker_id}|>{self.text}"


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------


def parse_podcast_script(
    script: str,
    speakers: List[SpeakerConfig],
) -> List[PodcastTurn]:
    """Parse a "Name: text" podcast script into :class:`PodcastTurn` objects.

    Handles:

    * ``"SpeakerName: text"``
    * ``"SpeakerName (emotion): text"`` — parenthetical is stripped
    * Multi-line continuation (no prefix → appended to the previous turn)
    * Blank lines — skipped

    Parameters
    ----------
    script:
        Raw podcast script string.
    speakers:
        List of :class:`SpeakerConfig` defining valid speaker names.

    Returns
    -------
    List[PodcastTurn]
        Turns in script order.
    """
    # Case-insensitive name → SpeakerConfig map
    name_map: dict = {sp.name.lower().strip(): sp for sp in speakers if sp.name.strip()}

    turns: List[PodcastTurn] = []

    for raw_line in script.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Match "Name:" or "Name (anything):" at the start of the line
        m = re.match(r"^([^:()\n]+?)(?:\s*\([^)]*\))?\s*:\s*(.+)$", line)
        if m:
            raw_name = m.group(1).strip()
            text = m.group(2).strip()
            sp = name_map.get(raw_name.lower())
            if sp is not None:
                turns.append(
                    PodcastTurn(
                        speaker_name=sp.name,
                        speaker_id=sp.speaker_id,
                        text=text,
                    )
                )
            else:
                # Unknown speaker name — treat as a continuation of the previous turn
                if turns:
                    turns[-1].text += " " + line
        else:
            # No "Name:" prefix — continuation of the previous turn
            if turns:
                turns[-1].text += " " + line

    return turns


def turns_to_model_text(turns: List[PodcastTurn]) -> str:
    """Convert :class:`PodcastTurn` list to ``<|speaker:N|>``-tagged text.

    The resulting string is passed directly to ``generate_long`` as the
    ``text`` argument.
    """
    return "\n".join(t.to_tagged() for t in turns)


def script_preview_html(
    turns: List[PodcastTurn],
    speakers: List[SpeakerConfig],
) -> str:
    """Render a colored HTML table previewing parsed script turns.

    Each speaker name is rendered in their accent color.  Unknown speaker IDs
    fall back to gray.
    """
    id_to_color = {sp.speaker_id: sp.color for sp in speakers}
    id_to_name = {sp.speaker_id: sp.name for sp in speakers}

    if not turns:
        return (
            "<p style='color:#64748b;font-style:italic;padding:8px;'>"
            "No turns parsed — check that speaker names in the script match "
            "the configured speaker names exactly (case-insensitive)."
            "</p>"
        )

    rows = "".join(
        f"<tr>"
        f"<td style='color:{id_to_color.get(t.speaker_id, '#888')};"
        f"font-weight:700;white-space:nowrap;padding:5px 16px 5px 6px;"
        f"vertical-align:top;font-size:13px;'>"
        f"{id_to_name.get(t.speaker_id, t.speaker_name)}</td>"
        f"<td style='padding:5px 6px;font-size:13px;line-height:1.5;"
        f"color:#1e293b;'>{t.text}</td>"
        f"</tr>"
        for t in turns
    )

    total = len(turns)
    unique = len({t.speaker_id for t in turns})
    header = (
        f"<p style='font-size:12px;color:#64748b;margin-bottom:6px;'>"
        f"✓ {total} turn{'s' if total != 1 else ''} parsed across "
        f"{unique} speaker{'s' if unique != 1 else ''}</p>"
    )

    table = (
        "<table style='width:100%;border-collapse:collapse;"
        "border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;'>"
        + rows
        + "</table>"
    )
    return header + table


# ---------------------------------------------------------------------------
# Helper: build SpeakerConfig list from Gradio inputs
# ---------------------------------------------------------------------------


def build_speaker_configs(
    n_speakers: int,
    names: List[str],
    audio_paths: List[Optional[str]],
    ref_texts: List[str],
) -> List[SpeakerConfig]:
    """Construct a :class:`SpeakerConfig` list from Gradio widget values.

    Parameters
    ----------
    n_speakers:
        Number of active speakers (1–MAX_SPEAKERS).
    names:
        Speaker name strings (length MAX_SPEAKERS, trailing entries may be
        unused).
    audio_paths:
        File paths returned by ``gr.Audio(type='filepath')``.
    ref_texts:
        Reference text strings.

    Returns
    -------
    List[SpeakerConfig]
        One entry per active speaker, with ``speaker_id == i``.
    """
    speakers = []
    for i in range(int(n_speakers)):
        name = (names[i] or _DEFAULT_NAMES[i]).strip() or f"Speaker {i + 1}"
        audio_bytes: Optional[bytes] = None
        if audio_paths[i]:
            try:
                with open(audio_paths[i], "rb") as fh:
                    audio_bytes = fh.read()
            except OSError as exc:
                logger.warning(f"Could not read audio for '{name}': {exc}")
        speakers.append(
            SpeakerConfig(
                name=name,
                speaker_id=i,
                reference_audio=audio_bytes,
                reference_text=(ref_texts[i] or "").strip(),
            )
        )
    return speakers


# ---------------------------------------------------------------------------
# PodcastSynthesizer
# ---------------------------------------------------------------------------


class PodcastSynthesizer:
    """High-level multi-speaker podcast TTS synthesizer.

    Wraps :class:`fish_speech.inference_engine.TTSInferenceEngine` to support
    ordered multi-speaker voice cloning and natural-language podcast scripts.

    Parameters
    ----------
    tts_engine:
        An initialised ``TTSInferenceEngine`` instance.
    whisper_model_dir:
        Optional path to a local Whisper model directory used for reference
        audio auto-transcription.  Pass ``None`` to disable Whisper support.
    """

    def __init__(self, tts_engine, whisper_model_dir: Optional[str] = None):
        self.engine = tts_engine
        self.whisper_model_dir = whisper_model_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_preview(
        self,
        speaker: SpeakerConfig,
        preview_text: str = (
            "Hello! This is a preview of my voice. "
            "The quick brown fox jumps over the lazy dog."
        ),
        temperature: float = 0.8,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        """Generate a short voice preview for a single speaker.

        Parameters
        ----------
        speaker:
            The speaker whose voice should be previewed.
        preview_text:
            Text to synthesise during the preview (default: a short phrase).

        Returns
        -------
        Tuple[int, np.ndarray]
            ``(sample_rate, audio_array)`` ready for a Gradio ``gr.Audio``.
        """
        references = []
        if speaker.has_reference:
            # Pre-tag as speaker 0 for a single-speaker preview
            references = [
                ServeReferenceAudio(
                    audio=speaker.reference_audio,
                    text=f"<|speaker:0|>{speaker.reference_text}",
                )
            ]

        req = ServeTTSRequest(
            text=f"<|speaker:0|>{preview_text}",
            references=references,
            max_new_tokens=512,
            chunk_length=200,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
        )

        for result in self.engine.inference(req):
            if result.code == "final":
                return result.audio
            if result.code == "error":
                raise RuntimeError(f"Preview generation failed: {result.error}")

        raise RuntimeError("No audio was generated for the preview.")

    def synthesize_podcast(
        self,
        script: str,
        speakers: List[SpeakerConfig],
        temperature: float = 0.8,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        chunk_length: int = 200,
        max_new_tokens: int = 0,
        seed: Optional[int] = None,
        progress_fn=None,
    ) -> Tuple[int, np.ndarray]:
        """Synthesise a full multi-speaker podcast.

        The ``script`` is parsed into speaker turns, reference audio is
        encoded in speaker-ID order, and the model generates audio for the
        entire script in a single ``generate_long`` call.

        Parameters
        ----------
        script:
            Podcast script in ``"Name: text"`` format.
        speakers:
            Ordered :class:`SpeakerConfig` list — index == speaker_id.
            Speakers **with** reference audio get their voices cloned;
            speakers without reference audio receive a generic voice.
        temperature, top_p, repetition_penalty, chunk_length, max_new_tokens, seed:
            Standard ``ServeTTSRequest`` generation parameters.
        progress_fn:
            Optional ``Callable[[float, str], None]`` invoked with
            ``(fraction, message)`` for Gradio progress updates.

        Returns
        -------
        Tuple[int, np.ndarray]
            ``(sample_rate, audio_array)`` for the full podcast.

        Raises
        ------
        ValueError
            If the script produces no parseable turns.
        RuntimeError
            If the generation engine reports an error or produces no audio.
        """
        # ── 1. Parse script ──────────────────────────────────────────────
        turns = parse_podcast_script(script, speakers)
        if not turns:
            raise ValueError(
                "No turns could be parsed from the script.\n"
                "Make sure speaker names in the script exactly match the "
                "names configured in the Speakers tab (case-insensitive)."
            )

        model_text = turns_to_model_text(turns)
        unique_ids = sorted({t.speaker_id for t in turns})
        logger.info(
            f"Podcast: {len(turns)} turns, {len(unique_ids)} unique speaker(s), "
            f"{len(model_text)} chars"
        )

        # ── 2. Build ordered references (one per speaker, in speaker_id order) ──
        #
        # IMPORTANT: Reference texts are pre-tagged with the correct
        # <|speaker:N|> ID so that generate_long does NOT re-index them by
        # list position.  This ensures speaker 2's reference is labelled
        # <|speaker:2|> even if speaker 1 has no reference and is absent
        # from the list.
        references = []
        for sp in sorted(speakers, key=lambda s: s.speaker_id):
            if sp.has_reference:
                tagged_text = f"<|speaker:{sp.speaker_id}|>{sp.reference_text}"
                references.append(
                    ServeReferenceAudio(
                        audio=sp.reference_audio,
                        text=tagged_text,
                    )
                )
            else:
                logger.warning(
                    f"Speaker '{sp.name}' (id={sp.speaker_id}) has no reference "
                    "audio — a generic voice will be generated."
                )

        if not references:
            logger.warning(
                "No reference audio was provided for any speaker.  "
                "The model will generate voices without voice cloning."
            )

        if progress_fn:
            progress_fn(0.05, f"Generating podcast ({len(turns)} turns)…")

        # ── 3. Run inference ─────────────────────────────────────────────
        req = ServeTTSRequest(
            text=model_text,
            references=references,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
        )

        segments = []
        sample_rate: Optional[int] = None

        for result in self.engine.inference(req):
            if result.code == "final":
                if result.audio:
                    sample_rate, arr = result.audio
                    segments.append(arr)
                if progress_fn:
                    progress_fn(1.0, "Done!")
                break
            elif result.code == "segment" and result.audio:
                sample_rate, arr = result.audio
                segments.append(arr)
            elif result.code == "error":
                raise RuntimeError(f"Podcast generation failed: {result.error}")

        if not segments or sample_rate is None:
            raise RuntimeError(
                "The model produced no audio.  "
                "Check the script content and model checkpoint."
            )

        return sample_rate, np.concatenate(segments, axis=0)

    def transcribe_with_whisper(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe a reference audio file using the local Whisper model.

        Parameters
        ----------
        audio_path:
            Path to the audio file (as returned by Gradio).
        language:
            Optional ISO-639 language code hint; ``None`` = auto-detect.

        Returns
        -------
        str
            Transcribed text.

        Raises
        ------
        RuntimeError
            If the Whisper model directory was not configured.
        """
        if not self.whisper_model_dir:
            raise RuntimeError(
                "Whisper model directory was not configured at startup.  "
                "Pass --whisper-model-dir when launching the Podcast Studio."
            )
        from tools.webui.whisper_utils import transcribe_reference_audio

        return transcribe_reference_audio(
            audio_path=audio_path,
            model_dir=self.whisper_model_dir,
            language=language,
        )
