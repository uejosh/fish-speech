"""
Gradio WebUI — Fish Speech Podcast Studio.

Layout
------
┌─────────────────────────────────────────────────────────────────────┐
│  🎙️  Fish Speech — Podcast Studio                                   │
├────────────────────────────┬────────────────────────────────────────┤
│  [Tab: 🎤 Speakers]        │  [Tab: 📝 Script]   [Tab: ⚙️ Settings] │
│                            │                                        │
│  Num speakers slider       │  Format guide (collapsible)            │
│  ┌──────────────────────┐  │  Script textbox (large)                │
│  │ ● Speaker 1 (blue)   │  │  Word count + 🔍 Preview button        │
│  │   Name / Audio /     │  │  Parsed-turns HTML preview             │
│  │   Ref text /         │  └────────────────────────────────────────┤
│  │   Transcribe /       │                                           │
│  │   Preview Voice      │  ⚙️ Temperature / Top-P / Rep-Pen /       │
│  ├──────────────────────┤     Chunk length / Max tokens / Seed      │
│  │ ● Speaker 2 (orange) │                                           │
│  │   …same…             │                                           │
│  └──────────────────────┘                                           │
│                                                                     │
│             [ 🎙️  Generate Podcast ]  ← primary, full-width        │
│                                                                     │
│  Output: error HTML + gr.Audio (full podcast)                       │
└─────────────────────────────────────────────────────────────────────┘
"""

import html as html_lib
from typing import List, Optional

import gradio as gr
from loguru import logger

from podcast.engine import (
    MAX_SPEAKERS,
    SpeakerConfig,
    SPEAKER_COLORS,
    _DEFAULT_NAMES,
    build_speaker_configs,
    parse_podcast_script,
    script_preview_html,
    PodcastSynthesizer,
)


# ---------------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------------

_HEADER_MD = """
# 🎙️ Fish Speech — Podcast Studio
**Multi-speaker long-form podcast generation powered by [Fish Audio S2-Pro](https://fish.audio/)**

Configure up to **4 speakers** with reference voice clips, write your podcast script in
natural dialogue format, then generate the full multi-speaker audio in one click.
"""

_FORMAT_GUIDE_MD = """
### 📋 Script Format

One turn per line — `SpeakerName: Text of what they say.`

```
Alice: Welcome to AI Frontiers! I'm your host, Alice.
Bob: [excited] Thanks so much for having me — I've been looking forward to this!
Alice: Let's start with the big picture. What's the most exciting trend you're seeing?
Bob: Honestly? [pause] It's the convergence of language models and speech synthesis.
Alice: [laugh] Well, we're literally demonstrating that right now.
```

**Emotion & prosody tags** (inline):
`[laugh]` · `[whisper]` · `[excited]` · `[pause]` · `[short pause]` · `[sigh]` ·
`[angry]` · `[sad]` · `[surprised]` · `[chuckle]` · `[emphasis]` · `[inhale]` ·
`[exhale]` · `[singing]` · `[screaming]` · `[shouting]`

Free-form tags also work: `[speaking quickly]`, `[in a dramatic tone]`
"""

_EXAMPLE_SCRIPT = """\
Alice: Welcome to AI Frontiers, the podcast where we explore the cutting edge of \
artificial intelligence. I'm your host Alice, and today we have a very special guest.
Bob: Thanks for having me, Alice! [excited] I've been looking forward to this conversation \
for weeks.
Alice: Bob, you've been working on large language models for years now. What do you \
think is the most exciting recent development?
Bob: That's a great question. [pause] I think the most fascinating thing is how these \
models are starting to understand context over incredibly long sequences — and the \
breakthroughs in voice synthesis are remarkable too.
Alice: [laugh] And here we are, using one to generate this very conversation!
Bob: [chuckle] Exactly! It's a bit meta, isn't it? But on a serious note, I think \
the next frontier is real-time, emotionally expressive speech that sounds completely human.
Alice: Well, speaking of which — listeners, you can find all our episodes and transcripts \
at our website. Until next time, keep exploring the frontiers!
Bob: Thanks everyone. [excited] See you next week!\
"""

_APP_CSS = """
.speaker-card {
    border: 1.5px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 14px 10px 14px;
    margin-bottom: 10px;
    background: #fafafa;
}
.status-badge {
    font-size: 12px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
}
.generate-row button {
    font-size: 1.08em !important;
    min-height: 52px !important;
}
"""


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------


def build_podcast_app(
    synthesizer: PodcastSynthesizer,
    whisper_model_dir: Optional[str] = None,
) -> gr.Blocks:
    """Build and return the Podcast Studio Gradio application.

    Parameters
    ----------
    synthesizer:
        A ready-to-use :class:`podcast.engine.PodcastSynthesizer` instance.
    whisper_model_dir:
        Directory containing a Whisper model for auto-transcription; pass
        ``None`` to disable the transcribe buttons.

    Returns
    -------
    gr.Blocks
        The assembled Gradio application (not yet launched).
    """
    has_whisper = bool(whisper_model_dir)

    with gr.Blocks(
        title="🎙️ Fish Speech — Podcast Studio",
        theme=gr.themes.Soft(),
        css=_APP_CSS,
    ) as app:

        gr.Markdown(_HEADER_MD)

        # ── Speaker state & generation settings ───────────────────────────
        # These are read by both the Speakers tab and the generate callback,
        # so we define the Gradio component variables before the tabs.

        # We will collect component references as we build the UI
        name_inputs: List[gr.Textbox] = []
        audio_inputs: List[gr.Audio] = []
        reftext_inputs: List[gr.Textbox] = []
        transcribe_btns: List[gr.Button] = []
        preview_btns: List[gr.Button] = []
        preview_audios: List[gr.Audio] = []
        status_badges: List[gr.HTML] = []
        speaker_groups: List[gr.Group] = []

        with gr.Tabs():

            # ──────────────────────────────────────────────────────────────
            # TAB 1 — SPEAKERS
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("🎤 Speakers"):

                gr.Markdown(
                    "Upload a **5–30 second** reference audio clip per speaker to clone "
                    "their voice.  Provide the exact transcription of the clip for best "
                    "results.  Speakers without a reference clip receive a generic voice."
                )

                num_speakers = gr.Slider(
                    label="Number of speakers",
                    minimum=1,
                    maximum=MAX_SPEAKERS,
                    value=2,
                    step=1,
                    info=f"Supports up to {MAX_SPEAKERS} speakers",
                )

                for i in range(MAX_SPEAKERS):
                    color = SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
                    with gr.Group(
                        visible=(i < 2),
                        elem_classes=["speaker-card"],
                    ) as sg:

                        with gr.Row():
                            gr.HTML(
                                f"<span style='color:{color};font-weight:700;"
                                f"font-size:1.05em;'>● Speaker {i + 1}</span>"
                            )
                            sb = gr.HTML(
                                _badge("⚠️ No reference", color="#f59e0b"),
                                elem_classes=["status-badge"],
                            )

                        with gr.Row():
                            ni = gr.Textbox(
                                label="Name",
                                value=_DEFAULT_NAMES[i],
                                placeholder=f"e.g. {_DEFAULT_NAMES[i]}",
                                scale=1,
                            )

                        with gr.Row():
                            ai = gr.Audio(
                                label="Reference Audio",
                                type="filepath",
                                scale=3,
                            )
                            tb = gr.Button(
                                "🎤 Transcribe",
                                variant="secondary",
                                size="sm",
                                scale=1,
                                interactive=has_whisper,
                            )
                        if not has_whisper:
                            gr.Markdown(
                                "<small style='color:#94a3b8;'>Whisper not configured — "
                                "enter reference text manually.</small>"
                            )

                        rti = gr.Textbox(
                            label="Reference Text",
                            placeholder="Exact words spoken in the reference audio clip above",
                            lines=2,
                            info="Improves voice-cloning accuracy",
                        )

                        with gr.Row():
                            pb = gr.Button(
                                f"▶ Preview Voice",
                                variant="secondary",
                                size="sm",
                            )

                        pa = gr.Audio(
                            label="Voice Preview",
                            type="numpy",
                            interactive=False,
                            visible=False,
                        )

                    speaker_groups.append(sg)
                    name_inputs.append(ni)
                    audio_inputs.append(ai)
                    reftext_inputs.append(rti)
                    transcribe_btns.append(tb)
                    preview_btns.append(pb)
                    preview_audios.append(pa)
                    status_badges.append(sb)

            # ──────────────────────────────────────────────────────────────
            # TAB 2 — SCRIPT
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("📝 Script"):

                with gr.Accordion("📖 Format Guide", open=False):
                    gr.Markdown(_FORMAT_GUIDE_MD)

                script_box = gr.Textbox(
                    label="Podcast Script  —  format: SpeakerName: Text goes here.",
                    placeholder=(
                        "Alice: Welcome to the show!\n"
                        "Bob: [excited] Thanks for having me!\n"
                        "Alice: Let's dive right in..."
                    ),
                    lines=20,
                    max_lines=80,
                    value=_EXAMPLE_SCRIPT,
                )

                with gr.Row():
                    word_count_md = gr.Markdown("Words: 0", scale=2)
                    parse_btn = gr.Button(
                        "🔍 Preview Script Parsing",
                        variant="secondary",
                        size="sm",
                        scale=1,
                    )

                script_preview_out = gr.HTML(
                    value=(
                        "<p style='color:#94a3b8;font-style:italic;padding:8px;'>"
                        "Click '🔍 Preview Script Parsing' to validate speaker names "
                        "and see how the script will be split into turns."
                        "</p>"
                    ),
                )

                gr.Examples(
                    label="📚 Example Scripts",
                    examples=[
                        [_EXAMPLE_SCRIPT],
                        [
                            "Alice: Hello and welcome to the show!\n"
                            "Bob: Great to be here.\n"
                            "Alice: Let's get started with today's topic.\n"
                            "Bob: [excited] Can't wait!"
                        ],
                        [
                            "Host: Today we're discussing the future of AI.\n"
                            "Guest: It's an exciting time — [pause] the pace of progress "
                            "is simply staggering.\n"
                            "Host: [laugh] You can say that again!\n"
                            "Guest: And what's remarkable is how quickly these tools "
                            "are becoming accessible to everyone."
                        ],
                    ],
                    inputs=[script_box],
                )

            # ──────────────────────────────────────────────────────────────
            # TAB 3 — SETTINGS
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("⚙️ Settings"):

                gr.Markdown(
                    "Generation parameters.  The defaults work well for most podcasts; "
                    "adjust only if you experience quality issues."
                )

                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        info="Higher = more expressive & varied speech",
                        minimum=0.7,
                        maximum=1.0,
                        value=0.8,
                        step=0.01,
                    )
                    top_p = gr.Slider(
                        label="Top-P",
                        minimum=0.7,
                        maximum=0.95,
                        value=0.8,
                        step=0.01,
                    )

                with gr.Row():
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        info="Reduces stuck/looping generation artifacts",
                        minimum=1.0,
                        maximum=1.2,
                        value=1.1,
                        step=0.01,
                    )
                    chunk_length = gr.Slider(
                        label="Chunk Length (bytes per batch)",
                        info="How many turns are grouped per generation batch",
                        minimum=100,
                        maximum=300,
                        value=200,
                        step=10,
                    )

                with gr.Row():
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens per batch (0 = unlimited)",
                        minimum=0,
                        maximum=2048,
                        value=0,
                        step=64,
                    )
                    seed = gr.Number(
                        label="Seed  (0 = random)",
                        value=0,
                        precision=0,
                        info="Fixed seed for reproducible generation",
                    )

        # ── Generate button ───────────────────────────────────────────────
        with gr.Row(elem_classes=["generate-row"]):
            generate_btn = gr.Button(
                "🎙️  Generate Podcast",
                variant="primary",
                size="lg",
            )

        # ── Output ────────────────────────────────────────────────────────
        gr.Markdown("### 🎵 Output")
        error_out = gr.HTML(value="")
        audio_out = gr.Audio(
            label="Full Podcast Audio",
            type="numpy",
            interactive=False,
            autoplay=False,
        )

        # ══════════════════════════════════════════════════════════════════
        # Event callbacks
        # ══════════════════════════════════════════════════════════════════

        # 1 ── Dynamic speaker group visibility ───────────────────────────
        def _update_visibility(n: int):
            return [gr.update(visible=(i < int(n))) for i in range(MAX_SPEAKERS)]

        num_speakers.change(
            fn=_update_visibility,
            inputs=[num_speakers],
            outputs=speaker_groups,
        )

        # 2 ── Status badge when reference audio is uploaded / cleared ────
        def _update_status(audio_path: Optional[str]) -> str:
            if audio_path:
                return _badge("✅ Reference loaded", color="#16a34a")
            return _badge("⚠️ No reference", color="#f59e0b")

        for i in range(MAX_SPEAKERS):
            audio_inputs[i].change(
                fn=_update_status,
                inputs=[audio_inputs[i]],
                outputs=[status_badges[i]],
            )

        # 3 ── Live word count ─────────────────────────────────────────────
        def _word_count(text: str) -> str:
            if not text or not text.strip():
                return "Words: 0"
            return f"Words: {len(text.split())}"

        script_box.change(
            fn=_word_count,
            inputs=[script_box],
            outputs=[word_count_md],
        )

        # 4 ── Script preview / validation ────────────────────────────────
        def _preview_script(script: str, n_sp: int, *names) -> str:
            sps = [
                SpeakerConfig(
                    name=(names[i] or _DEFAULT_NAMES[i]).strip() or f"Speaker {i + 1}",
                    speaker_id=i,
                )
                for i in range(int(n_sp))
            ]
            turns = parse_podcast_script(script, sps)
            return script_preview_html(turns, sps)

        parse_btn.click(
            fn=_preview_script,
            inputs=[script_box, num_speakers, *name_inputs],
            outputs=[script_preview_out],
        )

        # 5 ── Transcribe buttons ─────────────────────────────────────────
        for i in range(MAX_SPEAKERS):
            if has_whisper:
                transcribe_btns[i].click(
                    fn=lambda ap: (
                        synthesizer.transcribe_with_whisper(ap) if ap else ""
                    ),
                    inputs=[audio_inputs[i]],
                    outputs=[reftext_inputs[i]],
                )
            else:
                transcribe_btns[i].click(
                    fn=lambda: (
                        "Whisper is not configured.  "
                        "Please enter the reference text manually."
                    ),
                    inputs=[],
                    outputs=[reftext_inputs[i]],
                )

        # 6 ── Per-speaker voice preview ──────────────────────────────────
        for i in range(MAX_SPEAKERS):

            def _make_preview_fn(idx: int):
                def _preview(audio_path, ref_text, temp, tp, rep_pen):
                    if not audio_path:
                        return (
                            None,
                            gr.update(visible=False),
                            _badge("⚠️ Upload audio first", color="#f59e0b"),
                        )
                    try:
                        with open(audio_path, "rb") as fh:
                            ab = fh.read()
                        sp = SpeakerConfig(
                            name="Preview",
                            speaker_id=0,
                            reference_audio=ab,
                            reference_text=(ref_text or "").strip(),
                        )
                        audio = synthesizer.synthesize_preview(
                            sp,
                            temperature=temp,
                            top_p=tp,
                            repetition_penalty=rep_pen,
                        )
                        return (
                            audio,
                            gr.update(visible=True),
                            _badge("✅ Reference loaded", color="#16a34a"),
                        )
                    except Exception as exc:
                        logger.error(f"Speaker preview failed: {exc}")
                        return (
                            None,
                            gr.update(visible=False),
                            _badge("❌ Preview failed", color="#dc2626"),
                        )

                return _preview

            preview_btns[i].click(
                fn=_make_preview_fn(i),
                inputs=[
                    audio_inputs[i],
                    reftext_inputs[i],
                    temperature,
                    top_p,
                    repetition_penalty,
                ],
                outputs=[preview_audios[i], preview_audios[i], status_badges[i]],
            )

        # 7 ── Main generate callback ──────────────────────────────────────
        def _generate(
            script,
            n_sp,
            *all_vals,
            progress=gr.Progress(),
        ):
            """Unpack Gradio inputs and run podcast synthesis."""
            # Layout of all_vals:
            #   [0 .. MAX_SPEAKERS-1]              → name strings
            #   [MAX_SPEAKERS .. 2*MAX_SPEAKERS-1] → audio file paths
            #   [2*MAX_SPEAKERS .. 3*MAX_SPEAKERS-1] → ref texts
            #   [3*MAX_SPEAKERS + 0] temperature
            #   [3*MAX_SPEAKERS + 1] top_p
            #   [3*MAX_SPEAKERS + 2] repetition_penalty
            #   [3*MAX_SPEAKERS + 3] chunk_length
            #   [3*MAX_SPEAKERS + 4] max_new_tokens
            #   [3*MAX_SPEAKERS + 5] seed
            S = MAX_SPEAKERS
            names_v = list(all_vals[:S])
            audio_v = list(all_vals[S : 2 * S])
            texts_v = list(all_vals[2 * S : 3 * S])
            temp_v = float(all_vals[3 * S])
            tp_v = float(all_vals[3 * S + 1])
            rep_v = float(all_vals[3 * S + 2])
            chunk_v = int(all_vals[3 * S + 3])
            maxtok_v = int(all_vals[3 * S + 4])
            seed_v = all_vals[3 * S + 5]

            speakers = build_speaker_configs(int(n_sp), names_v, audio_v, texts_v)

            def _prog(frac: float, msg: str):
                progress(frac, desc=msg)

            try:
                sr, audio_arr = synthesizer.synthesize_podcast(
                    script=script,
                    speakers=speakers,
                    temperature=temp_v,
                    top_p=tp_v,
                    repetition_penalty=rep_v,
                    chunk_length=chunk_v,
                    max_new_tokens=maxtok_v,
                    seed=int(seed_v) if seed_v else None,
                    progress_fn=_prog,
                )
                return (sr, audio_arr), ""
            except Exception as exc:
                logger.error(f"Podcast generation error: {exc}")
                err = html_lib.escape(str(exc))
                return (
                    None,
                    f'<div style="color:#dc2626;font-weight:600;padding:10px;'
                    f'border:1px solid #fecaca;border-radius:6px;background:#fff5f5;">'
                    f"❌ {err}</div>",
                )

        generate_btn.click(
            fn=_generate,
            inputs=[
                script_box,
                num_speakers,
                *name_inputs,
                *audio_inputs,
                *reftext_inputs,
                temperature,
                top_p,
                repetition_penalty,
                chunk_length,
                max_new_tokens,
                seed,
            ],
            outputs=[audio_out, error_out],
            concurrency_limit=1,
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _badge(text: str, color: str = "#3B82F6") -> str:
    """Return a small inline HTML badge."""
    return (
        f"<span style='color:{color};font-size:12px;font-weight:600;"
        f"white-space:nowrap;'>{text}</span>"
    )
