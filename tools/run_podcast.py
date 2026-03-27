"""
Local launcher for Fish Speech Podcast Studio.

Usage
-----
    python tools/run_podcast.py \\
        --llama-checkpoint-path  checkpoints/s2-pro \\
        --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \\
        [--whisper-model-dir checkpoints/whisper-small-pt] \\
        [--device cuda] [--half] [--compile] \\
        [--port 7862] [--share]

The WebUI opens at http://localhost:<port> (default 7862) so it doesn't
conflict with the standard WebUI on port 7860.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Suppress einx traceback noise
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from podcast.app import build_podcast_app
from podcast.engine import PodcastSynthesizer


def parse_args():
    p = ArgumentParser(description="Fish Speech — Podcast Studio")
    p.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro",
        help="Directory containing the S2-Pro LLaMA weights.",
    )
    p.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro/codec.pth",
        help="Path to the DAC codec checkpoint.",
    )
    p.add_argument(
        "--decoder-config-name",
        type=str,
        default="modded_dac_vq",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: cuda / mps / cpu (auto-detected if omitted).",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 instead of BF16.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for faster decoding (first generation is slow).",
    )
    p.add_argument(
        "--whisper-model-dir",
        type=Path,
        default=None,
        help="Path to a local Whisper model for reference-audio transcription.",
    )
    p.add_argument("--port", type=int, default=7862)
    p.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precision = torch.half if args.half else torch.bfloat16

    # ── Auto-detect device ────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("Running on Apple MPS.")
    elif torch.xpu.is_available():
        args.device = "xpu"
        logger.info("Running on Intel XPU.")
    elif not torch.cuda.is_available():
        args.device = "cpu"
        logger.info("CUDA not available — running on CPU (slow).")
    else:
        logger.info(f"Running on {torch.cuda.get_device_name(0)}.")

    # ── Load models ───────────────────────────────────────────────────────
    logger.info("Loading LLaMA model…")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=precision,
        compile=args.compile,
    )

    logger.info("Loading DAC decoder model…")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    # ── Warm-up pass ──────────────────────────────────────────────────────
    logger.info("Warming up inference engine…")
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=args.compile,
    )
    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="<|speaker:0|>Hello.",
                references=[],
                max_new_tokens=64,
                chunk_length=200,
                top_p=0.8,
                repetition_penalty=1.1,
                temperature=0.8,
            )
        )
    )
    logger.info("Warm-up done.")

    # ── Create PodcastSynthesizer ─────────────────────────────────────────
    synthesizer = PodcastSynthesizer(
        tts_engine=inference_engine,
        whisper_model_dir=(
            str(args.whisper_model_dir) if args.whisper_model_dir else None
        ),
    )

    # ── Build & launch WebUI ──────────────────────────────────────────────
    logger.info("Launching Podcast Studio WebUI…")
    app = build_podcast_app(
        synthesizer=synthesizer,
        whisper_model_dir=(
            str(args.whisper_model_dir) if args.whisper_model_dir else None
        ),
    )
    app.launch(server_port=args.port, share=args.share)
