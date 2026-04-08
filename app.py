"""
Module: app
SRS Reference: FR-DEP-001-010
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import torch
import yaml

from src.audio.dsp import preprocess_audio
from src.cv.gradcam import run_gradcam
from src.cv.infer import export_to_onnx, load_onnx_session, run_onnx_inference
from src.cv.model import DSDBAModel
from src.nlp.explain import generate_explanation
from src.utils.errors import DSDBAError
from src.utils.logger import log_error, log_info, log_warning


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _models_dir() -> Path:
    return _project_root() / "models" / "checkpoints"


def _ensure_onnx_session(cfg, model):
    from huggingface_hub import hf_hub_download
    try:
        onnx_path = hf_hub_download(repo_id="narcissablack/fake67", filename="dsdba_efficientnet_b4.onnx")
        log_info(stage="deployment", message="onnx_loaded_from_hub", data={"repo": "narcissablack/fake67"})
    except Exception:
        local_path = _models_dir() / "dsdba_efficientnet_b4.onnx"
        if local_path.exists():
            onnx_path = str(local_path)
        else:
            onnx_path = export_to_onnx(model=model, cfg=cfg)
    return load_onnx_session(onnx_path=onnx_path, cfg=cfg)


def _maybe_load_weights(model, cfg):
    from huggingface_hub import hf_hub_download
    ckpt_name = str(cfg.get("training", {}).get("best_checkpoint_filename", "best_model.pth"))
    try:
        ckpt_path = hf_hub_download(repo_id="narcissablack/fake67", filename=ckpt_name)
        log_info(stage="deployment", message="checkpoint_loaded_from_hub", data={"repo": "narcissablack/fake67", "filename": ckpt_name})
    except Exception:
        local_path = _models_dir() / ckpt_name
        if local_path.exists():
            ckpt_path = str(local_path)
        else:
            return
    try:
        payload = torch.load(str(ckpt_path), map_location="cpu")
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state, strict=False)
        log_info(stage="deployment", message="checkpoint_loaded", data={"path": str(ckpt_path)})
    except Exception as exc:
        log_warning(stage="deployment", message="checkpoint_load_failed", data={"reason": str(exc)})


def _band_df(band_pct):
    order = ["low", "low_mid", "high_mid", "high"]
    bands = [b for b in order if b in band_pct]
    perc = [float(band_pct[b]) for b in bands]
    return pd.DataFrame({"band": bands, "percent": perc})


def _confidence_percent(conf):
    return float(conf) * 100.0


def _verdict_html(label, confidence):
    pct = max(0.0, min(100.0, _confidence_percent(confidence)))
    color = "#ef4444" if str(label).lower() == "spoof" else "#22c55e"
    return (
        "<div style='width: 100%; background: #e5e7eb; border-radius: 8px; overflow: hidden;'>"
        f"<div style='width: {pct:.2f}%; background: {color}; padding: 6px 0; color: white; text-align: center;'>"
        f"{pct:.2f}%</div></div>"
    )


def _spectrogram_image_from_tensor(tensor):
    import matplotlib.pyplot as plt
    x = tensor.detach().cpu()
    img = x[0].numpy().astype(np.float32, copy=False)
    out = Path(tempfile.gettempdir()) / f"dsdba_spec_{int(time.time() * 1000)}.png"
    plt.figure(figsize=(5, 4), dpi=120)
    plt.imshow(img, aspect="auto", origin="lower", cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close()
    return out


def ensure_demo_samples(cfg):
    import soundfile as sf
    root = _project_root() / "data" / "samples"
    root.mkdir(parents=True, exist_ok=True)
    sr = int(cfg["audio"]["sample_rate"])
    n = int(cfg["audio"]["n_samples"])
    t = np.linspace(0.0, float(cfg["audio"]["duration_sec"]), num=n, endpoint=False, dtype=np.float32)
    samples = [
        ("bonafide_01.wav", 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)),
        ("bonafide_02.wav", 0.1 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)),
    ]
    rng = np.random.default_rng(0)
    noise = (0.03 * rng.standard_normal(size=n)).astype(np.float32)
    samples.append(("spoof_01.wav", np.clip(noise + 0.06 * np.sin(2.0 * np.pi * 3200.0 * t).astype(np.float32), -1.0, 1.0)))
    samples.append(("spoof_02.wav", np.clip(noise + 0.06 * np.sin(2.0 * np.pi * 5200.0 * t).astype(np.float32), -1.0, 1.0)))
    paths = []
    for name, wav in samples:
        p = root / name
        if not p.exists():
            sf.write(str(p), wav, sr, subtype="PCM_16")
        paths.append(p)
    return paths


# ── Startup ────────────────────────────────────────────────────────────────────
CFG = load_config("config.yaml")
MODEL = DSDBAModel(cfg=CFG, pretrained=False)
_maybe_load_weights(model=MODEL, cfg=CFG)
MODEL.eval()
ONNX_SESSION = _ensure_onnx_session(cfg=CFG, model=MODEL)


def _input_path(audio_path: Any) -> str | None:
    """Normalize Gradio File / Audio payloads (str, list, dict, pathlib) to a path string."""
    if audio_path is None:
        return None
    if isinstance(audio_path, (list, tuple)):
        if not audio_path:
            return None
        audio_path = audio_path[0]
    if isinstance(audio_path, dict):
        audio_path = audio_path.get("path") or audio_path.get("name")
    if audio_path is None:
        return None
    if isinstance(audio_path, Path):
        return str(audio_path)
    path_attr = getattr(audio_path, "path", None)
    if isinstance(path_attr, str) and path_attr:
        return path_attr
    s = str(audio_path)
    return s if s else None


def ui_run(audio_path: str | list | tuple | None):
    """Sync function — Gradio 4.x compatible."""
    audio_path = _input_path(audio_path)
    empty_bands = pd.DataFrame(columns=["band", "percent"])
    if not audio_path:
        return "", 0.0, "", None, None, None, empty_bands, "Please upload a WAV/FLAC file."

    start = time.perf_counter()
    try:
        ap = Path(audio_path)
        max_mb = float(CFG["deployment"]["max_upload_mb"])
        if ap.stat().st_size > int(max_mb * 1024 * 1024):
            return "", 0.0, "", None, None, None, empty_bands, "File too large (> 20 MB)."

        tensor = preprocess_audio(file_path=ap, cfg=CFG)
        label, confidence = run_onnx_inference(session=ONNX_SESSION, tensor=tensor, cfg=CFG)
        heatmap_path, band_pct = run_gradcam(tensor=tensor, model=MODEL, cfg=CFG)
        spec_path = _spectrogram_image_from_tensor(tensor)

        # NLP explanation — sync dengan asyncio.run
        try:
            explanation_text, _ = asyncio.run(
                generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=CFG)
            )
        except Exception:
            explanation_text = "Explanation unavailable."

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log_info(stage="deployment", message="ui_run_complete", data={"latency_ms": round(elapsed_ms, 3)})

        return (
            label,
            _confidence_percent(confidence),
            _verdict_html(label=label, confidence=confidence),
            str(audio_path),
            str(spec_path),
            str(heatmap_path),
            _band_df(band_pct),
            explanation_text,
        )

    except DSDBAError as exc:
        msg = {"AUD-001": "Audio too short (< 0.5 s).", "AUD-002": "Unsupported format."}.get(exc.code, "Audio processing failed.")
        return "", 0.0, "", None, None, None, empty_bands, msg
    except Exception as exc:
        log_error(stage="deployment", message="ui_exception", data={"reason": str(exc)})
        return "", 0.0, "", None, None, None, empty_bands, f"Error: {str(exc)}"


def build_demo():
    demo_samples = ensure_demo_samples(CFG)

    with gr.Blocks(title="DSDBA — Deepfake Speech Detection") as demo:
        gr.Markdown("## DSDBA — Deepfake Speech Detection & Explainability (Gradio 4.x)")
        gr.Markdown("Upload a WAV/FLAC file (≤ 20 MB). CV results appear first; explanation loads asynchronously.")

        with gr.Row():
            with gr.Column(scale=1):
                # gr.File is more reliable than gr.Audio for uploads on Hugging Face Spaces (WebSocket / client quirks).
                audio_in = gr.File(
                    label="Upload audio (WAV/FLAC)",
                    file_count="single",
                    type="filepath",
                    file_types=[".wav", ".flac"],
                )
                run_btn = gr.Button("Run", variant="primary")
                gr.Markdown("**Demo examples:**")
                btn0 = gr.Button(demo_samples[0].name)
                btn1 = gr.Button(demo_samples[1].name)
                btn2 = gr.Button(demo_samples[2].name)
                btn3 = gr.Button(demo_samples[3].name)

            with gr.Column(scale=2):
                with gr.Row():
                    verdict = gr.Label(label="Verdict")
                    confidence_pct = gr.Number(label="Confidence (%)", precision=2)
                conf_bar = gr.HTML(label="Confidence bar")
                waveform = gr.Audio(label="Waveform", type="filepath")
                spec_img = gr.Image(label="Spectrogram (proxy)", type="filepath")
                gradcam_img = gr.Image(label="Grad-CAM overlay", type="filepath")
                # BarPlot pulls Altair/Vega in the browser; on some Spaces/iframes it breaks the whole UI (no upload, no Run).
                band_table = gr.Dataframe(label="Band attribution (%)", headers=["band", "percent"], interactive=False)
                gr.Markdown("**AI-generated explanation (English)**")
                explanation = gr.Textbox(label="Explanation", lines=6)

        outputs = [verdict, confidence_pct, conf_bar, waveform, spec_img, gradcam_img, band_table, explanation]

        run_btn.click(fn=ui_run, inputs=[audio_in], outputs=outputs)

        # One click: load sample + run (works even if manual upload is flaky in the browser).
        for i, b in enumerate((btn0, btn1, btn2, btn3)):
            path = str(demo_samples[i])
            b.click(fn=lambda p=path: p, inputs=[], outputs=[audio_in]).then(
                fn=ui_run, inputs=[audio_in], outputs=outputs
            )

        with gr.Accordion("About", open=False):
            gr.Markdown("\n".join([
                "### About",
                "- **Pipeline**: Audio DSP → EfficientNet-B4 (ONNX) → Grad-CAM → LLM explanation.",
                "- **Dataset**: Abdel-Dayem, M. (2023). Fake-or-Real (FoR) Dataset. Kaggle.",
                "- **Team**: Ferel, Safa — ITS Informatics | KCVanguard ML Workshop.",
            ]))

    demo.queue()
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )