"""
Module: src.cv.train
SRS Reference: FR-CV-003-008
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: B
Pipeline Stage: CV Inference
Purpose: Train EfficientNet-B4 for binary classification (bonafide vs spoof) and produce checkpoints.
Dependencies: torch, torchvision.
Interface Contract:
  Input:  torch.utils.data.DataLoader of (tensor [3,224,224] float32, label int)
  Output: Path to saved checkpoint (.pth/.pt) for Sprint C inference
Latency Target: <= 3,000 ms per forward proxy stage (training wall time validated in Sprint B)
Open Questions Resolved: Q4/Q5/Q6 resolved (downstream only)
Open Questions Blocking: Q3 - VRAM feasibility affects training viability and checkpoint strategy
MCP Tools Used: context7-mcp | huggingface-mcp | stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.audio.dsp import preprocess_audio
from src.cv.model import DSDBAModel
from src.utils.logger import log_info, log_warning


class AudioClassificationDataset(Dataset[tuple[Tensor, int]]):
  """Dataset that applies DSP preprocessing on each audio file path."""

  def __init__(
    self,
    file_paths: list[Path],
    labels: list[int],
    cfg: dict[str, Any],
    transform: Callable[[Tensor], Tensor] | None = None,
  ) -> None:
    self.file_paths = file_paths
    self.labels = labels
    self.cfg = cfg
    self.transform = transform

  def __len__(self) -> int:
    return len(self.file_paths)

  def __getitem__(self, idx: int) -> tuple[Tensor, int]:
    tensor = preprocess_audio(self.file_paths[idx], self.cfg)
    if self.transform is not None:
      tensor = self.transform(tensor)
    return tensor, int(self.labels[idx])


def _resolve_dataset_paths(root: Path) -> tuple[list[Path], list[int]]:
  """Resolve audio file paths and labels from bonafide/spoof folders."""
  exts = {".wav", ".flac", ".mp3", ".ogg"}
  pairs = [("bonafide", 0), ("spoof", 1)]
  file_paths: list[Path] = []
  labels: list[int] = []
  for class_name, label in pairs:
    class_dir = root / class_name
    files = [p for p in sorted(class_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    file_paths.extend(files)
    labels.extend([label] * len(files))
  return file_paths, labels


def get_class_weights(dataset: Any) -> Tensor:
  """[FR-CV-005] Compute inverse-frequency class weights tensor."""
  labels = getattr(dataset, "labels", None)
  if labels is None:
    raise ValueError("Dataset must expose 'labels' for class weight computation")

  labels_arr = np.asarray(labels, dtype=np.int64)
  if labels_arr.size == 0:
    raise ValueError("Dataset labels are empty")

  counts = np.bincount(labels_arr, minlength=2).astype(np.float32)
  counts[counts == 0.0] = 1.0
  total = float(np.sum(counts))
  weights = total / (2.0 * counts)
  return torch.tensor(weights, dtype=torch.float32)


def build_augmentations(cfg: dict[str, Any]) -> Callable[[Tensor], Tensor]:
  """[FR-CV-006] Build lightweight spectrogram-domain augmentation callable."""
  aug_cfg = cfg["training"]["augmentation"]
  enabled = bool(aug_cfg.get("specaugment_enabled", False))
  time_mask_pct = float(aug_cfg.get("time_mask_pct", 0.0))
  freq_mask_pct = float(aug_cfg.get("freq_mask_pct", 0.0))
  time_shift_sec = float(aug_cfg.get("time_shift_sec", 0.0))
  snr_db = float(aug_cfg.get("gaussian_noise_snr_db", 20.0))
  sample_rate = int(cfg["audio"]["sample_rate"])
  n_samples = int(cfg["audio"]["n_samples"])

  def _augment(x: Tensor) -> Tensor:
    if not enabled:
      return x

    out = x.clone()
    _, h, w = out.shape

    # Frequency masking.
    max_f = max(1, int(h * freq_mask_pct))
    f = int(torch.randint(0, max_f + 1, (1,)).item())
    if f > 0:
      f0 = int(torch.randint(0, max(1, h - f + 1), (1,)).item())
      out[:, f0 : f0 + f, :] = 0.0

    # Time masking.
    max_t = max(1, int(w * time_mask_pct))
    t = int(torch.randint(0, max_t + 1, (1,)).item())
    if t > 0:
      t0 = int(torch.randint(0, max(1, w - t + 1), (1,)).item())
      out[:, :, t0 : t0 + t] = 0.0

    # Approximate time shift using width roll.
    max_shift_samples = int(time_shift_sec * sample_rate)
    max_shift_frames = int((max_shift_samples / max(n_samples, 1)) * w)
    if max_shift_frames > 0:
      shift = int(torch.randint(-max_shift_frames, max_shift_frames + 1, (1,)).item())
      out = torch.roll(out, shifts=shift, dims=-1)

    # Additive Gaussian noise based on SNR target.
    signal_power = torch.mean(out**2)
    noise_power = signal_power / max(1e-6, 10 ** (snr_db / 10.0))
    noise = torch.randn_like(out) * torch.sqrt(noise_power)
    out = torch.clamp(out + noise, 0.0, 1.0)
    return out

  return _augment


def compute_eer(y_true: list[int] | np.ndarray, y_scores: list[float] | np.ndarray) -> float:
  """[FR-CV-008] Compute Equal Error Rate from labels and spoof scores."""
  y_true_np = np.asarray(y_true)
  y_scores_np = np.asarray(y_scores)
  if len(np.unique(y_true_np)) < 2:
    return 1.0

  fpr, tpr, _ = roc_curve(y_true_np, y_scores_np)
  fnr = 1.0 - tpr
  idx = int(np.nanargmin(np.abs(fnr - fpr)))
  return float((fnr[idx] + fpr[idx]) / 2.0)


def train_epoch(
  model: DSDBAModel,
  loader: DataLoader[tuple[Tensor, int]],
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  cfg: dict[str, Any],
) -> dict[str, float]:
  """Train one epoch and return aggregated metrics."""
  del cfg  # Reserved for future scheduler/AMP config expansion.
  device = next(model.parameters()).device
  model.train()

  total_loss = 0.0
  correct = 0
  total = 0

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    total_loss += float(loss.item()) * x.size(0)
    pred = torch.argmax(logits, dim=1)
    correct += int((pred == y).sum().item())
    total += int(x.size(0))

  avg_loss = total_loss / max(total, 1)
  acc = correct / max(total, 1)
  return {"train_loss": float(avg_loss), "train_acc": float(acc)}


@torch.no_grad()
def validate_epoch(
  model: DSDBAModel,
  loader: DataLoader[tuple[Tensor, int]],
  cfg: dict[str, Any],
) -> dict[str, float]:
  """Validate one epoch and return EER and AUC-ROC."""
  del cfg
  device = next(model.parameters()).device
  model.eval()

  y_true: list[int] = []
  y_scores: list[float] = []

  for x, y in loader:
    x = x.to(device)
    logits = model(x)
    spoof_scores = torch.sigmoid(logits[:, 1]).detach().cpu().numpy()
    y_scores.extend(spoof_scores.tolist())
    y_true.extend(y.numpy().tolist())

  eer = compute_eer(y_true, y_scores)
  try:
    auc = float(roc_auc_score(y_true, y_scores))
  except Exception:
    auc = 0.5

  return {"eer": float(eer), "auc_roc": float(auc)}


def _save_checkpoint(path: Path, model: DSDBAModel, epoch: int, metrics: dict[str, float]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "metrics": metrics,
    },
    path,
  )


def _upload_checkpoint_to_hf(path: Path, cfg: dict[str, Any]) -> None:
  repo_id = str(cfg["training"].get("hf_model_repo", "")).strip()
  if not repo_id:
    log_warning(stage="cv_train", message="hf_upload_skipped_empty_repo", data={})
    return

  token_env = str(cfg["nlp"].get("hf_token_env_var", "HF_TOKEN"))
  token = __import__("os").environ.get(token_env)
  if not token:
    log_warning(stage="cv_train", message="hf_upload_skipped_no_token", data={"env": token_env})
    return

  api = HfApi(token=token)
  api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
  api.upload_file(
    path_or_fileobj=str(path),
    path_in_repo=path.name,
    repo_id=repo_id,
    repo_type="model",
  )
  log_info(stage="cv_train", message="hf_upload_ok", data={"repo_id": repo_id, "file": path.name})


def run_training(cfg: dict[str, Any]) -> DSDBAModel:
  """[FR-CV-003..008] Run two-phase training and return trained model."""
  root = Path(__file__).resolve().parents[2]
  train_root = root / "data" / "train"
  val_root = root / "data" / "validation"

  train_files, train_labels = _resolve_dataset_paths(train_root)
  val_files, val_labels = _resolve_dataset_paths(val_root)
  if not train_files or not val_files:
    raise ValueError("Training/validation data not found. Expected data/{train,validation}/{bonafide,spoof}")

  train_dataset = AudioClassificationDataset(
    train_files,
    train_labels,
    cfg,
    transform=build_augmentations(cfg),
  )
  val_dataset = AudioClassificationDataset(val_files, val_labels, cfg, transform=None)

  batch_size = int(cfg["training"]["batch_size"])
  num_workers = int(cfg["training"].get("num_workers", 0))
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = DSDBAModel(cfg=cfg, pretrained=True).to(device)
  model.freeze_backbone()

  class_weights = get_class_weights(train_dataset).to(device)
  criterion = nn.CrossEntropyLoss(weight=class_weights)

  frozen_epochs = int(cfg["model"].get("frozen_epochs", 5))
  max_epochs = int(cfg["training"].get("max_epochs", 30))
  finetune_lr = float(cfg["model"].get("finetune_lr", 1e-4))
  phase1_lr = min(1e-3, finetune_lr)

  best_auc = -math.inf
  best_path = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "best_model.pth"
  patience = int(cfg["training"].get("early_stopping_patience", 5))
  no_improve = 0

  # Phase 1: head-only training.
  optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=phase1_lr)
  for epoch in range(1, min(frozen_epochs, max_epochs) + 1):
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, cfg)
    val_metrics = validate_epoch(model, val_loader, cfg)
    merged = {**train_metrics, **val_metrics}
    _save_checkpoint(best_path.with_name(f"epoch_{epoch:02d}.pth"), model, epoch, merged)
    if val_metrics["auc_roc"] > best_auc:
      best_auc = val_metrics["auc_roc"]
      _save_checkpoint(best_path, model, epoch, merged)
      no_improve = 0
    else:
      no_improve += 1

    log_info(stage="cv_train", message="phase1_epoch_complete", data={"epoch": epoch, **merged})
    if no_improve >= patience:
      break

  # Phase 2: fine-tuning top layers.
  if max_epochs > frozen_epochs:
    model.unfreeze_top_n(n=2)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=min(finetune_lr, 1e-4))
    for epoch in range(frozen_epochs + 1, max_epochs + 1):
      train_metrics = train_epoch(model, train_loader, optimizer, criterion, cfg)
      val_metrics = validate_epoch(model, val_loader, cfg)
      merged = {**train_metrics, **val_metrics}
      _save_checkpoint(best_path.with_name(f"epoch_{epoch:02d}.pth"), model, epoch, merged)
      if val_metrics["auc_roc"] > best_auc:
        best_auc = val_metrics["auc_roc"]
        _save_checkpoint(best_path, model, epoch, merged)
        no_improve = 0
      else:
        no_improve += 1

      log_info(stage="cv_train", message="phase2_epoch_complete", data={"epoch": epoch, **merged})
      if no_improve >= patience:
        break

  _upload_checkpoint_to_hf(best_path, cfg)
  return model


if __name__ == "__main__":
  cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
  cfg_data = yaml.safe_load(cfg_path.read_text())
  run_training(cfg_data)