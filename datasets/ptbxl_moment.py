"""PTB-XL dataset wrapper for MOMENT-enhanced MedTsLLM.

The standard MedTsLLM input is a 512-step resampling of the complete ECG.
Additionally, this wrapper returns two overlapping 512-sample windows from the
original 100 Hz, 10-second recording so MOMENT can retain higher-resolution
morphology as global context.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ptbxl import PTBXLClassificationDataset


class PTBXLMomentClassificationDataset(PTBXLClassificationDataset):
    supported_tasks = ["classification"]

    def _moment_config(self):
        return self.config.models.medtsllm.moment

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Conservative ECG augmentations applied only to the training split."""
        cfg = self._moment_config().get("augmentation")
        if self.split != "train" or cfg is None or not cfg.get("enabled", False):
            return x

        x = x.clone()

        amplitude_scale = float(cfg.get("amplitude_scale", 0.0))
        if amplitude_scale > 0:
            scale = 1.0 + (2.0 * torch.rand(1, x.size(1)) - 1.0) * amplitude_scale
            x = x * scale

        noise_std = float(cfg.get("noise_std", 0.0))
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        lead_dropout_prob = float(cfg.get("lead_dropout_prob", 0.0))
        if lead_dropout_prob > 0:
            drop = torch.rand(x.size(1)) < lead_dropout_prob
            if drop.all():
                drop[torch.randint(0, x.size(1), (1,)).item()] = False
            x[:, drop] = 0.0

        temporal_mask_ratio = float(cfg.get("temporal_mask_ratio", 0.0))
        if temporal_mask_ratio > 0:
            mask_len = max(1, min(x.size(0), int(round(x.size(0) * temporal_mask_ratio))))
            start = torch.randint(0, x.size(0) - mask_len + 1, (1,)).item()
            x[start : start + mask_len] = 0.0

        return x

    @staticmethod
    def _resize(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(0) == target_len:
            return x
        resized = F.interpolate(
            x.transpose(0, 1).unsqueeze(0),
            size=target_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0).transpose(0, 1)

    def _make_moment_windows(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self._moment_config()
        window_len = int(cfg.get("sequence_length", 512))
        n_windows = int(cfg.get("n_highres_windows", 2))

        if n_windows < 1:
            raise ValueError("models.medtsllm.moment.n_highres_windows must be >= 1")

        if x.size(0) <= window_len:
            window = self._resize(x, window_len)
            return window.unsqueeze(0).repeat(n_windows, 1, 1)

        max_start = x.size(0) - window_len
        starts = torch.linspace(0, max_start, steps=n_windows).round().long()
        windows = [x[start : start + window_len] for start in starts.tolist()]
        return torch.stack(windows, dim=0)  # [W, T, C]

    def __getitem__(self, idx):
        full_record = self._augment(self.records[idx])
        x_enc = self.resample_to_history(full_record)
        label = self.record_labels[idx]

        out = {
            "x_enc": x_enc,
            "x_moment_windows": self._make_moment_windows(full_record),
            "labels": label,
        }
        if self.record_descriptions is not None:
            out["descriptions"] = self.record_descriptions[idx]
        return out


ptbxl_moment_datasets = {
    "classification": PTBXLMomentClassificationDataset,
}
