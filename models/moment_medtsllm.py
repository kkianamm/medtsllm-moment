"""MOMENT-enhanced MedTsLLM for sequence-level time-series classification.

This module keeps the original MedTsLLM reprogramming path and adds a
pretrained MOMENT encoder. MOMENT patch tokens are aggregated across leads,
optionally enriched with two full-resolution ECG windows, projected into the
LLM hidden space, and fused with MedTsLLM tokens through a learned residual
 gate.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from momentfm import MOMENTPipeline
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "momentfm is required for MomentMedTsLLM. Install it with "
        "`pip install -r requirements-moment.txt`."
    ) from exc

from .medtsllm import MedTsLLM


class LeadAttention(nn.Module):
    """Aggregate channel/lead-specific MOMENT tokens at every patch."""

    def __init__(self, d_model: int, n_leads: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.lead_embedding = nn.Parameter(torch.empty(1, n_leads, 1, d_model))
        nn.init.normal_(self.lead_embedding, mean=0.0, std=0.02)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, leads, patches, d_model]

        Returns:
            tokens: [batch, patches, d_model]
            weights: [batch, leads, patches]
        """
        if x.ndim != 4:
            raise ValueError(f"LeadAttention expects [B,C,N,D], received {tuple(x.shape)}")
        if x.size(1) > self.n_leads:
            raise ValueError(
                f"Input has {x.size(1)} leads, but n_leads={self.n_leads}. "
                "Increase models.medtsllm.moment.max_leads."
            )

        x = x + self.lead_embedding[:, : x.size(1)]
        weights = torch.softmax(self.score(x), dim=1)
        tokens = (weights * x).sum(dim=1)
        return tokens, weights.squeeze(-1)


class AttentionPool(nn.Module):
    """Learned attention pooling over a token dimension."""

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(d_model // 2, 32)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"AttentionPool expects [B,N,D], received {tuple(x.shape)}")
        weights = torch.softmax(self.score(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class WindowAttention(nn.Module):
    """Aggregate global summaries from multiple full-resolution ECG windows."""

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(d_model // 2, 32)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"WindowAttention expects [B,W,D], received {tuple(x.shape)}")
        weights = torch.softmax(self.score(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class ContextInjection(nn.Module):
    """Inject one global MOMENT vector into every aligned MOMENT patch token."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = context.unsqueeze(1).expand(-1, tokens.size(1), -1)
        gate = torch.sigmoid(self.gate(torch.cat([tokens, context], dim=-1)))
        return self.norm(tokens + gate * context)


class GatedTokenFusion(nn.Module):
    """Project MOMENT tokens and fuse them residually with MedTsLLM tokens."""

    def __init__(self, d_moment: int, d_llm: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.moment_projection = nn.Sequential(
            nn.Linear(d_moment, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_llm),
            nn.LayerNorm(d_llm),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_llm, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_llm),
        )
        self.norm = nn.LayerNorm(d_llm)

    def project_moment(self, moment_tokens: torch.Tensor) -> torch.Tensor:
        return self.moment_projection(moment_tokens)

    def forward(
        self,
        med_tokens: torch.Tensor,
        moment_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        moment_projected = self.project_moment(moment_tokens)
        gate = torch.sigmoid(self.gate(torch.cat([med_tokens, moment_projected], dim=-1)))
        fused = self.norm(med_tokens + gate * moment_projected)
        return fused, moment_projected


class MomentMedTsLLM(MedTsLLM):
    """MOMENT-conditioned MedTsLLM classifier.

    Expected inputs:
        x_enc: [B, history_len, C]
        labels: [B]
        x_moment_windows (optional): [B, W, moment_seq_len, C]
        descriptions (optional): patient metadata strings
    """

    supported_tasks = ["classification"]
    supported_modes = ["multivariate"]

    def __init__(self, config, dataset) -> None:
        if config.task != "classification":
            raise ValueError("MomentMedTsLLM currently supports only classification.")

        original_task = config.task
        original_pred_len = config.pred_len

        # The upstream MedTsLLM class currently rejects classification in its
        # constructor. Semantic segmentation has the same K-way output setup,
        # so initialize through that path and replace its output head below.
        config.task = "semantic_segmentation"
        config.pred_len = 1
        try:
            super().__init__(config, dataset)
        finally:
            config.task = original_task
            config.pred_len = original_pred_len

        self.task = "classification"
        self.pred_len = 1
        self.n_classes = int(dataset.n_classes)
        self.n_outputs_per_step = self.n_classes
        self.n_outputs = self.n_classes

        if self.covariate_mode not in {"concat", "add", "weighted-average"}:
            raise ValueError(
                "MomentMedTsLLM supports covariate_mode in "
                "{'concat', 'add', 'weighted-average'}. Use 'concat' for PTB-XL."
            )

        # Remove the original fixed-size segmentation head. Classification uses
        # token attention pooling and a compact linear head instead.
        self.output_projection = nn.Identity()
        if hasattr(self, "embedding_downsample_layer"):
            self.embedding_downsample_layer = nn.Identity()

        self.moment_cfg = self.model_config.moment
        self.use_highres_windows = bool(self.moment_cfg.get("use_highres_windows", True))
        self.save_frozen_backbone = bool(self.moment_cfg.get("save_frozen_backbone", False))

        model_kwargs = {
            "task_name": "embedding",
            "freeze_encoder": bool(self.moment_cfg.get("freeze_encoder", True)),
            "freeze_embedder": bool(self.moment_cfg.get("freeze_embedder", True)),
            "enable_gradient_checkpointing": bool(
                self.moment_cfg.get("enable_gradient_checkpointing", True)
            ),
        }
        self.moment = MOMENTPipeline.from_pretrained(
            self.moment_cfg.model_id,
            model_kwargs=model_kwargs,
        )
        self.moment.init()

        self.moment_seq_len = int(self.moment.config.seq_len)
        self.moment_dim = int(self.moment.config.d_model)
        self.moment_patch_len = int(self.moment.config.patch_len)

        max_leads = int(self.moment_cfg.get("max_leads", dataset.n_features))
        lead_hidden = int(self.moment_cfg.get("lead_attention_hidden", 128))
        fusion_hidden = int(self.moment_cfg.get("fusion_hidden", 256))
        head_dropout = float(self.moment_cfg.get("head_dropout", 0.2))

        self.lead_attention = LeadAttention(
            d_model=self.moment_dim,
            n_leads=max_leads,
            hidden_dim=lead_hidden,
        )
        self.window_patch_pool = AttentionPool(self.moment_dim, lead_hidden)
        self.window_attention = WindowAttention(self.moment_dim, lead_hidden)
        self.context_injection = ContextInjection(self.moment_dim)
        self.fusion = GatedTokenFusion(
            d_moment=self.moment_dim,
            d_llm=self.d_llm,
            hidden_dim=fusion_hidden,
            dropout=self.dropout,
        )

        llm_output_dim = self.d_llm if self.llm_enabled else self.d_ff
        self.ecg_pool = AttentionPool(llm_output_dim)
        self.med_aux_pool = AttentionPool(self.d_llm)
        self.moment_aux_pool = AttentionPool(self.d_llm)

        self.classifier = nn.Sequential(
            nn.LayerNorm(llm_output_dim),
            nn.Dropout(head_dropout),
            nn.Linear(llm_output_dim, self.n_classes),
        )
        self.med_aux_classifier = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Dropout(head_dropout),
            nn.Linear(self.d_llm, self.n_classes),
        )
        self.moment_aux_classifier = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Dropout(head_dropout),
            nn.Linear(self.d_llm, self.n_classes),
        )

        self._configure_moment_trainability()
        self._print_new_parameter_summary()

    def _configure_moment_trainability(self) -> None:
        freeze_encoder = bool(self.moment_cfg.get("freeze_encoder", True))
        freeze_embedder = bool(self.moment_cfg.get("freeze_embedder", True))
        unfreeze_last_n = int(self.moment_cfg.get("unfreeze_last_n", 0))

        for parameter in self.moment.encoder.parameters():
            parameter.requires_grad = not freeze_encoder
        for parameter in self.moment.patch_embedding.parameters():
            parameter.requires_grad = not freeze_embedder

        if unfreeze_last_n > 0:
            for parameter in self.moment.encoder.parameters():
                parameter.requires_grad = False
            blocks = self._get_transformer_blocks(self.moment.encoder)
            if not blocks:
                raise RuntimeError(
                    "Could not locate MOMENT encoder blocks for partial unfreezing. "
                    "Set unfreeze_last_n=0 or freeze_encoder=false."
                )
            for block in blocks[-unfreeze_last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        # Embedding mode uses an identity head, but make the policy explicit.
        for parameter in self.moment.head.parameters():
            parameter.requires_grad = False

        self.moment_has_trainable_parameters = any(
            p.requires_grad for p in self.moment.parameters()
        )

    @staticmethod
    def _get_transformer_blocks(module: nn.Module) -> Iterable[nn.Module]:
        candidates = (
            ("block",),
            ("layers",),
            ("encoder", "block"),
            ("encoder", "layers"),
        )
        for path in candidates:
            current = module
            valid = True
            for name in path:
                if not hasattr(current, name):
                    valid = False
                    break
                current = getattr(current, name)
            if valid and isinstance(current, (nn.ModuleList, list, tuple)):
                return list(current)
        return []

    def _print_new_parameter_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        moment_total = sum(p.numel() for p in self.moment.parameters())
        moment_trainable = sum(p.numel() for p in self.moment.parameters() if p.requires_grad)
        print("MomentMedTsLLM parameter summary:")
        print(f"  total parameters: {total:,}")
        print(f"  trainable parameters: {trainable:,}")
        print(f"  MOMENT parameters: {moment_total:,}")
        print(f"  trainable MOMENT parameters: {moment_trainable:,}")

    @staticmethod
    def _resize_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resize [B,T,C] to target_len while preserving channels."""
        if x.size(1) == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    @staticmethod
    def _align_token_count(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """Align token count by cropping one padded token or interpolating."""
        if x.size(1) == target_tokens:
            return x
        if x.size(1) == target_tokens + 1:
            return x[:, :target_tokens]
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_tokens, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def _encode_prompts(self, inputs: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        x_enc = inputs["x_enc"]
        batch_size = x_enc.size(0)

        if not self.llm_enabled:
            return torch.zeros(
                batch_size,
                0,
                self.d_llm,
                device=x_enc.device,
                dtype=dtype,
            )

        prompts = self.build_prompt(inputs)
        if not prompts or not prompts[0]:
            return torch.zeros(
                batch_size,
                0,
                self.d_llm,
                device=x_enc.device,
                dtype=dtype,
            )

        encoded = [[self.encode_part(part) for part in prompt] for prompt in prompts]
        encoded = [torch.cat(parts, dim=1) for parts in encoded]
        max_len = max(item.size(1) for item in encoded)
        encoded = [self.pad_sequence(item, max_len) for item in encoded]
        encoded = torch.cat(encoded, dim=0)
        return encoded.to(device=x_enc.device, dtype=dtype)

    def _encode_moment(
        self,
        x_enc: torch.Tensor,
        x_moment_windows: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return lead-aggregated aligned MOMENT tokens [B,N,D]."""
        batch_size, _, n_leads = x_enc.shape
        aligned = self._resize_time(x_enc, self.moment_seq_len)
        aligned = aligned.transpose(1, 2).contiguous()  # [B,C,T]

        window_count = 0
        model_input = aligned
        if self.use_highres_windows and x_moment_windows is not None:
            if x_moment_windows.ndim != 4:
                raise ValueError(
                    "x_moment_windows must have shape [B,W,T,C], received "
                    f"{tuple(x_moment_windows.shape)}"
                )
            if x_moment_windows.size(0) != batch_size or x_moment_windows.size(-1) != n_leads:
                raise ValueError("x_moment_windows batch/lead dimensions do not match x_enc.")

            window_count = x_moment_windows.size(1)
            windows = x_moment_windows.reshape(
                batch_size * window_count,
                x_moment_windows.size(2),
                n_leads,
            )
            windows = self._resize_time(windows, self.moment_seq_len)
            windows = windows.transpose(1, 2).contiguous()  # [B*W,C,T]
            model_input = torch.cat([aligned, windows], dim=0)

        input_mask = torch.ones(
            model_input.size(0),
            model_input.size(-1),
            device=model_input.device,
            dtype=model_input.dtype,
        )
        moment_output = self.moment(
            x_enc=model_input,
            input_mask=input_mask,
            reduction="none",
        )
        raw_tokens = moment_output.embeddings
        if raw_tokens.ndim != 4:
            raise RuntimeError(
                "MOMENT embedding mode with reduction='none' must return [B,C,N,D], "
                f"received {tuple(raw_tokens.shape)}"
            )

        aligned_raw = raw_tokens[:batch_size]
        aligned_tokens, aligned_lead_weights = self.lead_attention(aligned_raw)

        diagnostics: Dict[str, torch.Tensor] = {
            "lead_attention": aligned_lead_weights,
        }

        if window_count > 0:
            window_raw = raw_tokens[batch_size:].reshape(
                batch_size * window_count,
                n_leads,
                raw_tokens.size(2),
                raw_tokens.size(3),
            )
            window_tokens, _ = self.lead_attention(window_raw)
            window_summary, _ = self.window_patch_pool(window_tokens)
            window_summary = window_summary.reshape(batch_size, window_count, -1)
            global_context, window_weights = self.window_attention(window_summary)
            aligned_tokens = self.context_injection(aligned_tokens, global_context)
            diagnostics["window_attention"] = window_weights

        return aligned_tokens, diagnostics

    def _run_llm(self, prompt_tokens: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        if not self.llm_enabled:
            return self.llm_replacement(fused_tokens)

        if self.llm.config.is_encoder_decoder:
            output = self.llm(
                inputs_embeds=prompt_tokens,
                decoder_inputs_embeds=fused_tokens,
            ).last_hidden_state
            return output[:, -fused_tokens.size(1) :].to(fused_tokens.dtype)

        llm_input = torch.cat([prompt_tokens, fused_tokens], dim=1)
        output = self.llm(inputs_embeds=llm_input).last_hidden_state
        return output[:, -fused_tokens.size(1) :].to(fused_tokens.dtype)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_enc = inputs["x_enc"]
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)
        if x_enc.ndim != 3:
            raise ValueError(f"x_enc must have shape [B,T,C], received {tuple(x_enc.shape)}")
        if x_enc.size(-1) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} input features, received {x_enc.size(-1)}."
            )

        if self.device is None:
            self.device = x_enc.device

        med_tokens = self.encode_ts(x_enc)
        moment_tokens, diagnostics = self._encode_moment(
            x_enc,
            inputs.get("x_moment_windows"),
        )

        med_tokens = self._align_token_count(med_tokens, moment_tokens.size(1))
        fused_tokens, moment_projected = self.fusion(med_tokens, moment_tokens)

        prompt_tokens = self._encode_prompts(inputs, dtype=fused_tokens.dtype)
        ecg_tokens = self._run_llm(prompt_tokens, fused_tokens)

        ecg_repr, ecg_attention = self.ecg_pool(ecg_tokens)
        med_repr, _ = self.med_aux_pool(med_tokens)
        moment_repr, _ = self.moment_aux_pool(moment_projected)

        output: Dict[str, torch.Tensor] = {
            "logits": self.classifier(ecg_repr),
            "med_logits": self.med_aux_classifier(med_repr),
            "moment_logits": self.moment_aux_classifier(moment_repr),
            "med_repr": med_repr,
            "moment_repr": moment_repr,
            "ecg_attention": ecg_attention,
        }
        output.update(diagnostics)
        return output

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compatibility helper used by some external scripts."""
        return self.forward(inputs)["logits"]

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen backbones should not activate dropout when only fusion layers train.
        if self.llm_enabled and not self.lora_enabled:
            self.llm.eval()
        if not self.moment_has_trainable_parameters:
            self.moment.eval()
        return self

    def state_dict(self):
        """Save adapters/heads plus any trainable backbone parameters.

        Frozen LLM and MOMENT weights are reloaded from their pretrained model IDs,
        which keeps checkpoints small. Trainable LoRA or partially-unfrozen MOMENT
        parameters are retained.
        """
        state = nn.Module.state_dict(self)
        trainable_names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}

        for key in list(state.keys()):
            if key == "word_embeddings":
                del state[key]
                continue
            if key.startswith("llm.") and key not in trainable_names:
                del state[key]
                continue
            if (
                key.startswith("moment.")
                and not self.save_frozen_backbone
                and key not in trainable_names
            ):
                del state[key]

        return state
