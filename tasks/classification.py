"""Sequence-level classification task with multi-branch fusion losses."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from .base import BaseTask


ModelOutput = Union[torch.Tensor, Dict[str, torch.Tensor]]


class ClassificationTask(BaseTask):
    """Whole-record classification for MedTsLLM and MomentMedTsLLM."""

    def __init__(self, run_id, config, newrun=True):
        self.task = "classification"
        super().__init__(run_id, config, newrun)

        self.gradient_accumulation_steps = int(
            self.config.training.get("gradient_accumulation_steps", 1)
        )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        self.aux_weight = float(self.config.training.get("aux_weight", 0.0))
        self.alignment_weight = float(
            self.config.training.get("alignment_weight", 0.0)
        )
        self.alignment_temperature = float(
            self.config.training.get("alignment_temperature", 0.1)
        )
        self.gradient_clip = float(self.config.training.get("gradient_clip", 0.0))

        # Evaluate the held-out test set during training when enabled.
        # Defaults match medtsllm_blip: test after every epoch.
        self.evaluate_test_each_epoch = bool(
            self.config.training.get("evaluate_test_each_epoch", True)
        )
        self.test_eval_interval = max(
            1,
            int(self.config.training.get("test_eval_interval", 1)),
        )

    def build_optimizer(self):
        """Use lower learning rates for any unfrozen pretrained backbones."""
        if self.config.model != "moment_medtsllm":
            return super().build_optimizer()

        base_lr = float(self.config.training.learning_rate)
        moment_lr = float(self.config.training.get("moment_learning_rate", 1e-5))
        llm_lr = float(self.config.training.get("llm_learning_rate", 2e-5))
        weight_decay = float(self.config.training.get("weight_decay", 0.01))

        groups = {"fusion": [], "moment": [], "llm": []}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("moment."):
                groups["moment"].append(parameter)
            elif name.startswith("llm."):
                groups["llm"].append(parameter)
            else:
                groups["fusion"].append(parameter)

        parameter_groups = []
        if groups["fusion"]:
            parameter_groups.append({"params": groups["fusion"], "lr": base_lr})
        if groups["moment"]:
            parameter_groups.append({"params": groups["moment"], "lr": moment_lr})
        if groups["llm"]:
            parameter_groups.append({"params": groups["llm"], "lr": llm_lr})

        optimizer_name = self.config.training.optimizer
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(parameter_groups, lr=base_lr)
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=base_lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                parameter_groups,
                lr=base_lr,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(
                "MomentMedTsLLM differential learning rates support "
                "adam, adamw, or sgd; received " + str(optimizer_name)
            )
        return self.optimizer

    @staticmethod
    def _main_logits(output: ModelOutput) -> torch.Tensor:
        if isinstance(output, dict):
            if "logits" not in output:
                raise KeyError("Model output dictionary must contain 'logits'.")
            return output["logits"]
        return output

    def _cross_modal_alignment_loss(
        self,
        med_repr: torch.Tensor,
        moment_repr: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric supervised contrastive alignment across the two branches."""
        if med_repr.size(0) < 2:
            return med_repr.new_zeros(())

        med_repr = F.normalize(med_repr.float(), dim=-1)
        moment_repr = F.normalize(moment_repr.float(), dim=-1)
        logits = med_repr @ moment_repr.transpose(0, 1)
        logits = logits / self.alignment_temperature

        positive_mask = labels[:, None].eq(labels[None, :]).float()
        positive_count = positive_mask.sum(dim=1).clamp_min(1.0)

        med_to_moment = F.log_softmax(logits, dim=1)
        moment_to_med = F.log_softmax(logits.transpose(0, 1), dim=1)

        loss_med = -((positive_mask * med_to_moment).sum(dim=1) / positive_count).mean()
        loss_moment = -(
            (positive_mask.transpose(0, 1) * moment_to_med).sum(dim=1)
            / positive_count
        ).mean()
        return 0.5 * (loss_med + loss_moment)

    def _compute_loss(
        self,
        output: ModelOutput,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self._main_logits(output)
        main_loss = self.loss_fn(logits, labels)
        total_loss = main_loss

        stats = {"main": float(main_loss.detach())}

        if isinstance(output, dict) and self.aux_weight > 0:
            med_logits = output.get("med_logits")
            moment_logits = output.get("moment_logits")
            if med_logits is not None and moment_logits is not None:
                aux_loss = 0.5 * (
                    self.loss_fn(med_logits, labels)
                    + self.loss_fn(moment_logits, labels)
                )
                total_loss = total_loss + self.aux_weight * aux_loss
                stats["aux"] = float(aux_loss.detach())

        if isinstance(output, dict) and self.alignment_weight > 0:
            med_repr = output.get("med_repr")
            moment_repr = output.get("moment_repr")
            if med_repr is not None and moment_repr is not None:
                alignment_loss = self._cross_modal_alignment_loss(
                    med_repr,
                    moment_repr,
                    labels,
                )
                total_loss = total_loss + self.alignment_weight * alignment_loss
                stats["alignment"] = float(alignment_loss.detach())

        stats["total"] = float(total_loss.detach())
        return total_loss, stats

    def train(self):
        for epoch_index in range(self.config.training.epochs):
            print(f"Epoch {epoch_index + 1}/{self.config.training.epochs}")
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            epoch_loss = 0.0
            n_batches = 0
            progress = tqdm(self.train_dataloader, total=len(self.train_dataloader))

            for batch_index, inputs in enumerate(progress):
                inputs = self.prepare_batch(inputs)
                labels = inputs["labels"].long()

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.mixed and self.device.type in {"cuda", "cpu"},
                ):
                    output = self.model(inputs)
                    loss, loss_parts = self._compute_loss(output, labels)
                    scaled_loss = loss / self.gradient_accumulation_steps

                scaled_loss.backward()

                should_step = (
                    (batch_index + 1) % self.gradient_accumulation_steps == 0
                    or (batch_index + 1) == len(self.train_dataloader)
                )
                if should_step:
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clip,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                loss_value = float(loss.detach())
                epoch_loss += loss_value
                n_batches += 1
                self.log_step(loss_value)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    main=f"{loss_parts['main']:.4f}",
                )

            val_scores = self.val()

            epoch_number = epoch_index + 1
            should_evaluate_test = (
                self.evaluate_test_each_epoch
                and epoch_number % self.test_eval_interval == 0
            )

            if should_evaluate_test:
                test_scores = self.test()
            else:
                test_scores = {}

            mean_train_loss = epoch_loss / max(n_batches, 1)
            epoch_scores = {
                **val_scores,
                **test_scores,
            }
            self.log_epoch(
                epoch_scores,
                **{"train/epoch_loss": mean_train_loss},
            )
            self.scheduler.step()

            status = (
                f"[epoch {epoch_number}] train loss: {mean_train_loss:.5f}; "
                f"validation: {val_scores}"
            )
            if test_scores:
                status += f"; test: {test_scores}"
            print(status)

        if self.config.training.get("restore_best", True):
            self._restore_best_checkpoint()
        self.model.eval()

    def _restore_best_checkpoint(self):
        checkpoint_path = self.logger.logdir / "checkpoints" / "best.pt"
        if not checkpoint_path.exists():
            print("Best checkpoint was not found; keeping the final-epoch weights.")
            return

        state = torch.load(checkpoint_path, map_location=self.device)
        incompatible = self.model.load_state_dict(state["model"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys while restoring best checkpoint: "
                f"{incompatible.unexpected_keys}"
            )
        print(f"Restored best validation checkpoint from {checkpoint_path}.")

    def val(self):
        pred_scores, targets = self.predict(self.val_dataloader)
        scores = self.score(pred_scores, targets)
        return {f"val/{metric}": value for metric, value in scores.items()}

    def test(self):
        pred_scores, targets = self.predict(self.test_dataloader)
        scores = self.score(pred_scores, targets)
        scores = {f"test/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def predict(self, dataloader):
        self.model.eval()
        all_scores = []
        all_targets = []

        with torch.no_grad():
            for inputs in tqdm(dataloader, total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.mixed and self.device.type in {"cuda", "cpu"},
                ):
                    output = self.model(inputs)
                    logits = self._main_logits(output)

                all_scores.append(logits.float().cpu())
                all_targets.append(inputs["labels"].long().cpu())

        return torch.cat(all_scores, dim=0), torch.cat(all_targets, dim=0)

    def build_loss(self):
        match self.config.training.loss:
            case "ce" | "cross_entropy" | "auto":
                weight = None
                if self.config.training.get("class_weights", False):
                    weight = self.train_dataset.class_weights.to(self.device)
                label_smoothing = float(
                    self.config.training.get("label_smoothing", 0.0)
                )
                self.loss_fn = torch.nn.CrossEntropyLoss(
                    weight=weight,
                    label_smoothing=label_smoothing,
                )
            case _:
                raise ValueError(
                    f"Invalid loss function selection: {self.config.training.loss}"
                )
        return self.loss_fn

    def score(self, pred_scores, target):
        pred = pred_scores.argmax(dim=1).numpy()
        target = target.numpy()
        average = "binary" if pred_scores.size(1) == 2 else "macro"
        return {
            "accuracy": accuracy_score(target, pred),
            "balanced_accuracy": balanced_accuracy_score(target, pred),
            "f1": f1_score(target, pred, average=average, zero_division=0),
            "precision": precision_score(
                target,
                pred,
                average=average,
                zero_division=0,
            ),
            "recall": recall_score(
                target,
                pred,
                average=average,
                zero_division=0,
            ),
        }
