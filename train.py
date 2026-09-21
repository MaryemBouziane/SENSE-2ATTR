#!/usr/bin/env python3
"""Train SENSE-2ATTR on prepared Common Voice train/dev CSV manifests.

Semantic targets are computed from transcripts by the frozen BGE-M3 model.
Speaker targets are computed by a frozen ECAPA-TDNN.
Each attribute has its own layer projections, learned mixture, and pooling.

Usage:
    python train.py hparams/train_2attr.yaml
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import speechbrain as sb
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler
from speechbrain.inference import EncoderClassifier
from speechbrain.utils.distributed import run_on_main


class SenseBrain(sb.core.Brain):
    """Joint semantic and speaker distillation with independent layer mixtures."""

    @staticmethod
    def _cosine_loss(student, teacher, scale):
        """Return the scaled sum of cosine distances over the batch."""
        cosine = torch.sum(student.float() * teacher.float(), dim=-1)
        return (1.0 - cosine).sum() * float(scale)

    def compute_forward(self, batch, stage):
        """Compute the two student representations and their teacher targets."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Shared encoder: [layers, batch, time, feature_dim].
        feats_all = self.modules.wav2vec2(wavs, wav_lens)
        if not torch.is_tensor(feats_all) or feats_all.dim() != 4:
            raise RuntimeError(
                "Expected all encoder hidden states with shape [L, B, T, D]. "
                "Set output_all_hiddens: true in YAML."
            )

        feats_sem = self.modules.layer_mix_sem(feats_all)
        feats_spk = self.modules.layer_mix_spk(feats_all)

        # Semantic branch: mixing -> pooling -> normalization (1024 dimensions).
        u_sem = self.modules.attn_pooling_sem(feats_sem)
        u_sem = F.normalize(u_sem.float(), p=2, dim=-1)

        # Speaker branch: mixing -> pooling -> projection (1024 -> 192).
        u_spk = self.modules.attn_pooling_spk(feats_spk)
        u_spk = self.modules.projection_spk(u_spk)
        u_spk = F.normalize(u_spk.float(), p=2, dim=-1)

        # Teacher text encoder: BGE-M3.
        src_text = batch.wrd
        text_embeddings = self.modules.bge_model(src_text)

        with torch.no_grad():
            # ECAPA stays outside the student modules and remains in eval mode.
            # ECAPA returns [B, 1, 192]; retain the batch dimension.
            t_spk = self.spk_teacher.encode_batch(wavs, wav_lens).squeeze(1)
            t_spk = F.normalize(t_spk.float(), p=2, dim=-1)

        pairs = {"sem": (u_sem, text_embeddings), "spk": (u_spk, t_spk)}
        for name, (student, teacher) in pairs.items():
            if student.shape != teacher.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: student {tuple(student.shape)} "
                    f"vs teacher {tuple(teacher.shape)}."
                )
        return pairs

    def compute_objectives(self, predictions, batch, stage):
        """Combine semantic and speaker cosine losses with their task weights."""
        loss_sem = self._cosine_loss(
            *predictions["sem"], self.hparams.loss_scale_sem
        )
        loss_spk = self._cosine_loss(
            *predictions["spk"], self.hparams.loss_scale_spk
        )
        return (
            self.hparams.lambda_sem * loss_sem
            + self.hparams.lambda_spk * loss_spk
        )

    def init_optimizers(self):
        """Train both mixtures and heads, plus the encoder when it is unfrozen."""
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict = {"model_optimizer": self.model_optimizer}
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        # Save optimizer state as well as model weights when resuming training.
        if self.checkpointer is not None:
            for name, optimizer in self.optimizers_dict.items():
                self.checkpointer.add_recoverable(name, optimizer)

    def freeze_optimizers(self, optimizers):
        """Always update the heads; update w2v-BERT only when it is unfrozen."""
        active = {"model_optimizer": optimizers["model_optimizer"]}
        if not self.hparams.wav2vec2_frozen:
            active["wav2vec_optimizer"] = optimizers["wav2vec_optimizer"]
        return active

    def on_stage_end(self, stage, stage_loss, epoch):
        """Track epoch loss, update learning rates, and save checkpoints."""
        if stage not in (sb.Stage.TRAIN, sb.Stage.VALID):
            return

        # All ranks use the same development loss for learning-rate scheduling.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stage_loss = torch.tensor(stage_loss, device=self.device)
            torch.distributed.all_reduce(stage_loss)
            stage_loss = (stage_loss / torch.distributed.get_world_size()).item()
        stats = {"loss": float(stage_loss)}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stats
            return

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing_model(stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr)
            stats_meta = {"epoch": current_epoch, "lr_model": old_lr}

            if not self.hparams.wav2vec2_frozen:
                old_lr, new_lr = self.hparams.lr_annealing_wav2vec(stats["loss"])
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr
                )
                stats_meta["lr_wav2vec"] = old_lr

            def log_and_checkpoint():
                self.hparams.train_logger.log_stats(
                    stats_meta=stats_meta,
                    train_stats=self.train_stats,
                    valid_stats=stats,
                )
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stats["loss"], "epoch": current_epoch},
                    name=f"checkpoint_epoch{current_epoch}",
                    num_to_keep=10,
                    min_keys=["loss"],
                )

            run_on_main(log_and_checkpoint)


def validate_manifests(hparams):
    """Check prepared Common Voice manifests and return the training rows."""
    train_csv = Path(hparams["train_csv"]).expanduser().resolve()
    valid_csv = Path(hparams["valid_csv"]).expanduser().resolve()
    for path, is_train in ((train_csv, True), (valid_csv, False)):
        manifest = pd.read_csv(path, keep_default_na=False)
        required = {"ID", "wav", "wrd", "lang", "duration"}
        if is_train:
            required.add("ratio")
        missing = required.difference(manifest.columns)
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}.")
        if manifest.empty:
            raise ValueError(f"{path}: the CSV contains no examples.")
        for column in ("ID", "wav", "wrd"):
            if manifest[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{path}: empty values in column '{column}'.")
        if is_train:
            train_manifest = manifest
    return train_manifest


def dataio_prepare(hparams):
    """Read Common Voice audio and transcripts from prepared train/dev CSVs."""

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Read audio already prepared as mono at 16 kHz."""
        return sb.dataio.dataio.read_audio(wav)

    datasets = {}
    for split, key in (("train", "train_csv"), ("valid", "valid_csv")):
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=str(Path(hparams[key]).expanduser()),
            dynamic_items=[audio_pipeline],
            output_keys=["id", "lang", "sig", "duration", "wrd"],
        )
    return datasets


def main():
    """Initialize the student, frozen teachers, sampling weights, and training."""
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    # Forward --device to YAML so BGE and the student use the same device.
    if run_opts.get("device") is not None:
        if isinstance(overrides, str):
            overrides = load_hyperpyyaml(overrides) or {}
        else:
            overrides = dict(overrides)
        overrides["device"] = run_opts["device"]
    with open(hparams_file, encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(stream, overrides)

    train_manifest = validate_manifests(hparams)
    sample_ratios = train_manifest["ratio"].astype(float).to_numpy()
    if (not np.isfinite(sample_ratios).all() or (sample_ratios < 0).any()
            or not (sample_ratios > 0).any()):
        raise ValueError("Training ratios must be finite, non-negative, and not all zero.")

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    datasets = dataio_prepare(hparams)

    brain = SenseBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Use the Brain device, including its rank-local CUDA device under DDP.
    spk_teacher = EncoderClassifier.from_hparams(
        source=hparams["teacher_source"],
        savedir=hparams["teacher_savedir"],
        run_opts={"device": str(brain.device)},
    )
    spk_teacher.mods.eval()
    spk_teacher.mods.requires_grad_(False)
    brain.spk_teacher = spk_teacher

    # Sample training examples using the ratio column in train.csv.
    train_sampler = ReproducibleWeightedRandomSampler(
        sample_ratios.tolist(),
        replacement=True,
        num_samples=len(sample_ratios),
        seed=int(hparams["seed"]),
    )
    brain.train_sampler = train_sampler
    train_loader_kwargs = dict(hparams["dataloader_options"])
    train_loader_kwargs["sampler"] = train_sampler
    train_loader_kwargs["shuffle"] = False

    brain.fit(
        brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )


if __name__ == "__main__":
    main()
