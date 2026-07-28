# MOMENT-conditioned MedTsLLM

This extension adds a sequence-classification model named `moment_medtsllm` to
`kkianamm/medtsllm2`. It is designed for the repository's five-class,
single-label PTB-XL setup.

## Architecture implemented

1. The original MedTsLLM branch performs RevIN, patch embedding, and vocabulary
   reprogramming.
2. `AutonLab/MOMENT-1-base` produces unreduced `[lead, patch, embedding]`
   representations.
3. Learned lead attention creates one MOMENT vector per patch.
4. Two 512-sample windows from the original 1,000-sample ECG are encoded by
   MOMENT and pooled into a high-resolution global context vector.
5. A gated context module injects that vector into the aligned MOMENT tokens.
6. A projection and residual gate fuse MOMENT and MedTsLLM tokens in the LLM
   hidden space.
7. Patient metadata prompts and fused ECG tokens pass through the LLM.
8. Attention pooling and a compact classifier produce the final prediction.
9. Auxiliary branch classifiers and supervised cross-modal contrastive
   alignment stabilize fusion training.

## Install

Run the installer from this bundle. It is idempotent and creates one-time
`.before-moment-medtsllm` backups of files it modifies.

```bash
python install_moment_medtsllm.py --repo /path/to/medtsllm2
cd /path/to/medtsllm2
```

Create a clean Python 3.11 environment and install the merged requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-full-moment.txt
```

The original repository pins older NumPy and Transformers versions that are
incompatible with the current `momentfm` package. Use
`requirements-full-moment.txt`, or install the original requirements first and
then `requirements-moment.txt` last.

## PTB-XL files

The existing dataset implementation expects this structure:

```text
data/ptbxl/
├── ptbxl_database.csv
├── scp_statements.csv
├── records100/
└── ...
```

On first use, each split is cached as `cache_100hz_{split}.npz`.

## Train

```bash
python train.py configs/datasets/ptbxl_moment_medtsllm.toml
```

The default configuration is the frozen-backbone stage:

- MOMENT encoder frozen
- LLM frozen
- MedTsLLM reprogramming, lead attention, context injection, fusion, pooling,
  and classification heads trainable
- Effective batch size: `4 × 4 = 16`
- Validation selection: fold 9 macro F1
- Test evaluation: fold 10 once, after restoring the best validation checkpoint

## Gradual adaptation

After the frozen run converges, edit the configuration:

```toml
[training]
learning_rate = 1e-4
moment_learning_rate = 1e-5
llm_learning_rate = 2e-5

[models.medtsllm.moment]
unfreeze_last_n = 2

[models.medtsllm.lora]
enabled = true
```

The task automatically places fusion parameters, MOMENT parameters, and LLM
LoRA parameters into different optimizer groups.

## Important ablations

Use the following changes one at a time:

- No high-resolution context: `use_highres_windows = false`
- No auxiliary classifiers: `aux_weight = 0.0`
- No cross-modal alignment: `alignment_weight = 0.0`
- Frozen MOMENT versus `unfreeze_last_n = 2`
- LLM frozen versus LoRA enabled
- Lead attention versus a manual mean over leads

Report accuracy, balanced accuracy, macro F1, macro precision, and macro recall
for at least three random seeds.

## Checkpoints

Frozen pretrained LLM and MOMENT tensors are omitted from checkpoints. They are
reloaded from their model IDs. Trainable LoRA weights and partially unfrozen
MOMENT weights are retained.

## Troubleshooting

### Out of memory

First reduce `batch_size` to 2 or 1 while increasing
`gradient_accumulation_steps`. Next set `use_highres_windows = false`. Enabling
4-bit LLM loading may help, but requires installing `bitsandbytes` and is best
used only on supported CUDA systems.

### Hugging Face download failure

Ensure the machine can access Hugging Face and has sufficient cache space. Both
MOMENT and the configured LLM are downloaded on first initialization.

### Existing PTB-XL caches

Delete `data/ptbxl/cache_100hz_*.npz` only when changing the underlying PTB-XL
records or target construction. The MOMENT wrapper uses the same cached raw
recordings as the existing PTB-XL dataset.
