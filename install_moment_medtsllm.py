DEBUG = false

task = "classification"
model = "moment_medtsllm"

# MedTsLLM receives a full-record 512-step resampling.
history_len = 512
pred_len = 1

[data]
dataset = "PTB-XL-MOMENT"
mode = "multivariate"
cols = "all"
normalize = true
step = 1

[training]
epochs = 20
batch_size = 4
gradient_accumulation_steps = 4
optimizer = "adamw"
learning_rate = 3e-4
moment_learning_rate = 1e-5
llm_learning_rate = 2e-5
weight_decay = 0.01
lr_scheduler = "constant"
dropout = 0.1
loss = "ce"
class_weights = true
label_smoothing = 0.05

# L = CE(fused) + aux_weight * CE(branches)
#     + alignment_weight * supervised cross-modal contrastive loss
aux_weight = 0.20
alignment_weight = 0.05
alignment_temperature = 0.10
gradient_clip = 1.0

eval_metric = "f1"
eval_metric_direction = "max"
save_best = true
restore_best = true

[datasets.PTB-XL-MOMENT]
version = "v1"

[models.medtsllm]
d_model = 32
d_ff = 128
n_heads = 8
num_tokens = 256
covariate_mode = "concat"
embedding_downsample_mode = "linear"

[models.medtsllm.patching]
# MOMENT uses 8-sample disjoint patches. MedTsLLM adds one replication-padded
# patch; MomentMedTsLLM removes that final token before fusion.
patch_len = 8
stride = 8

[models.medtsllm.prompting]
dataset = true
clip = true
task = true
input_stats = false
examples = false
input_stats_dim = 0
input_stats_select = "all"
example_pool = 0

[models.medtsllm.llm]
enabled = true
llm = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
load_in_4bit = false
load_in_8bit = false
llm_layers = -1

[models.medtsllm.lora]
# Stage C: leave false while validating the fusion modules.
# Stage D: set true after the frozen-backbone run converges.
enabled = false
layers = "auto"
rank = 8
alpha = 16
dropout = 0.05
rslora = true

[models.medtsllm.moment]
model_id = "AutonLab/MOMENT-1-base"
sequence_length = 512
max_leads = 12

# Frozen-backbone stage.
freeze_encoder = true
freeze_embedder = true
unfreeze_last_n = 0
enable_gradient_checkpointing = true
save_frozen_backbone = false

lead_attention_hidden = 128
fusion_hidden = 256
head_dropout = 0.20

# Two original-resolution windows: samples 0:512 and 488:1000 for PTB-XL.
use_highres_windows = true
n_highres_windows = 2

[models.medtsllm.moment.augmentation]
enabled = true
amplitude_scale = 0.05
noise_std = 0.01
lead_dropout_prob = 0.05
temporal_mask_ratio = 0.02

[setup]
seed = 0
device = "auto"
dtype = "mixed"
num_workers = 4
logger = "print"
