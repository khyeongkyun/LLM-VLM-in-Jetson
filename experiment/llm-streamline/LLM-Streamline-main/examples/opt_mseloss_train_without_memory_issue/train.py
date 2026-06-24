"""
Memory-efficient MSE training for OPT layer pruning.

How it differs from the top-level mseloss_entry.py
────────────────────────────────────────────────────
mseloss_entry.py pre-collects ALL hidden states from the entire dataset into
RAM before training starts.  For large models / datasets this runs out of
memory.

This example embeds replace_layer inside the model's forward pass (via
CustomOPTDecoder in modeling_opt.py), so hidden states are computed and
discarded batch-by-batch.  The only thing kept in GPU memory at each step is
the current batch.

Prerequisite: cosine similarity analysis
─────────────────────────────────────────
Run the main mseloss_entry.py (or get_cosine.py) with your target OPT model
first to identify BEST_LAYER.  Then set BEST_LAYER and LAST_PRUNED_LAYER in
modeling_opt.py before running this script.

Example for facebook/opt-1.3b (24 layers), pruning 4 layers:
  BEST_LAYER       = 12   # layer index found by cosine similarity
  LAST_PRUNED_LAYER = 16   # BEST_LAYER + 4
  → training model has 17 layers (indices 0-16)
  → replace_layer learns to approximate what layers 13-16 compute
"""

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from itertools import chain
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm

from modeling_opt import CustomOPTModel, BEST_LAYER, LAST_PRUNED_LAYER
from scheduler import get_cosine_schedule_with_warmup


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
OPT_MODEL_NAME = "facebook/opt-6.7b"  # change to opt-2.7b, opt-6.7b, etc.
TRAIN_DATA_PATH = None                 # local path for load_from_disk, or None to auto-download
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
LR = 2e-4
MIN_LR = 5e-6
WEIGHT_DECAY = 1e-3
EPOCHS = 1
EVAL_EVERY = 500   # evaluate every N gradient steps

# All parameter names inside an OPTDecoderLayer (weights + biases).
# Used to copy the pretrained best-layer weights into replace_layer.
LAYER_PARAM_KEYS = [
    "self_attn.q_proj.weight",     "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",     "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",     "self_attn.v_proj.bias",
    "self_attn.out_proj.weight",   "self_attn.out_proj.bias",
    "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
    "fc1.weight",  "fc1.bias",
    "fc2.weight",  "fc2.bias",
    "final_layer_norm.weight",     "final_layer_norm.bias",
]


# ─────────────────────────────────────────────────────────────────────────────
# Accelerator
# ─────────────────────────────────────────────────────────────────────────────
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model: build custom model and copy pretrained weights
# ─────────────────────────────────────────────────────────────────────────────
config = AutoConfig.from_pretrained(OPT_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Only run layers 0..LAST_PRUNED_LAYER during training.
# The remaining layers (LAST_PRUNED_LAYER+1 … end) stay frozen in the final
# pruned model but are not needed while training replace_layer.
NUM_TRAINING_LAYERS = LAST_PRUNED_LAYER + 1
config.num_hidden_layers = NUM_TRAINING_LAYERS

model = CustomOPTModel(config)

print(f"Loading pretrained weights from {OPT_MODEL_NAME} ...")
pretrained = AutoModelForCausalLM.from_pretrained(OPT_MODEL_NAME)
pretrained_dict = pretrained.state_dict()  # keys: model.decoder.*  and  lm_head.*
model_dict = model.state_dict()            # keys: decoder.*

# ── Copy shared weights (all decoder weights except replace_layer) ──────────
# OPTForCausalLM stores the model under 'model.', so pretrained_dict keys
# are 'model.decoder.X' while our CustomOPTModel keys are 'decoder.X'.
for key in model_dict.keys():
    if "replace_layer" in key:
        continue
    pretrained_key = "model." + key
    if pretrained_key in pretrained_dict:
        model_dict[key] = pretrained_dict[pretrained_key]

# ── Initialise replace_layer from the pretrained BEST_LAYER ─────────────────
print(f"Initialising replace_layer from pretrained layer {BEST_LAYER} ...")
for param in LAYER_PARAM_KEYS:
    src = f"model.decoder.layers.{BEST_LAYER}.{param}"
    dst = f"decoder.replace_layer.{param}"
    if src in pretrained_dict and dst in model_dict:
        model_dict[dst] = pretrained_dict[src]

model.load_state_dict(model_dict)
del pretrained
torch.cuda.empty_cache()

# ── Freeze everything except replace_layer ───────────────────────────────────
for name, param in model.named_parameters():
    param.requires_grad = "replace_layer" in name

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
if TRAIN_DATA_PATH is not None:
    dataset = load_from_disk(TRAIN_DATA_PATH)
    eval_dataset = dataset["validation"]
    dataset = dataset["train"].train_test_split(
        test_size=300_000 / len(dataset["train"])
    )["test"]
else:
    print("Downloading SlimPajama-6B ...")
    raw = load_dataset("DKYoon/SlimPajama-6B")["train"]
    split = raw.train_test_split(test_size=3_000 / len(raw))
    dataset, eval_dataset = split["train"], split["test"]

    block_size = 2048

    def tokenize(examples):
        return tokenizer(examples["text"])

    def group(examples):
        concat = {k: list(chain(*examples[k])) for k in examples.keys()}
        total = (len(concat[list(concat.keys())[0]]) // block_size) * block_size
        return {
            k: [concat[k][i : i + block_size] for i in range(0, total, block_size)]
            for k in concat
        }

    col_names = dataset.column_names
    dataset = dataset.map(tokenize, batched=True, remove_columns=col_names)
    dataset = dataset.map(group, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=col_names)
    eval_dataset = eval_dataset.map(group, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(
    dataset, shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE
)
eval_dataloader = DataLoader(
    eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE * 2
)


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser & scheduler
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) * 0.03),
    num_training_steps=len(train_dataloader) * EPOCHS,
    max_learning_rate=LR,
    min_learning_rate=MIN_LR,
)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

mse_loss = nn.MSELoss()
best_eval_loss = float("inf")
global_step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}"):
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            # last_hidden_state[0] = frozen target  (raw output of last training layer)
            # last_hidden_state[1] = replace_layer prediction
            target = outputs.last_hidden_state[0]
            pred   = outputs.last_hidden_state[1]
            loss = mse_loss(pred, target)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % EVAL_EVERY == 0:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for eval_batch in tqdm(eval_dataloader, desc="Eval", leave=False):
                    out = model(
                        input_ids=eval_batch["input_ids"],
                        attention_mask=eval_batch["attention_mask"],
                    )
                    tgt = out.last_hidden_state[0]
                    prd = out.last_hidden_state[1]
                    eval_losses.append(
                        accelerator.gather_for_metrics(
                            mse_loss(prd, tgt).repeat(BATCH_SIZE * 2)
                        )
                    )

            eval_loss = torch.cat(eval_losses).mean().item()
            if accelerator.is_main_process:
                print(f"Step {global_step} — eval MSE loss: {eval_loss:.6f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                # Uncomment to save the replace_layer state dict:
                torch.save(
                    accelerator.unwrap_model(model).decoder.replace_layer.state_dict(),
                    f"replace_layer_step{global_step}.pt",
                )

            model.train()
