"""
Memory-efficient MSE training for layer pruning — supports LLaMA and OPT.

Usage:
    python train.py --model llama [--model_name <path_or_hf_id>] [--data_path <local_dir>]
    python train.py --model opt   [--model_name facebook/opt-6.7b] [--data_path <local_dir>]

Set BEST_LAYER and LAST_PRUNED_LAYER in the respective modeling file before running:
  - LLaMA: edit modeling_llama.py
  - OPT:   edit modeling_opt.py
"""

import argparse
from itertools import chain

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from scheduler import get_cosine_schedule_with_warmup

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama", "opt"], required=True,
                    help="Model family to train.")
parser.add_argument("--model_name", type=str, default=None,
                    help="HuggingFace model ID or local path. Defaults to a sensible value per model.")
parser.add_argument("--data_path", type=str, default=None,
                    help="Local path for load_from_disk. If None, downloads SlimPajama-6B.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--grad_accum", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--min_lr", type=float, default=5e-6)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--eval_every", type=int, default=500,
                    help="Evaluate every N gradient steps.")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Model-specific settings
# ─────────────────────────────────────────────────────────────────────────────
if args.model == "llama":
    from modeling_llama import LlamaModel, BEST_LAYER, LAST_PRUNED_LAYER
    MODEL_CLASS = LlamaModel
    DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
    LAYER_PARAM_KEYS = [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
    ]
    # In LlamaForCausalLM, all sub-module keys are prefixed with "model."
    # LlamaModel (custom) has no such prefix, so pretrained key = "model." + model key.
    PRETRAINED_PREFIX = "model."
    REPLACE_LAYER_DST_PREFIX = "replace_layer."
    REPLACE_LAYER_SRC_PREFIX = f"model.layers.{BEST_LAYER}."

else:  # opt
    from modeling_opt import CustomOPTModel, BEST_LAYER, LAST_PRUNED_LAYER
    MODEL_CLASS = CustomOPTModel
    DEFAULT_MODEL_NAME = "facebook/opt-6.7b"
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
    # In OPTForCausalLM, keys are "model.decoder.*"; CustomOPTModel keys are "decoder.*".
    PRETRAINED_PREFIX = "model."
    REPLACE_LAYER_DST_PREFIX = "decoder.replace_layer."
    REPLACE_LAYER_SRC_PREFIX = f"model.decoder.layers.{BEST_LAYER}."

MODEL_NAME = args.model_name or DEFAULT_MODEL_NAME

# ─────────────────────────────────────────────────────────────────────────────
# Accelerator
# ─────────────────────────────────────────────────────────────────────────────
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=args.grad_accum,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model: build custom model and copy pretrained weights
# ─────────────────────────────────────────────────────────────────────────────
config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

config.num_hidden_layers = LAST_PRUNED_LAYER + 1

model = MODEL_CLASS(config)

print(f"Loading pretrained weights from {MODEL_NAME} ...")
pretrained = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
pretrained_dict = pretrained.state_dict()
model_dict = model.state_dict()

# Copy all shared weights (skip replace_layer — handled separately below)
for key in model_dict.keys():
    if "replace_layer" in key:
        continue
    pretrained_key = PRETRAINED_PREFIX + key
    if pretrained_key in pretrained_dict:
        model_dict[key] = pretrained_dict[pretrained_key]

# Initialise replace_layer from the best layer's pretrained weights
print(f"Initialising replace_layer from pretrained layer {BEST_LAYER} ...")
for param in LAYER_PARAM_KEYS:
    src = REPLACE_LAYER_SRC_PREFIX + param
    dst = REPLACE_LAYER_DST_PREFIX + param
    if src in pretrained_dict and dst in model_dict:
        model_dict[dst] = pretrained_dict[src]

model.load_state_dict(model_dict)
del pretrained
torch.cuda.empty_cache()

# Freeze everything except replace_layer
for name, param in model.named_parameters():
    param.requires_grad = "replace_layer" in name

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
if args.data_path is not None:
    dataset = load_from_disk(args.data_path)
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
    dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
)
eval_dataloader = DataLoader(
    eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size * 2
)

# ─────────────────────────────────────────────────────────────────────────────
# Optimiser & scheduler
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.95),
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) * 0.03),
    num_training_steps=len(train_dataloader) * args.epochs,
    max_learning_rate=args.lr,
    min_learning_rate=args.min_lr,
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
for epoch in range(args.epochs):
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

        if global_step % args.eval_every == 0:
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
                            mse_loss(prd, tgt).repeat(args.batch_size * 2)
                        )
                    )

            eval_loss = torch.cat(eval_losses).mean().item()
            if accelerator.is_main_process:
                print(f"Step {global_step} — eval MSE loss: {eval_loss:.6f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                unwrapped = accelerator.unwrap_model(model)
                replace_layer = (
                    unwrapped.replace_layer if args.model == "llama"
                    else unwrapped.decoder.replace_layer
                )
                torch.save(
                    replace_layer.state_dict(),
                    f"replace_layer_step{global_step}.pt",
                )
            model.train()
