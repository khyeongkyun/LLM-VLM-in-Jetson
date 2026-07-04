"""
Retrain a replacement layer for a pruned block of layers, without ever
materialising the full model or the whole dataset's hidden states in memory.

Given the [pruning_start_layer, pruning_end_layer] range identified by
search_pruning_layer.py, this builds a truncated model (only layers
0..pruning_end_layer are instantiated) with a single trainable replace_layer
inserted right after layer (pruning_start_layer - 1), and trains it with MSE
loss to approximate the frozen output of layer pruning_end_layer.

Unifies __pruning/llm_streamline/replace_with_mlp_no_memory_issue/train.py and
replace_with_tf_no_memory_issue/train.py (those two were byte-identical aside
from which `modeling_*` module they imported) by picking the right modeling
class from ./modeling/ at runtime based on --model/--replace.

--replace none skips training entirely and produces a pruned model by directly
wiring layer (pruning_start_layer - 1) to layer (pruning_end_layer + 1), then
saving the result with save_pretrained.

Usage:
    python retrain_pruned_layer.py --model <llama|opt> --replace <none|mlp|tf> \
        --pruning_start_layer 19 --pruning_end_layer 29 \
        [--model_name <path_or_hf_id>] [--data_path <local_dir>]
"""

import importlib
import os
import re
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
    HfArgumentParser,
)

from args import ModelArguments, TrainingArguments
from LLM_Streamline.scheduler import get_cosine_schedule_with_warmup


if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────
    # CLI
    # ─────────────────────────────────────────────────────────────────────────
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Model-specific settings: pick the modeling module matching (model, replace)
    # ─────────────────────────────────────────────────────────────────────────
    modeling = importlib.import_module(f"modeling.modeling_{args.model}_{args.replace}")
    print(f"modeling_{args.model}_{args.replace}.py imported for {args.model} + {args.replace} replacement layer.")

    # The MLP replacement (fc1/fc2, a fresh 2-layer feed-forward net) only has a real
    # pretrained counterpart for OPT, whose FFN sublayer is itself named fc1/fc2; for
    # LLaMA (mlp.gate/up/down_proj) none of these keys exist in pretrained_dict, so
    # it's silently left at random init by the "if src in pretrained_dict" guard below.
    MLP_LAYER_PARAM_KEYS = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]

    if args.model == "llama":
        MODEL_CLASS = modeling.LlamaModel
        REPLACE_LAYER_DST_PREFIX = "replace_layer."
        REPLACE_LAYER_SRC_PREFIX = f"model.layers.{args.pruning_start_layer}."
        TF_LAYER_PARAM_KEYS = [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
        ]
    else:  # opt
        MODEL_CLASS = modeling.CustomOPTModel
        REPLACE_LAYER_DST_PREFIX = "decoder.replace_layer."
        REPLACE_LAYER_SRC_PREFIX = f"model.decoder.layers.{args.pruning_start_layer}."
        TF_LAYER_PARAM_KEYS = [
            "self_attn.q_proj.weight",     "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",     "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",     "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",   "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
            "fc1.weight",  "fc1.bias",
            "fc2.weight",  "fc2.bias",
            "final_layer_norm.weight",     "final_layer_norm.bias",
        ]

    # In both families, the pretrained checkpoint's top-level module is "model."
    PRETRAINED_PREFIX = "model."
    LAYER_PARAM_KEYS = TF_LAYER_PARAM_KEYS if args.replace == "tf" else MLP_LAYER_PARAM_KEYS

    MODEL_NAME = model_args.model_name

    if args.replace == "none":
        # ─────────────────────────────────────────────────────────────────────
        # "none" replace: pure layer pruning, no training
        # ─────────────────────────────────────────────────────────────────────
        n_pruned = args.pruning_end_layer - args.pruning_start_layer + 1

        print(f"Loading pretrained weights from {MODEL_NAME} ...")
        pretrained = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        pretrained_dict = pretrained.state_dict()
        del pretrained
        torch.cuda.empty_cache()

        config = AutoConfig.from_pretrained(MODEL_NAME)
        original_n_layers = config.num_hidden_layers
        config.num_hidden_layers -= n_pruned

        pruned_model = AutoModelForCausalLM.from_config(config)
        pruned_dict = pruned_model.state_dict()

        if args.model == "llama":
            layer_re = re.compile(r"^model\.layers\.(\d+)\.(.*)")
            layer_fmt = "model.layers.{j}.{rest}"
        else:  # opt
            layer_re = re.compile(r"^model\.decoder\.layers\.(\d+)\.(.*)")
            layer_fmt = "model.decoder.layers.{j}.{rest}"

        for key, value in pretrained_dict.items():
            m = layer_re.match(key)
            if m:
                i, rest = int(m.group(1)), m.group(2)
                if args.pruning_start_layer <= i <= args.pruning_end_layer:
                    continue  # drop pruned layers
                j = i if i < args.pruning_start_layer else i - n_pruned
                new_key = layer_fmt.format(j=j, rest=rest)
            else:
                new_key = key
            if new_key in pruned_dict:
                pruned_dict[new_key] = value

        save_path = os.path.join(args.output_dir, f"{args.model}_prune_from{args.pruning_start_layer}to{args.pruning_end_layer}_{args.replace}.pt")
        pruned_model.load_state_dict(pruned_dict)
        torch.save(pruned_model.state_dict(), save_path)
        print(
            f"Pruned model saved to {save_path}  "
            f"(layers {args.pruning_start_layer}–{args.pruning_end_layer} removed, "
            f"{original_n_layers} → {config.num_hidden_layers} layers)"
        )

    else:
        # ─────────────────────────────────────────────────────────────────────
        # Accelerator
        # ─────────────────────────────────────────────────────────────────────
        accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=args.grad_accum,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Model: build custom model and copy pretrained weights
        # ─────────────────────────────────────────────────────────────────────
        config = AutoConfig.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        config.num_hidden_layers = args.pruning_end_layer + 1

        model = MODEL_CLASS(config, start_pruned_layer=args.pruning_start_layer)

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

        # Initialise replace_layer from the layer just before the pruned range, where possible
        print(f"Initialising replace_layer from pretrained layer {args.pruning_start_layer} ...")
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

        # ─────────────────────────────────────────────────────────────────────
        # Dataset
        # ─────────────────────────────────────────────────────────────────────
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
                    k: [concat[k][i: i + block_size] for i in range(0, total, block_size)]
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

        # ─────────────────────────────────────────────────────────────────────
        # Optimiser & scheduler
        # ─────────────────────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────────────────────
        # Training loop
        # ─────────────────────────────────────────────────────────────────────
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
                    pred = outputs.last_hidden_state[1]
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
                            os.path.join(
                                args.output_dir,
                                f"{args.model}_prune_from{args.pruning_start_layer}to{args.pruning_end_layer}_{args.replace}_step{global_step}.pt",
                            ),
                        )
                    model.train()
