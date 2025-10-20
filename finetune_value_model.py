import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, get_peft_model
#from peft.utils.other import ALL_LINEAR_LAYERS
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class ModelConfig:
    loss_type: str
    learning_rate: float
    weight_decay: float
    epochs: int
    warmup_steps: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    gradient_accumulation: int
    log_steps: int


def _to_chat_messages(prompt_field: Any) -> List[Dict[str, str]]:
    if isinstance(prompt_field, str):
        return [{"role": "user", "content": prompt_field}]
    if isinstance(prompt_field, Sequence):
        messages: List[Dict[str, str]] = []
        for item in prompt_field:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append({"role": item["role"], "content": item["content"]})
        if messages:
            return messages
    raise ValueError("Unsupported prompt format for chat template rendering.")


def _render_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        try:
            model_dtype = next(backbone.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16
        self.value_head = nn.Linear(hidden_size, 1).to(dtype=model_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # get inner transformer (varies by arch: "model" or "transformer")
        core = getattr(self.backbone, "model", None)
        if core is None:
            core = getattr(self.backbone, "transformer", None)
        if core is None:
            core = self.backbone  # fallback

        # ask ONLY for the last hidden state from the core
        core_out = core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = core_out.last_hidden_state  # [B, L, H]

        # pool at the last non-pad token
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        values = self.value_head(pooled).squeeze(-1)
        return values


class ValueCollator:
    def __init__(self, tokenizer: AutoTokenizer, label_dtype: torch.dtype):
        self.tokenizer = tokenizer
        self.label_dtype = label_dtype

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([f["labels"] for f in features], dtype=self.label_dtype)
        model_inputs = []
        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f["attention_mask"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
            model_inputs.append({"input_ids": input_ids, "attention_mask": attention_mask})
        batch = self.tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def prepare_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int,
) -> Dict[str, Dataset]:
    ds = Dataset.from_parquet(dataset_path)

    def has_ground_truth(example: Dict[str, Any]) -> bool:
        reward = example.get("reward_model", None)
        if not isinstance(reward, dict):
            return False
        return reward.get("ground_truth", None) is not None

    ds = ds.filter(has_ground_truth)

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = _to_chat_messages(example["prompt"])
        rendered = _render_prompt(tokenizer, messages)
        enc = tokenizer(rendered, truncation=True, max_length=max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": float(example["reward_model"]["ground_truth"]),
        }

    ds = ds.map(preprocess, desc="Tokenizing prompts")

    total = len(ds)
    if total < 2:
        raise ValueError("Need at least two samples to create train/validation splits.")

    if total > 2000:
        val_size = 2000
    else:
        val_size = max(1, total // 2)
    val_size = min(val_size, total - 1)

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    val_indices = np.sort(indices[:val_size])
    train_indices = np.sort(indices[val_size:])

    train_ds = ds.select(train_indices.tolist())
    val_ds = ds.select(val_indices.tolist())

    return {"train": train_ds, "val": val_ds}


def init_wandb(project: Optional[str], run_name: Optional[str], config: Dict[str, Any]) -> Optional["wandb.wandb_sdk.wandb_run.Run"]:
    if project is None or wandb is None:
        return None
    return wandb.init(project=project, name=run_name, config=config)


def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        os.makedirs(args.output_model, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"

    datasets = prepare_dataset(args.dataset, tokenizer, args.max_length, args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    base_model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # if supported
    )

    # memory savers
    base_model.config.use_cache = False
    #base_model.gradient_checkpointing_enable()
    #base_model.enable_input_require_grads()

    # LoRA for a base model: use FEATURE_EXTRACTION (or SEQ_CLS)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules='all-linear',
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = get_peft_model(base_model, lora_config)
    #for module in peft_model.modules():
    #    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
    #        for lora_a in module.lora_A.values():
    #            nn.init.zeros_(lora_a.weight)
    #        for lora_b in module.lora_B.values():
    #            nn.init.zeros_(lora_b.weight)

    value_model = ValueModel(peft_model, hidden_size=peft_model.config.hidden_size)

    collator = ValueCollator(tokenizer, label_dtype=torch_dtype)

    train_ds = datasets["train"].with_format(type="torch")
    val_ds = datasets["val"].with_format(type="torch")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(value_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = max(len(train_loader) // accelerator.gradient_accumulation_steps, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, max(total_steps // 10, 0))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))

    value_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        value_model, optimizer, train_loader, val_loader, scheduler
    )

    if args.loss_type == "mse":
        criterion = nn.MSELoss()
    elif args.loss_type == "mae":
        criterion = nn.L1Loss()
    elif args.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")

    config = ModelConfig(
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        max_length=args.max_length,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_accumulation=args.gradient_accumulation_steps,
        log_steps=args.log_steps,
    )

    run = None
    if accelerator.is_main_process:
        run = init_wandb(args.wandb_project, args.wandb_run_name, config.__dict__)
        Path(args.output_model).mkdir(parents=True, exist_ok=True)

    global_step = 0
    cuda_device = accelerator.device.type == "cuda"

    for epoch in range(1, args.epochs + 1):
        value_model.train()
        epoch_losses: List[float] = []
        epoch_pred_means: List[float] = []
        step_losses: List[float] = []
        step_pred_means: List[float] = []
        for step, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(value_model):
                logits = value_model(batch["input_ids"], batch["attention_mask"])
                labels = batch["labels"].to(logits.dtype)
                if args.loss_type == "bce":
                    preds = torch.sigmoid(logits)
                    loss = criterion(logits, labels)
                else:
                    preds = logits
                    loss = criterion(preds, labels)

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_losses.append(loss.detach().item())
            epoch_pred_means.append(preds.detach().mean().float().item())
            step_losses.append(loss.detach().item())
            step_pred_means.append(preds.detach().mean().float().item())

            del logits, preds, labels, loss
            if cuda_device:
                torch.cuda.empty_cache()
            del batch

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.log_steps == 0:
                    mean_loss = float(np.mean(step_losses)) if step_losses else float("nan")
                    mean_pred = float(np.mean(step_pred_means)) if step_pred_means else float("nan")
                    step_losses.clear()
                    step_pred_means.clear()
                    if run is not None:
                        run.log({"train/loss": mean_loss, "train/pred_mean": mean_pred, "train/step": global_step})

        train_loss_epoch = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_pred_mean = float(np.mean(epoch_pred_means)) if epoch_pred_means else float("nan")
        if accelerator.is_main_process and run is not None:
            run.log({"train/loss_epoch": train_loss_epoch, "train/pred_mean_epoch": train_pred_mean, "epoch": epoch})

        value_model.eval()
        val_losses: List[float] = []
        val_pred_means: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                logits = value_model(batch["input_ids"], batch["attention_mask"])
                labels = batch["labels"].to(logits.dtype)
                if args.loss_type == "bce":
                    preds = torch.sigmoid(logits)
                    val_loss = criterion(logits, labels)
                else:
                    preds = logits
                    val_loss = criterion(preds, labels)
                gathered = accelerator.gather(val_loss.detach()).mean().item()
                val_losses.append(gathered)
                val_pred_means.append(preds.detach().mean().float().item())

                del logits, preds, labels, val_loss
                if cuda_device:
                    torch.cuda.empty_cache()
                del batch

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        mean_val_pred = float(np.mean(val_pred_means)) if val_pred_means else float("nan")
        if accelerator.is_main_process and run is not None:
            run.log({"val/loss": mean_val_loss, "val/pred_mean": mean_val_pred, "epoch": epoch})

        accelerator.print(f"Epoch {epoch}: train_loss={train_loss_epoch:.6f}, val_loss={mean_val_loss:.6f}")

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            unwrapped: ValueModel = accelerator.unwrap_model(value_model)  # type: ignore
            backbone = unwrapped.backbone
            if hasattr(backbone, "merge_and_unload"):
                backbone = backbone.merge_and_unload()
            epoch_dir = Path(args.output_model) / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            backbone.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            torch.save(unwrapped.value_head.state_dict(), epoch_dir / "value_head.pt")
            head_config = {"hidden_size": backbone.config.hidden_size, "loss_type": args.loss_type}
            with open(epoch_dir / "value_head_config.json", "w", encoding="utf-8") as f:
                json.dump(head_config, f, indent=2)

    accelerator.wait_for_everyone()

    if run is not None:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA finetuning script for scalar value prediction.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset", default="/pscratch/sd/s/siddart2/gql_data/qwen4b/combined.parquet")
    parser.add_argument("--output-model", default="/pscratch/sd/s/siddart2/checkpoints/value_baselines/qwen4b/bce")
    parser.add_argument("--loss-type", choices=["mse", "mae", "bce"], default="bce")
    parser.add_argument("--max-length", type=int, default=10000)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.00)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wandb-project", default='gql-value-baseline')
    parser.add_argument("--wandb-run-name", default='bce')
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
