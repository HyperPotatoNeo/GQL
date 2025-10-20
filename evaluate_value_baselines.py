import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def _extract_question(text: str) -> str:
    return text.strip()


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        try:
            example_param = next(backbone.parameters())
            model_dtype = example_param.dtype
            model_device = example_param.device
        except StopIteration:
            model_dtype = torch.float32
            model_device = torch.device("cpu")
        self.value_head = nn.Linear(hidden_size, 1, dtype=model_dtype, device=model_device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        last_indices = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices]
        values = self.value_head(pooled).squeeze(-1)
        return values


class EvalCollator:
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
        batch["question"] = [f["question"] for f in features]
        return batch


def _dtype_from_string(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _prepare_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
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
        question_text = messages[0]["content"] if messages else ""
        enc = tokenizer(
            rendered,
            truncation=True,
            max_length=max_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": float(example["reward_model"]["ground_truth"]),
            "question": _extract_question(question_text),
        }

    return ds.map(preprocess, desc="Tokenizing prompts")


def run_evaluation(
    model_dir: Path,
    dataset_path: str,
    batch_size: int,
    max_length: int,
    sample_path: Path,
    sample_count: int,
    error_type: str,
    torch_dtype: str,
) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    dtype = _dtype_from_string(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    value_model = ValueModel(base_model)
    head_path = model_dir / "value_head.pt"
    head_config_path = model_dir / "value_head_config.json"
    if not head_path.exists():
        raise FileNotFoundError(f"Expected value head weights at {head_path}")
    state_dict = torch.load(head_path, map_location="cpu")
    value_model.value_head.load_state_dict(state_dict)
    if head_config_path.exists():
        with open(head_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        expected_hidden = config.get("hidden_size")
        if expected_hidden is not None and expected_hidden != value_model.backbone.config.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: head expects {expected_hidden}, backbone provides {value_model.backbone.config.hidden_size}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_model.to(device)
    value_model.eval()

    ds = _prepare_dataset(dataset_path, tokenizer, max_length)
    if len(ds) == 0:
        print("No rows with ground truth available for evaluation.")
        return

    collator = EvalCollator(tokenizer, label_dtype=torch.float32)
    ds = ds.with_format(type=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_errors: List[float] = []
    all_predictions: List[float] = []
    all_targets: List[float] = []
    all_questions: List[str] = []
    all_logits: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = value_model(input_ids, attention_mask)
            logits_float = logits.float()
            targets_float = labels.float()

            if error_type == "bce":
                errors = F.binary_cross_entropy_with_logits(logits_float, targets_float, reduction="none")
                preds = torch.sigmoid(logits_float)
            elif error_type == "mae":
                preds = logits_float
                errors = torch.clamp(torch.abs(preds - targets_float), min=0.0, max=1.0)
            elif error_type == "mse":
                preds = logits_float
                errors = torch.clamp((preds - targets_float) ** 2, min=0.0, max=1.0)
            else:
                raise ValueError(f"Unsupported error type: {error_type}")

            all_errors.extend(errors.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())
            all_targets.extend(targets_float.cpu().tolist())
            all_logits.extend(logits_float.cpu().tolist())
            all_questions.extend(batch["question"])

    errors_array = np.array(all_errors, dtype=np.float64)
    mean_error = float(np.mean(errors_array)) if len(errors_array) > 0 else math.nan
    std_error = float(np.std(errors_array)) if len(errors_array) > 0 else math.nan

    sample_lines: List[str] = [
        "Evaluation Summary",
        f"Mean {error_type.upper()} Error: {mean_error:.6f}" if not math.isnan(mean_error) else "Mean Error: undefined",
        f"Std {error_type.upper()} Error: {std_error:.6f}" if not math.isnan(std_error) else "Std Error: undefined",
        "",
        "Samples",
        "",
    ]

    for idx in range(min(sample_count, len(all_questions))):
        question = all_questions[idx]
        pred = all_predictions[idx]
        target = all_targets[idx]
        error_val = all_errors[idx]
        logit_val = all_logits[idx]
        prob_str = ""
        if error_type == "bce":
            prob_str = f" (prob {pred:.6f})"
        sample_lines.extend(
            [
                "Question:",
                question,
                "",
                f"Logit Prediction: {logit_val:.6f}",
                f"Prediction{prob_str}: {pred:.6f}",
                f"Ground Truth: {target:.6f}",
                f"Error ({error_type}): {error_val:.6f}",
                "-" * 60,
                "",
            ]
        )

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("\n".join(sample_lines), encoding="utf-8")

    total = len(all_questions)
    print(f"Evaluated {total} rows.")
    print(f"Mean {error_type.upper()} Error: {mean_error:.6f}" if not math.isnan(mean_error) else "Mean Error: undefined")
    print(f"Std {error_type.upper()} Error: {std_error:.6f}" if not math.isnan(std_error) else "Std Error: undefined")
    print(f"Sample output written to {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a finetuned value function model.")
    parser.add_argument("--model-dir", required=True, help="Directory containing the merged model and value head weights.")
    parser.add_argument("--dataset", required=True, help="Parquet dataset path generated by generate_rollouts.py.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--max-length", type=int, default=10000, help="Maximum sequence length for tokenization.")
    parser.add_argument("--sample-output", required=True, help="Path to save sample predictions.")
    parser.add_argument("--sample-count", type=int, default=20, help="Number of sample rows to record.")
    parser.add_argument("--error-type", choices=["bce", "mae", "mse"], default="mae", help="Error metric to compute.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model dtype.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        model_dir=Path(args.model_dir),
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        sample_path=Path(args.sample_output),
        sample_count=args.sample_count,
        error_type=args.error_type,
        torch_dtype=args.torch_dtype,
    )


if __name__ == "__main__":
    main()
