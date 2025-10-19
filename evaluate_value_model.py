import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^}]*)\}")
_QUESTION_PATTERN = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)


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
    match = _QUESTION_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_boxed_value(text: str) -> Optional[float]:
    matches = _BOXED_PATTERN.findall(text)
    if not matches:
        return None
    content = matches[-1]
    numeric = re.search(r"-?\d+(?:\.\d+)?", content.replace(",", ""))
    if not numeric:
        return None
    try:
        return float(numeric.group())
    except ValueError:
        return None


def _batch_iter(dataset: Dataset, batch_size: int) -> Iterable[Tuple[int, int, Dict[str, List[Any]]]]:
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end, dataset[start:end]


def run_evaluation(
    model_name: str,
    dataset_path: str,
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    dtype: str,
    seed: int,
    batch_size: int,
    sample_path: Path,
    sample_count: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype=dtype,
        trust_remote_code=True,
        seed=seed,
    )
    ds = Dataset.from_parquet(dataset_path)

    sampling = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    predictions: List[float] = []
    ground_truths: List[float] = []
    sample_records: List[Tuple[str, str, Optional[float], Optional[float]]] = []
    missing_boxed: int = 0

    for start, end, batch in _batch_iter(ds, batch_size):
        batch_len = end - start
        prompts = batch["prompt"]
        reward_models = batch.get("reward_model", [{} for _ in range(batch_len)])

        rendered_prompts: List[str] = []
        batch_questions: List[str] = []
        batch_targets: List[float] = []

        for idx in range(batch_len):
            prompt_field = prompts[idx]
            messages = _to_chat_messages(prompt_field)
            rendered_prompts.append(_render_prompt(tokenizer, messages))

            question_source = messages[0]["content"] if messages else ""
            batch_questions.append(_extract_question(question_source))

            reward_cell = reward_models[idx] if idx < len(reward_models) else {}
            if isinstance(reward_cell, dict):
                gt_value = reward_cell.get("ground_truth", None)
            else:
                gt_value = None
            batch_targets.append(float(gt_value) if gt_value is not None else math.nan)

        outputs = llm.generate(rendered_prompts, sampling)

        for idx, out in enumerate(outputs):
            text = out.outputs[0].text
            pred_value = _extract_boxed_value(text)
            target = batch_targets[idx]
            if pred_value is None:
                missing_boxed += 1
            predictions.append(pred_value if pred_value is not None else math.nan)
            ground_truths.append(target)

            if len(sample_records) < sample_count:
                sample_records.append((batch_questions[idx], text.strip(), pred_value, target))

    preds_array = np.array(predictions, dtype=np.float64)
    preds_array = np.nan_to_num(preds_array, nan=0.0)
    targets_array = np.array(ground_truths, dtype=np.float64)
    abs_errors = np.clip(np.abs(preds_array - targets_array), a_min=0.0, a_max=1.0)
    valid = ~np.isnan(abs_errors)

    if np.any(valid):
        mean_error = float(np.nanmean(abs_errors))
        std_error = float(np.nanstd(abs_errors))
    else:
        mean_error = math.nan
        std_error = math.nan

    summary_lines = [
        "Evaluation Summary",
        f"Mean Absolute Error: {mean_error:.6f}" if not math.isnan(mean_error) else "Mean Absolute Error: undefined",
        f"Std Absolute Error: {std_error:.6f}" if not math.isnan(std_error) else "Std Absolute Error: undefined",
        "",
        "Samples",
        "",
    ]

    sample_lines: List[str] = []
    for question, response, pred, target in sample_records:
        pred_str = "None" if pred is None else f"{pred:.6f}"
        target_str = "None" if target is None or math.isnan(target) else f"{target:.6f}"
        err_str = "None"
        if pred is not None and target is not None and not math.isnan(target):
            err_str = f"{abs(pred - target):.6f}"
        sample_lines.extend(
            [
                "Question:",
                question,
                "",
                "Response:",
                response,
                "",
                f"Predicted: {pred_str}",
                f"Ground Truth: {target_str}",
                f"Absolute Error: {err_str}",
                "-" * 60,
                "",
            ]
        )

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("\n".join(summary_lines + sample_lines), encoding="utf-8")

    total = len(ds)
    evaluated = int(np.sum(valid))
    print(f"Evaluated {evaluated} / {total} rows with extractable boxed predictions.")
    print(f"Missing boxed answers: {missing_boxed}")
    print(
        f"Mean absolute error: {mean_error:.6f}" if not math.isnan(mean_error) else "Mean absolute error: undefined"
    )
    print(f"Std absolute error: {std_error:.6f}" if not math.isnan(std_error) else "Std absolute error: undefined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a generative value model on stored value prompts.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name or path for the value function LLM.")
    parser.add_argument("--dataset", default="/pscratch/sd/s/siddart2/gql_data/qwen4b/aime.parquet", help="Path to the parquet dataset generated by generate_rollouts.py.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Number of tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallel size for vLLM.")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"], help="Model dtype.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of prompts to evaluate per batch.")
    parser.add_argument(
        "--sample-output",
        default="/pscratch/sd/s/siddart2/gql_evals/td/aime_value_samples.txt",
        help="File to save example questions and responses.",
    )
    parser.add_argument("--sample-count", type=int, default=20, help="Number of examples to save in the sample output file.")
    return parser.parse_args()


def main():
    args = parse_args()
    sample_path = Path(args.sample_output)
    run_evaluation(
        model_name=args.model,
        dataset_path=args.dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tp_size,
        dtype=args.dtype,
        seed=args.seed,
        batch_size=args.batch_size,
        sample_path=sample_path,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
