import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm.auto import tqdm

from rewards.unified import compute_score


@dataclass
class QueryState:
    index: int
    prompt_messages: List[Dict[str, str]]
    question_text: str
    data_source: Optional[str]
    extra_info: Optional[Dict[str, Any]]
    ground_truth: Any
    prefix_text: str = ""
    rounds: int = 0
    completed: bool = False
    final_answer: Optional[str] = None
    final_score: Optional[float] = None
    finish_reason: Optional[str] = None
    best_answer: Optional[str] = None
    best_value: float = float("-inf")
    best_complete_answer: Optional[str] = None
    best_complete_value: float = float("-inf")


@dataclass
class CandidatePrefix:
    query_idx: int
    text: str
    finish_reason: Optional[str]
    score: float = 0.0
    raw_samples: Optional[List[float]] = None


_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^}]*)\}")


def _extract_boxed_value(text: str) -> Optional[float]:
    matches = _BOXED_PATTERN.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    numeric = re.search(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))
    if numeric is None:
        return None
    try:
        return float(numeric.group())
    except ValueError:
        return None


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


def _extract_question_from_messages(messages: List[Dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def _count_response_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    if "input_ids" not in encoded:
        return 0
    ids = encoded["input_ids"]
    return len(ids) if isinstance(ids, list) else len(ids[0])


def build_base_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]], assistant_prefix: str) -> str:
    convo = list(messages)
    if assistant_prefix:
        convo = convo + [{"role": "assistant", "content": assistant_prefix}]
    return tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)


def build_value_prompt(tokenizer: AutoTokenizer, question: str, prefix_text: str) -> str:
    instructions = (
        "You are given a question and a complete, partial, or empty response that could have been written by you. "
        "Evaluate how likely this response prefix is to lead to a correct final answer. "
        "Reason about the quality of the reasoning and conclude with a numerical score between 0.0 and 1.0. "
        "Output your final scalar score inside \\boxed{}.\n\n"
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        "<response_prefix>\n"
        f"{prefix_text}\n"
        "</response_prefix>\n\n"
        "Provide your analysis followed by the score in \\boxed{}."
    )
    chat = [{"role": "user", "content": instructions}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def _dtype_from_string(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


class DirectValueModel(nn.Module):
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
        return torch.sigmoid(self.value_head(pooled).squeeze(-1))


def _load_dataset(dataset_path: str) -> Dataset:
    ds = Dataset.from_parquet(dataset_path)
    if len(ds) == 0:
        raise ValueError("Dataset is empty.")
    return ds


def _prepare_queries(ds: Dataset) -> List[QueryState]:
    queries: List[QueryState] = []
    for idx, row in enumerate(ds):
        messages = _to_chat_messages(row["prompt"])
        question = _extract_question_from_messages(messages)
        data_source = row.get("data_source")
        extra_info = row.get("extra_info")
        reward_model = row.get("reward_model", {}) if isinstance(row.get("reward_model"), dict) else {}
        ground_truth = reward_model.get("ground_truth")
        if not isinstance(extra_info, dict):
            extra_info = {}
        queries.append(
            QueryState(
                index=idx,
                prompt_messages=messages,
                question_text=question,
                data_source=data_source,
                extra_info=extra_info,
                ground_truth=ground_truth,
            )
        )
    return queries


def _finish_reason_is_complete(reason: Optional[str]) -> bool:
    return reason is not None and reason.lower() in {"stop", "eos_token"}
    #if reason is None:
    #    return False
    #reason_lower = reason.lower()
    #return reason_lower not in {"length", "max_tokens"}


def _evaluate_generative_value(
    candidates: List[CandidatePrefix],
    engine: LLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    k_samples: int,
    max_tokens: int,
    temperature: float,
    lora_request: Optional[LoRARequest] = None,
) -> None:
    if not candidates:
        return
    prompts = [build_value_prompt(tokenizer, questions[c.query_idx], c.text) for c in candidates]
    sampling = SamplingParams(
        n=k_samples,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    outputs = engine.generate(prompts, sampling, lora_request=lora_request)
    for idx in tqdm(range(len(candidates)), desc='Value estimation (verifier)', leave=False):
        cand = candidates[idx]
        out = outputs[idx]
        scores: List[float] = []
        for sample in out.outputs:
            value = _extract_boxed_value(sample.text)
            if value is not None:
                scores.append(value)
            else:
                scores.append(0.0)
        cand.raw_samples = scores if scores else None
        avg_score = float(np.mean(scores)) if scores else 0.0
        cand.score = float(np.clip(avg_score, 0.0, 1.0))


def _evaluate_direct_value(
    candidates: List[CandidatePrefix],
    value_model: DirectValueModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    questions: List[str],
    batch_size: int,
    max_length: int,
) -> None:
    if not candidates:
        return
    texts = [build_value_prompt(tokenizer, questions[c.query_idx], c.text) for c in candidates]
    scores: List[float] = []
    value_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc='Value estimation (value head)', leave=False):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = value_model(enc["input_ids"], enc["attention_mask"])
            scores.extend(logits.detach().cpu().tolist())
    for cand, score in zip(candidates, scores):
        cand.score = float(np.clip(score, 0.0, 1.0))
        cand.raw_samples = None


def _update_best_answer(q_state: QueryState, candidate: CandidatePrefix, is_complete: bool) -> None:
    value = float(np.clip(candidate.score, 0.0, 1.0))
    if value >= q_state.best_value:
        q_state.best_value = value
        q_state.best_answer = candidate.text
    if is_complete and value >= q_state.best_complete_value:
        q_state.best_complete_value = value
        q_state.best_complete_answer = candidate.text


def _finalize_if_needed(q_state: QueryState, reason: str) -> None:
    q_state.completed = True
    q_state.finish_reason = reason
    best_answer = q_state.best_complete_answer or q_state.best_answer or q_state.prefix_text
    best_value = q_state.best_complete_value
    if best_value <= float("-inf"):
        best_value = q_state.best_value
    if best_value <= float("-inf"):
        best_value = None
    q_state.final_answer = best_answer
    q_state.final_score = best_value


def run_generative_mode(
    queries: List[QueryState],
    engine: LLM,
    tokenizer: AutoTokenizer,
    n_trajectories: int,
    state_size: int,
    k_samples: int,
    value_max_tokens: int,
    base_temperature: float,
    value_temperature: float,
    max_total_tokens: int,
    value_lora: Optional[LoRARequest] = None,
) -> List[QueryState]:
    active = list(queries)
    questions = [q.question_text for q in queries]

    max_loops = max(1, (max_total_tokens + max(1, state_size) - 1) // max(1, state_size))
    for loop_step in range(max_loops):
        eligible_queries: List[QueryState] = []
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []

        for q_state in active:
            prefix_tokens = _count_response_tokens(tokenizer, q_state.prefix_text)
            remaining = max_total_tokens - prefix_tokens
            if remaining <= 0:
                q_state.final_answer = q_state.prefix_text
                _finalize_if_needed(q_state, "max_total_tokens")
                continue
            max_tokens = min(state_size, remaining)
            sampling_params.append(
                SamplingParams(
                    n=n_trajectories,
                    max_tokens=max_tokens,
                    temperature=base_temperature,
                )
            )
            prompts.append(build_base_prompt(tokenizer, q_state.prompt_messages, q_state.prefix_text))
            eligible_queries.append(q_state)

        if not eligible_queries:
            break

        print('Round base generation: generating candidate continuations... (loop {}/{})'.format(loop_step + 1, max_loops))
        base_outputs = engine.generate(prompts, sampling_params)

        candidates: List[CandidatePrefix] = []
        per_query_candidates: Dict[int, List[CandidatePrefix]] = {q.index: [] for q in eligible_queries}

        for q_state, result in zip(eligible_queries, base_outputs):
            q_state.rounds += 1
            for out in result.outputs:
                new_text = q_state.prefix_text + out.text if q_state.prefix_text else out.text
                candidate = CandidatePrefix(
                    query_idx=q_state.index,
                    text=new_text,
                    finish_reason=getattr(out, "finish_reason", None),
                )
                per_query_candidates[q_state.index].append(candidate)
                candidates.append(candidate)

        print('Round value estimation: scoring candidate prefixes...')
        _evaluate_generative_value(
            candidates=candidates,
            engine=engine,
            tokenizer=tokenizer,
            questions=questions,
            k_samples=k_samples,
            max_tokens=value_max_tokens,
            temperature=value_temperature,
            lora_request=value_lora,
        )

        next_active: List[QueryState] = []
        for q_state in eligible_queries:
            cand_list = per_query_candidates[q_state.index]
            if not cand_list:
                q_state.final_answer = q_state.prefix_text
                _finalize_if_needed(q_state, "no_candidate")
                continue

            for cand in cand_list:
                is_complete = _finish_reason_is_complete(cand.finish_reason)
                _update_best_answer(q_state, cand, is_complete)

            best_candidate = max(cand_list, key=lambda c: c.score)

            if _finish_reason_is_complete(best_candidate.finish_reason):
                _finalize_if_needed(q_state, best_candidate.finish_reason or "completed")
            else:
                q_state.prefix_text = best_candidate.text
                next_active.append(q_state)

        active = [q for q in next_active if not q.completed]

    for q_state in queries:
        if not q_state.completed:
            _finalize_if_needed(q_state, "search_complete")
        if q_state.best_answer is None:
            q_state.best_answer = q_state.final_answer
            if q_state.best_value <= float("-inf"):
                q_state.best_value = 0.0

    return queries


def run_direct_mode(
    queries: List[QueryState],
    engine: LLM,
    base_tokenizer: AutoTokenizer,
    value_model_dir: Path,
    n_trajectories: int,
    state_size: int,
    base_temperature: float,
    value_batch_size: int,
    value_max_length: int,
    torch_dtype: str,
    max_total_tokens: int,
) -> List[QueryState]:
    value_dtype = _dtype_from_string(torch_dtype)
    backbone = AutoModelForCausalLM.from_pretrained(
        value_model_dir,
        torch_dtype=value_dtype,
        trust_remote_code=True,
    )
    value_model = DirectValueModel(backbone)
    head_path = value_model_dir / "value_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Expected value head weights at {head_path}")
    state_dict = torch.load(head_path, map_location="cpu")
    value_model.value_head.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_model.to(device)
    value_model.eval()

    active = list(queries)
    questions = [q.question_text for q in queries]

    max_loops = max(1, (max_total_tokens + max(1, state_size) - 1) // max(1, state_size))
    for _ in range(max_loops):
        eligible_queries: List[QueryState] = []
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []

        for q_state in active:
            prefix_tokens = _count_response_tokens(base_tokenizer, q_state.prefix_text)
            remaining = max_total_tokens - prefix_tokens
            if remaining <= 0:
                q_state.final_answer = q_state.prefix_text
                _finalize_if_needed(q_state, "max_total_tokens")
                continue
            max_tokens = min(state_size, remaining)
            sampling_params.append(
                SamplingParams(
                    n=n_trajectories,
                    max_tokens=max_tokens,
                    temperature=base_temperature,
                )
            )
            prompts.append(build_base_prompt(base_tokenizer, q_state.prompt_messages, q_state.prefix_text))
            eligible_queries.append(q_state)

        if not eligible_queries:
            break

        print('Round base generation: generating candidate continuations...')
        base_outputs = engine.generate(prompts, sampling_params)

        candidates: List[CandidatePrefix] = []
        per_query_candidates: Dict[int, List[CandidatePrefix]] = {q.index: [] for q in eligible_queries}

        for q_state, result in zip(eligible_queries, base_outputs):
            q_state.rounds += 1
            for out in result.outputs:
                new_text = q_state.prefix_text + out.text if q_state.prefix_text else out.text
                candidate = CandidatePrefix(
                    query_idx=q_state.index,
                    text=new_text,
                    finish_reason=getattr(out, "finish_reason", None),
                )
                per_query_candidates[q_state.index].append(candidate)
                candidates.append(candidate)

        print('Round value estimation: scoring candidate prefixes...')
        _evaluate_direct_value(
            candidates=candidates,
            value_model=value_model,
            tokenizer=base_tokenizer,
            device=device,
            questions=questions,
            batch_size=value_batch_size,
            max_length=value_max_length,
        )

        next_active: List[QueryState] = []
        for q_state in eligible_queries:
            cand_list = per_query_candidates[q_state.index]
            if not cand_list:
                q_state.final_answer = q_state.prefix_text
                _finalize_if_needed(q_state, "no_candidate")
                continue

            for cand in cand_list:
                is_complete = _finish_reason_is_complete(cand.finish_reason)
                _update_best_answer(q_state, cand, is_complete)

            best_candidate = max(cand_list, key=lambda c: c.score)

            if _finish_reason_is_complete(best_candidate.finish_reason):
                _finalize_if_needed(q_state, best_candidate.finish_reason or "completed")
            else:
                q_state.prefix_text = best_candidate.text
                next_active.append(q_state)

        active = [q for q in next_active if not q.completed]

    for q_state in queries:
        if not q_state.completed:
            _finalize_if_needed(q_state, "search_complete")
        if q_state.best_answer is None:
            q_state.best_answer = q_state.final_answer
            if q_state.best_value <= float("-inf"):
                q_state.best_value = 0.0

    return queries


def save_results(output_path: Path, queries: List[QueryState]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for q in queries:
        data.append({
            "index": q.index,
            "question": q.question_text,
            "completed": q.completed,
            "rounds": q.rounds,
            "final_answer": q.final_answer,
            "final_score": q.final_score,
            "finish_reason": q.finish_reason,
            "best_answer": q.best_answer,
            "best_value": None if q.best_value == float("-inf") else q.best_value,
            "best_complete_answer": q.best_complete_answer,
            "best_complete_value": None if q.best_complete_value == float("-inf") else q.best_complete_value,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_reward_metrics(metrics_path: Path, queries: List[QueryState]) -> None:
    rewards: List[float] = []
    for q in queries:
        answer = q.best_complete_answer or q.best_answer or q.final_answer or ""
        data_source = q.data_source or ""
        extra_info = q.extra_info or {}
        try:
            reward = compute_score(data_source, answer, q.ground_truth, extra_info)
        except Exception:
            reward = 0.0
        rewards.append(float(reward))
    rewards_array = np.array(rewards, dtype=np.float64)
    mean_reward = float(np.mean(rewards_array)) if len(rewards_array) > 0 else math.nan
    std_reward = float(np.std(rewards_array)) if len(rewards_array) > 0 else math.nan
    metrics_lines = [
        f"Queries: {len(rewards)}",
        f"Mean reward: {mean_reward:.6f}" if not math.isnan(mean_reward) else "Mean reward: undefined",
        f"Std reward: {std_reward:.6f}" if not math.isnan(std_reward) else "Std reward: undefined",
    ]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("\n".join(metrics_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test-time scaling with generative or direct value functions.")
    parser.add_argument("--mode", choices=["generative", "direct"], default="generative")
    parser.add_argument("--dataset", default="/pscratch/sd/s/siddart2/data/aime/train.parquet", help="Parquet dataset containing prompts.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507", help="Base model name or path for trajectory generation.")
    parser.add_argument("--value-lora", help="Path to a LoRA adapter for the generative value function.")
    parser.add_argument("--value-model-dir", help="Directory with merged backbone and value head for direct scoring.")
    parser.add_argument("--output-path", default="/pscratch/sd/s/siddart2/gql_evals/evals/gvf/aime.json", help="File to save JSON results.")
    parser.add_argument("--metrics-output", default="/pscratch/sd/s/siddart2/gql_evals/evals/gvf/aime_metrics.txt", help="Path to save reward metrics text file.")
    parser.add_argument("--n-trajectories", type=int, default=4, help="Number of trajectories per query per round.")
    parser.add_argument("--state-size", type=int, default=1024, help="Max tokens generated per round by the base model.")
    parser.add_argument("--k-samples", type=int, default=4, help="Samples per prefix for generative value estimation.")
    parser.add_argument("--value-max-tokens", type=int, default=4096, help="Max tokens for generative value responses.")
    parser.add_argument("--value-temperature", type=float, default=1.0, help="Temperature for generative value sampling.")
    parser.add_argument("--base-temperature", type=float, default=1.0, help="Temperature for base model sampling.")
    parser.add_argument("--value-batch-size", type=int, default=8, help="Batch size for direct value scoring.")
    parser.add_argument("--value-max-length", type=int, default=4096, help="Token limit for direct value tokenizer.")
    parser.add_argument("--max-total-tokens", type=int, default=8192, help="Maximum total response tokens per query.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size for vLLM engines.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Dtype for direct value model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = _load_dataset(args.dataset)
    queries = _prepare_queries(ds)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    enable_lora = args.value_lora is not None
    engine = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8 if args.mode=="generative" else 0.5,
        trust_remote_code=True,
        enable_lora=enable_lora,
    )

    value_lora_request = None
    if args.value_lora:
        value_lora_request = LoRARequest("value_adapter", 1, args.value_lora)

    if args.mode == "generative":
        results = run_generative_mode(
            queries=queries,
            engine=engine,
            tokenizer=base_tokenizer,
            n_trajectories=args.n_trajectories,
            state_size=args.state_size,
            k_samples=args.k_samples,
            value_max_tokens=args.value_max_tokens,
            base_temperature=args.base_temperature,
            value_temperature=args.value_temperature,
            max_total_tokens=args.max_total_tokens,
            value_lora=value_lora_request,
        )
    else:
        if not args.value_model_dir:
            raise ValueError("--value-model-dir is required in direct mode.")
        value_model_dir = Path(args.value_model_dir)
        results = run_direct_mode(
            queries=queries,
            engine=engine,
            base_tokenizer=base_tokenizer,
            value_model_dir=value_model_dir,
            n_trajectories=args.n_trajectories,
            state_size=args.state_size,
            base_temperature=args.base_temperature,
            value_batch_size=args.value_batch_size,
            value_max_length=args.value_max_length,
            torch_dtype=args.torch_dtype,
            max_total_tokens=args.max_total_tokens,
        )

    save_results(Path(args.output_path), results)
    compute_reward_metrics(Path(args.metrics_output), results)


if __name__ == "__main__":
    main()
