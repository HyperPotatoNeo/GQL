import re
import json
from typing import Optional

from reasoning_gym.factory import get_score_answer_fn
from math_verify import parse, verify


def get_task_name(data_source: str) -> str:
    """Infer task type from a dataset source identifier.

    Returns one of: "math", "rg".
    """
    if (
        ("aime" in data_source)
        or ("hmmt" in data_source)
        or ("MATH" in data_source)
        or ("DeepScaleR" in data_source)
        or ("agentica-org/DeepScaleR-Preview-Dataset" in data_source)
    ):
        return "math"
    if "reasoning_gym" in data_source:
        return "rg"
    raise ValueError(f"Unknown task for data_source: {data_source}")


# --------------------- math ---------------------
def is_equiv(str1, str2, verbose=False):
    if '$' not in str1:
        str1 = '$' + str1 + '$'
    if '$' not in str2:
        str2 = '$' + str2 + '$'
    gold = parse(str2)
    pred = parse(str1)
    return verify(gold, pred)


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval


def compute_math_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


# --------------------- reasoning gym ---------------------
def extract_rg_solution(completion: str) -> Optional[str]:
    """Extract the model's predicted answer for reasoning-gym style prompts.

    Priority order:
      1) <answer> ... </answer>
      2) text after "Final Answer:" following an optional "</think>" tag
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    parts = completion.split("</think>", 1)
    if len(parts) == 1:
        return None

    tail = parts[1].strip()
    final_response = tail.rsplit("Final Answer:", 1)
    if len(final_response) == 1:
        return None

    return final_response[1].strip()


def compute_rg_score(solution_str, extra_info):
    if not extra_info or "dataset_name" not in extra_info or "entry" not in extra_info:
        raise ValueError("extra_info with keys 'dataset_name' and 'entry' is required for reasoning_gym scoring")

    answer = extract_rg_solution(solution_str) or ""
    score_answer_fn = get_score_answer_fn(name=extra_info["dataset_name"])
    return float(score_answer_fn(answer=answer, entry=json.loads(extra_info["entry"])))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
) -> float:
    """Unified reward function for math and reasoning-gym.

    Args:
        data_source: dataset identifier used to infer task type
        solution_str: model-produced solution text
        ground_truth: ground-truth answer (string for math; unused for RG)
        extra_info: additional info for RG containing keys {"dataset_name", "entry"}

    Returns:
        Floating-point reward in [0, 1].
    """
    task = get_task_name(str(data_source))

    if task == "rg":
        return compute_rg_score(solution_str, extra_info)
    elif task == "math":
        return compute_math_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Unknown task: {task}")