import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_last_boxed_content(text: str) -> Optional[str]:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    content = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            content.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content).strip()
            content.append(ch)
        else:
            content.append(ch)
        i += 1
    return None


def extract_mcq_choice(text: str, choices: str = "abcd") -> Optional[str]:
    allowed = {c.lower() for c in choices}

    def _normalize_candidate(src: str) -> str:
        s = src
        # Unwrap simple latex text blocks like \text{A. }
        for _ in range(3):
            new_s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
            if new_s == s:
                break
            s = new_s
        s = s.replace("~", " ")
        return s.strip()

    boxed = extract_last_boxed_content(text)
    if boxed:
        s = _normalize_candidate(boxed)
        match = re.search(r"(?i)^\s*[^a-z0-9]*([a-z])(?:\b|[^a-z])", s)
        if match:
            letter = match.group(1).lower()
            if letter in allowed:
                return letter

    keyword_patterns = [
        r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*\**\s*([a-z])\b",
        r"(?i)\b([a-z])\s*[\).,:]\s*",
    ]
    for pattern in keyword_patterns:
        matches = re.findall(pattern, text)
        for m in reversed(matches):
            letter = m.lower()
            if letter in allowed:
                return letter

    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def strip_structured_uncertainty_blocks(text: str) -> str:
    out = re.sub(
        r"<agent_uncertainty\b[^>]*>.*?</agent_uncertainty>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(r"<agent_uncertainty\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<peer_influence\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<message_uncertainty\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<message_adoption\b[^>]*/>", "", out, flags=re.IGNORECASE)
    # Backward compatibility: also strip legacy confidence tags.
    out = re.sub(r"<agent_confidence\b[^>]*>.*?</agent_confidence>", "", out, flags=re.IGNORECASE | re.DOTALL)
    out = re.sub(r"<agent_confidence\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<message_confidence\b[^>]*/>", "", out, flags=re.IGNORECASE)

    # Plain-text fallback formats.
    out = re.sub(
        r"(?im)^[ \t>*`-]*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty|message\s+adoption)\s*[:=]\s*[^\n\r]*$",
        "",
        out,
    )
    out = re.sub(
        r"(?im)^[ \t>*`-]*message[_\s-]*adoption[_\s-]*weight\s*[:=]\s*[^\n\r]*$",
        "",
        out,
    )
    tag_group = (
        "evidence_gap|memory_gap|reasoning_gap|ambiguous_task|context_conflict|tool_risk|weak_signal|guess|"
        "verify|critique|selective_use|balanced_use|follow|execute|refine|low_relevance"
    )
    out = re.sub(
        rf"(?im)^[ \t>*`-]*tag\s*[:=]\s*(?:{tag_group})\s*$",
        "",
        out,
    )
    return out


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)
