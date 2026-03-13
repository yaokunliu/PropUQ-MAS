def _truncate_context(context: str, max_len: int) -> str:
    if max_len is None or max_len <= 0:
        return context
    return context[:max_len]


def _system_message_for_model(args) -> str:
    model_name = str(getattr(args, "model_name", "") if args is not None else "").lower()
    if "qwen" in model_name:
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    if "gemma" in model_name:
        return "You are Gemma, created by Google. You are a helpful assistant."
    if "ministral" in model_name or "mistral" in model_name:
        return "You are Ministral, created by Mistral AI. You are a helpful assistant."
    return "You are a helpful assistant."


def _answer_format_reminder(args, *, role: str = "", mas_style: str = "") -> str:
    task = getattr(args, "task", None) if args is not None else None
    role = (role or "").lower()
    mas_style = (mas_style or "").lower()

    if mas_style == "sequential":
        if role == "planner":
            return "Follow the planner format above and output the plan first."
        if role == "critic":
            return "Follow the critic format above and output `Original Plan:` and `Feedback:` first."
        if role == "refiner":
            return "Follow the refiner format above and output the refined plan first."
        if role == "judger":
            if task in ["mbppplus", "humanevalplus"]:
                return "Follow the solver format above and output the final solution first as markdown python code block(s)."
            return "Follow the solver format above and output the final answer first in the exact required format (including \\boxed{...} when required)."

    if mas_style == "hierarchical":
        if task in ["mbppplus", "humanevalplus"]:
            if role == "judger":
                return "Follow the Task Summarizer format above and output the final answer first as markdown python code block(s)."
            return "Follow the resposnse format above and output your response first as markdown python code block(s)."
        if role == "judger":
            return "Follow the Task Summarizer format above and output the final answer first in the exact required format (including \\boxed{...} when required)."
        if role in ["planner", "critic", "refiner"]:
            return "Follow the resposnse format above and output your response first in the exact required format (including \\boxed{...} when required)."

    if task in ["mbppplus", "humanevalplus"]:
        return (
            "Output the required solution first as markdown python code block(s). "
        )
    return (
        "Output the required main answer first in the task/role format above "
        "(including \\boxed{...} when required)."
    )


def _build_uncertainty_instruction(
    args,
    *,
    allow_message_adoption: bool,
    role: str = "",
    mas_style: str = "",
) -> str:
    if args is None:
        return ""
    score_mode = getattr(args, "uncertainty_mode", None)
    if score_mode not in {"anchor", "continuous"}:
        return ""
    if score_mode == "continuous":
        lines = ["", "After providing your response according to the requirements, report the following values as numbers in the range [0,1]:"]
        lines.append("- Report your own self_uncertainty (self_uncertainty indicates the probability that your current answer or action may be wrong.)")
        if allow_message_adoption:
            lines.append(
                "- Report your message_adoption_weight (message_adoption_weight indicates the percentage to which you agree with an incoming message in your current step. A higher value means you largely follow its requirements or content, while a lower value means you rely less on it and instead critique, verify, filter, or only partially use it.)"
            )
            lines.append(_build_message_adoption_guidance(args=args, role=role, mas_style=mas_style))
        # lines.extend(
        #     [
        #         "",
        #         "Soft ranges for self_uncertainty:",
        #     "- [0.00, 0.10]: almost certainly correct",
        #     "- (0.10, 0.30]: low self_uncertainty",
        #     "- (0.30, 0.70): meaningful self_uncertainty",
        #     "- [0.70, 0.90): high self_uncertainty",
        #     "- [0.90, 1.00]: almost certainly wrong",
        #     "",
        #     "Soft ranges for message_adoption_weight:",
        #     "- [0.00, 0.10]: almost no reliance",
        #     "- (0.10, 0.30]: weak reliance",
        #     "- (0.30, 0.70): partial reliance",
        #     "- [0.70, 0.90): strong reliance",
        #     "- [0.90, 1.00]: near-total reliance",
        #     "",
        #     ]
        # )
        lines.extend(
            [
                "",
                "Soft ranges for self_uncertainty:",
                "- [0.00, 0.05]: almost certainly correct; use only in rare boundary cases",
                "- (0.05, 0.20]: low self_uncertainty",
                "- (0.20, 0.60]: moderate self_uncertainty (default range)",
                "- (0.60, 0.85]: high self_uncertainty",
                "- (0.85, 1.00]: almost certainly wrong; use only in rare boundary cases",
                "",
                "Soft ranges for message_adoption_weight:",
                "- [0.00, 0.10]: almost no reliance",
                "- (0.10, 0.30]: weak reliance",
                "- (0.30, 0.70): partial reliance",
                "- [0.70, 0.90): strong reliance",
                "- [0.90, 1.00]: near-total reliance",
                "",
            ]
        )
        lines.extend(
            [
                "",
                "## After the answer, you MUST append (one line per item):",
                "<agent_uncertainty score=\"FLOAT_0_TO_1\"/>",
            ]
        )
        if allow_message_adoption:
            for target in _message_adoption_targets(role=role, mas_style=mas_style):
                lines.append(
                    f"<message_adoption agent=\"{target}\" score=\"FLOAT_0_TO_1\"/>"
                )
    else:
        lines = ["", "After providing your response according to the requirements, report the following using only one of the allowed values [\"VL\", \"L\", \"M\", \"H\", \"VH\"] (VL = very low, L = low, M = medium, H = high, VH = very high):"]
        lines.append("- report your own self_uncertainty (self_uncertainty indicates the level at which your current answer or action may be wrong).")
        if allow_message_adoption:
            lines.append(
                "- report a message_adoption score for each specified agent whose message was provided to you (message_adoption_weight indicates the level to which you agree with an incoming message in your current step. A higher value means you largely follow its requirements or content, while a lower value means you rely less on it and instead critique, verify, filter, or only partially use it.)"
            )
            lines.append(_build_message_adoption_guidance(args=args, role=role, mas_style=mas_style))
        lines.extend(
            [
                "",
            "self_uncertainty anchors:",
            "- VL: task is clear and evidence is sufficient",
            "- L: mostly reliable, with a small local gap",
            "- M: mixed confidence with real chance of error",
            "- H: significant uncertainty remains because of missing evidence, weak reasoning, or unclear context",
            "- VH: highly unreliable",
            "",
            "message_adoption_weight anchors:",
            "- VL: barely rely on incoming message; mostly inspect/challenge/correct",
            "- L: weak reference only",
            "- M: partial reliance with independent judgment",
            "- H: important input for this step",
            "- VH: strongly follow incoming message",
            "",
            ]
        )
        lines.extend(
            [
                "",
                "## After the answer, you MUST append (one line per item):",
                "<agent_uncertainty level=\"ONE_OF_VL_L_M_H_VH\"/>",
            ]
        )
        if allow_message_adoption:
            for target in _message_adoption_targets(role=role, mas_style=mas_style):
                lines.append(
                    f"<message_adoption agent=\"{target}\" level=\"ONE_OF_VL_L_M_H_VH\"/>"
                )

    return "\n".join(lines)


def _message_adoption_targets(*, role: str = "", mas_style: str = ""):
    role = (role or "").lower()
    mas_style = (mas_style or "").lower()

    if mas_style == "sequential":
        if role == "critic":
            return ["Planner Agent"]
        if role == "refiner":
            return ["Critic Agent"]
        if role == "judger":
            return ["Refiner Agent"]

    if mas_style == "hierarchical" and role == "judger":
        return ["Math Agent", "Science Agent", "Code Agent"]

    return []


def _build_message_adoption_guidance(*, args, role: str = "", mas_style: str = "") -> str:
    if args is None or getattr(args, "uncertainty_mode", None) not in {"anchor", "continuous"}:
        return ""

    targets = _message_adoption_targets(role=role, mas_style=mas_style)
    if not targets:
        return ""

    if len(targets) == 1:
        return f"Score the message_adoption_weight for the message from {targets[0]}."

    if len(targets) == 2:
        target_text = f"{targets[0]} and {targets[1]}"
    else:
        target_text = ", ".join(targets[:-1]) + f", and {targets[-1]}"
    return f"Score the message_adoption_weight for messages from {target_text}."


def _attach_uncertainty_instruction(prompt_text: str, uncertainty_text: str) -> str:
    if not uncertainty_text:
        return prompt_text

    text = prompt_text.rstrip()
    after_answer_marker = "## After the answer, you MUST append (one line per item):"
    split_idx = uncertainty_text.find(after_answer_marker)
    if split_idx != -1:
        uncertainty_intro = uncertainty_text[:split_idx].rstrip()
        uncertainty_appendix = uncertainty_text[split_idx:].strip()
    else:
        uncertainty_intro = uncertainty_text.strip()
        uncertainty_appendix = ""

    format_marker = "## Format your response as follows:"
    format_idx = text.find(format_marker)
    if format_idx != -1:
        head = text[:format_idx].rstrip()
        tail = text[format_idx:]
        output_markers = [
            "Now, output your response below.",
            "Now, output your refined plan below.",
            "Now output your plan to solve the question below.",
        ]
        appendix_insert_idx = -1
        for marker in output_markers:
            appendix_insert_idx = tail.find(marker)
            if appendix_insert_idx != -1:
                break

        if appendix_insert_idx != -1:
            format_block = tail[:appendix_insert_idx].rstrip()
            output_tail = tail[appendix_insert_idx:].lstrip()
            parts = [head]
            if uncertainty_intro:
                parts.append(uncertainty_intro)
            parts.append(format_block)
            if uncertainty_appendix:
                parts.append(uncertainty_appendix)
            parts.append(output_tail)
            return "\n\n".join(part for part in parts if part) + "\n"

    insertion_markers = [
        "Now, output your response below.",
        "Now, output your refined plan below.",
        "Now output your plan to solve the question below.",
        "Give your response below.",
        "Now, reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}.",
        "Now, reason step by step and present your final answer clearly at the end.",
        "Now, reason step by step and output the final answer:",
    ]

    for marker in insertion_markers:
        idx = text.find(marker)
        if idx != -1:
            head = text[:idx].rstrip()
            tail = text[idx:].lstrip()
            parts = [head]
            if uncertainty_intro:
                parts.append(uncertainty_intro)
            if uncertainty_appendix:
                parts.append(uncertainty_appendix)
            parts.append(tail)
            return "\n\n".join(part for part in parts if part) + "\n"

    parts = [text]
    if uncertainty_intro:
        parts.append(uncertainty_intro)
    if uncertainty_appendix:
        parts.append(uncertainty_appendix)
    return "\n\n".join(part for part in parts if part) + "\n"


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = _system_message_for_model(args)

    assert method in ["mas"], "only for mas method"

    # truncate context if needed
    ctx = _truncate_context(context, args.mas_context_length)

    if role == "planner":
        user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below.
"""

    elif role == "critic":
        user_content = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below.
"""

    elif role == "refiner":
        user_content = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below.
"""

    elif role == "judger":
        task = getattr(args, "task", None)

        if task in ["gsm8k", "aime2024", "aime2025"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["mbppplus", "humanevalplus"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
"""
            
        elif task in ["winogrande"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
    else:
        user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and present your final answer clearly at the end.
"""

    user_content = _attach_uncertainty_instruction(
        user_content,
        _build_uncertainty_instruction(
            args, allow_message_adoption=(role != "planner"), role=role, mas_style="sequential"
        ),
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = _system_message_for_model(args)
    
    assert method in ["mas"], "this prompt only for mas method"
    
    if args.task in ['gsm8k', 'aime2024', 'aime2025']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Give your response below.
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Give your response below.
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Give your response below.       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Content from Previous Agent:
{_truncate_context(context, args.mas_context_length)}

Give your response below.
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Give your response below.
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Give your response below.
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Give your response below.       
"""
        elif role == "judger":

            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Content from Previous Agent:
{_truncate_context(context, args.mas_context_length)}

Give your response below.
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Give your response below.
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Give your response below.
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Give your response below.
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer in markdown python code block.

Input Question: {question}

Content from Previous Agent:
{_truncate_context(context, args.mas_context_length)}

Give your response below.
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Give your response below.
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Give your response below.
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Give your response below.       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Content from Previous Agent:
{_truncate_context(context, args.mas_context_length)}

"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Give your response below.
"""

    user_content = _attach_uncertainty_instruction(
        user_content,
        _build_uncertainty_instruction(
            args, allow_message_adoption=(role == "judger"), role=role, mas_style="hierarchical"
        ),
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_single_agent(question: str, args=None):

    system_message = _system_message_for_model(args)

    assert args.method in ["single_agent"], "this prompt only for single_agent method"

    task = args.task

    if task in ["gsm8k", "aime2024", "aime2025"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    user_content = _attach_uncertainty_instruction(
        user_content,
        _build_uncertainty_instruction(
            args, allow_message_adoption=False, role="singleagent", mas_style="single"
        ),
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_text_mas_gemma(
    role: str, question: str, context: str = "", method=None, args=None
):
    # Alias to keep compatibility with existing imports/routes.
    return build_agent_messages_sequential_text_mas(
        role=role, question=question, context=context, method=method, args=args
    )


def build_agent_messages_hierarchical_text_mas_gemma(
    role: str, question: str, context: str = "", method=None, args=None
):
    # Alias to keep compatibility with existing imports/routes.
    return build_agent_messages_hierarchical_text_mas(
        role=role, question=question, context=context, method=method, args=args
    )


def build_agent_messages_single_agent_gemma(question: str, args=None):
    # Alias to keep compatibility with existing imports/routes.
    return build_agent_messages_single_agent(question=question, args=args)
