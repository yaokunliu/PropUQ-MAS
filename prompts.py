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
            if task == "mbppplus":
                return "Follow the solver format above and output the final solution first as markdown python code block(s)."
            return "Follow the solver format above and output the final answer first in the exact required format (including \\boxed{...} when required)."

    if mas_style == "hierarchical":
        if task == "mbppplus":
            if role == "judger":
                return "Follow the Task Summarizer format above and output the final answer first as markdown python code block(s)."
            return "Follow the resposnse format above and output your response first as markdown python code block(s)."
        if role == "judger":
            return "Follow the Task Summarizer format above and output the final answer first in the exact required format (including \\boxed{...} when required)."
        if role in ["planner", "critic", "refiner"]:
            return "Follow the resposnse format above and output your response first in the exact required format (including \\boxed{...} when required)."

    if task == "mbppplus":
        return (
            "Output the required solution first as markdown python code block(s). "
        )
    return (
        "Output the required main answer first in the task/role format above "
        "(including \\boxed{...} when required)."
    )


def _uses_verb(args) -> bool:
    return "Verb" in _selected_uq_methods(args)


def _selected_uq_methods(args) -> list[str]:
    if args is None:
        return []
    value = getattr(args, "local_uncertainty_modes", getattr(args, "local_uncertainty_mode", None))
    if value is None:
        return ["Verb"]
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    out = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out or ["Verb"]


def _build_uncertainty_instruction(
    args,
    *,
    allow_alpha: bool,
    role: str = "",
    mas_style: str = "",
) -> str:
    if args is None:
        return ""
    use_verb = _uses_verb(args)
    use_alpha = allow_alpha and use_verb
    if not use_verb and not use_alpha:
        return ""
    lines = [""]
    if use_verb:
        lines.append("After providing your response according to the requirements, report the following values as numbers in the range [0,1]:")
        lines.append("- Report your own local_uncertainty (local_uncertainty indicates the probability that your current answer or action may be wrong.)")
    elif use_alpha:
        lines.append("After providing your response according to the requirements, report alpha only. Do not report local_uncertainty.")
    if use_alpha:
        lines.append(
            "- Report alpha for each incoming message in your current step. A higher alpha means you largely accept or reuse that message, while a lower alpha means you critique, revise, reject, or only weakly rely on it."
        )
        lines.append(_build_alpha_guidance(args=args, role=role, mas_style=mas_style))
    lines.extend(["", "## After the answer, you MUST append (one line per item):"])
    if use_verb:
        lines.append("<local_uncertainty score=\"FLOAT_0_TO_1\"/>")
    if use_alpha:
        for target in _alpha_targets(role=role, mas_style=mas_style):
            lines.append(
                f"<alpha agent=\"{target}\" score=\"FLOAT_0_TO_1\"/>"
            )

    return "\n".join(lines)


def _alpha_targets(*, role: str = "", mas_style: str = ""):
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


def _build_alpha_guidance(*, args, role: str = "", mas_style: str = "") -> str:
    if args is None:
        return ""

    targets = _alpha_targets(role=role, mas_style=mas_style)
    if not targets:
        return ""

    if len(targets) == 1:
        return f"Score alpha for the message from {targets[0]}."

    if len(targets) == 2:
        target_text = f"{targets[0]} and {targets[1]}"
    else:
        target_text = ", ".join(targets[:-1]) + f", and {targets[-1]}"
    return f"Score alpha for messages from {target_text}."


def _build_alpha_guidance_for_targets(targets) -> str:
    targets = [str(target).strip() for target in (targets or []) if str(target).strip()]
    if not targets:
        return ""
    if len(targets) == 1:
        return f"Score alpha for the message from {targets[0]}."
    if len(targets) == 2:
        return f"Score alpha for messages from {targets[0]} and {targets[1]}."
    return f"Score alpha for messages from {', '.join(targets[:-1])}, and {targets[-1]}."


def _build_uncertainty_instruction_for_targets(args, *, alpha_targets=None) -> str:
    if args is None:
        return ""
    use_verb = _uses_verb(args)
    alpha_targets = [str(target).strip() for target in (alpha_targets or []) if str(target).strip()]
    use_alpha = bool(alpha_targets) and use_verb
    if not use_verb and not use_alpha:
        return ""

    lines = [""]
    if use_verb:
        lines.append("After providing your response according to the requirements, report the following values as numbers in the range [0,1]:")
        lines.append("- Report your own local_uncertainty (local_uncertainty indicates the probability that your current answer or action may be wrong.)")
    elif use_alpha:
        lines.append("After providing your response according to the requirements, report alpha only. Do not report local_uncertainty.")
    if use_alpha:
        lines.append("- Report one alpha for each directly connected incoming agent whose answer was provided to you.")
        lines.append(_build_alpha_guidance_for_targets(alpha_targets))
    lines.extend(["", "## After the answer, you MUST append (one line per item):"])
    if use_verb:
        lines.append("<local_uncertainty score=\"FLOAT_0_TO_1\"/>")
    if use_alpha:
        for target in alpha_targets:
            lines.append(f"<alpha agent=\"{target}\" score=\"FLOAT_0_TO_1\"/>")
    return "\n".join(lines)


def _task_answer_instruction(args) -> str:
    task = getattr(args, "task", None) if args is not None else None
    if task == "mbppplus":
        return (
            "Your answer must be self-contained Python function(s) in markdown code block(s). "
            "Do not put feedback inside code blocks."
        )
    if task == "gsm8k":
        return "Your final answer must appear inside \\boxed{YOUR_FINAL_ANSWER}."
    if task == "medqa":
        return "Your final answer must be selected from A, B, C, D and appear inside \\boxed{YOUR_FINAL_ANSWER}."
    return "Present a clear final answer at the end of your response."


def _task_answer_placeholder(args) -> str:
    task = getattr(args, "task", None) if args is not None else None
    if task == "mbppplus":
        return "```python\nYOUR_PYTHON_CODE\n```"
    if task in ["gsm8k", "medqa"]:
        return "\\boxed{YOUR_FINAL_ANSWER}"
    return "[Your own answer here]"


def build_agent_messages_graph_text_mas(
    *,
    agent_label: str,
    question: str,
    context: str = "",
    incoming_agents=None,
    method=None,
    args=None,
):
    system_message = _system_message_for_model(args)
    assert method in ["mas"], "only for mas method"

    incoming_agents = [str(agent).strip() for agent in (incoming_agents or []) if str(agent).strip()]
    ctx = _truncate_context(context, args.mas_context_length)
    answer_instruction = _task_answer_instruction(args)
    answer_placeholder = _task_answer_placeholder(args)

    if not incoming_agents:
        user_content = f"""
You are {agent_label} in a multi-agent system. Each node in the MAS is one agent.

Solve the input question independently and provide only your own answer. Do not provide feedback because you are the first agent in this information flow.

## Input Question:
{question}

{answer_instruction}

## Format your response as follows:
## Answer
{answer_placeholder}

Now, output your response below.
"""
    else:
        incoming_text = "\n".join(f"- {agent_name}" for agent_name in incoming_agents)
        user_content = f"""
You are {agent_label} in a multi-agent system. Each node in the MAS is one agent.

You are given answers from the directly connected previous agents listed below. Independently verify each incoming answer, decide whether you agree, and identify what should be improved. If you disagree, provide an alternative solution. Then provide feedback and your own answer.

## Input Question:
{question}

## Directly Connected Incoming Agents:
{incoming_text}

## Answers From Directly Connected Incoming Agents:
{ctx}

{answer_instruction}

## Format your response as follows:
## Feedback
- Agent X: [State whether you agree, what is correct, and what should be improved]

## Answer
{answer_placeholder}

Now, output your response below.
"""

    user_content = _attach_uncertainty_instruction(
        user_content,
        _build_uncertainty_instruction_for_targets(args, alpha_targets=incoming_agents),
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


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

        if task == "gsm8k":
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

        elif task == "medqa":
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

        elif task == "mbppplus":
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
            args, allow_alpha=(role != "planner"), role=role, mas_style="sequential"
        ),
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = _system_message_for_model(args)
    
    assert method in ["mas"], "this prompt only for mas method"
    
    if args.task == "gsm8k":
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

    elif args.task == "medqa":
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

    elif args.task == "mbppplus":
        
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

    user_content = _attach_uncertainty_instruction(
        user_content,
        _build_uncertainty_instruction(
            args, allow_alpha=(role == "judger"), role=role, mas_style="hierarchical"
        ),
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_text_mas_gemma(
    role: str, question: str, context: str = "", method=None, args=None
):
    return build_agent_messages_sequential_text_mas(
        role=role, question=question, context=context, method=method, args=args
    )


def build_agent_messages_hierarchical_text_mas_gemma(
    role: str, question: str, context: str = "", method=None, args=None
):
    return build_agent_messages_hierarchical_text_mas(
        role=role, question=question, context=context, method=method, args=args
    )
