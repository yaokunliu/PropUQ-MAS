# Node-Level Medical QA Judge Prompt Template

This template evaluates whether an intermediate node in a multi-agent system is medically valid and useful. Replace every value enclosed in braces with the corresponding input. The placeholders must not contain personally identifiable information.

## System Prompt

```text
You are a strict but fair medical QA judge. Return valid JSON only. /no_think
```

## User Prompt

```text
You are an expert medical QA judge.

Task question:
{TASK_QUESTION}

Gold answer option: {GOLD_ANSWER_OPTION}
Final MAS prediction: {FINAL_MAS_PREDICTION}

Previous intermediate context:
{PREVIOUS_INTERMEDIATE_CONTEXT}

Node to judge: {NODE_ROLE}
{NODE_OUTPUT}

Judging instruction:
{ROLE_SPECIFIC_JUDGING_INSTRUCTION}

Return only a JSON object with these fields:
- "label": one of "helpful", "mixed", "harmful"
- "node_error": true if the node contains a substantive medical/reasoning error likely to mislead downstream agents, otherwise false
- "confidence": a number from 0 to 1
- "rationale": one concise sentence

/no_think
```

## Role-Specific Judging Instructions

### Critic

```text
Evaluate whether the Critic output gives a medically valid critique of the previous plan. It should identify real issues or confirm correct reasoning without introducing a substantive error.
```

### Refiner

```text
Evaluate whether the Refiner output produces a medically valid refined plan. It should preserve or improve the reasoning and should not introduce a substantive error.
```

### Generic Intermediate Node

```text
Evaluate whether this intermediate MAS node is medically valid and useful.
```

## Expected Output Schema

```json
{
  "label": "helpful",
  "node_error": false,
  "confidence": 0.95,
  "rationale": "The node provides medically valid reasoning without introducing a substantive error."
}
```

## Placeholder Notes

- `{TASK_QUESTION}`: The complete multiple-choice medical question, including answer options.
- `{GOLD_ANSWER_OPTION}`: The reference answer option.
- `{FINAL_MAS_PREDICTION}`: The final answer produced by the multi-agent system.
- `{PREVIOUS_INTERMEDIATE_CONTEXT}`: Outputs from nodes preceding the node under evaluation; use `[none]` when no prior context exists.
- `{NODE_ROLE}`: The role of the evaluated node, such as `Critic` or `Refiner`.
- `{NODE_OUTPUT}`: The complete output of the node under evaluation.
- `{ROLE_SPECIFIC_JUDGING_INSTRUCTION}`: One of the role-specific instructions above.
