from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    index: int
    name: str
    role: str


def default_agents(node_num: int = 4) -> List[Agent]:
    return [
        Agent(index=idx, name=f"Agent {idx + 1}", role="agent")
        for idx in range(node_num)
    ]


__all__ = ["Agent", "default_agents"]
