from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MASEdge:
    source: int
    target: int


@dataclass
class MASGraph:
    topology: str
    node_num: int
    edges: list[MASEdge]

    def edge_strings(self) -> list[str]:
        return [f"{edge.source}->{edge.target}" for edge in self.edges]

    def incoming(self, node_idx: int) -> list[int]:
        return sorted(edge.source for edge in self.edges if edge.target == node_idx)

    def outgoing(self, node_idx: int) -> list[int]:
        return sorted(edge.target for edge in self.edges if edge.source == node_idx)

    def levels(self) -> list[int]:
        levels = [0 for _ in range(self.node_num)]
        for node_idx in range(self.node_num):
            incoming_nodes = self.incoming(node_idx)
            if incoming_nodes:
                levels[node_idx] = max(levels[parent] for parent in incoming_nodes) + 1
        return levels

    def edge_count(self) -> int:
        return len(self.edges)

    def density(self) -> float:
        if self.node_num <= 1:
            return 0.0
        return self.edge_count() / (self.node_num * (self.node_num - 1))

    def step_to_agents(self) -> dict[int, list[str]]:
        step_map: dict[int, list[str]] = {}
        for node_idx, level in enumerate(self.levels()):
            step_map.setdefault(int(level), []).append(f"Agent {node_idx + 1}")
        return step_map

    def to_metadata(self) -> dict:
        levels = self.levels()
        return {
            "topology": self.topology,
            "node_num": self.node_num,
            "edge_count": self.edge_count(),
            "density": self.density(),
            "steps": {f"step_{step}": agents for step, agents in self.step_to_agents().items()},
            "edges": self.edge_strings(),
            "incoming": {
                f"Agent {idx + 1}": [f"Agent {src + 1}" for src in self.incoming(idx)]
                for idx in range(self.node_num)
            },
            "outgoing": {
                f"Agent {idx + 1}": [f"Agent {dst + 1}" for dst in self.outgoing(idx)]
                for idx in range(self.node_num)
            },
            "levels": {f"Agent {idx + 1}": int(levels[idx]) for idx in range(self.node_num)},
        }


def build_mas_graph(topology: str, node_num: int) -> MASGraph:
    if node_num < 1:
        raise ValueError("mas_node_num must be at least 1.")

    edges: list[MASEdge] = []
    if topology == "sequential":
        edges = [MASEdge(idx, idx + 1) for idx in range(node_num - 1)]
    elif topology == "hierarchical":
        final_node = node_num - 1
        edges = [MASEdge(idx, final_node) for idx in range(node_num - 1)]
    elif topology == "decentralized":
        edges = [MASEdge(u, v) for u in range(node_num) for v in range(u + 1, node_num)]
    else:
        raise ValueError(f"Unsupported MAS topology: {topology}")

    edge_pairs = {(edge.source, edge.target) for edge in edges}
    final_node = node_num - 1
    for node_idx in range(node_num - 1):
        has_outgoing = any(source == node_idx for source, _ in edge_pairs)
        if not has_outgoing and node_idx != final_node:
            edge_pairs.add((node_idx, final_node))
    normalized_edges = [MASEdge(source, target) for source, target in sorted(edge_pairs)]
    return MASGraph(topology=topology, node_num=node_num, edges=normalized_edges)


def graphviz_rendering_available() -> tuple[bool, str | None]:
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return False, "Python package `graphviz` is not installed."
    if shutil.which("dot") is None:
        return False, "`dot` executable is not available in PATH."
    return True, None


def render_graph_svg(graph: MASGraph, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from graphviz import Digraph

    dot = Digraph(name=f"mas_{graph.topology}_{graph.node_num}", format="svg")
    dot.attr(rankdir="LR", splines="spline", bgcolor="white", nodesep="0.5", ranksep="0.9")
    dot.attr("node", shape="circle", style="filled", fillcolor="#f8fafc", color="#111827", fontname="Arial", fontsize="16", width="0.7", fixedsize="true")
    dot.attr("edge", color="#4b5563", penwidth="1.6", arrowsize="0.8")

    levels = graph.levels()
    level_nodes: dict[int, list[int]] = {}
    for node_idx, level in enumerate(levels):
        level_nodes.setdefault(level, []).append(node_idx)

    for level, nodes in sorted(level_nodes.items()):
        with dot.subgraph(name=f"cluster_rank_{level}") as subgraph:
            subgraph.attr(rank="same")
            subgraph.attr(color="white")
            for node_idx in nodes:
                subgraph.node(str(node_idx), label=str(node_idx + 1))

    for edge in graph.edges:
        dot.edge(str(edge.source), str(edge.target))

    dot_path = output_path.with_suffix(".dot")
    dot_path.write_text(dot.source, encoding="utf-8")
    dot.render(outfile=str(output_path), cleanup=True)
    return str(dot_path)


def export_mas_graph_artifacts(graph: MASGraph, output_dir: Path, *, render_svg: bool = False) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = graph.to_metadata()
    json_path = output_dir / "mas_graph.json"
    txt_path = output_dir / "mas_graph.txt"
    svg_path = output_dir / "mas_graph.svg"
    dot_file_path = output_dir / "mas_graph.dot"
    had_json = json_path.exists()
    had_txt = txt_path.exists()
    had_svg = svg_path.exists()
    had_dot = dot_file_path.exists()

    if not json_path.exists():
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not txt_path.exists():
        lines = [
            f"topology: {graph.topology}",
            f"node_num: {graph.node_num}",
            f"edge_count: {metadata['edge_count']}",
            f"density: {metadata['density']:.6f}",
            "steps:",
            *[f"- step_{step}: {', '.join(agents)}" for step, agents in graph.step_to_agents().items()],
            "",
            "edges:",
            *[f"- {edge}" for edge in metadata["edges"]],
            "",
            "message_passing:",
        ]
        for agent_name, incoming_agents in metadata["incoming"].items():
            incoming_text = ", ".join(incoming_agents) if incoming_agents else "(no incoming agents)"
            outgoing_agents = metadata["outgoing"].get(agent_name, [])
            outgoing_text = ", ".join(outgoing_agents) if outgoing_agents else "(no outgoing agents)"
            lines.append(f"- {agent_name}: receive from {incoming_text}; send to {outgoing_text}")
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    svg_available = False
    warning = "SVG export disabled. Pass --render_mas_graph_svg to enable Graphviz rendering."
    dot_path = None
    if render_svg:
        if svg_path.exists() and dot_file_path.exists():
            svg_available = True
            warning = None
            dot_path = str(dot_file_path)
        else:
            svg_available, warning = graphviz_rendering_available()
            if svg_available:
                dot_path = render_graph_svg(graph, svg_path)
    return {
        "json": str(json_path),
        "text": str(txt_path),
        "svg": str(svg_path) if svg_available else None,
        "dot": dot_path,
        "svg_warning": warning,
        "reused_existing": had_json and had_txt and (not render_svg or (had_svg and had_dot)),
    }
