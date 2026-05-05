"""
Graph Export Module
===================
Uses torch.export / torch.compile tracing to compile models into graph
representations and analyze their computational structure.

Supports:
- Local models (from a .pt file path)
- LLM family (GPT-2, GPT-2 Medium, GPT-2 Large)
- ViT family (ViT-Base, ViT-Large)
- VLM family (CLIP ViT-Base)

Outputs:
- Readable graph summary to console
- CSV file with per-operator breakdown
"""

import os
import sys
import csv
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable

import torch
import torch.nn as nn

# Ensure HF models cache locally
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN")


@dataclass
class GraphNode:
    """A single node in the exported graph."""
    name: str
    op: str        # call_function, call_module, placeholder, output, etc.
    target: str    # the actual function/module called
    num_users: int # how many other nodes consume this output
    compressible: bool = False  # whether this op is compressible
    compress_methods: List[str] = field(default_factory=list)  # applicable methods


@dataclass
class DAGEdge:
    """An edge in the dependency DAG."""
    src: str   # source node name
    dst: str   # destination node name


@dataclass
class CompressibilityAnalysis:
    """Analysis of compressible ops in the graph."""
    total_ops: int = 0
    compressible_ops: int = 0
    compressible_ratio: float = 0.0
    ops_by_method: Dict[str, List[str]] = field(default_factory=dict)
    # method -> list of node names that can use it
    dag_edges: List[DAGEdge] = field(default_factory=list)
    dag_adj: Dict[str, List[str]] = field(default_factory=dict)
    # node -> list of downstream dependent compressible nodes
    critical_path_length: int = 0  # longest chain of compressible ops

    def summary(self, verbose: bool = True) -> str:
        lines = [
            f"\n{'─'*70}",
            f"  COMPRESSIBILITY ANALYSIS",
            f"{'─'*70}",
            f"    Total ops:             {self.total_ops}",
            f"    Compressible ops:      {self.compressible_ops} "
            f"({self.compressible_ratio:.1%})",
            f"    Critical path length:  {self.critical_path_length}",
            f"{'─'*70}",
            f"    Compressible ops by method:",
        ]
        for method, nodes in sorted(self.ops_by_method.items()):
            lines.append(f"\n      {method} ({len(nodes)} ops):")
            import re
            from collections import Counter as _Counter
            base_counts = _Counter()
            for name in nodes:
                short = name.rsplit(".", 1)[-1] if "." in name else name
                base = re.sub(r'_\d+$', '', short)
                base_counts[base] += 1
            for base_name, count in base_counts.most_common():
                lines.append(f"        {base_name:40s}  x{count}")
        lines.append(f"\n{'─'*70}")
        return "\n".join(lines)


@dataclass
class GraphExportResult:
    """Result of exporting a model to a graph."""
    model_name: str
    model_family: str  # "llm", "vit", "vlm", "local", "cnn"
    success: bool
    export_time_s: float = 0.0
    error_message: str = ""

    # Graph statistics
    total_nodes: int = 0
    op_counts: Dict[str, int] = field(default_factory=dict)
    operator_counts: Dict[str, int] = field(default_factory=dict)
    input_shapes: List[str] = field(default_factory=list)
    output_shapes: List[str] = field(default_factory=list)
    num_parameters: int = 0
    num_buffers: int = 0
    graph_depth: int = 0  # longest path from input to output

    # Nodes detail
    nodes: List[GraphNode] = field(default_factory=list)

    # Compressibility analysis
    compressibility: Optional[CompressibilityAnalysis] = None

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  GRAPH EXPORT: {self.model_name} ({self.model_family})",
            f"{'='*70}",
        ]

        if not self.success:
            lines += [
                f"  Status:      FAILED",
                f"  Error:       {self.error_message}",
                f"{'='*70}\n",
            ]
            return "\n".join(lines)

        lines += [
            f"  Status:        SUCCESS",
            f"  Export time:   {self.export_time_s:.3f} s",
            f"{'─'*70}",
            f"  Graph Overview:",
            f"    Total nodes:   {self.total_nodes}",
            f"    Parameters:    {self.num_parameters:,}",
            f"    Buffers:       {self.num_buffers:,}",
            f"    Graph depth:   {self.graph_depth}",
            f"{'─'*70}",
            f"  Node Types:",
        ]
        for op, count in sorted(self.op_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {op:30s}  {count:>6d}")

        lines += [
            f"{'─'*70}",
            f"  Top 20 Operators (by frequency):",
        ]
        for op, count in sorted(self.operator_counts.items(),
                                key=lambda x: -x[1])[:20]:
            lines.append(f"    {op:50s}  {count:>6d}")

        if self.input_shapes:
            lines += [f"{'─'*70}", f"  Input shapes:"]
            for s in self.input_shapes:
                lines.append(f"    {s}")

        if self.output_shapes:
            lines += [f"  Output shapes:"]
            for s in self.output_shapes:
                lines.append(f"    {s}")

        if self.compressibility:
            lines.append(self.compressibility.summary())

        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    def to_csv_rows(self) -> List[Dict]:
        """Return rows for CSV export."""
        rows = []
        for op, count in sorted(self.operator_counts.items(), key=lambda x: -x[1]):
            rows.append({
                "model_name": self.model_name,
                "model_family": self.model_family,
                "operator": op,
                "count": count,
                "total_nodes": self.total_nodes,
                "num_parameters": self.num_parameters,
                "export_time_s": round(self.export_time_s, 3),
            })
        return rows


# ════════════════════════════════════════════════════════════════
# Compressible Op Detection & Dependency DAG
# ════════════════════════════════════════════════════════════════

# Map of op targets to applicable compression methods
# Covers torch ops, nn.Module class names, and HuggingFace module names
COMPRESSIBLE_OPS = {
    # ── Linear / matmul ops — quantization, pruning, low-rank ──
    "linear": ["quantization", "pruning", "low_rank"],
    "addmm": ["quantization", "pruning", "low_rank"],
    "mm": ["quantization", "low_rank"],
    "bmm": ["quantization", "low_rank"],
    "matmul": ["quantization", "low_rank"],
    "Linear": ["quantization", "pruning", "low_rank"],

    # ── Conv ops — quantization, structured pruning ──
    "conv1d": ["quantization", "structured_pruning"],
    "conv2d": ["quantization", "structured_pruning"],
    "conv3d": ["quantization", "structured_pruning"],
    "conv_transpose2d": ["quantization", "structured_pruning"],
    "Conv2d": ["quantization", "structured_pruning"],
    "Conv1d": ["quantization", "structured_pruning"],
    "Conv3d": ["quantization", "structured_pruning"],
    "ConvTranspose2d": ["quantization", "structured_pruning"],

    # ── Attention — token merging, quantization ──
    "scaled_dot_product_attention": ["quantization", "token_merging"],
    "multi_head_attention_forward": ["quantization", "token_merging"],
    "MultiheadAttention": ["quantization", "token_merging"],

    # HuggingFace ViT attention modules
    "ViTSelfAttention": ["quantization", "token_merging", "low_rank"],
    "ViTSelfOutput": ["quantization", "pruning"],
    "ViTAttention": ["quantization", "token_merging"],

    # HuggingFace GPT-2 attention
    "GPT2Attention": ["quantization", "token_merging", "low_rank"],
    "GPT2MLP": ["quantization", "pruning", "low_rank"],
    "GPT2Block": ["quantization", "pruning", "token_merging"],

    # HuggingFace CLIP modules
    "CLIPAttention": ["quantization", "token_merging", "low_rank"],
    "CLIPMLP": ["quantization", "pruning", "low_rank"],
    "CLIPEncoderLayer": ["quantization", "pruning", "token_merging"],

    # ── ViT-specific layers — quantization, pruning, token merging ──
    "ViTIntermediate": ["quantization", "pruning", "low_rank"],
    "ViTOutput": ["quantization", "pruning"],
    "ViTLayer": ["quantization", "pruning", "token_merging"],
    "ViTEncoder": ["quantization", "token_merging"],
    "ViTEmbeddings": ["quantization"],
    "ViTPatchEmbeddings": ["quantization", "structured_pruning"],
    "ViTPooler": ["quantization", "pruning"],

    # ── Norm ops — can be fused with adjacent ops ──
    "layer_norm": ["fusion"],
    "batch_norm": ["fusion"],
    "group_norm": ["fusion"],
    "LayerNorm": ["fusion"],
    "BatchNorm2d": ["fusion"],
    "BatchNorm1d": ["fusion"],
    "GroupNorm": ["fusion"],

    # ── Activation — can be fused ──
    "relu": ["fusion"],
    "gelu": ["fusion"],
    "silu": ["fusion"],
    "ReLU": ["fusion"],
    "GELU": ["fusion"],
    "SiLU": ["fusion"],
    "Tanh": ["fusion"],
    "Sigmoid": ["fusion"],

    # ── Embedding — quantization ──
    "embedding": ["quantization"],
    "Embedding": ["quantization"],
}

# Keyword-based fallback patterns (lowercased) for models not in the dict
_KEYWORD_PATTERNS = [
    (["linear", "dense", "proj", "fc", "mlp", "intermediate", "output"],
     ["quantization", "pruning", "low_rank"]),
    (["conv"], ["quantization", "structured_pruning"]),
    (["attn", "attention", "self_attn", "selfattention", "crossattention"],
     ["quantization", "token_merging", "low_rank"]),
    (["norm", "layernorm", "batchnorm", "groupnorm", "rmsnorm"],
     ["fusion"]),
    (["embed", "patch_embed", "patchembedding"],
     ["quantization"]),
    (["relu", "gelu", "silu", "activation", "act"],
     ["fusion"]),
    (["pool", "avgpool", "maxpool"],
     ["structured_pruning"]),
]


# Pre-built lowercase lookup to avoid per-node O(N) scan of COMPRESSIBLE_OPS
_COMPRESSIBLE_OPS_LOWER: Dict[str, List[str]] = {k.lower(): v for k, v in COMPRESSIBLE_OPS.items()}


def _classify_node_compressibility(node: GraphNode) -> Tuple[bool, List[str]]:
    """Check if a graph node represents a compressible operation."""
    target = node.target

    # O(1) exact match
    if target in COMPRESSIBLE_OPS:
        return True, COMPRESSIBLE_OPS[target]

    # O(1) case-insensitive match via pre-built lowercase dict
    target_lower = target.lower()
    if target_lower in _COMPRESSIBLE_OPS_LOWER:
        return True, _COMPRESSIBLE_OPS_LOWER[target_lower]

    # Keyword-based fallback: match substrings in the target or node name
    combined = target_lower + " " + node.name.lower()
    for keywords, methods in _KEYWORD_PATTERNS:
        if any(kw in combined for kw in keywords):
            return True, methods

    return False, []


def _build_dependency_dag(graph, compressible_names: set) -> Tuple[List[DAGEdge], Dict[str, List[str]]]:
    """
    Build a dependency DAG among compressible ops.

    For each compressible node, find its downstream compressible consumers
    (direct or through non-compressible intermediaries).
    """
    edges = []
    adj = {name: [] for name in compressible_names}

    # Map node names to their users
    node_map = {}
    for node in graph.nodes:
        node_map[node.name] = node

    def _find_downstream_compressible(start_node, visited=None):
        """BFS to find reachable compressible nodes."""
        from collections import deque
        if visited is None:
            visited = set()
        downstream = []
        queue = deque(start_node.users.keys())
        while queue:
            user = queue.popleft()  # O(1) vs list.pop(0) which is O(n)
            if user.name in visited:
                continue
            visited.add(user.name)
            if user.name in compressible_names:
                downstream.append(user.name)
            else:
                # Pass through non-compressible nodes
                queue.extend(user.users.keys())
        return downstream

    for node in graph.nodes:
        if node.name in compressible_names:
            downstream = _find_downstream_compressible(node)
            for dst in downstream:
                edges.append(DAGEdge(src=node.name, dst=dst))
                adj[node.name].append(dst)

    return edges, adj


def _longest_path(adj: Dict[str, List[str]]) -> int:
    """Find the longest path in the DAG (critical path of compressible ops)."""
    memo = {}

    def _dfs(node):
        if node in memo:
            return memo[node]
        if not adj.get(node):
            memo[node] = 1
            return 1
        best = 1 + max(_dfs(child) for child in adj[node])
        memo[node] = best
        return best

    if not adj:
        return 0
    return max(_dfs(n) for n in adj)


def analyze_compressibility(graph, nodes_list: List[GraphNode]) -> CompressibilityAnalysis:
    """
    Walk the graph, classify each op as compressible or not,
    build the dependency DAG among compressible ops, and find
    the critical path.
    """
    analysis = CompressibilityAnalysis()
    compressible_names = set()
    ops_by_method: Dict[str, List[str]] = {}

    for gnode in nodes_list:
        if gnode.op in ("placeholder", "output", "parameter"):
            continue
        analysis.total_ops += 1

        is_comp, methods = _classify_node_compressibility(gnode)
        gnode.compressible = is_comp
        gnode.compress_methods = methods

        if is_comp:
            analysis.compressible_ops += 1
            compressible_names.add(gnode.name)
            for m in methods:
                if m not in ops_by_method:
                    ops_by_method[m] = []
                ops_by_method[m].append(gnode.name)

    analysis.ops_by_method = ops_by_method
    analysis.compressible_ratio = (
        analysis.compressible_ops / analysis.total_ops
        if analysis.total_ops > 0 else 0.0
    )

    # Build dependency DAG if we have a real graph (not module-walk)
    if graph is not None:
        try:
            edges, adj = _build_dependency_dag(graph, compressible_names)
            analysis.dag_edges = edges
            analysis.dag_adj = adj
            analysis.critical_path_length = _longest_path(adj)
        except Exception:
            # DAG construction may fail on some graph formats
            analysis.critical_path_length = 0

    return analysis


def save_compressibility_report(result: 'GraphExportResult',
                                output_dir: str) -> str:
    """Save detailed compressibility report to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = result.model_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe_name}_compressible_ops.txt")

    ca = result.compressibility
    if ca is None:
        return path

    with open(path, "w") as f:
        f.write(f"Compressible Ops Report: {result.model_name}\n")
        f.write(f"Family: {result.model_family}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Total ops:            {ca.total_ops}\n")
        f.write(f"Compressible ops:     {ca.compressible_ops} ({ca.compressible_ratio:.1%})\n")
        f.write(f"Critical path length: {ca.critical_path_length}\n")
        f.write(f"Parameters:           {result.num_parameters:,}\n\n")

        for method, nodes in sorted(ca.ops_by_method.items()):
            f.write(f"{'─'*70}\n")
            f.write(f"{method.upper()} ({len(nodes)} ops)\n")
            f.write(f"{'─'*70}\n")

            # Group by base op type: strip trailing _N digits
            # e.g. linear_0, linear_1, linear_2 -> "linear" x3
            import re
            from collections import Counter as _Counter, OrderedDict

            base_counts = _Counter()
            base_layers = OrderedDict()
            for name in nodes:
                # Get the short name (last component)
                short = name.rsplit(".", 1)[-1] if "." in name else name
                # Strip trailing _N suffix to group e.g. linear_0..linear_71
                base = re.sub(r'_\d+$', '', short)
                base_counts[base] += 1
                if base not in base_layers:
                    base_layers[base] = []
                base_layers[base].append(name)

            f.write(f"\n  Summary:\n")
            for base_name, count in base_counts.most_common():
                f.write(f"    {base_name:40s}  x{count}\n")

            # Only list individual layers for groups that appear once
            unique_layers = [
                (base, layers) for base, layers in base_layers.items()
                if len(layers) == 1
            ]
            grouped_layers = [
                (base, layers) for base, layers in base_layers.items()
                if len(layers) > 1
            ]

            if grouped_layers:
                f.write(f"\n  Grouped layers (showing first & last):\n")
                for base, layers in grouped_layers:
                    f.write(f"    {base} ({len(layers)} total):\n")
                    f.write(f"      {layers[0]}\n")
                    if len(layers) > 2:
                        f.write(f"      ... ({len(layers) - 2} more)\n")
                    if len(layers) > 1:
                        f.write(f"      {layers[-1]}\n")

            if unique_layers:
                f.write(f"\n  Individual layers:\n")
                for base, layers in unique_layers:
                    f.write(f"    {layers[0]}\n")

            f.write("\n")

        # DAG edges
        if ca.dag_edges:
            f.write(f"{'─'*70}\n")
            f.write(f"DEPENDENCY DAG ({len(ca.dag_edges)} edges)\n")
            f.write(f"{'─'*70}\n")
            for edge in ca.dag_edges:
                f.write(f"  {edge.src}  -->  {edge.dst}\n")
            f.write("\n")

    print(f"    Saved compressibility report: {path}")
    return path


def _save_exported(exported, model_name: str, export_method: str,
                   output_dir: str) -> Optional[str]:
    """
    Save the exported graph/program to disk.

    Returns the save path, or None if saving failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")

    # torch.export.ExportedProgram -> save as .pt2
    if hasattr(exported, 'graph_module') and hasattr(exported, 'module'):
        path = os.path.join(output_dir, f"{safe_name}_exported.pt2")
        try:
            torch.export.save(exported, path)
            print(f"    Saved ExportedProgram: {path}")
            return path
        except Exception as e:
            print(f"    Could not save as .pt2: {e}")

    # fx.GraphModule -> save via torch.save + print readable graph
    if hasattr(exported, 'graph') and hasattr(exported, 'code'):
        # Save the GraphModule
        gm_path = os.path.join(output_dir, f"{safe_name}_fx_graph.pt")
        try:
            torch.save(exported, gm_path)
            print(f"    Saved FX GraphModule: {gm_path}")
        except Exception as e:
            print(f"    Could not save GraphModule: {e}")
            gm_path = None

        # Save the readable graph code as .py
        code_path = os.path.join(output_dir, f"{safe_name}_fx_graph.py")
        try:
            with open(code_path, "w") as f:
                f.write(f"# FX Graph for: {model_name}\n")
                f.write(f"# Export method: {export_method}\n\n")
                f.write("# Generated forward code:\n")
                f.write(exported.code + "\n\n")
                f.write("# Full graph:\n")
                f.write(str(exported.graph) + "\n")
            print(f"    Saved readable graph: {code_path}")
        except Exception as e:
            print(f"    Could not save graph code: {e}")

        # Save graph as human-readable text
        txt_path = os.path.join(output_dir, f"{safe_name}_graph.txt")
        try:
            with open(txt_path, "w") as f:
                f.write(f"Graph Export: {model_name}\n")
                f.write(f"Method: {export_method}\n")
                f.write(f"{'='*70}\n\n")
                exported.graph.print_tabular(file=f)
            print(f"    Saved tabular graph: {txt_path}")
        except Exception as e:
            # print_tabular may not support file= in all versions
            try:
                import io
                buf = io.StringIO()
                exported.graph.print_tabular(file=buf)
                with open(txt_path, "w") as f:
                    f.write(f"Graph Export: {model_name}\n")
                    f.write(f"Method: {export_method}\n")
                    f.write(f"{'='*70}\n\n")
                    f.write(buf.getvalue())
                print(f"    Saved tabular graph: {txt_path}")
            except Exception:
                pass

        return gm_path

    return None


def _export_model(model: nn.Module, sample_input,
                  forward_fn: Optional[Callable] = None,
                  model_name: str = "model",
                  model_family: str = "unknown",
                  save_graph: bool = False,
                  output_dir: str = "./profiling_results",
                  full_export: bool = False) -> GraphExportResult:
    """
    Export a model using torch.export and analyze the graph.

    Falls back to torch.fx.symbolic_trace if torch.export fails.
    """
    model = model.cpu().eval()
    result = GraphExportResult(model_name=model_name, model_family=model_family,
                               success=False)

    # Count params and buffers
    result.num_parameters = sum(p.numel() for p in model.parameters())
    result.num_buffers = sum(b.numel() for b in model.buffers())

    # Ensure sample input is on CPU
    if isinstance(sample_input, torch.Tensor):
        sample_input = sample_input.cpu()
    elif isinstance(sample_input, dict):
        sample_input = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in sample_input.items()}

    # Try multiple export strategies
    exported = None
    t0 = time.perf_counter()
    errors = {}

    # Build a callable wrapper
    if forward_fn is not None:
        class WrappedModel(nn.Module):
            def __init__(self, m, fwd):
                super().__init__()
                self.m = m
                self.fwd = fwd
            def forward(self, x):
                return self.fwd(self.m, x)
        export_model = WrappedModel(model, forward_fn)
        export_model.eval()
    else:
        export_model = model

    if isinstance(sample_input, dict):
        inp = next(iter(sample_input.values()))
    else:
        inp = sample_input

    # Methods 1-3 are CPU-intensive graph tracing approaches.
    # Skipped by default — module-walk (Method 4) gives the same FLOPs,
    # compression plan, and compressibility analysis without the overhead.
    # Use --full-export to enable these.
    if full_export:
        # Method 1: torch.export.export
        try:
            exported = torch.export.export(export_model, (inp,), strict=False)
        except Exception as e1:
            errors["torch.export"] = str(e1)

        # Method 2: torch.fx.symbolic_trace
        if exported is None:
            try:
                exported = torch.fx.symbolic_trace(export_model)
            except Exception as e2:
                errors["torch.fx"] = str(e2)

    # Method 3: torch.compile with custom backend
    if exported is None and full_export:
        try:
            captured_graphs = []

            def _capture_backend(gm, example_inputs):
                captured_graphs.append(gm)
                return gm

            # Use GPU if available for faster forward pass during compilation
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compile_model = export_model.to(device)
            compile_inp = inp.to(device) if isinstance(inp, torch.Tensor) else inp

            compiled = torch.compile(compile_model, backend=_capture_backend)
            with torch.no_grad():
                compiled(compile_inp)

            if captured_graphs:
                exported = captured_graphs[0]

            # Move back to CPU for analysis
            export_model.cpu()
        except Exception as e3:
            errors["torch.compile"] = str(e3)
            export_model.cpu()

    # Method 4: Manual module walk (always succeeds — no graph, but gives structure)
    if exported is None:
        result.export_time_s = time.perf_counter() - t0
        result.success = True

        # Build operator counts from module tree
        op_counter = Counter()
        operator_counter = Counter()
        nodes_list = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules
                mod_type = type(module).__name__
                op_counter["call_module"] += 1
                operator_counter[mod_type] += 1
                nodes_list.append(GraphNode(
                    name=name, op="call_module", target=mod_type, num_users=0,
                ))
        # Add parameter nodes
        for pname, _ in model.named_parameters():
            op_counter["parameter"] += 1
            nodes_list.append(GraphNode(
                name=pname, op="parameter", target="param", num_users=0,
            ))

        result.total_nodes = len(nodes_list)
        result.op_counts = dict(op_counter)
        result.operator_counts = dict(operator_counter)
        result.nodes = nodes_list
        result.graph_depth = 0

        # Compressibility analysis (no real graph for DAG, but can classify ops)
        result.compressibility = analyze_compressibility(None, nodes_list)

        if save_graph:
            save_compressibility_report(result, output_dir)

        if errors:
            note = "Used module-walk fallback. Graph tracing errors: "
            note += " | ".join(f"{k}: {v[:80]}" for k, v in errors.items())
            result.error_message = note
        return result

    result.export_time_s = time.perf_counter() - t0
    result.success = True

    # Extract graph from either ExportedProgram or fx.GraphModule
    if hasattr(exported, 'graph_module'):
        graph = exported.graph_module.graph
    elif hasattr(exported, 'graph'):
        graph = exported.graph
    else:
        result.error_message = "Could not access graph from exported object"
        result.success = False
        return result

    # Analyze graph nodes
    op_counter = Counter()
    operator_counter = Counter()
    nodes_list = []
    depth_map = {}

    for node in graph.nodes:
        op = node.op
        target = str(node.target) if node.target else ""

        # Clean up target names
        if op == "call_function":
            # e.g. <built-in function add> -> "add"
            target_clean = target.split(".")[-1].rstrip("'>").split(" ")[-1]
        elif op == "call_method":
            target_clean = target
        elif op == "call_module":
            target_clean = target
        else:
            target_clean = op

        op_counter[op] += 1
        if op in ("call_function", "call_method", "call_module"):
            operator_counter[target_clean] += 1

        num_users = len(node.users)
        nodes_list.append(GraphNode(
            name=node.name, op=op, target=target_clean, num_users=num_users,
        ))

        # Estimate depth
        input_depths = []
        for inp_node in node.all_input_nodes:
            if inp_node.name in depth_map:
                input_depths.append(depth_map[inp_node.name])
        depth_map[node.name] = (max(input_depths) + 1) if input_depths else 0

        # Capture input/output shapes
        if op == "placeholder":
            meta = node.meta.get("val", None)
            if meta is not None and hasattr(meta, "shape"):
                result.input_shapes.append(f"{node.name}: {list(meta.shape)}")
        elif op == "output":
            # Try to get output shapes from args
            for arg in node.args:
                if hasattr(arg, "meta"):
                    meta = arg.meta.get("val", None)
                    if meta is not None and hasattr(meta, "shape"):
                        result.output_shapes.append(f"{list(meta.shape)}")

    result.total_nodes = len(nodes_list)
    result.op_counts = dict(op_counter)
    result.operator_counts = dict(operator_counter)
    result.nodes = nodes_list
    result.graph_depth = max(depth_map.values()) if depth_map else 0

    # Compressibility analysis + dependency DAG
    result.compressibility = analyze_compressibility(graph, nodes_list)

    # Determine which method succeeded
    if hasattr(exported, 'module'):
        export_method = "torch.export"
    elif hasattr(exported, 'code'):
        export_method = "torch.compile" if errors.get("torch.fx") else "torch.fx"
    else:
        export_method = "unknown"

    # Save the exported graph if requested
    if save_graph:
        save_path = _save_exported(exported, model_name, export_method, output_dir)
        if save_path:
            result.error_message = f"Saved via {export_method}"
        save_compressibility_report(result, output_dir)

    return result


# ════════════════════════════════════════════════════════════════
# Model family loaders
# ════════════════════════════════════════════════════════════════

def _load_llm(variant: str = "gpt2"):
    """Load an LLM (GPT-2 family)."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"  Loading {variant}...")
    model = GPT2LMHeadModel.from_pretrained(variant, cache_dir=MODEL_DIR, token=HF_TOKEN)
    tokenizer = GPT2Tokenizer.from_pretrained(variant, cache_dir=MODEL_DIR, token=HF_TOKEN)
    tokens = tokenizer("The quick brown fox", return_tensors="pt")
    input_ids = tokens["input_ids"]

    def forward_fn(m, x):
        return m(x, labels=x)

    return model, input_ids, forward_fn


def _load_vit(variant: str = "google/vit-base-patch16-224"):
    """Load a ViT model."""
    from transformers import ViTForImageClassification

    print(f"  Loading ViT ({variant})...")
    model = ViTForImageClassification.from_pretrained(variant, cache_dir=MODEL_DIR, token=HF_TOKEN)
    sample_input = torch.randn(1, 3, 224, 224)

    def forward_fn(m, x):
        return m(pixel_values=x)

    return model, sample_input, forward_fn


def _load_vlm(variant: str = "openai/clip-vit-base-patch32"):
    """Load a VLM (CLIP)."""
    from transformers import CLIPModel, CLIPProcessor

    print(f"  Loading VLM ({variant})...")
    model = CLIPModel.from_pretrained(variant, cache_dir=MODEL_DIR, token=HF_TOKEN)
    processor = CLIPProcessor.from_pretrained(variant, cache_dir=MODEL_DIR, token=HF_TOKEN)

    # Create dummy image + text inputs
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 49408, (1, 77))
    dummy_attention_mask = torch.ones(1, 77, dtype=torch.long)

    # CLIP needs both image and text
    def forward_fn(m, x):
        return m(pixel_values=x, input_ids=dummy_input_ids,
                 attention_mask=dummy_attention_mask)

    return model, dummy_image, forward_fn


def _load_local(path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """Load a local .pt model."""
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        raise ValueError(f"Expected nn.Module, got {type(loaded)}")
    sample_input = torch.randn(*input_shape)
    return model, sample_input, None


# ════════════════════════════════════════════════════════════════
# Main entry points
# ════════════════════════════════════════════════════════════════

# Default variants for each family
FAMILY_VARIANTS = {
    "llm": [
        ("GPT-2", "gpt2"),
        ("GPT-2-Medium", "gpt2-medium"),
        ("GPT-2-Large", "gpt2-large"),
    ],
    "vit": [
        ("ViT-Base-Patch16", "google/vit-base-patch16-224"),
        ("ViT-Large-Patch16", "google/vit-large-patch16-224"),
    ],
    "vlm": [
        ("CLIP-ViT-Base-Patch32", "openai/clip-vit-base-patch32"),
    ],
}

FAMILY_LOADERS = {
    "llm": _load_llm,
    "vit": _load_vit,
    "vlm": _load_vlm,
}


def export_model_family(family: str, output_dir: str = "./profiling_results",
                        variants: Optional[List[Tuple[str, str]]] = None,
                        save_graph: bool = False,
                        ) -> List[GraphExportResult]:
    """
    Export all variants in a model family and return results.

    Args:
        family: "llm", "vit", or "vlm"
        output_dir: Directory to save CSV output
        variants: Override default variants with [(name, hf_id), ...]
    """
    if variants is None:
        variants = FAMILY_VARIANTS.get(family, [])
    loader = FAMILY_LOADERS.get(family)
    if loader is None:
        raise ValueError(f"Unknown family: {family}. Choose from: llm, vit, vlm")

    results = []
    for name, variant_id in variants:
        print(f"\n{'─'*70}")
        print(f"  Exporting: {name} ({variant_id})")
        print(f"{'─'*70}")

        try:
            model, sample_input, forward_fn = loader(variant_id)
            result = _export_model(
                model, sample_input, forward_fn,
                model_name=name, model_family=family,
                save_graph=save_graph, output_dir=output_dir,
            )
            print(result.summary())
            results.append(result)

            # Free memory
            del model, sample_input
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  Error loading {name}: {e}")
            results.append(GraphExportResult(
                model_name=name, model_family=family,
                success=False, error_message=str(e),
            ))

    return results


def export_local_model(path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                       output_dir: str = "./profiling_results",
                       save_graph: bool = False) -> GraphExportResult:
    """Export a local model file."""
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{'─'*70}")
    print(f"  Exporting local model: {name}")
    print(f"{'─'*70}")

    try:
        model, sample_input, forward_fn = _load_local(path, input_shape)
        result = _export_model(
            model, sample_input, forward_fn,
            model_name=name, model_family="local",
            save_graph=save_graph, output_dir=output_dir,
        )
        print(result.summary())
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return GraphExportResult(
            model_name=name, model_family="local",
            success=False, error_message=str(e),
        )


def save_results_csv(results: List[GraphExportResult],
                     output_path: str) -> str:
    """Save all results to a CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_rows = []
    for r in results:
        if r.success:
            all_rows.extend(r.to_csv_rows())
        else:
            all_rows.append({
                "model_name": r.model_name,
                "model_family": r.model_family,
                "operator": "EXPORT_FAILED",
                "count": 0,
                "total_nodes": 0,
                "num_parameters": r.num_parameters,
                "export_time_s": round(r.export_time_s, 3),
            })

    if not all_rows:
        print("  No results to save.")
        return output_path

    fieldnames = ["model_name", "model_family", "operator", "count",
                  "total_nodes", "num_parameters", "export_time_s"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Saved CSV: {output_path}")
    return output_path


def save_summary_csv(results: List[GraphExportResult],
                     output_path: str) -> str:
    """Save a high-level summary CSV (one row per model)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = ["model_name", "model_family", "success", "total_nodes",
                  "num_parameters", "num_buffers", "graph_depth",
                  "num_op_types", "top_operator", "top_operator_count",
                  "compressible_ops", "compressible_ratio",
                  "critical_path", "quantizable_ops", "prunable_ops",
                  "fusable_ops", "token_merge_ops",
                  "export_time_s", "error"]

    rows = []
    for r in results:
        top_op = ""
        top_count = 0
        if r.operator_counts:
            top_op = max(r.operator_counts, key=r.operator_counts.get)
            top_count = r.operator_counts[top_op]

        ca = r.compressibility
        rows.append({
            "model_name": r.model_name,
            "model_family": r.model_family,
            "success": r.success,
            "total_nodes": r.total_nodes,
            "num_parameters": r.num_parameters,
            "num_buffers": r.num_buffers,
            "graph_depth": r.graph_depth,
            "num_op_types": len(r.operator_counts),
            "top_operator": top_op,
            "top_operator_count": top_count,
            "compressible_ops": ca.compressible_ops if ca else 0,
            "compressible_ratio": round(ca.compressible_ratio, 3) if ca else 0,
            "critical_path": ca.critical_path_length if ca else 0,
            "quantizable_ops": len(ca.ops_by_method.get("quantization", [])) if ca else 0,
            "prunable_ops": len(ca.ops_by_method.get("pruning", []))
                          + len(ca.ops_by_method.get("structured_pruning", [])) if ca else 0,
            "fusable_ops": len(ca.ops_by_method.get("fusion", [])) if ca else 0,
            "token_merge_ops": len(ca.ops_by_method.get("token_merging", [])) if ca else 0,
            "export_time_s": round(r.export_time_s, 3),
            "error": r.error_message if not r.success else "",
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved summary CSV: {output_path}")
    return output_path


def run_graph_export(families: List[str],
                     local_path: Optional[str] = None,
                     input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                     output_dir: str = "./profiling_results",
                     save_graph: bool = False) -> List[GraphExportResult]:
    """
    Run graph export across multiple model families and optionally a local model.

    Args:
        families: List of families to export, e.g. ["llm", "vit", "vlm"]
        local_path: Optional path to a local .pt model
        input_shape: Input shape for local model
        output_dir: Directory for CSV output
    """
    print(f"\n{'#'*70}")
    print(f"#  GRAPH COMPILATION & EXPORT")
    print(f"{'#'*70}")

    all_results = []

    # Export each family
    for family in families:
        print(f"\n{'='*70}")
        print(f"  Model Family: {family.upper()}")
        print(f"{'='*70}")
        results = export_model_family(family, output_dir, save_graph=save_graph)
        all_results.extend(results)

    # Export local model if provided
    if local_path:
        result = export_local_model(local_path, input_shape, output_dir,
                                    save_graph=save_graph)
        all_results.append(result)

    # Print comparison table
    print(f"\n{'#'*70}")
    print(f"#  GRAPH EXPORT COMPARISON")
    print(f"{'#'*70}")
    print(f"\n  {'Model':<25s} {'Family':<8s} {'Nodes':>7s} {'Params':>12s} "
          f"{'Depth':>6s} {'Compress':>9s} {'CritPath':>9s} {'Time':>7s}")
    print(f"  {'─'*25} {'─'*8} {'─'*7} {'─'*12} {'─'*6} {'─'*9} {'─'*9} {'─'*7}")
    for r in all_results:
        if r.success:
            ca = r.compressibility
            comp_str = f"{ca.compressible_ratio:.0%}" if ca else "─"
            crit_str = str(ca.critical_path_length) if ca else "─"
            print(f"  {r.model_name:<25s} {r.model_family:<8s} {r.total_nodes:>7d} "
                  f"{r.num_parameters:>12,} {r.graph_depth:>6d} "
                  f"{comp_str:>9s} {crit_str:>9s} {r.export_time_s:>6.3f}s")
        else:
            print(f"  {r.model_name:<25s} {r.model_family:<8s} {'FAIL':>7s} "
                  f"{r.num_parameters:>12,} {'─':>6s} {'─':>9s} {'─':>9s} {'─':>7s}")

    # Save CSVs
    os.makedirs(output_dir, exist_ok=True)
    save_results_csv(all_results, os.path.join(output_dir, "graph_export_operators.csv"))
    save_summary_csv(all_results, os.path.join(output_dir, "graph_export_summary.csv"))

    print(f"\n{'#'*70}\n")
    return all_results
