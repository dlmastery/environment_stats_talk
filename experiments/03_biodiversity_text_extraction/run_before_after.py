"""Exp03 driver: BEFORE (keyword/regex) vs AFTER (structured extraction).

Runs both extractors over synthetic citizen-science comments, scores triple-level
precision/recall/F1 against the per-comment gold labels, builds a NetworkX
species-interaction graph from the AFTER triples, and writes:

    results/metrics.json
    results/before_after_bars.png
    results/interaction_network.png
    results/summary.md

Runs anywhere: deterministic synthetic data, CPU only, NO API key required.

Run from the REPO ROOT so ``import common`` resolves:

    python experiments/03_biodiversity_text_extraction/run_before_after.py
    python experiments/03_biodiversity_text_extraction/run_before_after.py --n 800
    python experiments/03_biodiversity_text_extraction/run_before_after.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- make repo root + this experiment's before/after importable ----------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (str(_REPO_ROOT), str(_HERE / "before"), str(_HERE / "after")):
    if p not in sys.path:
        sys.path.insert(0, p)

import networkx as nx  # noqa: E402

from common import citizen_comments, SPECIES, RELATIONS  # noqa: E402
from common import plotting  # noqa: E402  (headless Agg)

import keyword_extractor as before_mod  # noqa: E402
import structured_extractor as after_mod  # noqa: E402

RESULTS_DIR = _HERE / "results"


# ---------------------------------------------------------------------------
# Scoring (triple-level)
# ---------------------------------------------------------------------------

def score_extractor(extract_fn, comments: list[dict]) -> dict:
    """Triple-level precision/recall/F1 of ``extract_fn`` vs per-comment gold.

    Also tracks how many extracted triples violate the controlled vocabulary
    (subject/object in SPECIES, relation in RELATIONS) — should be zero.
    """
    tp = fp = fn = 0
    invalid = 0
    n_pred = 0
    for c in comments:
        gold = set(c["interactions"])
        pred = set(extract_fn(c["text"]))
        n_pred += len(pred)
        for s, r, o in pred:
            if s not in SPECIES or o not in SPECIES or r not in RELATIONS:
                invalid += 1
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_predicted": n_pred,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "invalid_triples": invalid,
    }


def collect_triples(extract_fn, comments: list[dict]) -> list[tuple[str, str, str]]:
    """Flatten all extracted triples (with multiplicity removed) across comments."""
    out: set[tuple[str, str, str]] = set()
    for c in comments:
        for t in extract_fn(c["text"]):
            out.add(t)
    return sorted(out)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(triples: list[tuple[str, str, str]]) -> nx.MultiDiGraph:
    """Directed multigraph: nodes = species, edges = relations (subject->object)."""
    g = nx.MultiDiGraph()
    for s, r, o in triples:
        g.add_node(s)
        g.add_node(o)
        g.add_edge(s, o, relation=r)
    return g


_REL_COLORS = {
    "pollinates": "#188038",     # green
    "feeds_on": "#d93025",       # red
    "parasitizes": "#8e24aa",    # purple
    "competes_with": "#f9ab00",  # amber
    "depends_on": "#1a73e8",     # blue
}


def plot_network(graph: nx.MultiDiGraph, path: Path) -> str:
    """Draw the AFTER interaction network, edges colour-coded by relation."""
    plt = plotting.plt
    fig, ax = plotting.new_fig(8.5, 6.5)
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no interactions extracted", ha="center", va="center")
        ax.set_axis_off()
        return plotting.save(fig, path)

    pos = nx.spring_layout(graph, seed=42, k=1.1)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#e8f0fe",
                           edgecolors="#1a73e8", node_size=1500, linewidths=1.2)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

    # group parallel edges by relation for colour + slight curvature
    for i, rel in enumerate(RELATIONS):
        edges = [(u, v) for u, v, d in graph.edges(data=True)
                 if d.get("relation") == rel]
        if not edges:
            continue
        rad = 0.12 + 0.06 * i
        nx.draw_networkx_edges(
            graph, pos, ax=ax, edgelist=edges,
            edge_color=_REL_COLORS.get(rel, "#666"),
            arrows=True, arrowsize=14, width=1.8,
            connectionstyle=f"arc3,rad={rad}",
            label=rel,
        )

    handles = [plt.Line2D([0], [0], color=_REL_COLORS[r], lw=2, label=r)
               for r in RELATIONS]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8)
    ax.set_title("AFTER: species-interaction network (structured extraction)")
    ax.set_axis_off()
    return plotting.save(fig, path)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(metrics: dict, graph: nx.MultiDiGraph, path: Path) -> None:
    b, a = metrics["before"], metrics["after"]
    recall_gap = a["recall"] - b["recall"]
    lines = [
        "# Experiment 03 — Biodiversity species-interaction extraction (BEFORE vs AFTER)",
        "",
        f"- Comments scored: **{metrics['n_comments']}** "
        f"(seed={metrics['seed']}, distractor_frac≈0.35)",
        f"- Gold triple types: **{metrics['n_gold_triple_types']}**",
        "",
        "## Triple-level scores",
        "",
        "| Approach | Precision | Recall | F1 | TP | FP | FN | invalid |",
        "|---|---|---|---|---|---|---|---|",
        f"| BEFORE — keyword/regex | {b['precision']:.3f} | {b['recall']:.3f} | "
        f"{b['f1']:.3f} | {b['tp']} | {b['fp']} | {b['fn']} | {b['invalid_triples']} |",
        f"| AFTER — structured extraction | {a['precision']:.3f} | {a['recall']:.3f} | "
        f"{a['f1']:.3f} | {a['tp']} | {a['fp']} | {a['fn']} | {a['invalid_triples']} |",
        "",
        f"**Recall gap (AFTER − BEFORE): {recall_gap:+.3f}** "
        f"({b['recall']:.0%} → {a['recall']:.0%}).",
        f"**F1 gap (AFTER − BEFORE): {a['f1'] - b['f1']:+.3f}.**",
        "",
        "## AFTER interaction network",
        "",
        f"- Nodes (species): **{graph.number_of_nodes()}**",
        f"- Edges (interactions): **{graph.number_of_edges()}**",
        "",
        "The BEFORE keyword matcher fires only on explicit active-voice verbs, so it",
        "misses the passive / indirect phrasings (\"visited by\", \"rely on\",",
        "\"clustered on\", \"chasing off\") that dominate real citizen-science text,",
        "and it confuses surface verbs with ecological relations (labels obligate-host",
        "\"caterpillars feeding on milkweed\" as feeds_on rather than depends_on). The",
        "AFTER structured extractor normalises synonyms, resolves passive voice, and is",
        "bound to a controlled vocabulary — recovering the buried interactions.",
        "",
        "> Rigor note: extracted triples are a *hypothesis set*. A human ecologist",
        "> verifies them before any downstream network analysis (see README).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400,
                        help="number of synthetic comments (default 400)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke run (n=60), still writes all artifacts")
    parser.add_argument("--outdir", type=str, default=str(RESULTS_DIR),
                        help="results directory (default ./results)")
    args = parser.parse_args(argv)

    n = 60 if args.quick else args.n
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comments = citizen_comments(n=n, seed=args.seed)

    before_scores = score_extractor(before_mod.extract, comments)
    after_scores = score_extractor(after_mod.extract, comments)

    after_triples = collect_triples(after_mod.extract, comments)
    graph = build_graph(after_triples)

    metrics = {
        "experiment": "03_biodiversity_text_extraction",
        "n_comments": n,
        "seed": args.seed,
        "relations": RELATIONS,
        "n_species_vocab": len(SPECIES),
        "n_gold_triple_types": len({(s, r, o) for c in comments
                                    for (s, r, o) in c["interactions"]}),
        "before": before_scores,
        "after": after_scores,
        "after_graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "triples": [list(t) for t in after_triples],
        },
        "recall_gap_after_minus_before": round(
            after_scores["recall"] - before_scores["recall"], 4),
        "f1_gap_after_minus_before": round(
            after_scores["f1"] - before_scores["f1"], 4),
    }

    # write artifacts
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plotting.before_after_bars(
        labels=["precision", "recall", "F1"],
        before_vals=[before_scores["precision"], before_scores["recall"],
                     before_scores["f1"]],
        after_vals=[after_scores["precision"], after_scores["recall"],
                    after_scores["f1"]],
        ylabel="score",
        title="Interaction extraction: keyword/regex vs structured",
        path=outdir / "before_after_bars.png",
        lower_is_better=False,
    )
    plot_network(graph, outdir / "interaction_network.png")
    write_summary(metrics, graph, outdir / "summary.md")

    # console summary
    print(f"[exp03] n={n} seed={args.seed}")
    print(f"  BEFORE  P={before_scores['precision']:.3f}  "
          f"R={before_scores['recall']:.3f}  F1={before_scores['f1']:.3f}")
    print(f"  AFTER   P={after_scores['precision']:.3f}  "
          f"R={after_scores['recall']:.3f}  F1={after_scores['f1']:.3f}")
    print(f"  recall gap (AFTER-BEFORE): {metrics['recall_gap_after_minus_before']:+.3f}")
    print(f"  AFTER graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")
    print(f"  wrote -> {outdir}")
    return metrics


if __name__ == "__main__":
    main()
