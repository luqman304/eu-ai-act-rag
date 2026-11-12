"""Render results/*.json into the markdown report (results/REPORT.md)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .stats import mean_ci, paired_delta_ci

RET_COLS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10", "precision@5"]
GEN_COLS = ["faithfulness", "answer_relevance", "correctness", "citation_precision"]


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None or x != x:
        return "–"
    return f"{x * 100:.1f}" if pct else f"{x:.3f}"


def _vals(items: list[dict], col: str) -> list[float]:
    return [it[col] for it in items if it.get(col) is not None and "error" not in it]


def _short(cfg: dict) -> tuple[str, str, str, str]:
    rr = (
        cfg["rerank_model"].split("/")[-1]
        if cfg["rerank"] and cfg["rerank_model"]
        else ("default" if cfg["rerank"] else "–")
    )
    return cfg["strategy"], cfg["mode"], rr, "yes" if cfg.get("recitals", True) else "no"


def retrieval_tables(payload: dict[str, Any], baseline: str | None = None) -> str:
    runs = payload["runs"]
    by_name = {r["config"]["name"]: r for r in runs}
    base = by_name.get(baseline) if baseline else runs[0]
    lines = [
        f"_{payload['n_items']} answerable questions · dense `{payload['models']['dense']}` · sparse "
        f"`{payload['models']['sparse']}` · metrics at the chunk cut-off, provision-level gold labels · "
        f"generated {payload['created']}_",
        "",
        "| chunking | retrieval | rerank | recitals | " + " | ".join(RET_COLS) + " | p50 ms |",
        "|---|---|---|---|" + "---:|" * (len(RET_COLS) + 1),
    ]
    best = {c: max(mean(_vals(r["items"], c)) for r in runs) for c in RET_COLS}
    for r in runs:
        cells = []
        for c in RET_COLS:
            m = mean(_vals(r["items"], c))
            s = _fmt(m, pct=True)
            cells.append(f"**{s}**" if abs(m - best[c]) < 1e-9 else s)
        lat = sorted(it["latency_ms"] for it in r["items"])
        lines.append(
            "| "
            + " | ".join(_short(r["config"]))
            + " | "
            + " | ".join(cells)
            + f" | {lat[len(lat) // 2]:.0f} |"
        )

    # Paired deltas vs the baseline for the headline metric, with bootstrap CIs.
    lines += [
        "",
        f"**Recall@5 vs. baseline `{base['config']['name']}`** (paired bootstrap, 95% CI, 5 000 resamples)",
        "",
        "| configuration | recall@5 | Δ vs baseline | 95% CI | p |",
        "|---|---:|---:|---|---:|",
    ]
    base_vals = [it["recall@5"] for it in base["items"]]
    for r in runs:
        if r is base:
            continue
        vals = [it["recall@5"] for it in r["items"]]
        d, lo, hi, p = paired_delta_ci(base_vals, vals)
        lines.append(
            f"| `{r['config']['name']}` | {_fmt(mean(vals), True)} | {d * 100:+.1f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] | {p:.3f} |"
        )
    return "\n".join(lines)


def style_breakdown(payload: dict[str, Any], names: list[str], metric: str = "recall@5") -> str:
    by_name = {r["config"]["name"]: r for r in payload["runs"]}
    styles = sorted({it["style"] for r in payload["runs"] for it in r["items"]})
    counts = defaultdict(int)
    for it in payload["runs"][0]["items"]:
        counts[it["style"]] += 1
    lines = [
        "| configuration | " + " | ".join(f"{s} (n={counts[s]})" for s in styles) + " |",
        "|---|" + "---:|" * len(styles),
    ]
    for n in names:
        if n not in by_name:
            continue
        row = []
        for s in styles:
            vals = [it[metric] for it in by_name[n]["items"] if it["style"] == s]
            row.append(_fmt(mean(vals), True) if vals else "–")
        lines.append(f"| `{n}` | " + " | ".join(row) + " |")
    return "\n".join(lines)


def generation_table(payload: dict[str, Any]) -> str:
    lines = [
        f"_generator `{payload['models']['generator']}` (effort {payload['efforts']['answer']}) · judge "
        f"`{payload['models']['judge']}` (effort {payload['efforts']['judge']}) · {payload['n_items']} questions · "
        f"generated {payload['created']}_",
        "",
        "| configuration | recall@k | faithfulness | answer relevance | correctness | citation precision | "
        "abstention acc. (unanswerable) | false abstentions | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["runs"]:
        items = r["items"]
        k = r["config"]["k"]
        ok = [it for it in items if "error" not in it]
        cells = [_fmt(mean(_vals(ok, f"recall@{k}")) if _vals(ok, f"recall@{k}") else None)]
        for c in GEN_COLS:
            m, lo, hi = mean_ci(_vals(ok, c))
            cells.append(f"{m:.3f} [{lo:.2f}–{hi:.2f}]" if m == m else "–")
        unans = [it for it in ok if not it["answerable_gold"]]
        ans = [it for it in ok if it["answerable_gold"]]
        cells.append(f"{sum(it['abstention_correct'] for it in unans)}/{len(unans)}" if unans else "–")
        cells.append(f"{sum(it['abstained'] for it in ans)}/{len(ans)}")
        cells.append(str(len(items) - len(ok)))
        lines.append(f"| `{r['config']['name']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generation_failures(payload: dict[str, Any], limit: int = 8) -> str:
    """Show concrete unsupported claims: the qualitative side of the faithfulness number."""
    out = []
    for r in payload["runs"]:
        rows = [it for it in r["items"] if it.get("unsupported_claims")]
        if not rows:
            continue
        out.append(f"**`{r['config']['name']}`**: {len(rows)} answers with ≥1 unsupported claim")
        for it in rows[:limit]:
            out.append(f"- `{it['id']}`: " + "; ".join(f"“{c}”" for c in it["unsupported_claims"][:2]))
        out.append("")
    return "\n".join(out) or "_No unsupported claims._"


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
