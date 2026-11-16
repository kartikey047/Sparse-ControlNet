"""Turn a finished experiment directory (rows.jsonl, timing.json, verify.json) into summary files.

Writes summary.json, summary.md and summary.png next to the rows. Pure numpy/matplotlib, no GPU.
"""

import json
import os

import numpy as np

from .settings import ALL_CONDITIONS, REFERENCES
from .stats import bootstrap_ci, paired_diff_ci, retention

# (key in the rows, column label, higher is better)
METRICS = [
    ("f1", "Canny F1 (100/200)", True),
    ("f1_multi_mean", "Canny F1 (3 thresholds)", True),
    ("soft_recall", "soft edge recall", True),
    ("chamfer_px", "Chamfer dist (px)", False),
    ("grad_corr", "gradient correlation", True),
    ("dino_dist", "DINO structure dist", False),
]
PAIRS = [
    ("early", "low_noise"),
    ("combined", "early"),
    ("combined", "low_noise"),
    ("early", "low_noise_6"),
]


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(ci):
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def summarize(out_dir, rows_file="rows.jsonl"):
    """Compute the summary tables for `out_dir` and write them to disk. Returns the summary dict."""
    all_rows = load_rows(os.path.join(out_dir, rows_file))
    rows = [r for r in all_rows if "res" in r]
    if not rows:
        print("no completed rows")
        return None

    def values(condition, key, subset=rows):
        return [r["res"][condition].get(key, float("nan")) for r in subset]

    keys = [k for k, _, _ in METRICS if k in rows[0]["res"]["full"]]
    labels = {k: label for k, label, _ in METRICS}
    higher_is_better = {k: up for k, _, up in METRICS}
    timed_rows = [r for r in rows if not r.get("warmup")]

    conditions = list(ALL_CONDITIONS) + [
        c for c in rows[0]["res"] if c not in ALL_CONDITIONS
    ]
    per_condition = {}
    for cond in conditions:
        entry = {k: bootstrap_ci(values(cond, k)) for k in keys}
        entry["clip_score"] = bootstrap_ci(values(cond, "clip"))
        entry["control_evals"] = rows[0]["res"][cond]["control_evals"]
        entry["peak_vram_mb_max"] = float(max(values(cond, "peak_vram_mb")))
        if timed_rows:
            entry["gen_s_median_in_loop"] = float(
                np.median(values(cond, "gen_s", timed_rows))
            )
        if cond not in REFERENCES:
            entry["retention"] = {
                k: retention(values(cond, k), values("full", k), values("none", k))
                for k in keys
            }
            entry["effect_recovery"] = bootstrap_ci(values(cond, "effect_recovery"))
        per_condition[cond] = entry

    paired = {
        f"{a} - {b}": {k: paired_diff_ci(values(a, k), values(b, k)) for k in keys}
        for a, b in PAIRS
        if a in conditions and b in conditions
    }

    n_skipped = sum("skipped" in r for r in all_rows)
    timing = _load_json(os.path.join(out_dir, "timing.json"))
    summary = {
        "n_images": len(rows),
        "n_skipped": n_skipped,
        "settings": per_condition,
        "paired_differences": paired,
        "timing": timing,
        "verify": _load_json(os.path.join(out_dir, "verify.json")),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    markdown = _markdown(summary, keys, labels, higher_is_better)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    print(markdown)

    _plot_retention(summary, keys, labels, os.path.join(out_dir, "summary.png"))
    return summary


def _markdown(summary, keys, labels, higher_is_better):
    settings, n = summary["settings"], summary["n_images"]
    lines = [
        "# Real SD1.5 + ControlNet (canny): ControlNet step schedules\n",
        f"{n} images analysed ({summary['n_skipped']} skipped), 20 DDIM steps, 512x512, guidance 7.5. "
        "Same prompt, control image and seed for every condition. 95% bootstrap CIs over images.\n",
        "## Control strength by metric (mean [95% CI])\n",
    ]

    header = [f"{labels[k]} ({'up' if higher_is_better[k] else 'down'})" for k in keys]
    lines.append("| condition | ControlNet calls | " + " | ".join(header) + " | CLIP |")
    lines.append("|---|---|" + "---|" * (len(keys) + 1))
    for cond in settings:
        row = settings[cond]
        cells = " | ".join(_fmt(row[k]) for k in keys)
        lines.append(
            f"| {cond} | {row['control_evals']}/20 | {cells} | {_fmt(row['clip_score'])} |"
        )

    lines += [
        "",
        "## Share of full ControlNet's gain over no control that each setting keeps "
        "(1 = as good as full, 0 = as bad as none)\n",
        "| setting | "
        + " | ".join(labels[k] for k in keys)
        + " | pixel effect recovery |",
        "|---|" + "---|" * (len(keys) + 1),
    ]
    for cond in settings:
        if cond in REFERENCES:
            continue
        cells = " | ".join(_fmt(settings[cond]["retention"][k]) for k in keys)
        lines.append(
            f"| {cond} | {cells} | {_fmt(settings[cond]['effect_recovery'])} |"
        )

    lines += [
        "",
        "## Paired differences (A - B), mean [95% CI]; the sign depends on the metric direction\n",
        "| A - B | " + " | ".join(labels[k] for k in keys) + " |",
        "|---|" + "---|" * len(keys),
    ]
    for name, diffs in summary["paired_differences"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(diffs[k]) for k in keys) + " |")

    timing = summary["timing"]
    if timing:
        lines += [
            "",
            "## Runtime (interleaved repeats, same inputs; denoising loop only)\n",
            "| condition | median s [IQR] | speedup vs full |",
            "|---|---|---|",
        ]
        for cond in settings:
            if cond not in timing:
                continue
            t = timing[cond]
            lines.append(
                f"| {cond} | {t['gen_s_median']:.2f} [{t['gen_s_q25']:.2f}, {t['gen_s_q75']:.2f}] "
                f"| {t['speedup_vs_full']:.2f}x |"
            )

    lines += [
        "",
        "Canny F1 compares output edges with the control edges within 2 px. Soft recall is "
        "mean exp(-d^2 / 2*3^2) over the distance d from each control edge to the nearest output "
        "edge, and Chamfer is that mean distance. Gradient correlation and DINO distance compare "
        "the output with the source image the control was extracted from.",
    ]
    return "\n".join(lines) + "\n"


def _plot_retention(summary, keys, labels, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping the plot")
        return

    settings = summary["settings"]
    gated = [c for c in settings if c not in REFERENCES]
    bar_width = 0.8 / len(gated)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, cond in enumerate(gated):
        cis = [settings[cond]["retention"][k] for k in keys]
        means = [c["mean"] for c in cis]
        errors = [
            [c["mean"] - c["lo"] for c in cis],
            [c["hi"] - c["mean"] for c in cis],
        ]
        ax.bar(
            x + (j - (len(gated) - 1) / 2) * bar_width,
            means,
            bar_width,
            yerr=errors,
            label=cond,
            capsize=2,
        )

    ax.axhline(1, linestyle=":", color="k")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("share of full-ControlNet gain kept")
    ax.set_title(
        f"Control strength retained ({summary['n_images']} images; "
        "early uses fewer ControlNet calls than the others)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
