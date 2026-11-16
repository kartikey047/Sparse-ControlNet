"""Real-model experiment: how much of full ControlNet's control do the gated schedules keep?

    python -m sparsecn.experiment run       --out runs/run1 --budget-min 120
    python -m sparsecn.experiment summarize --out runs/run1
    python -m sparsecn.experiment rescore   --out runs/run1      # new metrics from saved PNGs, no generation
    python -m sparsecn.experiment rerun     --src runs/run1 --out runs/run2 --condition early

For every prompt we first generate a base image without control (seed A) and take its Canny edges as
the control image. Then all five conditions are generated for the same prompt and control with a
different seed B, so any structure that matches the control has to come from ControlNet.

Rows are appended to <out>/rows.jsonl and finished prompts are skipped on restart. Every output image
is saved under <out>/imgs so that new metrics can be computed later without regenerating anything.
"""

import argparse
import json
import os
import platform
import random
import time

import numpy as np

from .edge_metrics import (
    canny,
    canny_multi_f1,
    edge_f1,
    effect_recovery_img,
    grad_corr,
    soft_recall,
)
from .prompts import make_prompts
from .report import load_rows, summarize
from .settings import ALL_CONDITIONS, setting_mask

BASE_SEED_OFFSET = 10_000  # seed of the uncontrolled base image is this + prompt index
OUTPUT_SEED_OFFSET = 20_000  # seed shared by all conditions of one prompt
MIN_CONTROL_EDGES = 300  # a control with fewer edge pixels says almost nothing


def to_uint8(img01):
    return (img01 * 255).round().astype(np.uint8)


def save_png(img01, path):
    from PIL import Image

    Image.fromarray(to_uint8(np.clip(img01, 0, 1))).save(path)


def load_png(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def edges_to_image(edges):
    """Control map as a 3-channel float image, the way it is stored on disk."""
    return np.repeat(edges[..., None].astype(np.float32), 3, axis=2)


def append_row(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def structure_metrics(dino, output, base, control_edges, base_similarity):
    """Metrics against the control map and against the base image the control was taken from."""
    output_u8 = to_uint8(output)
    output_edges = canny(output_u8)
    metrics = {
        **edge_f1(output_edges, control_edges, tol=2),
        **canny_multi_f1(output_u8, control_edges),
        **soft_recall(output_edges, control_edges),
        "grad_corr": grad_corr(output, base),
    }
    metrics["dino_dist"] = dino.distance(dino.self_similarity(output), base_similarity)
    return metrics


class Scorer:
    """Computes every per-image metric (edge metrics, DINO distance and CLIP score)."""

    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from .runner import DinoStructure

        self.torch = torch
        self.dino = DinoStructure()
        self.clip = (
            CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", dtype=torch.float16
            )
            .to("cuda")
            .eval()
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def structure_metrics(self, output, base, control_edges, base_similarity):
        return structure_metrics(
            self.dino, output, base, control_edges, base_similarity
        )

    def clip_score(self, img01, prompt):
        from PIL import Image

        torch = self.torch
        inputs = self.clip_processor(
            text=[prompt],
            images=Image.fromarray(to_uint8(img01)),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")
        inputs = {
            k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.clip(**inputs)
        image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((image_emb * text_emb).sum() * 100)


# ---------------------------------------------------------------- run


def check_against_stock_pipeline(runner, prompts, n, out_dir):
    """Our gated loop with every step enabled must reproduce the stock diffusers pipeline."""
    path = os.path.join(out_dir, "verify.json")
    if os.path.exists(path):
        return
    results = []
    for i in range(min(n, len(prompts))):
        base, _ = runner.generate(prompts[i], None, "none", BASE_SEED_OFFSET + i)
        edges = canny(to_uint8(base))
        ours, _ = runner.generate(prompts[i], edges, "full", OUTPUT_SEED_OFFSET + i)
        stock = runner.reference_pipeline_image(
            prompts[i], edges, OUTPUT_SEED_OFFSET + i
        )
        diff = np.abs(ours - stock)
        results.append(
            {
                "prompt": prompts[i],
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("gated full vs stock pipeline:", results, flush=True)


def measure_timing(runner, prompts, out_dir, rounds, conditions=ALL_CONDITIONS):
    """Denoising time per condition on one prompt, with conditions interleaved in shuffled order.

    Timing inside the main loop is noisy because the laptop GPU clocks drift, so runtime is measured
    separately by repeating all conditions many times in alternating order.
    """
    path = os.path.join(out_dir, "timing.json")
    if rounds <= 0 or os.path.exists(path):
        return

    prompt = prompts[0]
    base, _ = runner.generate(prompt, None, "none", BASE_SEED_OFFSET)
    edges = canny(to_uint8(base))
    for cond in conditions:  # warm-up
        runner.generate(prompt, edges, cond, 1)

    shuffler = random.Random(1)
    gen_times = {c: [] for c in conditions}
    control_times = {c: [] for c in conditions}
    for _ in range(rounds):
        order = list(conditions)
        shuffler.shuffle(order)
        for cond in order:
            _, cost = runner.generate(prompt, edges, cond, 1)
            gen_times[cond].append(cost["gen_s"])
            control_times[cond].append(cost["control_s"])

    timing = {}
    for cond in conditions:
        t = gen_times[cond]
        timing[cond] = {
            "gen_s_median": float(np.median(t)),
            "gen_s_q25": float(np.quantile(t, 0.25)),
            "gen_s_q75": float(np.quantile(t, 0.75)),
            "control_s_median": float(np.median(control_times[cond])),
            "n": len(t),
        }
    for cond in conditions:
        timing[cond]["speedup_vs_full"] = (
            timing["full"]["gen_s_median"] / timing[cond]["gen_s_median"]
        )
    timing["_note"] = (
        "interleaved repeats on one prompt; gen_s excludes text encoding and VAE decode"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=1)
    print(
        "median denoising seconds:",
        {c: round(timing[c]["gen_s_median"], 3) for c in conditions},
        flush=True,
    )


def write_preview(path, edges, outputs):
    """One row of tiles: control map followed by every condition."""
    from PIL import Image

    tiles = [edges_to_image(edges)] + [outputs[c] for c in ALL_CONDITIONS]
    strip = np.concatenate(tiles, axis=1)
    Image.fromarray(to_uint8(strip)).resize((len(tiles) * 256, 256)).save(
        path, quality=88
    )


def run(args):
    import torch
    import diffusers
    from .runner import SDRunner

    os.makedirs(os.path.join(args.out, "previews"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "imgs"), exist_ok=True)

    runner = SDRunner(
        sd_id=args.sd_id,
        controlnet_id=args.controlnet_id,
        steps=args.steps,
        size=args.size,
    )
    scorer = Scorer()

    masks = {
        c: "".join("1" if m else "0" for m in setting_mask(c, runner.step_infos))
        for c in ALL_CONDITIONS
    }
    settings_used = {k: v for k, v in vars(args).items() if k != "func"}
    meta = {
        "args": settings_used,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "diffusers": diffusers.__version__,
        "started": time.ctime(),
        "settings": masks,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("masks (1 = ControlNet runs):", json.dumps(masks), flush=True)

    prompts = make_prompts(args.max_images, args.seed)
    check_against_stock_pipeline(runner, prompts, args.verify, args.out)

    rows_path = os.path.join(args.out, "rows.jsonl")
    already_done = {r["idx"] for r in load_rows(rows_path)}
    shuffler = random.Random(args.seed)
    start = time.time()
    n_new = 0

    for idx, prompt in enumerate(prompts):
        if idx in already_done:
            continue
        if (time.time() - start) / 60 > args.budget_min:
            print("time budget reached", flush=True)
            break

        try:
            base, _ = runner.generate(prompt, None, "none", BASE_SEED_OFFSET + idx)
            edges = canny(to_uint8(base))
            n_edge = int(edges.sum())
            if n_edge < MIN_CONTROL_EDGES:
                append_row(
                    rows_path,
                    {
                        "idx": idx,
                        "prompt": prompt,
                        "skipped": "few_edges",
                        "n_edge": n_edge,
                    },
                )
                continue

            # Conditions are generated in a random order so GPU warm-up effects do not favour any of them.
            order = list(ALL_CONDITIONS)
            shuffler.shuffle(order)
            outputs, costs = {}, {}
            for cond in order:
                outputs[cond], costs[cond] = runner.generate(
                    prompt, edges, cond, OUTPUT_SEED_OFFSET + idx
                )

            base_similarity = scorer.dino.self_similarity(base)
            results = {}
            for cond in ALL_CONDITIONS:
                results[cond] = {
                    **scorer.structure_metrics(
                        outputs[cond], base, edges, base_similarity
                    ),
                    "clip": scorer.clip_score(outputs[cond], prompt),
                    "effect_recovery": effect_recovery_img(
                        outputs[cond], outputs["full"], outputs["none"]
                    ),
                    **costs[cond],
                }

            img_dir = os.path.join(args.out, "imgs")
            save_png(base, os.path.join(img_dir, f"{idx:04d}_base.png"))
            save_png(
                edges_to_image(edges), os.path.join(img_dir, f"{idx:04d}_edges.png")
            )
            for cond in ALL_CONDITIONS:
                save_png(outputs[cond], os.path.join(img_dir, f"{idx:04d}_{cond}.png"))

            append_row(
                rows_path,
                {
                    "idx": idx,
                    "prompt": prompt,
                    "n_edge": n_edge,
                    "warmup": n_new < args.warmup,
                    "order": order,
                    "res": results,
                },
            )
            n_new += 1

            if n_new <= args.previews:
                write_preview(
                    os.path.join(args.out, "previews", f"{idx:04d}.jpg"), edges, outputs
                )
            if n_new % 10 == 0:
                minutes = (time.time() - start) / 60
                print(
                    f"[{minutes:6.1f} min] {n_new} images done ({minutes / n_new * 60:.1f} s/image)",
                    flush=True,
                )

        except torch.cuda.OutOfMemoryError:
            print(
                f"out of memory at prompt {idx}, clearing the cache and moving on",
                flush=True,
            )
            torch.cuda.empty_cache()
        except (
            Exception
        ) as err:  # keep a long unattended run alive and record what failed
            print(f"prompt {idx} failed: {err!r}", flush=True)
            append_row(
                rows_path,
                {
                    "idx": idx,
                    "prompt": prompt,
                    "skipped": "error",
                    "error": repr(err)[:300],
                },
            )

    print(
        f"finished: {n_new} new images in {(time.time() - start) / 60:.1f} min",
        flush=True,
    )
    measure_timing(runner, prompts, args.out, args.timing_rounds)
    summarize(args.out)


# ---------------------------------------------------------------- rescore / rerun


def rescore(args):
    """Recompute every image metric from the saved PNGs. Writes rows_rescored.jsonl."""
    from .runner import DinoStructure

    dino = DinoStructure()
    img_dir = os.path.join(args.out, "imgs")
    rescored = []
    for row in load_rows(os.path.join(args.out, "rows.jsonl")):
        if "res" not in row:
            continue
        i = row["idx"]
        base = load_png(os.path.join(img_dir, f"{i:04d}_base.png"))
        edges = load_png(os.path.join(img_dir, f"{i:04d}_edges.png"))[..., 0] > 0.5
        base_similarity = dino.self_similarity(base)

        results = {}
        for cond in ALL_CONDITIONS:
            output = load_png(os.path.join(img_dir, f"{i:04d}_{cond}.png"))
            new_metrics = structure_metrics(dino, output, base, edges, base_similarity)
            results[cond] = {**row["res"][cond], **new_metrics}
        rescored.append({**row, "res": results})

    path = os.path.join(args.out, "rows_rescored.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rescored)
    print("rescored", len(rescored), "images ->", path)


def rerun_condition(args):
    """Regenerate one or more conditions for every prompt of an existing run.

    The saved base images, control maps and the other conditions are reused unchanged, and every
    condition keeps the prompt, control and seed it had, so the new images are paired with the old
    ones. Results go to a new directory; the source run is not touched.

    `--keep-old early=early_10` stores the previous result of "early" under the name "early_10"
    before it is replaced, so both versions end up in the same summary.
    """
    from .runner import SDRunner

    os.makedirs(os.path.join(args.out, "imgs"), exist_ok=True)
    runner = SDRunner()
    scorer = Scorer()

    keep_old = dict(item.split("=") for item in args.keep_old)
    for cond in args.condition:
        mask = "".join("1" if m else "0" for m in setting_mask(cond, runner.step_infos))
        print(f"{cond} mask:", mask, flush=True)

    src_images = os.path.join(args.src, "imgs")
    rows = [r for r in load_rows(os.path.join(args.src, "rows.jsonl")) if "res" in r]
    if args.limit:
        rows = rows[: args.limit]
    new_rows = []
    start = time.time()
    for n, row in enumerate(rows):
        i = row["idx"]
        base = load_png(os.path.join(src_images, f"{i:04d}_base.png"))
        edges = load_png(os.path.join(src_images, f"{i:04d}_edges.png"))[..., 0] > 0.5
        full = load_png(os.path.join(src_images, f"{i:04d}_full.png"))
        none = load_png(os.path.join(src_images, f"{i:04d}_none.png"))
        base_similarity = scorer.dino.self_similarity(base)

        results = dict(row["res"])
        for cond in args.condition:
            if cond in keep_old and cond in results:
                results[keep_old[cond]] = results[cond]
            image, cost = runner.generate(
                row["prompt"], edges, cond, OUTPUT_SEED_OFFSET + i
            )
            results[cond] = {
                **scorer.structure_metrics(image, base, edges, base_similarity),
                "clip": scorer.clip_score(image, row["prompt"]),
                "effect_recovery": effect_recovery_img(image, full, none),
                **cost,
            }
            save_png(image, os.path.join(args.out, "imgs", f"{i:04d}_{cond}.png"))
        new_rows.append({**row, "res": results})
        if n % 25 == 0:
            print(f"{n}/{len(rows)} prompts, {time.time() - start:.0f}s", flush=True)

    with open(os.path.join(args.out, "rows.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in new_rows)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "note": f"{args.condition} regenerated from {args.src}",
                "keep_old": keep_old,
            },
            f,
            indent=1,
        )

    verify_src = os.path.join(args.src, "verify.json")
    if os.path.exists(verify_src):
        with open(verify_src, encoding="utf-8") as f_in:
            with open(
                os.path.join(args.out, "verify.json"), "w", encoding="utf-8"
            ) as f_out:
                f_out.write(f_in.read())

    all_conditions = list(new_rows[0]["res"])
    measure_timing(
        runner, make_prompts(1200, 0), args.out, args.timing_rounds, all_conditions
    )
    summarize(args.out)


# ---------------------------------------------------------------- command line


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sparsecn.experiment", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_p = commands.add_parser("run", help="generate images and score them")
    run_p.add_argument("--out", required=True)
    run_p.add_argument(
        "--budget-min",
        type=float,
        default=120,
        help="stop generating after this many minutes",
    )
    run_p.add_argument("--max-images", type=int, default=1200)
    run_p.add_argument("--steps", type=int, default=20)
    run_p.add_argument("--size", type=int, default=512)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="first images excluded from in-loop timing",
    )
    run_p.add_argument(
        "--verify",
        type=int,
        default=3,
        help="prompts used to check against the stock pipeline",
    )
    run_p.add_argument("--previews", type=int, default=8)
    run_p.add_argument("--timing-rounds", type=int, default=15)
    run_p.add_argument("--sd-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    run_p.add_argument("--controlnet-id", default="lllyasviel/sd-controlnet-canny")
    run_p.set_defaults(func=run)

    summ_p = commands.add_parser(
        "summarize", help="write summary.md/json/png from rows.jsonl"
    )
    summ_p.add_argument("--out", required=True)
    summ_p.add_argument("--rows", default="rows.jsonl")
    summ_p.set_defaults(func=lambda a: summarize(a.out, a.rows))

    rescore_p = commands.add_parser(
        "rescore", help="recompute metrics from saved images"
    )
    rescore_p.add_argument("--out", required=True)
    rescore_p.set_defaults(func=rescore)

    rerun_p = commands.add_parser(
        "rerun", help="regenerate one condition for an existing run"
    )
    rerun_p.add_argument("--src", required=True)
    rerun_p.add_argument("--out", required=True)
    rerun_p.add_argument(
        "--condition",
        nargs="+",
        default=["early"],
        choices=["early", "low_noise", "combined", "early_10", "low_noise_6"],
    )
    rerun_p.add_argument(
        "--keep-old",
        nargs="*",
        default=[],
        metavar="OLD=NEW",
        help="keep the previous result of a regenerated condition under a new name",
    )
    rerun_p.add_argument(
        "--limit", type=int, help="only the first N prompts of the run"
    )
    rerun_p.add_argument("--timing-rounds", type=int, default=15)
    rerun_p.set_defaults(func=rerun_condition)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
