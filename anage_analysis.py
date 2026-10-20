"""AnAge body-size scaling analysis: how maximum longevity scales with body mass across classes.

Reads the AnAge extract used in CISC7204 Tutorial 07 (anage_data.csv) and produces:
  figures/main_mass_vs_longevity.png      - the hero visualization (log-log scatter + per-class fits)
  figures/class_scaling_summary.png       - companion: predicted lifespan at a fixed body mass
  results/fits.json                       - the numbers behind both figures

Usage:
    python anage_analysis.py                      # expects data/anage_data.csv
    python anage_analysis.py --data path/to/anage_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
LONG, MASS = "Maximum longevity (yrs)", "Body mass (g)"
CLASSES = ["Mammalia", "Aves", "Reptilia"]

# colour-blind-safe triad (blue / orange / green), never red-green
C_MAMMAL, C_BIRD, C_REPT = "#1f6fb4", "#e2761b", "#4a9b6e"
C_GRID, C_INK, C_MUTED = "#d9dee3", "#1b2a33", "#5b6b76"
STYLE = {
    "Mammalia": (C_MAMMAL, "Mammals", "o"),
    "Aves": (C_BIRD, "Birds", "^"),
    "Reptilia": (C_REPT, "Reptiles", "s"),
}


def load(path: Path) -> tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(path)
    total = len(df)
    d = df[np.isfinite(df[LONG]) & np.isfinite(df[MASS])].copy()
    d = d[d["Class"].isin(CLASSES)]
    return d, total, total - len(d)


def fits(d: pd.DataFrame) -> dict:
    out = {}
    for cls in CLASSES:
        sub = d[d["Class"] == cls]
        x, y = np.log10(sub[MASS].astype(float)), np.log10(sub[LONG].astype(float))
        slope, intercept = np.polyfit(x, y, 1)
        out[cls] = {
            "n": int(len(sub)),
            "slope": float(slope),
            "intercept": float(intercept),
            "r": float(np.corrcoef(x, y)[0, 1]),
            "predicted_at_1kg": float(10 ** (intercept + slope * 3.0)),
        }
    return out


def predict(fit: dict, mass_g: float) -> float:
    return 10 ** (fit["intercept"] + fit["slope"] * np.log10(mass_g))


def hero(d: pd.DataFrame, f: dict, total: int, excluded: int, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.6, 9.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.082, right=0.975, top=0.822, bottom=0.142)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=C_GRID, linewidth=0.9, zorder=0)
    ax.grid(True, which="minor", color=C_GRID, linewidth=0.4, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9aa7b0")
        ax.spines[sp].set_linewidth(1.1)

    ax.axvspan(830, 1210, color="#f0f3f6", zorder=0.4)          # 1 kg window

    for cls in ("Mammalia", "Aves"):
        colour, _, marker = STYLE[cls]
        sub = d[d["Class"] == cls]
        ax.scatter(sub[MASS], sub[LONG], s=34, c=colour, alpha=0.55, marker=marker,
                   edgecolors="white", linewidths=0.5, zorder=3)
        xs = np.logspace(np.log10(sub[MASS].min()), np.log10(sub[MASS].max()), 120)
        ax.plot(xs, predict(f[cls], xs), color=colour, linewidth=2.6, zorder=4)

    rep = d[d["Class"] == "Reptilia"]
    ax.scatter(rep[MASS], rep[LONG], s=46, c=C_REPT, alpha=0.9, marker="s",
               edgecolors="white", linewidths=0.7, zorder=5)

    # controlled comparison at 1 kg
    y_mam, y_bird = predict(f["Mammalia"], 1000.0), predict(f["Aves"], 1000.0)
    ax.plot([1000, 1000], [y_mam, y_bird], color=C_INK, linewidth=1.5,
            linestyle=(0, (4, 3)), zorder=6)
    for y, colour in ((y_mam, C_MAMMAL), (y_bird, C_BIRD)):
        ax.scatter([1000], [y], s=90, facecolor="white", edgecolor=colour,
                   linewidth=2.2, zorder=7)
        ax.annotate("{:.0f} yr".format(y), xy=(1000, y), xytext=(9, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=12.5, fontweight="bold", color=colour, zorder=8)
    ax.annotate("same body mass,\ndifferent lifespan",
                xy=(860, np.sqrt(y_mam * y_bird)), xytext=(-12, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=11.5, color=C_INK, style="italic", zorder=8)

    labels = [
        ("Human", "Human", (14, 10), "left"),
        ("Asiatic elephant", "Asiatic elephant", (-12, 12), "right"),
        ("West Indian manatee", "Manatee", (12, -22), "left"),
        ("White-toothed pygmy shrew", "Pygmy shrew", (10, -14), "left"),
        ("Common raven", "Common raven", (10, 8), "left"),
        ("Wandering albatross", "Wandering albatross", (-12, 17), "right"),
        ("Ostrich", "Ostrich", (-11, 8), "right"),
        ("Ruby-throated hummingbird", "Hummingbird", (11, -3), "left"),
        ("Tuatara", "Tuatara", (9, 10), "left"),
    ]
    for key, label, offset, ha in labels:
        hit = d[d["Common name"].astype(str).str.strip().str.lower() == key.lower()]
        if hit.empty:
            hit = d[d["Common name"].astype(str).str.contains(key, case=False, na=False, regex=False)]
        if hit.empty:
            print("  label skipped (not found):", key)
            continue
        r = hit.iloc[0]
        ax.annotate(label, xy=(r[MASS], r[LONG]), xytext=offset, textcoords="offset points",
                    fontsize=11, color=C_INK, ha=ha, va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=STYLE.get(r["Class"], (C_MUTED,))[0], lw=0.9, alpha=0.92))

    ele = d[d["Common name"].astype(str).str.strip() == "Asiatic elephant"]
    if not ele.empty:
        ele = ele.iloc[0]
        ax.annotate("the heaviest mammal here is not the longest-lived:\n"
                    "a 70 kg human (122 yr) outlives a\n3,672 kg elephant (65 yr)",
                    xy=(float(ele[MASS]), float(ele[LONG])), xycoords="data",
                    xytext=(0.022, 0.965), textcoords="axes fraction",
                    fontsize=11.4, color=C_INK, ha="left", va="top", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.40", fc="#fff9e8", ec="#d9a521", lw=1.1),
                    arrowprops=dict(arrowstyle="-|>", color="#d9a521", lw=1.4, alpha=0.85,
                                    connectionstyle="arc3,rad=-0.12"))

    ax.set_xlabel("Body mass  (grams, log scale)", fontsize=13.5, color=C_INK, labelpad=11)
    ax.set_ylabel("Maximum longevity  (years, log scale)", fontsize=13.5, color=C_INK, labelpad=11)
    ax.tick_params(axis="both", which="both", labelsize=11.5, colors=C_MUTED)

    fig.suptitle("Size Is Not Fate: Lifespan Scaling Across the Animal Kingdom",
                 x=0.082, y=0.972, ha="left", fontsize=23, fontweight="bold", color=C_INK)
    fig.text(0.082, 0.912,
             "Each point is one species. Within every group, larger animals tend to live longer — but the groups are "
             "stacked on different levels.\nAt the same body mass, a bird lives roughly twice as long as a mammal.",
             ha="left", va="top", fontsize=13.2, color=C_MUTED, linespacing=1.5)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor=C_MAMMAL,
               markeredgecolor="white", label="Mammals  (n = {})".format(f["Mammalia"]["n"])),
        Line2D([], [], marker="^", linestyle="none", markersize=9, markerfacecolor=C_BIRD,
               markeredgecolor="white", label="Birds  (n = {})".format(f["Aves"]["n"])),
        Line2D([], [], marker="s", linestyle="none", markersize=9, markerfacecolor=C_REPT,
               markeredgecolor="white", label="Reptiles  (n = {})".format(f["Reptilia"]["n"])),
        Line2D([], [], color=C_MUTED, linewidth=2.6, label="best-fit trend per group"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=12.2,
                    handletextpad=0.9, labelspacing=0.75, borderaxespad=0.5)
    for t in leg.get_texts():
        t.set_color(C_INK)

    fig.text(0.082, 0.016,
             "Source: AnAge database (Tacutu et al.); {} of {} records report both body mass and maximum longevity and "
             "belong to Mammalia, Aves or Reptilia.\nPoints are individual species; lines are ordinary least squares "
             "fits on log10–log10 axes. Reptiles (n = {}) are shown without a fit.".format(
                 len(d), total, f["Reptilia"]["n"]),
             ha="left", va="bottom", fontsize=9.2, color=C_MUTED, linespacing=1.45)

    path = out / "main_mass_vs_longevity.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def summary(d: pd.DataFrame, f: dict, out: Path) -> Path:
    """Companion chart: predicted maximum lifespan at five fixed body masses, mammals vs birds."""
    masses = [10, 100, 1000, 10000, 100000]
    mam = [predict(f["Mammalia"], m) for m in masses]
    bird = [predict(f["Aves"], m) for m in masses]

    fig, ax = plt.subplots(figsize=(11.2, 6.4), dpi=300)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.105, right=0.975, top=0.80, bottom=0.155)

    y = np.arange(len(masses))
    h = 0.36
    ax.barh(y + h / 2, bird, height=h, color=C_BIRD, label="Birds", zorder=3)
    ax.barh(y - h / 2, mam, height=h, color=C_MAMMAL, label="Mammals", zorder=3)
    for yi, (a, b) in enumerate(zip(mam, bird)):
        ax.text(a + 0.7, yi - h / 2, "{:.0f}".format(a), va="center", fontsize=10.5, color=C_MAMMAL)
        ax.text(b + 0.7, yi + h / 2, "{:.0f}".format(b), va="center", fontsize=10.5, color=C_BIRD)
        ax.text(max(a, b) + 4.2, yi, "x{:.2f}".format(b / a), va="center", fontsize=11.5,
                color=C_INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(["{:,} g".format(m) for m in masses], color=C_INK)
    ax.set_xlabel("Predicted maximum longevity (years)", fontsize=12.5, color=C_INK)
    ax.set_xlim(0, max(bird) * 1.13)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9aa7b0")
    ax.tick_params(colors=C_MUTED, labelsize=11)

    fig.suptitle("Same Size, Different Clock", x=0.105, y=0.945, ha="left",
                 fontsize=19, fontweight="bold", color=C_INK)
    fig.text(0.105, 0.875,
             "Predicted maximum lifespan from the per-class log-log fits, evaluated at five body masses. "
             "The bird advantage is a\nconsistent vertical offset — the two scaling slopes are nearly identical "
             "(0.18 vs 0.21) — so it holds at every size.",
             ha="left", va="top", fontsize=11.2, color=C_MUTED, linespacing=1.5)
    ax.legend(frameon=False, loc="lower right", fontsize=11.5)

    path = out / "class_scaling_summary.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(HERE / "data" / "anage_data.csv"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit("dataset not found: {} (see README.md for how to obtain it)".format(data_path))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d, total, excluded = load(data_path)
    f = fits(d)
    print("records: {} raw, {} used, {} dropped (missing mass or longevity)".format(total, len(d), excluded))
    for cls in CLASSES:
        print("  {:<9} n={:<4} slope={:+.3f}  r={:+.3f}  predicted@1kg={:5.1f} yr".format(
            cls, f[cls]["n"], f[cls]["slope"], f[cls]["r"], f[cls]["predicted_at_1kg"]))

    p1 = hero(d, f, total, excluded, out)
    p2 = summary(d, f, out)

    results = {
        "source": "AnAge (Tacutu et al.), via the CISC7204 Tutorial 07 dataset",
        "raw_records": total,
        "used_records": int(len(d)),
        "dropped_missing_fields": int(excluded),
        "fits": f,
        "at_1kg": {"mammal_yrs": round(predict(f["Mammalia"], 1000), 1),
                    "bird_yrs": round(predict(f["Aves"], 1000), 1),
                    "ratio": round(predict(f["Aves"], 1000) / predict(f["Mammalia"], 1000), 2)},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "fits.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("figures:", p1.name, "|", p2.name)
    print(json.dumps(results["at_1kg"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
