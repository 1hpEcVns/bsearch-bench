#!/usr/bin/env python3
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def log_cross(x1, y1a, y1b, x2, y2a, y2b):
    lx1, lx2 = math.log10(x1), math.log10(x2)
    la1, lb1 = math.log10(y1a), math.log10(y1b)
    la2, lb2 = math.log10(y2a), math.log10(y2b)
    da, db = la2 - la1, lb2 - lb1
    denom = da - db
    if denom == 0:
        return None
    t = (lb1 - la1) / denom
    if not (0.0 <= t <= 1.0):
        return None
    return 10.0 ** (lx1 + t * (lx2 - lx1))


def crossover(df, col_brute, col_bin):
    xs = df["n"].to_numpy()
    a = (
        df[col_brute]
        .rolling(5, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    b = (
        df[col_bin]
        .rolling(5, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    faster = (a < b).astype(int)  # 1: brute faster
    for i in range(len(xs) - 1):
        if faster[i] == 1 and faster[i + 1] == 0:
            if i + 2 < len(xs) and faster[i + 2] == 0:
                return log_cross(xs[i], a[i], b[i], xs[i + 1], a[i + 1], b[i + 1])
    return None


def plot_source(csv_path, out_prefix, title_label):
    df = pd.read_csv(
        csv_path,
        names=["type", "n", "avx2_brute_ns", "branchless_ns", "branchy_ns"],
    )
    markers = {
        "avx2_brute_ns": ("AVX2 brute", "o", "#d62728"),
        "branchless_ns": ("branchless", "s", "#1f77b4"),
        "branchy_ns": ("normal (branchy)", "^", "#2ca02c"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, t in zip(axes, ["u8", "u16", "u32"]):
        sub = df[df["type"] == t]
        for col, (label, marker, c) in markers.items():
            ax.plot(sub["n"], sub[col], marker=marker, markersize=5,
                    linewidth=1.5, color=c, label=label)
        cb = crossover(sub, "avx2_brute_ns", "branchless_ns")
        cn = crossover(sub, "avx2_brute_ns", "branchy_ns")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(t)
        ax.set_xlabel("n (array size)")
        ax.set_ylabel("ns / query")
        ax.grid(True, which="both", alpha=0.25)
        if cb is not None:
            ax.axvline(cb, color="#1f77b4", ls=":", lw=1.2)
        if cn is not None:
            ax.axvline(cn, color="#2ca02c", ls=":", lw=1.2)
        ax.annotate(
            f"binary > brute until\n≈ n={cb:.0f}" if cb else "no crossover",
            xy=(cb, ax.get_ylim()[0]) if cb else (0, 0),
            xytext=(0.03, 0.88), textcoords="axes fraction",
            color="#1f77b4", fontsize=10,
        )
        ax.annotate(
            f"normal > brute until\n≈ n={cn:.0f}" if cn else "no crossover",
            xy=(0.03, 0.72), xytext=(0.03, 0.72), textcoords="axes fraction",
            color="#2ca02c", fontsize=10,
        )
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"{title_label} — AVX2 brute vs branchless vs normal binary",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_prefix}_crossover.webp", dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
    for ax, t in zip(axes2, ["u8", "u16", "u32"]):
        sub = df[df["type"] == t]
        ax.plot(sub["n"], sub["branchless_ns"] / sub["avx2_brute_ns"],
                marker="o", color="#1f77b4", label="branchless / brute")
        ax.plot(sub["n"], sub["branchy_ns"] / sub["avx2_brute_ns"],
                marker="s", color="#ff7f0e", label="normal / brute")
        ax.axhline(1.0, color="black", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(t)
        ax.set_xlabel("n (array size)")
        ax.set_ylabel("ratio (binary / brute)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    fig2.suptitle(f"{title_label} — ratio below 1 means AVX2 brute is faster",
                  fontsize=14)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(f"{out_prefix}_ratio.webp", dpi=150)
    plt.close(fig2)

    print(f"\n{title_label} approximate crossover (binary search becomes faster):")
    for t in ("u8", "u16", "u32"):
        sub = df[df["type"] == t]
        cb = crossover(sub, "avx2_brute_ns", "branchless_ns")
        cn = crossover(sub, "avx2_brute_ns", "branchy_ns")
        print(
            f"  {t}: vs branchless ≈ {cb:.0f}   vs normal ≈ {cn:.0f}"
            if cb is not None and cn is not None
            else f"  {t}: vs branchless {cb}   vs normal {cn}"
        )


if __name__ == "__main__":
    plot_source("results.csv", "cpp23", "C++23 (-O3 -mavx2)")
    plot_source("results_rs.csv", "rust", "Rust (-O target-cpu=native)")
