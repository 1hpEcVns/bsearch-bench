#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def aligned_boundary(df, col_brute, col_bin):
    """Return (last aligned n where brute wins, first aligned n where binary wins)."""
    xs = df["n"].to_numpy().astype(int)
    a = df[col_brute].to_numpy()
    b = df[col_bin].to_numpy()
    faster = (a < b).astype(int)  # 1: brute faster
    for i in range(len(xs) - 1):
        if faster[i] == 1 and faster[i + 1] == 0:
            return int(xs[i]), int(xs[i + 1])
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
        cb = aligned_boundary(sub, "avx2_brute_ns", "branchless_ns")
        cn = aligned_boundary(sub, "avx2_brute_ns", "branchy_ns")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(t)
        ax.set_xlabel("n (array size)")
        ax.set_ylabel("ns / query")
        ax.grid(True, which="both", alpha=0.25)
        if cb is not None:
            ax.axvline(cb[1], color="#1f77b4", ls=":", lw=1.2)
        if cn is not None:
            ax.axvline(cn[1], color="#2ca02c", ls=":", lw=1.2)
        ax.annotate(
            f"brute ≤ {cb[0]} / binary ≥ {cb[1]}" if cb else "no crossover",
            xy=(cb[1], ax.get_ylim()[0]) if cb else (0, 0),
            xytext=(0.03, 0.88), textcoords="axes fraction",
            color="#1f77b4", fontsize=10,
        )
        ax.annotate(
            f"brute ≤ {cn[0]} / normal ≥ {cn[1]}" if cn else "no crossover",
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

    print(f"\n{title_label} aligned boundary (brute last win / binary first win):")
    for t in ("u8", "u16", "u32"):
        sub = df[df["type"] == t]
        cb = aligned_boundary(sub, "avx2_brute_ns", "branchless_ns")
        cn = aligned_boundary(sub, "avx2_brute_ns", "branchy_ns")
        print(
            f"  {t}: branchless brute≤{cb[0]} bin≥{cb[1]}   normal brute≤{cn[0]} bin≥{cn[1]}"
            if cb is not None and cn is not None
            else f"  {t}: vs branchless {cb}   vs normal {cn}"
        )


def plot_cpp_vs_rust():
    cpp = pd.read_csv(
        "results.csv",
        names=["type", "n", "avx2_brute_ns", "branchless_ns", "branchy_ns"],
    )
    rust = pd.read_csv(
        "results_rs.csv",
        names=["type", "n", "avx2_brute_ns", "branchless_ns", "branchy_ns"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, t in zip(axes, ["u8", "u16", "u32"]):
        c = cpp[cpp["type"] == t]
        r = rust[rust["type"] == t]
        ax.plot(c["n"], c["branchless_ns"], marker="o", color="#1f77b4",
                label="C++23 branchless")
        ax.plot(r["n"], r["branchless_ns"], marker="s", color="#d62728",
                label="Rust branchless")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(t)
        ax.set_xlabel("n (array size)")
        ax.set_ylabel("ns / query")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Branchless binary search: C++23 vs Rust", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("cpp_vs_rust_branchless.webp", dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
    for ax, t in zip(axes2, ["u8", "u16", "u32"]):
        c = cpp[cpp["type"] == t]
        r = rust[rust["type"] == t]
        ax.plot(c["n"], r["branchless_ns"] / c["branchless_ns"],
                marker="o", color="#7f7f7f")
        ax.axhline(1.0, color="black", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(t)
        ax.set_xlabel("n (array size)")
        ax.set_ylabel("Rust / C++ time")
        ax.grid(True, which="both", alpha=0.25)
    fig2.suptitle("Branchless binary search ratio (below 1 = Rust faster)",
                  fontsize=14)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig("cpp_vs_rust_branchless_ratio.webp", dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    plot_source("results.csv", "cpp23", "C++23 (-O3 -mavx2)")
    plot_source("results_rs.csv", "rust", "Rust (-O target-cpu=native)")
    plot_cpp_vs_rust()
