#!/usr/bin/env python3
"""
plot_results.py — Genera i grafici di performance
Tutti i grafici vengono salvati come PNG in --outdir (default: grafici_output/).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette fissa e coerente per kernel/modalità, riusata in tutti i grafici
PALETTE = {
    "CUDA_Naive":       "#2a78d6",
    "CUDA_Opt2D":       "#eb6834",
    "CUDA_Tiled":       "#1baf7a",
    "CUDA_WarpRow":     "#eda100",
    "CUDA_Transposed":  "#e87ba4",
    "Serial":           "#898781",
    "Naive_SMP":        "#2a78d6",
    "Naive_Hybrid":     "#4a3aa7",
    "Opt_SMP":          "#1baf7a",
    "Opt_Hybrid":       "#eda100",
    "CUDA_best":        "#8e44ad",
    "OMP_best":         "#c0392b",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


CUDA_MODES = ["CUDA_Naive", "CUDA_Opt2D", "CUDA_Tiled", "CUDA_WarpRow", "CUDA_Transposed"]
OMP_MODES = ["Naive_SMP", "Naive_Hybrid", "Opt_SMP", "Opt_Hybrid"]


def add_best_rows(df):
    group_cols = ["M", "N", "k", "MPI_Ranks", "Run"]
    frames = [df]
    for label, modes in [("CUDA_best", CUDA_MODES), ("OMP_best", OMP_MODES)]:
        sub = df[df["Mode"].isin(modes)]
        if sub.empty:
            continue
        idx = sub.groupby(group_cols)["GFLOPS"].idxmax()
        best = sub.loc[idx].copy()
        best["BestKernel"] = best["Mode"]
        best["Mode"] = label
        frames.append(best)
    return pd.concat(frames, ignore_index=True)


def load_data(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source"] = os.path.basename(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Solo run valide (scarta eventuali righe con GFLOPS mancante/NaN)
    df = df.dropna(subset=["GFLOPS"])
    df["ratio_label"] = df.apply(_ratio_label, axis=1)
    return df


def _ratio_label(row):
    m, n = row["M"], row["N"]
    if m == n:
        return "quadrata (1:1)"
    g = np.gcd(int(m), int(n))
    return f"{int(m // g)}:{int(n // g)}"


def _color_for(mode):
    return PALETTE.get(mode, "#52514e")


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# 1. GFLOPS al crescere della dimensione del problema (M*N), per kernel
# ---------------------------------------------------------------------------
def plot_gflops_vs_size(df, outdir, k_fixed):
    sub = df[df["k"] == k_fixed]
    if sub.empty:
        print(f"[skip] nessun dato per k={k_fixed} in plot_gflops_vs_size")
        return
    sub = sub.copy()
    sub["size"] = sub["M"] * sub["N"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, mode in enumerate(sorted(sub["Mode"].unique())):
        d = sub[sub["Mode"] == mode].groupby(["size", "M", "N"])["GFLOPS"].mean().reset_index()
        d = d.sort_values("size")
        ax.plot(d["size"], d["GFLOPS"], marker=MARKERS[i % len(MARKERS)],
                label=mode, color=_color_for(mode))
        for _, r in d.iterrows():
            ax.annotate(f"{int(r.M)}x{int(r.N)}", (r["size"], r["GFLOPS"]),
                        fontsize=7, textcoords="offset points", xytext=(0, 6),
                        ha="center", color="#52514e")

    ax.set_xlabel("Dimensione del problema (M x N, elementi totali)")
    ax.set_ylabel("GFLOPS (media sulle run)")
    ax.set_title(f"Throughput al crescere della dimensione della matrice (k={k_fixed})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, outdir, f"01_gflops_vs_size_k{k_fixed}.png")


# ---------------------------------------------------------------------------
# 2. GFLOPS in funzione di k, per kernel, a una taglia di matrice fissata
# ---------------------------------------------------------------------------
def plot_gflops_vs_k(df, outdir, m_fixed, n_fixed):
    sub = df[(df["M"] == m_fixed) & (df["N"] == n_fixed)]
    if sub.empty:
        print(f"[skip] nessun dato per {m_fixed}x{n_fixed} in plot_gflops_vs_k")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, mode in enumerate(sorted(sub["Mode"].unique())):
        d = sub[sub["Mode"] == mode].groupby("k")["GFLOPS"].mean().reset_index().sort_values("k")
        ax.plot(d["k"], d["GFLOPS"], marker=MARKERS[i % len(MARKERS)],
                label=mode, color=_color_for(mode))

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["k"].unique()))
    ax.set_xticklabels(sorted(sub["k"].unique()))
    ax.set_xlabel("k (colonne del multivettore)")
    ax.set_ylabel("GFLOPS (media sulle run)")
    ax.set_title(f"Throughput in funzione di k — matrice {m_fixed}x{n_fixed}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, outdir, f"02_gflops_vs_k_{m_fixed}x{n_fixed}.png")


# ---------------------------------------------------------------------------
# 3. Speedup rispetto al seriale
# ---------------------------------------------------------------------------
def plot_speedup_vs_serial(df, outdir, k_fixed):
    sub = df[(df["k"] == k_fixed) & (df["SpeedUp"] > 0)]
    if sub.empty:
        print(f"[skip] nessun dato di SpeedUp per k={k_fixed}")
        return
    sub = sub.copy()
    sub["size_label"] = sub["M"].astype(int).astype(str) + "x" + sub["N"].astype(int).astype(str)

    pivot = sub.pivot_table(index="size_label", columns="Mode", values="SpeedUp", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index, key=lambda s: np.prod([int(x) for x in s.split("x")])))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for i, mode in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[mode], width=width, label=mode, color=_color_for(mode))

    ax.axhline(1.0, color="#898781", linewidth=1, linestyle="--")
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Speedup vs seriale")
    ax.set_title(f"Speedup rispetto all'implementazione seriale (k={k_fixed})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, outdir, f"03_speedup_vs_serial_k{k_fixed}.png")


# ---------------------------------------------------------------------------
# 4. Confronto kernel x k a una taglia fissata
# ---------------------------------------------------------------------------
def plot_kernel_comparison(df, outdir, m_fixed, n_fixed):
    sub = df[(df["M"] == m_fixed) & (df["N"] == n_fixed)]
    if sub.empty:
        print(f"[skip] nessun dato per {m_fixed}x{n_fixed} in plot_kernel_comparison")
        return

    pivot = sub.pivot_table(index="k", columns="Mode", values="GFLOPS", aggfunc="mean")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for i, mode in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[mode], width=width, label=mode, color=_color_for(mode))

    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("k")
    ax.set_ylabel("GFLOPS (media sulle run)")
    ax.set_title(f"Confronto tra kernel al variare di k — matrice {m_fixed}x{n_fixed}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, outdir, f"04_kernel_comparison_{m_fixed}x{n_fixed}.png")


# ---------------------------------------------------------------------------
# 5. GFLOPS in funzione del numero di processi MPI (scaling)
# ---------------------------------------------------------------------------
def plot_gflops_vs_np(df, outdir, m_fixed, n_fixed, k_fixed):
    sub = df[(df["M"] == m_fixed) & (df["N"] == n_fixed) & (df["k"] == k_fixed)]
    if sub.empty or sub["MPI_Ranks"].nunique() < 2:
        print(f"[skip] dati insufficienti su più MPI_Ranks per {m_fixed}x{n_fixed}, k={k_fixed}")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, mode in enumerate(sorted(sub["Mode"].unique())):
        d = sub[sub["Mode"] == mode].groupby("MPI_Ranks")["GFLOPS"].mean().reset_index().sort_values("MPI_Ranks")
        ax.plot(d["MPI_Ranks"], d["GFLOPS"], marker=MARKERS[i % len(MARKERS)],
                label=mode, color=_color_for(mode))

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["MPI_Ranks"].unique()))
    ax.set_xticklabels(sorted(sub["MPI_Ranks"].unique()))
    ax.set_xlabel("Numero di processi MPI")
    ax.set_ylabel("GFLOPS (media sulle run)")
    ax.set_title(f"Scaling al variare del numero di processi — {m_fixed}x{n_fixed}, k={k_fixed}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, outdir, f"05_gflops_vs_np_{m_fixed}x{n_fixed}_k{k_fixed}.png")


# ---------------------------------------------------------------------------
# 6. Effetto della forma della matrice (M:N) a parità di elementi totali
# ---------------------------------------------------------------------------
def plot_gflops_vs_shape(df, outdir, k_fixed):
    sub = df[df["k"] == k_fixed].copy()
    if sub.empty:
        print(f"[skip] nessun dato per k={k_fixed} in plot_gflops_vs_shape")
        return
    sub["size"] = sub["M"] * sub["N"]
    # tiene solo le taglie totali per cui esiste più di una forma (es. 8M, 12M)
    sizes_with_multi_shape = sub.groupby("size")["ratio_label"].nunique()
    sizes_with_multi_shape = sizes_with_multi_shape[sizes_with_multi_shape > 1].index
    sub = sub[sub["size"].isin(sizes_with_multi_shape)]
    if sub.empty:
        print("[skip] nessuna taglia con più forme diverse disponibile in plot_gflops_vs_shape")
        return

    pivot = sub.pivot_table(index=["size", "ratio_label"], columns="Mode", values="GFLOPS", aggfunc="mean")
    pivot = pivot.reset_index().sort_values(["size", "ratio_label"])
    labels = [f"{int(r['size']/1e6)}M elem.\n{r['ratio_label']}" for _, r in pivot.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot))
    modes = [c for c in pivot.columns if c not in ("size", "ratio_label")]
    width = 0.8 / max(1, len(modes))
    for i, mode in enumerate(modes):
        ax.bar(x + i * width, pivot[mode], width=width, label=mode, color=_color_for(mode))

    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("GFLOPS (media sulle run)")
    ax.set_title(f"Effetto della forma M:N a parità di elementi totali (k={k_fixed})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, outdir, f"06_gflops_vs_shape_k{k_fixed}.png")


# ---------------------------------------------------------------------------
# 7. Variabilità tra le run (boxplot GFLOPS per kernel)
# ---------------------------------------------------------------------------
def plot_variability(df, outdir):
    modes = sorted(df["Mode"].unique())
    data = [df[df["Mode"] == m]["GFLOPS"].values for m in modes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        bp = ax.boxplot(data, tick_labels=modes, patch_artist=True, showfliers=False)
    except TypeError:  # matplotlib < 3.9 non conosce ancora tick_labels
        bp = ax.boxplot(data, labels=modes, patch_artist=True, showfliers=False)
    for patch, mode in zip(bp["boxes"], modes):
        patch.set_facecolor(_color_for(mode))
        patch.set_alpha(0.5)

    ax.set_ylabel("GFLOPS (tutte le configurazioni e run)")
    ax.set_title("Variabilità del throughput per kernel/modalità")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, outdir, "07_variabilita_per_kernel.png")


# ---------------------------------------------------------------------------
# 8. Picco massimo di GFLOPS raggiunto da ciascun kernel
# ---------------------------------------------------------------------------
def plot_peak_gflops(df, outdir):
    modes = sorted(df["Mode"].unique())
    idx = df.groupby("Mode")["GFLOPS"].idxmax()
    peaks = df.loc[idx].set_index("Mode").reindex(modes)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(peaks.index, peaks["GFLOPS"],
                  color=[_color_for(m) for m in peaks.index])

    for bar, (_, row) in zip(bars, peaks.iterrows()):
        label_val = f"{row['GFLOPS']:.0f} GFLOPS"
        label_cfg = f"{int(row['M'])}x{int(row['N'])}, k={int(row['k'])}, np={int(row['MPI_Ranks'])}"
        ax.annotate(label_val, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.annotate(label_cfg, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7, color="#52514e", xytext=(0, 14),
                    textcoords="offset points")

    ax.set_ylabel("GFLOPS (massimo osservato)")
    ax.set_title("Picco massimo di throughput raggiunto per kernel/modalità")
    ax.tick_params(axis="x", rotation=20)
    ax.margins(y=0.15)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, outdir, "08_picco_gflops_per_kernel.png")


def _grid_shape(n, max_cols=3):
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    return rows, cols


def _facet_figure(n_facets, title, xlabel, ylabel, max_cols=3, figsize_per=(4.2, 3.2)):
    """Crea una figura con una griglia di subplot, una per elemento da facettare."""
    rows, cols = _grid_shape(n_facets, max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(figsize_per[0] * cols, figsize_per[1] * rows))
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.supxlabel(xlabel, fontsize=10)
    fig.supylabel(ylabel, fontsize=10)
    return fig, list(axes.flat)


def _finish_facet_figure(fig, axes_flat, n_used, modes, outdir, name):
    for ax in list(axes_flat)[n_used:]:
        ax.axis("off")
    handles, labels = list(axes_flat)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(modes), 5),
               fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, outdir, name)


def _sorted_sizes(df):
    sizes = df[["M", "N"]].drop_duplicates()
    sizes["total"] = sizes["M"] * sizes["N"]
    return list(sizes.sort_values("total")[["M", "N"]].itertuples(index=False, name=None))


# ---------------------------------------------------------------------------
# 9. Griglia: GFLOPS vs k, un sottografico per ciascuna dimensione di matrice
# ---------------------------------------------------------------------------
def plot_gflops_vs_k_grid(df, outdir):
    sizes = _sorted_sizes(df)
    modes = sorted(df["Mode"].unique())
    fig, axes = _facet_figure(len(sizes), "Throughput vs k, per ciascuna dimensione di matrice",
                              "k (colonne del multivettore)", "GFLOPS (media sulle run)")
    ks_all = sorted(df["k"].unique())
    for ax, (m, n) in zip(axes, sizes):
        sub = df[(df["M"] == m) & (df["N"] == n)]
        for i, mode in enumerate(modes):
            d = sub[sub["Mode"] == mode].groupby("k")["GFLOPS"].mean().reset_index().sort_values("k")
            ax.plot(d["k"], d["GFLOPS"], marker=MARKERS[i % len(MARKERS)], color=_color_for(mode),
                    label=mode, markersize=4, linewidth=1.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks_all)
        ax.set_xticklabels(ks_all)
        ax.minorticks_off()
        ax.set_title(f"{int(m)}x{int(n)}", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    _finish_facet_figure(fig, axes, len(sizes), modes, outdir, "09_gflops_vs_k_grid.png")


# ---------------------------------------------------------------------------
# 10. Griglia: confronto kernel x k, un sottografico per ciascuna dimensione
# ---------------------------------------------------------------------------
def plot_kernel_comparison_grid(df, outdir):
    sizes = _sorted_sizes(df)
    modes = sorted(df["Mode"].unique())
    fig, axes = _facet_figure(len(sizes), "Confronto tra kernel al variare di k, per dimensione",
                              "k", "GFLOPS (media sulle run)")
    for ax, (m, n) in zip(axes, sizes):
        sub = df[(df["M"] == m) & (df["N"] == n)]
        pivot = sub.pivot_table(index="k", columns="Mode", values="GFLOPS", aggfunc="mean").sort_index()
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, len(pivot.columns))
        for i, mode in enumerate(pivot.columns):
            ax.bar(x + i * width, pivot[mode], width=width, color=_color_for(mode), label=mode)
        ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(pivot.index, fontsize=7)
        ax.set_title(f"{int(m)}x{int(n)}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    _finish_facet_figure(fig, axes, len(sizes), modes, outdir, "10_kernel_comparison_grid.png")


# ---------------------------------------------------------------------------
# 11. Griglia: scaling MPI (GFLOPS vs NP), un sottografico per dimensione,
#     a un k rappresentativo fissato
# ---------------------------------------------------------------------------
def plot_gflops_vs_np_grid_by_size(df, outdir, k_fixed):
    sub_all = df[df["k"] == k_fixed]
    if sub_all.empty or sub_all["MPI_Ranks"].nunique() < 2:
        print(f"[skip] dati insufficienti su più MPI_Ranks per k={k_fixed}")
        return
    sizes = _sorted_sizes(sub_all)
    modes = sorted(sub_all["Mode"].unique())
    fig, axes = _facet_figure(len(sizes), f"Scaling MPI per dimensione (k={k_fixed})",
                              "Numero di processi MPI", "GFLOPS (media sulle run)")
    nps_all = sorted(sub_all["MPI_Ranks"].unique())
    for ax, (m, n) in zip(axes, sizes):
        sub = sub_all[(sub_all["M"] == m) & (sub_all["N"] == n)]
        for i, mode in enumerate(modes):
            d = sub[sub["Mode"] == mode].groupby("MPI_Ranks")["GFLOPS"].mean().reset_index().sort_values("MPI_Ranks")
            ax.plot(d["MPI_Ranks"], d["GFLOPS"], marker=MARKERS[i % len(MARKERS)], color=_color_for(mode),
                    label=mode, markersize=4, linewidth=1.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(nps_all)
        ax.set_xticklabels(nps_all)
        ax.minorticks_off()
        ax.set_title(f"{int(m)}x{int(n)}", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    _finish_facet_figure(fig, axes, len(sizes), modes, outdir, f"11_gflops_vs_np_grid_size_k{k_fixed}.png")


# ---------------------------------------------------------------------------
# 12. Griglia: scaling MPI (GFLOPS vs NP), un sottografico per k,
#     a una dimensione rappresentativa fissata
# ---------------------------------------------------------------------------
def plot_gflops_vs_np_grid_by_k(df, outdir, m_fixed, n_fixed):
    sub_all = df[(df["M"] == m_fixed) & (df["N"] == n_fixed)]
    if sub_all.empty or sub_all["MPI_Ranks"].nunique() < 2:
        print(f"[skip] dati insufficienti su più MPI_Ranks per {m_fixed}x{n_fixed}")
        return
    ks = sorted(sub_all["k"].unique())
    modes = sorted(sub_all["Mode"].unique())
    fig, axes = _facet_figure(len(ks), f"Scaling MPI per k (matrice {m_fixed}x{n_fixed})",
                              "Numero di processi MPI", "GFLOPS (media sulle run)")
    nps_all = sorted(sub_all["MPI_Ranks"].unique())
    for ax, k in zip(axes, ks):
        sub = sub_all[sub_all["k"] == k]
        for i, mode in enumerate(modes):
            d = sub[sub["Mode"] == mode].groupby("MPI_Ranks")["GFLOPS"].mean().reset_index().sort_values("MPI_Ranks")
            ax.plot(d["MPI_Ranks"], d["GFLOPS"], marker=MARKERS[i % len(MARKERS)], color=_color_for(mode),
                    label=mode, markersize=4, linewidth=1.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(nps_all)
        ax.set_xticklabels(nps_all)
        ax.minorticks_off()
        ax.set_title(f"k={k}", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    _finish_facet_figure(fig, axes, len(ks), modes, outdir, f"12_gflops_vs_np_grid_k_{m_fixed}x{n_fixed}.png")

# ---------------------------------------------------------------------------
# 13. Griglia: Speedup vs seriale in funzione di k, un sottografico per dimensione
# ---------------------------------------------------------------------------
def plot_speedup_vs_k_grid(df, outdir):
    sub_all = df[df["SpeedUp"] > 0]
    if sub_all.empty:
        print("[skip] nessun dato di SpeedUp disponibile in plot_speedup_vs_k_grid")
        return
    sizes = _sorted_sizes(sub_all)
    modes = sorted(sub_all["Mode"].unique())
    fig, axes = _facet_figure(len(sizes), "Speedup vs seriale in funzione di k, per dimensione",
                              "k (colonne del multivettore)", "Speedup vs seriale")
    ks_all = sorted(sub_all["k"].unique())
    for ax, (m, n) in zip(axes, sizes):
        sub = sub_all[(sub_all["M"] == m) & (sub_all["N"] == n)]
        for i, mode in enumerate(modes):
            d = sub[sub["Mode"] == mode].groupby("k")["SpeedUp"].mean().reset_index().sort_values("k")
            if d.empty:
                continue
            ax.plot(d["k"], d["SpeedUp"], marker=MARKERS[i % len(MARKERS)], color=_color_for(mode),
                    label=mode, markersize=4, linewidth=1.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks_all)
        ax.set_xticklabels(ks_all)
        ax.minorticks_off()
        ax.axhline(1.0, color="#898781", linewidth=1, linestyle="--")
        ax.set_title(f"{int(m)}x{int(n)}", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    _finish_facet_figure(fig, axes, len(sizes), modes, outdir, "13_speedup_vs_k_grid.png")


# ---------------------------------------------------------------------------
# 14. Impatto geometrico aggregato per kernel
# ---------------------------------------------------------------------------
def _shape_category(row):
    m, n = row["M"], row["N"]
    if m == n:
        return "Square"
    return "Tall" if m > n else "Wide"


def plot_geometric_impact_by_kernel(df, outdir, k_fixed):
    sub = df[df["k"] == k_fixed].copy()
    if sub.empty:
        print(f"[skip] nessun dato per k={k_fixed} in plot_geometric_impact_by_kernel")
        return
    sub["shape_cat"] = sub.apply(_shape_category, axis=1)

    pivot = sub.pivot_table(index="Mode", columns="shape_cat", values="GFLOPS", aggfunc="mean")
    shape_order = [c for c in ["Tall", "Square", "Wide"] if c in pivot.columns]
    pivot = pivot[shape_order]

    shape_colors = {"Tall": "#2a78d6", "Square": "#1baf7a", "Wide": "#eda100"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(shape_order))
    for i, shape in enumerate(shape_order):
        ax.bar(x + i * width, pivot[shape], width=width, label=shape, color=shape_colors[shape])

    ax.set_xticks(x + width * (len(shape_order) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("GFLOPS (media sulle run)")
    ax.set_title(f"Impatto della geometria della matrice sui kernel (GFLOPS medi, k={k_fixed})")
    ax.legend(title="Forma", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, outdir, f"14_impatto_geometrico_per_kernel_k{k_fixed}.png")


# ---------------------------------------------------------------------------
# 15. Impatto della topologia di griglia MPI (P x Q) a NP fissato
# ---------------------------------------------------------------------------
def plot_grid_topology_impact(df, outdir, k_fixed):
    sub = df[(df["k"] == k_fixed) & (df["GridRows"] > 0) & (df["GridCols"] > 0)].copy()
    if sub.empty:
        print(f"[skip] nessun dato con griglia esplicita (GridRows/GridCols>0) per k={k_fixed} "
              "in plot_grid_topology_impact — servono run con run_cuda_grid_topology.sh "
              "o le config Opt_Grid* di run_omp_benchmark.sh")
        return
    sub["shape_label"] = sub["GridRows"].astype(int).astype(str) + "x" + sub["GridCols"].astype(int).astype(str)
    sub["shape_cat"] = sub.apply(_shape_category, axis=1)

    modes = sorted(sub["Mode"].unique())
    fig, axes = _facet_figure(len(modes), f"Impatto della topologia di griglia MPI (k={k_fixed})",
                              "Griglia P x Q", "GFLOPS (media sulle run)", max_cols=3)
    shape_colors = {"Tall": "#2a78d6", "Square": "#1baf7a", "Wide": "#eda100"}
    for ax, mode in zip(axes, modes):
        s = sub[sub["Mode"] == mode]
        for shape_cat in [c for c in ["Tall", "Square", "Wide"] if c in s["shape_cat"].unique()]:
            d = s[s["shape_cat"] == shape_cat].groupby("shape_label")["GFLOPS"].mean().reset_index()
            # ordina le label PxQ per P crescente
            d["P"] = d["shape_label"].str.split("x").str[0].astype(int)
            d = d.sort_values("P")
            ax.plot(d["shape_label"], d["GFLOPS"], marker="o", label=shape_cat,
                    color=shape_colors[shape_cat], linewidth=1.5)
        ax.set_title(mode, fontsize=10)
        ax.tick_params(labelsize=7, axis="x", rotation=20)
        ax.grid(True, alpha=0.3)
    _finish_facet_figure(fig, axes, len(modes), ["Tall", "Square", "Wide"], outdir,
                         f"15_grid_topology_impact_k{k_fixed}.png")


# ---------------------------------------------------------------------------
# 16. un sottografico per k, TUTTI i kernel
#     (CUDA + OMP insieme) come barre raggruppate su tutte le taglie
# ---------------------------------------------------------------------------
def plot_all_kernels_grid_by_k(df, outdir, exclude_modes=("Serial", "Opt_Grid1x8", "Opt_Grid2x4", "Opt_Grid8x1"),
                               only_modes=None, title_suffix="", filename_suffix=""):
    sub = df[~df["Mode"].isin(exclude_modes)].copy()
    if only_modes is not None:
        sub = sub[sub["Mode"].isin(only_modes)]
    if sub.empty:
        print(f"[skip] nessun dato in plot_all_kernels_grid_by_k{filename_suffix}")
        return
    sub["size_label"] = sub["M"].astype(int).astype(str) + "x" + sub["N"].astype(int).astype(str)
    sub["size_total"] = sub["M"] * sub["N"]
    size_order = (sub[["size_label", "size_total"]].drop_duplicates()
                  .sort_values("size_total")["size_label"].tolist())

    ks = sorted(sub["k"].unique())
    modes = sorted(sub["Mode"].unique())
    fig, axes = _facet_figure(len(ks), f"Confronto kernel{title_suffix} per k, su tutte le taglie",
                              "Dimensione matrice", "GFLOPS (media sulle run)", max_cols=3,
                              figsize_per=(6.5, 4.2))
    for ax, k in zip(axes, ks):
        s = sub[sub["k"] == k]
        pivot = s.pivot_table(index="size_label", columns="Mode", values="GFLOPS", aggfunc="mean")
        pivot = pivot.reindex(size_order)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, len(modes))
        for i, mode in enumerate(modes):
            if mode not in pivot.columns:
                continue
            ax.bar(x + i * width, pivot[mode], width=width, color=_color_for(mode), label=mode)
        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"k = {k}", fontsize=11)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    _finish_facet_figure(fig, axes, len(ks), modes, outdir, f"16_tutti_kernel_grid_per_k{filename_suffix}.png")


def plot_champion_comparison(df, outdir, k_fixed, best_cuda_mode, best_omp_mode):
    """
    Confronta solo il miglior kernel CUDA, il miglior OMP e il Seriale.
    Filtra np=1 per mostrare il potenziale hardware puro.
    """
    # Filtriamo i campioni e fissiamo k e MPI_Ranks
    sub = df[(df["k"] == k_fixed) & (df["MPI_Ranks"] == 1)]
    sub = sub[sub["Mode"].isin([best_cuda_mode, best_omp_mode, "Serial"])].copy()

    if sub.empty:
        print("[skip] Dati campioni mancanti per il confronto.")
        return

    sub["size"] = sub["M"] * sub["N"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Ordine logico di plottaggio (dal più lento al più veloce) per visibilità
    for mode in ["Serial", best_omp_mode, best_cuda_mode]:
        d = sub[sub["Mode"] == mode].groupby("size")["GFLOPS"].mean().reset_index().sort_values("size")
        if not d.empty:
            ax.plot(d["size"], d["GFLOPS"], marker="o", linewidth=2,
                    label=mode, color=_color_for(mode))

    ax.set_xlabel("Dimensione Matrice (Elementi Totali)")
    ax.set_ylabel("GFLOPS (Throughput)")
    ax.set_title(f"Confronto Assoluto: CPU vs GPU vs Ibrido (k={k_fixed}, MPI_Ranks=1)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Architettura")

    _save(fig, outdir, f"COMP_champion_vs_champion_k{k_fixed}.png")

def plot_brute_force_comparison(df, outdir, k_fixed, best_cuda_mode, best_omp_mode):
    """
    Confronto diretto (a barre) con numeri visibili tra i campioni CUDA, OpenMP e Serial.
    """
    sub = df[(df["k"] == k_fixed) & (df["MPI_Ranks"] == 1)].copy()
    target_modes = ["Serial", best_omp_mode, best_cuda_mode]
    sub = sub[sub["Mode"].isin(target_modes)]

    if sub.empty:
        print(f"[skip] Dati campioni mancanti per il confronto bruto a k={k_fixed}.")
        return

    # Creiamo le etichette per l'asse X e calcoliamo la dimensione per ordinarle
    sub["size_label"] = sub["M"].astype(int).astype(str) + "x" + sub["N"].astype(int).astype(str)
    sub["elements"] = sub["M"] * sub["N"]

    # Pivot per allineare i dati
    pivot = sub.pivot_table(index="size_label", columns="Mode", values="GFLOPS", aggfunc="mean")

    # Ordiniamo le righe dalla matrice più piccola alla più grande
    sizes_ordered = sub[["size_label", "elements"]].drop_duplicates().sort_values("elements")["size_label"].tolist()
    pivot = pivot.reindex(sizes_ordered)

    # Ordiniamo le colonne: Serial, OpenMP, CUDA (per avere un climax visivo)
    cols_ordered = [m for m in target_modes if m in pivot.columns]
    pivot = pivot[cols_ordered]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pivot.index))
    width = 0.25  # Larghezza delle barre

    for i, mode in enumerate(cols_ordered):
        offset = (i - len(cols_ordered)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot[mode], width=width, label=mode, color=_color_for(mode))
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f"{height:.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),  # offset verticale di 4 punti
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Throughput (GFLOPS)", fontsize=11)
    ax.set_title(f"CPU vs GPU: Potenza Computazionale Pura (k={k_fixed}, 1 Nodo MPI)", fontsize=14, fontweight="bold")
    ax.legend(title="Architettura", fontsize=10)

    # Linee di griglia leggere dietro le barre
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Aumentiamo il margine superiore per non tagliare i numeri stampati
    ax.margins(y=0.15)

    _save(fig, outdir, f"COMP_brute_force_k{k_fixed}.png")
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="+", help="uno o più CSV di risultati (es. risultati_cuda.csv risultati_omp.csv)")
    parser.add_argument("--outdir", default="grafici_output", help="cartella di output per i PNG")
    parser.add_argument("--k-fixed", type=int, default=32, help="k usato nei grafici che richiedono un k fisso")
    parser.add_argument("--m-fixed", type=int, default=None, help="M usato nei grafici che richiedono una taglia fissa (default: la più piccola quadrata disponibile)")
    parser.add_argument("--n-fixed", type=int, default=None, help="N usato nei grafici che richiedono una taglia fissa")
    args = parser.parse_args()

    print("Caricamento dati da:", ", ".join(args.csv_files))
    df = load_data(args.csv_files)
    print(f"  {len(df)} righe caricate, kernel/modalità presenti: {sorted(df['Mode'].unique())}")
    df_with_best = add_best_rows(df)  # solo per i grafici di confronto CPU/GPU

    m_fixed, n_fixed = args.m_fixed, args.n_fixed
    if m_fixed is None or n_fixed is None:
        squares = df[df["M"] == df["N"]]
        if not squares.empty:
            m_fixed = n_fixed = int(squares["M"].min())
        else:
            m_fixed, n_fixed = int(df["M"].iloc[0]), int(df["N"].iloc[0])
    print(f"Taglia di riferimento per i grafici a M,N fissi: {m_fixed}x{n_fixed}")

    print("Generazione grafici...")
    plot_gflops_vs_size(df, args.outdir, args.k_fixed)
    plot_gflops_vs_k(df, args.outdir, m_fixed, n_fixed)
    plot_speedup_vs_serial(df, args.outdir, args.k_fixed)
    plot_kernel_comparison(df, args.outdir, m_fixed, n_fixed)
    plot_gflops_vs_np(df, args.outdir, m_fixed, n_fixed, args.k_fixed)
    plot_gflops_vs_shape(df, args.outdir, args.k_fixed)
    plot_variability(df, args.outdir)
    plot_peak_gflops(df, args.outdir)
    plot_gflops_vs_k_grid(df, args.outdir)
    plot_kernel_comparison_grid(df, args.outdir)
    plot_gflops_vs_np_grid_by_size(df, args.outdir, args.k_fixed)
    plot_gflops_vs_np_grid_by_k(df, args.outdir, m_fixed, n_fixed)
    plot_speedup_vs_k_grid(df, args.outdir)
    plot_geometric_impact_by_kernel(df, args.outdir, args.k_fixed)
    plot_grid_topology_impact(df, args.outdir, args.k_fixed)
    plot_all_kernels_grid_by_k(df, args.outdir,
                               title_suffix=" (CUDA + OMP)", filename_suffix="_combinato")
    plot_all_kernels_grid_by_k(df, args.outdir, only_modes=CUDA_MODES,
                               title_suffix=" CUDA", filename_suffix="_cuda")
    plot_all_kernels_grid_by_k(df, args.outdir, only_modes=OMP_MODES,
                               title_suffix=" OpenMP", filename_suffix="_omp")
    plot_champion_comparison(df_with_best, args.outdir, args.k_fixed, "CUDA_best", "OMP_best")
    plot_brute_force_comparison(df_with_best, args.outdir, args.k_fixed, "CUDA_best", "OMP_best")
    print("Fatto.")


if __name__ == "__main__":
    sys.exit(main())