#!/usr/bin/env python3
"""
Legge results/risultati_tempi.csv (schema: Mode,Backend,M,N,k,MPI_Ranks,Run,
ParallelTime,DistribTime,TransferTime) e genera un grafico a griglia: un
sottografico per (Mode, taglia), barre impilate per k.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"distrib": "#6b6b6b", "transfer": "#1baf7a", "product": "#eda100"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--outdir", default="grafici_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    df["size_label"] = df["M"].astype(int).astype(str) + "x" + df["N"].astype(int).astype(str)
    agg = df.groupby(["Mode", "size_label", "k"])[["ParallelTime", "DistribTime", "TransferTime"]].mean().reset_index()

    modes = sorted(agg["Mode"].unique())
    sizes = sorted(agg["size_label"].unique(), key=lambda s: np.prod([int(x) for x in s.split("x")]))
    ks = sorted(agg["k"].unique())

    n_panels = len(modes) * len(sizes)
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(5 * cols, 3.6 * rows))
    axes_flat = list(axes.flat)

    panel_idx = 0
    for mode in modes:
        for size in sizes:
            ax = axes_flat[panel_idx]
            panel_idx += 1
            sub = agg[(agg["Mode"] == mode) & (agg["size_label"] == size)].set_index("k").reindex(ks)
            x = np.arange(len(ks))
            distrib_ms = sub["DistribTime"].values * 1000
            transfer_ms = sub["TransferTime"].values * 1000
            product_ms = sub["ParallelTime"].values * 1000
            ax.bar(x, distrib_ms, label="distribuzione", color=COLORS["distrib"])
            ax.bar(x, transfer_ms, bottom=distrib_ms, label="trasferimento", color=COLORS["transfer"])
            ax.bar(x, product_ms, bottom=distrib_ms + transfer_ms, label="prodotto", color=COLORS["product"])
            ax.set_xticks(x)
            ax.set_xticklabels(ks, fontsize=7)
            ax.set_title(f"{mode} — {size}", fontsize=9)
            ax.set_ylabel("ms", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat[panel_idx:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Tempi di distribuzione, trasferimento e calcolo", fontsize=13, y=1.05)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "17_time_breakdown.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()