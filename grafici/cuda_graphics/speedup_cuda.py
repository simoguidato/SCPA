import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np  

# ─── Configurazione ──────────────────────────────────────────────────────────
CSV_FILE   = '../risultati_cuda_stat.csv'
OUTPUT_DIR = '..'

PALETTE = {
    'CUDA_Naive':    '#4b0082',
    'CUDA_Opt2D':    '#c0396b',
    'CUDA_Tiled':    '#f08030',
    'CUDA_WarpRow':  '#8b5cf6',
}
LABELS = {
    'CUDA_Naive':    'Kernel 0 – 1D Baseline',
    'CUDA_Opt2D':    'Kernel 1 – 2D Grid',
    'CUDA_Tiled':    'Kernel 2 – Shared Mem Tiling',
    'CUDA_WarpRow':  'Kernel 3 – Warp-Row Reg Tiling',
}

SIZE_ORDER = ['1000x4000', '2000x2000', '2000x8000', '4000x1000',
              '4000x4000', '8000x2000', '8000x8000']

# ─── Caricamento e pulizia ────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
df['Size_Label'] = df['M'].astype(str) + "x" + df['N'].astype(str)

# Tempo seriale: mediana tra i 4 kernel per ogni (M, N, k)
# (riduce il rumore del carico server durante la singola misura di validazione)
serial_ref = (df[df['Run'] == 1]
              .groupby(['M', 'N', 'k'])['SequentialTime']
              .median()
              .reset_index()
              .rename(columns={'SequentialTime': 'SerialTime_median'}))

# Tempo GPU: media delle 5 run per ogni (Mode, M, N, k)
gpu_mean = (df.groupby(['Mode', 'M', 'N', 'k', 'Size_Label'])['ParallelTime']
            .mean()
            .reset_index()
            .rename(columns={'ParallelTime': 'GPUTime_mean'}))

# Merge e calcolo SpeedUp
merged = gpu_mean.merge(serial_ref, on=['M', 'N', 'k'])
merged['SpeedUp'] = merged['SerialTime_median'] / merged['GPUTime_mean']
merged['Size_Label'] = pd.Categorical(merged['Size_Label'],
                                       categories=SIZE_ORDER, ordered=True)

print("SpeedUp massimi per kernel:")
print(merged.groupby('Mode')['SpeedUp'].max().round(1).to_string())
print()
print("Tabella SpeedUp a k=32:")
pivot = (merged[merged['k'] == 32]
         .pivot_table(index='Size_Label', columns='Mode', values='SpeedUp')
         .round(1))
print(pivot.to_string())

# ─── GRAFICO 1: SpeedUp vs k per 4 taglie principali ─────────────────────────
SIZES_4 = ['2000x2000', '4000x4000', '8000x2000', '2000x8000']

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

k_vals = sorted(df['k'].unique())

for idx, size in enumerate(SIZES_4):
    ax = axes[idx]
    subset = merged[merged['Size_Label'] == size]

    for mode, grp in subset.groupby('Mode'):
        grp = grp.sort_values('k')
        ax.plot(grp['k'], grp['SpeedUp'],
                marker='o', linewidth=2.5, markersize=8,
                color=PALETTE[mode], label=LABELS[mode])
        # Annotazione a k=32
        last = grp[grp['k'] == 32]
        if not last.empty:
            ax.annotate(f"{last['SpeedUp'].values[0]:.0f}×",
                        xy=(32, last['SpeedUp'].values[0]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, color=PALETTE[mode], fontweight='bold')

    ax.axhline(1, color='gray', linewidth=1, linestyle='--', alpha=0.6,
               label='Pareggio con seriale')
    ax.set_title(f'Matrice {size}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Valore di k')
    ax.set_ylabel('SpeedUp vs Seriale')
    ax.set_xticks(k_vals)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if idx == 0:
        ax.legend(fontsize=9, loc='upper left')

fig.suptitle('SpeedUp dei Kernel CUDA rispetto all\'Implementazione Seriale',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/speedup_vs_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  speedup_vs_k.png")

# ─── GRAFICO 2: SpeedUp vs taglia a k=32 (linee per kernel) ──────────────────
df_k32 = merged[merged['k'] == 32].copy()

fig, ax = plt.subplots(figsize=(12, 6))

for mode, grp in df_k32.groupby('Mode'):
    grp = grp.sort_values('Size_Label')
    x = range(len(grp))
    ax.plot(list(x), grp['SpeedUp'].values,
            marker='s', linewidth=2.5, markersize=9,
            color=PALETTE[mode], label=LABELS[mode])
    for xi, (_, row) in zip(x, grp.iterrows()):
        ax.annotate(f"{row['SpeedUp']:.0f}×",
                    xy=(xi, row['SpeedUp']),
                    xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=8.5, color=PALETTE[mode], fontweight='bold')

ax.axhline(1, color='gray', linewidth=1.2, linestyle='--', alpha=0.7, label='Pareggio')
ax.set_xticks(range(len(SIZE_ORDER)))
ax.set_xticklabels(SIZE_ORDER, rotation=25, ha='right', fontsize=11)
ax.set_ylabel('SpeedUp vs Seriale', fontsize=12)
ax.set_ylim(0, df_k32['SpeedUp'].max() * 1.2)
ax.set_title('SpeedUp a k=32 al variare della Taglia', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/speedup_vs_taglia_k32.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  speedup_vs_taglia_k32.png")

# ─── GRAFICO 3: Heatmap SpeedUp (taglia × k) per kernel migliore ─────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

modes = ['CUDA_Naive', 'CUDA_Opt2D', 'CUDA_Tiled', 'CUDA_WarpRow']

for idx, mode in enumerate(modes):
    ax = axes[idx]
    sub = merged[merged['Mode'] == mode].copy()
    heat = sub.pivot_table(index='Size_Label', columns='k', values='SpeedUp').round(1)
    heat = heat.reindex(SIZE_ORDER)

    vmax = heat.values.max()
    im = ax.imshow(heat.values, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=vmax)

    # Annotazioni
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            val = heat.values[r, c]
            color = 'white' if val < vmax * 0.5 else 'black'
            ax.text(c, r, f'{val:.0f}×', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, fontsize=10)
    ax.set_yticks(range(len(SIZE_ORDER)))
    ax.set_yticklabels(SIZE_ORDER, fontsize=9)
    ax.set_xlabel('k')
    ax.set_title(LABELS[mode], fontsize=12, fontweight='bold',
                 color=PALETTE[mode])
    plt.colorbar(im, ax=ax, label='SpeedUp', shrink=0.8)

fig.suptitle('Heatmap SpeedUp GPU vs Seriale – tutti i kernel',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/speedup_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  speedup_heatmap.png")

# ─── GRAFICO 4: Confronto SpeedUp massimo per kernel (barplot) ───────────────
max_su = (merged[merged['k'] == 32]
          .groupby('Mode')['SpeedUp']
          .agg(['mean', 'max', 'min'])
          .reset_index())

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(max_su))
colors = [PALETTE[m] for m in max_su['Mode']]

bars = ax.bar(x, max_su['mean'], color=colors, width=0.55,
              edgecolor='white', linewidth=0.8, alpha=0.9)
ax.errorbar(x, max_su['mean'],
            yerr=[max_su['mean'] - max_su['min'],
                  max_su['max'] - max_su['mean']],
            fmt='none', color='black', capsize=5, linewidth=1.5)

for bar, (_, row) in zip(bars, max_su.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row['mean']:.0f}×\n(max {row['max']:.0f}×)",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(1, color='gray', linewidth=1.2, linestyle='--', alpha=0.7)
ax.set_xticks(list(x))
ax.set_xticklabels([LABELS[m] for m in max_su['Mode']],
                   rotation=15, ha='right', fontsize=11)
ax.set_ylabel('SpeedUp vs Seriale (media su taglie)', fontsize=12)
ax.set_title('SpeedUp medio a k=32 – confronto kernel\n'
             '(barre di errore: min–max tra le taglie)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/speedup_riepilogo.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  speedup_riepilogo.png")
