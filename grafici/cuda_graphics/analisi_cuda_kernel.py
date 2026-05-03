import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

print("Generazione grafici analisi kernel CUDA...")

df = pd.read_csv('../risultati_cuda_stat.csv')
df['Size_Label'] = df['M'].astype(str) + "x" + df['N'].astype(str)

# Ordine fisso per le taglie (crescente per area, poi per M)
SIZE_ORDER = ['1000x4000', '2000x2000', '2000x8000', '4000x1000',
              '4000x4000', '8000x2000', '8000x8000']
df['Size_Label'] = pd.Categorical(df['Size_Label'], categories=SIZE_ORDER, ordered=True)

# Media su 5 run
df_mean = df.groupby(['Mode', 'Size_Label', 'k'], observed=True)['GFLOPS'].mean().reset_index()

PALETTE = {
    'CUDA_Naive':  '#4b0082',   # viola scuro
    'CUDA_Opt2D':  '#c0396b',   # magenta
    'CUDA_Tiled':  '#f08030',   # arancio
    'CUDA_WarpRow': '#9b59b6'
}
LABELS = {
    'CUDA_Naive':  'Kernel 0 – 1D (baseline)',
    'CUDA_Opt2D':  'Kernel 1 – 2D Grid',
    'CUDA_Tiled':  'Kernel 2 – Shared Memory Tiling',
    'CUDA_WarpRow': 'WarpRow – Reg Tiling'
}

sns.set_theme(style="darkgrid", font_scale=1.05)

# =========================================================
# GRAFICO 1: Confronto a barre k=32, tutte le taglie
# =========================================================
df_k32 = df_mean[df_mean['k'] == 32].copy()

fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
axes = axes.flatten()

for idx, size in enumerate(SIZE_ORDER):
    ax = axes[idx]
    subset = df_k32[df_k32['Size_Label'] == size].sort_values('GFLOPS')
    colors = [PALETTE[m] for m in subset['Mode']]
    bars = ax.bar(subset['Mode'], subset['GFLOPS'], color=colors, width=0.55, edgecolor='white', linewidth=0.8)

    # Valori sopra le barre
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 4,
                f'{h:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

    ax.set_title(f'Matrice {size}', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('GFLOPS' if idx % 4 == 0 else '')
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels([LABELS[m].split(' – ')[0] for m in subset['Mode']], rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, df_k32['GFLOPS'].max() * 1.18)

axes[7].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PALETTE[m], label=LABELS[m]) for m in PALETTE]
fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.97, 0.08),
           fontsize=11, title='Kernel', title_fontsize=12, framealpha=0.9)

fig.suptitle('Confronto Kernel CUDA – Prestazioni a k=32 per ogni Taglia', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cuda_1_Confronto_Barre.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  cuda_1_Confronto_Barre.png")

# =========================================================
# GRAFICO 2: GFLOPS vs k per ogni taglia (4 taglie principali)
# =========================================================
SIZES_4 = ['2000x2000', '4000x4000', '8000x2000', '2000x8000']
df4 = df_mean[df_mean['Size_Label'].isin(SIZES_4)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

for idx, size in enumerate(SIZES_4):
    ax = axes[idx]
    subset = df4[df4['Size_Label'] == size]
    for mode, grp in subset.groupby('Mode', observed=True):
        grp = grp.sort_values('k')
        ax.plot(grp['k'], grp['GFLOPS'], marker='o', linewidth=2.5, markersize=8,
                color=PALETTE[mode], label=LABELS[mode])
        last = grp[grp['k'] == 32]
        if not last.empty:
            ax.annotate(f"{last['GFLOPS'].values[0]:.0f}",
                        xy=(32, last['GFLOPS'].values[0]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, color=PALETTE[mode], fontweight='bold')

    ax.set_title(f'Matrice {size}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Valore di k')
    ax.set_ylabel('Prestazioni (GFLOPS)')
    ax.set_xticks([3, 6, 8, 20, 32])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

axes[0].legend(fontsize=9, loc='upper left')
fig.suptitle("Scalabilità CUDA rispetto all'intensità aritmetica (k)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('cuda_2_GFLOPS_vs_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  cuda_2_GFLOPS_vs_k.png")