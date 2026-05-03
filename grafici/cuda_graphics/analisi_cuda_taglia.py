import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("Generazione Analisi Taglia CUDA...")

df = pd.read_csv('../risultati_cuda_stat.csv')
df['Size_Label'] = df['M'].astype(str) + "x" + df['N'].astype(str)
df['Area'] = df['M'] * df['N']

def assegna_geometria(row):
    ratio = row['M'] / row['N']
    if 0.6 <= ratio <= 1.6: return 'Quadrata (M ≈ N)'
    elif ratio > 1.6: return 'Tall (M >> N)'
    else: return 'Wide (N >> M)'

df['Geometria'] = df.apply(assegna_geometria, axis=1)

# Ordine taglie per area poi per M
SIZE_ORDER = ['1000x4000', '2000x2000', '2000x8000', '4000x1000',
              '4000x4000', '8000x2000', '8000x8000']
df['Size_Label'] = pd.Categorical(df['Size_Label'], categories=SIZE_ORDER, ordered=True)

df_k32 = df[df['k'] == 32].groupby(['Mode','Size_Label','Geometria','Area'], observed=True)['GFLOPS'].mean().reset_index()

PALETTE_GEO = {
    'Quadrata (M ≈ N)': '#e74c3c',
    'Tall (M >> N)':    '#3498db',
    'Wide (N >> M)':    '#2ecc71',
}
KERNELS = ['CUDA_Naive', 'CUDA_Opt2D', 'CUDA_Tiled', 'CUDA_WarpRow']
KERNEL_TITLES = {
    'CUDA_Naive':  'Kernel 0 – Baseline 1D (un thread per riga)',
    'CUDA_Opt2D':  'Kernel 1 – Griglia 2D (un thread per elemento)',
    'CUDA_Tiled':  'Kernel 2 – Tiling con Shared Memory',
    'CUDA_WarpRow': 'Kernel 3 – Warp-Row (Register Tiling)',
}

sns.set_theme(style="whitegrid", font_scale=1.05)

fig, axes = plt.subplots(4, 1, figsize=(13, 20), sharex=True)

x_pos = {s: i for i, s in enumerate(SIZE_ORDER)}

for row_idx, kernel in enumerate(KERNELS):
    ax = axes[row_idx]
    subset = df_k32[df_k32['Mode'] == kernel].copy()
    subset['x'] = subset['Size_Label'].map(x_pos)
    subset = subset.sort_values('x')

    for geo, grp in subset.groupby('Geometria', observed=True):
        grp = grp.sort_values('x')
        ax.plot(grp['x'], grp['GFLOPS'], marker='s', markersize=10,
                linewidth=2.2, color=PALETTE_GEO[geo], label=geo,
                linestyle='--' if 'Wide' in geo else ('-' if 'Quad' in geo else ':'))

    for _, r in subset.iterrows():
        ax.annotate(f"{r['GFLOPS']:.0f}",
                    xy=(r['x'], r['GFLOPS']),
                    xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold',
                    color=PALETTE_GEO[r['Geometria']])

    ax.set_title(KERNEL_TITLES[kernel], fontsize=13, fontweight='bold', pad=8)
    ax.set_ylabel('GFLOPS')

    max_val = subset['GFLOPS'].max() if not subset.empty else 100
    ax.set_ylim(0, max_val * 1.25)

    if row_idx == 0:
        ax.legend(title='Geometria', loc='upper left', fontsize=10, title_fontsize=11)

axes[-1].set_xticks(range(len(SIZE_ORDER)))
axes[-1].set_xticklabels(SIZE_ORDER, rotation=35, ha='right', fontsize=11)
axes[-1].set_xlabel('Taglia Matrice (M × N)', fontsize=12)

fig.suptitle('Crescita delle Prestazioni CUDA al variare della Taglia (k=32)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cuda_3_Crescita_Taglia.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  cuda_3_Crescita_Taglia.png")