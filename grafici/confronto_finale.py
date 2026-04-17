import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os, sys

print("Generazione Confronto Finale GPU vs CPU...")

# ── Caricamento GPU ──────────────────────────────────────────────────────────
df_gpu = pd.read_csv('risultati_cuda_stat.csv')
df_gpu['Size_Label'] = df_gpu['M'].astype(str) + "x" + df_gpu['N'].astype(str)
df_gpu_mean = df_gpu.groupby(['Mode', 'Size_Label', 'k'])['GFLOPS'].mean().reset_index()

# ── Caricamento CPU (opzionale) ──────────────────────────────────────────────
CPU_CSV = 'risultati_avanzati.csv'
if os.path.exists(CPU_CSV):
    df_cpu = pd.read_csv(CPU_CSV)
    best_cpu = (df_cpu[(df_cpu['MPI_Ranks'] == 2) & (df_cpu['OMP_Threads'] == 20)
                        & (df_cpu['Mode'] == 'Opt_Hybrid')]
                .groupby(['M', 'N', 'k'])['GFLOPS'].mean().reset_index())
    best_cpu['Mode'] = 'CPU_Best'
    best_cpu['Size_Label'] = best_cpu['M'].astype(str) + "x" + best_cpu['N'].astype(str)
    has_cpu = True
    print("  → Dati CPU trovati, confronto completo attivato")
else:
    has_cpu = False
    print("  → risultati_avanzati.csv non trovato – grafici solo GPU")

# ── Configurazione ────────────────────────────────────────────────────────────
SIZE_ORDER_GPU = ['1000x4000', '2000x2000', '2000x8000', '4000x1000',
                  '4000x4000', '8000x2000', '8000x8000']

# Solo le taglie comuni a OMP se disponibile
if has_cpu:
    cpu_sizes = set(best_cpu['Size_Label'])
    SIZE_ORDER = [s for s in SIZE_ORDER_GPU if s in cpu_sizes]
else:
    SIZE_ORDER = SIZE_ORDER_GPU

PALETTE = {
    'CUDA_Naive':  '#4b0082',
    'CUDA_Opt2D':  '#c0396b',
    'CUDA_Tiled':  '#f08030',
    'CPU_Best':    '#27ae60',
}
LABELS = {
    'CUDA_Naive':  'Kernel 0 – 1D Baseline',
    'CUDA_Opt2D':  'Kernel 1 – 2D Grid',
    'CUDA_Tiled':  'Kernel 2 – Shared Mem Tiling',
    'CPU_Best':    'CPU Best (OMP NP:2/T:20)',
}

# ── GRAFICO 1: Barre k=32, tutte le taglie ───────────────────────────────────
modes_to_plot = ['CUDA_Opt2D', 'CUDA_Tiled']
if has_cpu:
    modes_to_plot = ['CPU_Best'] + modes_to_plot

df_k32_gpu = df_gpu_mean[(df_gpu_mean['k'] == 32) & (df_gpu_mean['Mode'].isin(['CUDA_Opt2D', 'CUDA_Tiled']))].copy()
if has_cpu:
    df_k32_cpu = best_cpu[best_cpu['k'] == 32][['Mode', 'Size_Label', 'GFLOPS']].copy()
    df_k32 = pd.concat([df_k32_cpu, df_k32_gpu[['Mode','Size_Label','GFLOPS']]], ignore_index=True)
else:
    df_k32 = df_k32_gpu[['Mode','Size_Label','GFLOPS']].copy()

df_k32['Size_Label'] = pd.Categorical(df_k32['Size_Label'], categories=SIZE_ORDER, ordered=True)
df_k32 = df_k32[df_k32['Size_Label'].isin(SIZE_ORDER)].sort_values('Size_Label')

n_modes = len(modes_to_plot)
n_sizes = len(SIZE_ORDER)
bar_width = 0.22
x = np.arange(n_sizes)

fig, ax = plt.subplots(figsize=(14, 7))

for i, mode in enumerate(modes_to_plot):
    sub = df_k32[df_k32['Mode'] == mode].set_index('Size_Label').reindex(SIZE_ORDER)['GFLOPS']
    offset = (i - (n_modes - 1) / 2) * (bar_width + 0.03)
    bars = ax.bar(x + offset, sub.values, width=bar_width,
                  color=PALETTE[mode], label=LABELS[mode],
                  edgecolor='white', linewidth=0.7, alpha=0.92)
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 3,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(SIZE_ORDER, rotation=25, ha='right', fontsize=11)
ax.set_ylabel('Prestazioni (GFLOPS)', fontsize=12)
ax.set_ylim(0, df_k32['GFLOPS'].max() * 1.18)
ax.legend(fontsize=11, loc='upper left')
ax.set_title('Confronto Prestazioni Massime (k=32): GPU vs CPU', fontsize=15, fontweight='bold', pad=12)

# Nota se manca CPU
if not has_cpu:
    ax.text(0.02, 0.97, '* Dati CPU non disponibili in questa directory',
            transform=ax.transAxes, fontsize=9, color='gray', va='top')

plt.tight_layout()
plt.savefig('confronto_finale_BARRE.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  confronto_finale_BARRE.png")

# ── GRAFICO 2: Linee GFLOPS vs k, 4 taglie principali ─────────────────────
SIZES_4 = ['2000x2000', '4000x4000', '8000x2000', '2000x8000']
SIZES_4 = [s for s in SIZES_4 if s in SIZE_ORDER_GPU]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

k_vals = sorted(df_gpu['k'].unique())

for idx, size in enumerate(SIZES_4):
    ax = axes[idx]

    # GPU
    for mode in ['CUDA_Opt2D', 'CUDA_Tiled']:
        sub = df_gpu_mean[(df_gpu_mean['Mode'] == mode) & (df_gpu_mean['Size_Label'] == size)].sort_values('k')
        if not sub.empty:
            ax.plot(sub['k'], sub['GFLOPS'], marker='o', linewidth=2.5, markersize=8,
                    color=PALETTE[mode], label=LABELS[mode])

    # CPU
    if has_cpu:
        sub_cpu = best_cpu[(best_cpu['Size_Label'] == size)].sort_values('k')
        if not sub_cpu.empty:
            ax.plot(sub_cpu['k'], sub_cpu['GFLOPS'], marker='D', linewidth=2.2,
                    markersize=8, linestyle='--', color=PALETTE['CPU_Best'], label=LABELS['CPU_Best'])

    ax.set_title(f'Matrice {size}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Valore di k')
    ax.set_ylabel('GFLOPS')
    ax.set_xticks(k_vals)

    # Speedup annotato per k=32
    if has_cpu:
        gpu_k32 = df_gpu_mean[(df_gpu_mean['Mode']=='CUDA_Opt2D') &
                               (df_gpu_mean['Size_Label']==size) &
                               (df_gpu_mean['k']==32)]['GFLOPS']
        cpu_k32 = best_cpu[(best_cpu['Size_Label']==size) & (best_cpu['k']==32)]['GFLOPS']
        if not gpu_k32.empty and not cpu_k32.empty:
            ratio = gpu_k32.values[0] / cpu_k32.values[0]
            ax.text(0.97, 0.05, f'GPU/CPU = {ratio:.1f}×', ha='right', va='bottom',
                    transform=ax.transAxes, fontsize=10, color=PALETTE['CUDA_Opt2D'],
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    if idx == 0:
        ax.legend(fontsize=9, loc='upper left')

fig.suptitle('GPU vs CPU – Scalabilità rispetto all\'intensità aritmetica (k)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('confronto_finale_LINEE.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  confronto_finale_LINEE.png")
