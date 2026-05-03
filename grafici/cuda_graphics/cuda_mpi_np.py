import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  CARICAMENTO
# ═══════════════════════════════════════════════════════════════
CSV_FILE = '../risultati_cuda_mpi_all.csv'

df = pd.read_csv(CSV_FILE)
if 'GFLOPS' not in df.columns:
    df = pd.read_csv(CSV_FILE,
                     names=['Mode','M','N','k','MPI_Ranks','OMP_Threads','Run',
                            'ParallelTime','SequentialTime','RelativeError',
                            'GFLOPS','SpeedUp'],
                     skiprows=1)

df['Latency_ms'] = df['ParallelTime'] * 1000.0
df['Size_Label'] = df['M'].astype(str) + 'x' + df['N'].astype(str)
df['Geo'] = df.apply(
    lambda r: 'Tall' if r.M/r.N > 1.6 else ('Wide' if r.M/r.N < 0.6 else 'Square'),
    axis=1
)

# Ordine taglie: crescente per area, poi Tall→Square→Wide
sizes_info = (df[['Size_Label','M','N','Geo']].drop_duplicates()
              .assign(area=lambda d: d.M*d.N,
                      gk=lambda d: d['Geo'].map({'Tall':0,'Square':1,'Wide':2}))
              .sort_values(['area','gk']))
SIZE_ORDER = sizes_info['Size_Label'].tolist()
df['Size_Label'] = pd.Categorical(df['Size_Label'], categories=SIZE_ORDER, ordered=True)
df_m = (df[df['Run'] > 1]
        .groupby(['Mode','MPI_Ranks','Size_Label','M','N','k','Geo'], observed=True)
        .agg(GFLOPS=('GFLOPS','mean'),
             GFLOPS_std=('GFLOPS','std'),
             Latency_ms=('Latency_ms','mean'),
             Latency_std=('Latency_ms','std'))
        .reset_index())

K_VALS   = sorted(df['k'].unique())
NP_VALS  = sorted(df['MPI_Ranks'].unique())
KERNELS  = ['CUDA_Naive','CUDA_Opt2D','CUDA_Tiled','CUDA_WarpRow']

KERNEL_LBL = {
    'CUDA_Naive':   'K0 – 1D Baseline',
    'CUDA_Opt2D':   'K1 – 2D Grid',
    'CUDA_Tiled':   'K2 – Shared Mem Tiling',
    'CUDA_WarpRow': 'K3 – Warp-Row Reg Tiling',
}
KERNEL_PAL = {
    'CUDA_Naive':   '#4b0082',
    'CUDA_Opt2D':   '#c0396b',
    'CUDA_Tiled':   '#f08030',
    'CUDA_WarpRow': '#8b5cf6',
}
NP_PAL     = {1:'#1f4e79', 2:'#2e75b6', 4:'#70ad47', 8:'#ed7d31', 16:'#c00000'}
NP_MARKERS = {1:'o', 2:'s', 4:'^', 8:'D', 16:'*'}
GEO_CLR    = {'Tall':'#2980b9','Square':'#c0392b','Wide':'#27ae60'}
GEO_LS     = {'Tall':'-','Square':'--','Wide':':'}

plt.rcParams.update({
    'font.size':10,'axes.titlesize':11,'axes.titleweight':'bold',
    'figure.facecolor':'#F8F9FA','axes.facecolor':'#EAECEE',
    'axes.grid':True,'grid.alpha':0.35,'grid.linestyle':'--',
    'axes.spines.top':False,'axes.spines.right':False,
})

def annotate_last(ax, x, y, label, color, offset=(5,0)):
    ax.annotate(f'{y:.0f}', xy=(x,y), xytext=offset,
                textcoords='offset points', fontsize=7.5,
                color=color, fontweight='bold')

# ═══════════════════════════════════════════════════════════════
#  FIG 1 — GFLOPS vs NP (k=32): pannello per taglia, linee per kernel
# ═══════════════════════════════════════════════════════════════
print("FIG 1...")
df_k32 = df_m[df_m['k'] == 32]
ncols, nrows = 4, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 9), sharey=False)
fig.suptitle('GFLOPS vs Processi MPI (k=32) — pannello per taglia, linee per kernel',
             fontsize=14, fontweight='bold')

for idx, size in enumerate(SIZE_ORDER):
    ax = axes[idx // ncols][idx % ncols]
    sub = df_k32[df_k32['Size_Label'] == size]

    for mode in KERNELS:
        grp = sub[sub['Mode']==mode].sort_values('MPI_Ranks')
        if grp.empty: continue
        ax.plot(grp['MPI_Ranks'], grp['GFLOPS'],
                color=KERNEL_PAL[mode], marker='o', lw=2.2, ms=7,
                label=KERNEL_LBL[mode])
        ax.fill_between(grp['MPI_Ranks'],
                        grp['GFLOPS'] - grp['GFLOPS_std'].fillna(0),
                        grp['GFLOPS'] + grp['GFLOPS_std'].fillna(0),
                        color=KERNEL_PAL[mode], alpha=0.12)

    geo = df[df['Size_Label']==size]['Geo'].iloc[0]
    ax.set_title(f'{size}  [{geo}]', fontsize=10, color=GEO_CLR[geo])
    ax.set_xlabel('Processi MPI')
    ax.set_ylabel('GFLOPS' if idx % ncols == 0 else '')
    ax.set_xticks(NP_VALS)
    ax.set_ylim(bottom=0)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

axes[1][3].set_visible(False)
plt.tight_layout()
plt.savefig('fig1_gflops_vs_np_per_taglia.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig1_gflops_vs_np_per_taglia.png")

# ═══════════════════════════════════════════════════════════════
#  FIG 2 — GFLOPS vs NP (k=32): pannello per kernel, linee per taglia
# ═══════════════════════════════════════════════════════════════
print("FIG 2...")
size_colors = plt.cm.tab10(np.linspace(0, 0.85, len(SIZE_ORDER)))

fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=False)
fig.suptitle('GFLOPS vs Processi MPI (k=32) — pannello per kernel, linee per taglia',
             fontsize=14, fontweight='bold')

for ki, mode in enumerate(KERNELS):
    ax = axes[ki]
    sub = df_k32[df_k32['Mode']==mode]

    for si, size in enumerate(SIZE_ORDER):
        grp = sub[sub['Size_Label']==size].sort_values('MPI_Ranks')
        if grp.empty: continue
        geo = grp['Geo'].iloc[0]
        ax.plot(grp['MPI_Ranks'], grp['GFLOPS'],
                color=size_colors[si], linestyle=GEO_LS[geo],
                marker='o', lw=2, ms=6, label=size)
        last = grp.iloc[-1]
        annotate_last(ax, last['MPI_Ranks'], last['GFLOPS'], size,
                      size_colors[si])

    ax.set_title(KERNEL_LBL[mode], fontsize=11, color=KERNEL_PAL[mode])
    ax.set_xlabel('Processi MPI')
    ax.set_ylabel('GFLOPS' if ki == 0 else '')
    ax.set_xticks(NP_VALS)
    ax.set_ylim(bottom=0)
    if ki == 0:
        ax.legend(fontsize=7.5, loc='upper right')

fig.text(0.5, -0.02,
         'Stile linea:  ─── Tall   ╌╌╌ Square   ···  Wide',
         ha='center', fontsize=9, color='#555555')
plt.tight_layout()
plt.savefig('fig2_gflops_vs_np_per_kernel.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig2_gflops_vs_np_per_kernel.png")

# ═══════════════════════════════════════════════════════════════
#  FIG 3 — GFLOPS vs k: pannello per kernel, linee per NP
#           una riga per ogni taglia principale
# ═══════════════════════════════════════════════════════════════
print("FIG 3...")
TARGET_SIZES = [s for s in ['4000x4000','8000x2000','2000x8000'] if s in SIZE_ORDER]
fig, axes = plt.subplots(len(TARGET_SIZES), 4,
                         figsize=(22, 5*len(TARGET_SIZES)), sharey=False)
if len(TARGET_SIZES) == 1:
    axes = axes[np.newaxis, :]
fig.suptitle('GFLOPS vs k — pannello per kernel, linee per Processi MPI',
             fontsize=14, fontweight='bold')

for ri, size in enumerate(TARGET_SIZES):
    geo = df[df['Size_Label']==size]['Geo'].iloc[0]
    for ki, mode in enumerate(KERNELS):
        ax = axes[ri][ki]
        sub = df_m[(df_m['Mode']==mode) & (df_m['Size_Label']==size)]

        for np_ in NP_VALS:
            grp = sub[sub['MPI_Ranks']==np_].sort_values('k')
            if grp.empty: continue
            ax.plot(grp['k'], grp['GFLOPS'],
                    color=NP_PAL[np_], marker=NP_MARKERS[np_],
                    lw=2, ms=7, label=f'NP={np_}')
            v32 = grp[grp['k']==32]
            if not v32.empty:
                annotate_last(ax, 32, v32['GFLOPS'].values[0],
                              f'NP={np_}', NP_PAL[np_])

        t = KERNEL_LBL[mode] if ri==0 else ''
        ax.set_title((t+'\n' if t else '') + f'{size} [{geo}]',
                     fontsize=10, color=KERNEL_PAL[mode])
        ax.set_xticks(K_VALS)
        ax.set_xlabel('k')
        ax.set_ylabel('GFLOPS' if ki==0 else '')
        ax.set_ylim(bottom=0)
        if ri==0 and ki==0:
            ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('fig3_gflops_vs_k_per_np.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig3_gflops_vs_k_per_np.png")

# ═══════════════════════════════════════════════════════════════
#  FIG 4 — Heatmap GFLOPS (NP × k) per ogni kernel e taglia
# ═══════════════════════════════════════════════════════════════
print("FIG 4...")
vmax_top    = df_m[df_m['Mode']!='CUDA_Naive']['GFLOPS'].max()
vmax_naive  = df_m[df_m['Mode']=='CUDA_Naive']['GFLOPS'].max()

fig, axes = plt.subplots(len(SIZE_ORDER), 4,
                         figsize=(22, 4.2*len(SIZE_ORDER)))
fig.suptitle('Heatmap GFLOPS (Processi MPI × k) — righe: taglie, colonne: kernel',
             fontsize=14, fontweight='bold')

for ri, size in enumerate(SIZE_ORDER):
    geo = df[df['Size_Label']==size]['Geo'].iloc[0]
    for ki, mode in enumerate(KERNELS):
        ax = axes[ri][ki]
        sub = df_m[(df_m['Mode']==mode) & (df_m['Size_Label']==size)]
        pivot = (sub.pivot_table(index='MPI_Ranks', columns='k', values='GFLOPS')
                 .reindex(NP_VALS).reindex(columns=K_VALS))

        vm = vmax_naive if mode=='CUDA_Naive' else vmax_top
        im = ax.imshow(pivot.values, cmap='RdYlGn',
                       aspect='auto', vmin=0, vmax=vm)

        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                if not np.isnan(v):
                    col = 'white' if v < vm*0.5 else 'black'
                    ax.text(c, r, f'{v:.0f}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color=col)

        ax.set_xticks(range(len(K_VALS)))
        ax.set_xticklabels(K_VALS, fontsize=8)
        ax.set_yticks(range(len(NP_VALS)))
        ax.set_yticklabels(NP_VALS, fontsize=8)
        ax.set_xlabel('k' if ri==len(SIZE_ORDER)-1 else '')
        if ki == 0:
            ax.set_ylabel(f'{size}\nNP', fontsize=8,
                          color=GEO_CLR[geo], fontweight='bold')
        else:
            ax.set_ylabel('')
        if ri == 0:
            ax.set_title(KERNEL_LBL[mode], fontsize=10,
                         color=KERNEL_PAL[mode], fontweight='bold')

plt.tight_layout()
plt.savefig('fig4_heatmap_gflops.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig4_heatmap_gflops.png")

# ═══════════════════════════════════════════════════════════════
#  FIG 5 — Latenza (ms) vs NP: pannello per kernel, linee per k
# ═══════════════════════════════════════════════════════════════
print("FIG 5...")
K_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(K_VALS)))
K_PAL    = dict(zip(K_VALS, K_COLORS))

TARGET_SIZE_LAT = '4000x4000' if '4000x4000' in SIZE_ORDER else SIZE_ORDER[3]

fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=False)
fig.suptitle(f'Latenza (ms) vs Processi MPI — Matrice {TARGET_SIZE_LAT}\nLinee per k, pannello per kernel',
             fontsize=14, fontweight='bold')

for ki, mode in enumerate(KERNELS):
    ax = axes[ki]
    sub = df_m[(df_m['Mode']==mode) & (df_m['Size_Label']==TARGET_SIZE_LAT)]

    for k in K_VALS:
        grp = sub[sub['k']==k].sort_values('MPI_Ranks')
        if grp.empty: continue
        ax.plot(grp['MPI_Ranks'], grp['Latency_ms'],
                color=K_PAL[k], marker='o', lw=2, ms=7,
                label=f'k={k}')
        ax.fill_between(grp['MPI_Ranks'],
                        grp['Latency_ms'] - grp['Latency_std'].fillna(0),
                        grp['Latency_ms'] + grp['Latency_std'].fillna(0),
                        color=K_PAL[k], alpha=0.12)
        last = grp.iloc[-1]
        ax.annotate(f'{last["Latency_ms"]:.1f}ms',
                    xy=(last['MPI_Ranks'], last['Latency_ms']),
                    xytext=(4, 0), textcoords='offset points',
                    fontsize=7, color=K_PAL[k], fontweight='bold')

    ax.set_title(KERNEL_LBL[mode], fontsize=11, color=KERNEL_PAL[mode])
    ax.set_xlabel('Processi MPI')
    ax.set_ylabel('Latenza (ms)' if ki==0 else '')
    ax.set_xticks(NP_VALS)
    if ki == 0:
        ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('fig5_latency_vs_np.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig5_latency_vs_np.png")

# ═══════════════════════════════════════════════════════════════
#  FIG 6 — Overhead MPI: degrado % rispetto a NP=1
# ═══════════════════════════════════════════════════════════════
print("FIG 6...")
df_np1  = df_m[df_m['MPI_Ranks']==1][['Mode','Size_Label','k','GFLOPS']] \
    .rename(columns={'GFLOPS':'GFLOPS_np1'})
df_deg  = df_m.merge(df_np1, on=['Mode','Size_Label','k'])
df_deg['Degradation_pct'] = (df_deg['GFLOPS'] / df_deg['GFLOPS_np1']) * 100

fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
fig.suptitle('Overhead MPI — GFLOPS% rispetto a NP=1 (100% = nessuna perdita)\nk=32, linee per taglia',
             fontsize=14, fontweight='bold')

df_deg_k32 = df_deg[df_deg['k']==32]

for ki, mode in enumerate(KERNELS):
    ax = axes[ki]
    sub = df_deg_k32[df_deg_k32['Mode']==mode]

    for si, size in enumerate(SIZE_ORDER):
        grp = sub[sub['Size_Label']==size].sort_values('MPI_Ranks')
        if grp.empty: continue
        geo = grp['Geo'].iloc[0]
        ax.plot(grp['MPI_Ranks'], grp['Degradation_pct'],
                color=size_colors[si], linestyle=GEO_LS[geo],
                marker='o', lw=2, ms=6, label=size)

    ax.axhline(100, color='gray', lw=1.2, ls='--', alpha=0.6, label='NP=1 baseline')
    ax.axhline(50, color='red', lw=1, ls=':', alpha=0.4)
    ax.set_title(KERNEL_LBL[mode], fontsize=11, color=KERNEL_PAL[mode])
    ax.set_xlabel('Processi MPI')
    ax.set_ylabel('GFLOPS%' if ki==0 else '')
    ax.set_xticks(NP_VALS)
    ax.set_ylim(0, 115)
    if ki == 0:
        ax.legend(fontsize=7.5, loc='lower left')

plt.tight_layout()
plt.savefig('fig6_overhead_mpi.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig6_overhead_mpi.png")

# ═══════════════════════════════════════════════════════════════
#  FIG 7 — Multi-linea con doppio asse Y (GFLOPS + Latenza)
#           Stile come la figura di riferimento:
#           asse X = NP, linee per k, doppio Y per GFLOPS e Latenza
# ═══════════════════════════════════════════════════════════════
print("FIG 7...")
TARGET_SIZE_7 = '4000x4000' if '4000x4000' in SIZE_ORDER else SIZE_ORDER[3]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
fig.suptitle(f'GFLOPS e Latenza vs Processi MPI — {TARGET_SIZE_7}\n'
             f'Asse sx = GFLOPS (linee piene), Asse dx = Latenza ms (linee tratteggiate)',
             fontsize=14, fontweight='bold')

for ki, mode in enumerate(KERNELS):
    ax1 = axes[ki]
    ax2 = ax1.twinx()

    sub = df_m[(df_m['Mode']==mode) & (df_m['Size_Label']==TARGET_SIZE_7)]

    for k in K_VALS:
        grp = sub[sub['k']==k].sort_values('MPI_Ranks')
        if grp.empty: continue
        color = K_PAL[k]

        l1, = ax1.plot(grp['MPI_Ranks'], grp['GFLOPS'],
                       color=color, marker='o', lw=2.2, ms=8,
                       markerfacecolor=color, label=f'k={k}  GFLOPS')
        ax2.plot(grp['MPI_Ranks'], grp['Latency_ms'],
                 color=color, marker='o', lw=2.2, ms=8,
                 linestyle='--', markerfacecolor='white',
                 markeredgecolor=color, markeredgewidth=1.5)

        last = grp.iloc[-1]
        ax1.annotate(f'{last["GFLOPS"]:.0f}',
                     xy=(last['MPI_Ranks'], last['GFLOPS']),
                     xytext=(4, 2), textcoords='offset points',
                     fontsize=7, color=color, fontweight='bold')

    ax1.set_title(KERNEL_LBL[mode], fontsize=12, color=KERNEL_PAL[mode])
    ax1.set_xlabel('Processi MPI', fontsize=10)
    ax1.set_ylabel('GFLOPS', fontsize=10, color='#222222')
    ax2.set_ylabel('Latenza (ms)', fontsize=10, color='#555555')
    ax1.set_xticks(NP_VALS)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor='#222222')
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax2.spines['right'].set_visible(True)

    # Legenda unificata
    handles, labels = ax1.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([0],[0], color='gray', lw=2, ls='--',
                       marker='o', markerfacecolor='white',
                       label='─── GFLOPS (sx)   ╌╌╌ Latenza/ms (dx)')]
    ax1.legend(handles=handles, fontsize=7.5, loc='upper right',
               ncol=2, framealpha=0.85)

plt.tight_layout()
plt.savefig('fig7_dual_axis_gflops_latency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig7_dual_axis_gflops_latency.png")

print()
print("="*55)
print("✅  Tutte le figure generate:")
for i in range(1, 8):
    names = {
        1: 'GFLOPS vs NP  (pannelli per taglia)',
        2: 'GFLOPS vs NP  (pannelli per kernel)',
        3: 'GFLOPS vs k   (linee per NP)',
        4: 'Heatmap GFLOPS (NP × k)',
        5: 'Latenza vs NP  (pannelli per kernel)',
        6: 'Overhead MPI % rispetto a NP=1',
        7: 'Doppio asse GFLOPS + Latenza vs NP',
    }
    print(f"  fig{i}_*.png — {names[i]}")
print("="*55)