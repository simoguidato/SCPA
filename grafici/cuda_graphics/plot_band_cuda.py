import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PARAMETRI HARDWARE (DA PERSONALIZZARE)
# ==========================================
PEAK_BANDWIDTH_GBPS = 350

# ==========================================
# 2. CALCOLO DELLA BANDA EFFETTIVA
# ==========================================
df = pd.read_csv('../risultati_cuda_stat.csv')

# Calcolo dei byte minimi trasferiti (M*N + N*k + M*k) * 4 bytes per i float
df['Bytes_Moved'] = (df['M'] * df['N'] + df['N'] * df['k'] + df['M'] * df['k']) * 4
df['GB_Moved'] = df['Bytes_Moved'] / 1e9

# Calcolo della Banda Effettiva (GB/s)
df = df[df['ParallelTime'] > 0]
df['Effective_Bandwidth_GBps'] = df['GB_Moved'] / df['ParallelTime']

df_mean = df.groupby(['Mode', 'M', 'N', 'k'])['Effective_Bandwidth_GBps'].mean().reset_index()
# Matrice target, da testare altri valori
target_M, target_N = 8000, 2000
df_plot = df_mean[(df_mean['M'] == target_M) & (df_mean['N'] == target_N)]

if df_plot.empty:
    # Se non c'è 4000x4000, prendiamo la matrice più grande disponibile
    max_area = (df_mean['M'] * df_mean['N']).idxmax()
    target_M = df_mean.loc[max_area, 'M']
    target_N = df_mean.loc[max_area, 'N']
    df_plot = df_mean[(df_mean['M'] == target_M) & (df_mean['N'] == target_N)]

# ==========================================
# 3. CREAZIONE DEL GRAFICO ELEGANTE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#ffffff')
ax.set_facecolor('#ffffff')
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#b0b0b0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Plot della linea del LIMITE HARDWARE
ax.axhline(y=PEAK_BANDWIDTH_GBPS, color='#e74c3c', linestyle='--', linewidth=2, zorder=1)
ax.text(df_plot['k'].max(), PEAK_BANDWIDTH_GBPS + 5, f' Limite Hardware ({PEAK_BANDWIDTH_GBPS} GB/s)',
        color='#e74c3c', fontweight='bold', ha='right', va='bottom')

# Colori per i diversi Kernel
colors = {'CUDA_Naive': '#7f8c8d', 'CUDA_Opt2D': '#3498db', 'CUDA_Tiled': '#2ecc71', 'CUDA_WarpRow': '#9b59b6'}

for mode in df_plot['Mode'].unique():
    data = df_plot[df_plot['Mode'] == mode].sort_values('k')
    c = colors.get(mode, '#34495e') # Colore di default se il nome è diverso

    ax.plot(data['k'], data['Effective_Bandwidth_GBps'],
            marker='o', linewidth=3, markersize=8, label=mode, color=c, zorder=3)

# ==========================================
# 4. TITOLI E FORMATTAZIONE
# ==========================================
plt.title(f"Saturazione della Banda di Memoria GPU\n(Matrice {target_M}x{target_N})",
          fontsize=16, fontweight='bold', pad=20, loc='left')
plt.xlabel("Intensità Aritmetica (Valore di k)", fontsize=12, fontweight='bold', color='#333333')
plt.ylabel("Banda Effettiva (GB/s)\n", fontsize=12, fontweight='bold', color='#333333')

plt.ylim(0, PEAK_BANDWIDTH_GBPS * 1.15)

plt.legend(loc='upper right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('banda_effettiva_cuda.png', dpi=300)
print(f"Grafico della banda generato per la matrice {target_M}x{target_N}!")