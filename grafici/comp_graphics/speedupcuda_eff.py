import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
df_cuda = pd.read_csv('../risultati_cuda_stat.csv')
df_cuda['Size'] = df_cuda['M'].astype(str) + "x" + df_cuda['N'].astype(str)
df_cuda['Area'] = df_cuda['M'] * df_cuda['N']
df_omp = pd.read_csv('../risultati_avanzati.csv')
df_omp['Size'] = df_omp['M'].astype(str) + "x" + df_omp['N'].astype(str)

# Baseline 1: CPU Seriale (per lo SpeedUp Assoluto)
serial_base = df_omp[df_omp['Mode'] == 'Serial'].groupby(['Size', 'k'])['ParallelTime'].mean().reset_index()
serial_base.rename(columns={'ParallelTime': 'SerialTime'}, inplace=True)

# Baseline 2: CUDA Naive
cuda_naive_base = df_cuda[df_cuda['Mode'] == 'CUDA_Naive'].groupby(['Size', 'k'])['ParallelTime'].mean().reset_index()
cuda_naive_base.rename(columns={'ParallelTime': 'NaiveTime'}, inplace=True)

df_plot = df_cuda[df_cuda['Mode'] == 'CUDA_Tiled'].copy() # Usa CUDA_WarpRow se è il tuo migliore
df_plot = df_plot.merge(serial_base, on=['Size', 'k'], how='left')
df_plot = df_plot.merge(cuda_naive_base, on=['Size', 'k'], how='left')

# ==========================================
# 2. CALCOLO METRICHE HPC
# ==========================================
# Quanto è più veloce del singolo core della CPU?
df_plot['Absolute_SpeedUp'] = df_plot['SerialTime'] / df_plot['ParallelTime']

# Quanto è più veloce del codice GPU non ottimizzato?
df_plot['Relative_SpeedUp'] = df_plot['NaiveTime'] / df_plot['ParallelTime']

df_agg = df_plot[df_plot['Size'] == '4000x4000'].groupby('k').mean(numeric_only=True).reset_index()
# ==========================================
# 3. CREAZIONE DEL GRAFICO DOPPIO
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#ffffff')

# --- PLOT 1: SpeedUp Assoluto (Hardware Power) ---
ax1 = axes[0]
ax1.plot(df_agg['k'], df_agg['Absolute_SpeedUp'], marker='o', color='#2ecc71', linewidth=3, markersize=8)
ax1.set_title("CUDA SpeedUp Assoluto\n(GPU Tiled vs CPU Singolo Core)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Intensità Aritmetica (k)", fontsize=12)
ax1.set_ylabel("Fattore di SpeedUp (X volte più veloce)", fontsize=12)

max_abs = df_agg['Absolute_SpeedUp'].max()
ax1.annotate(f"Picco: {max_abs:.0f}x!", xy=(df_agg['k'].max(), max_abs),
             xytext=(-40, -20), textcoords='offset points', color='#2ecc71', fontweight='bold')

# --- PLOT 2: SpeedUp Relativo (Algorithmic Efficiency) ---
ax2 = axes[1]
ax2.plot(df_agg['k'], df_agg['Relative_SpeedUp'], marker='s', color='#f39c12', linewidth=3, markersize=8)
ax2.axhline(1.0, color='gray', linestyle='--', label='Baseline CUDA Naive (1.0x)')
ax2.set_title("Efficienza Algoritmica CUDA\n(Tiled vs Naive)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Intensità Aritmetica (k)", fontsize=12)
ax2.set_ylabel("Miglioramento Relativo (X volte)", fontsize=12)
ax2.legend(loc='lower right')

max_rel = df_agg['Relative_SpeedUp'].max()
ax2.annotate(f"L'ottimizzazione\nha reso il codice\n{max_rel:.1f}x più veloce!",
             xy=(df_agg['k'].max(), max_rel),
             xytext=(-90, -40), textcoords='offset points', color='#d35400', fontweight='bold')

plt.tight_layout()
plt.savefig('4_CUDA_SpeedUp_Efficienza.png', dpi=200)
print("Grafico CUDA SpeedUp generato!")