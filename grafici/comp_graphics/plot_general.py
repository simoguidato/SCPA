import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PREPARAZIONE DATI E RICALCOLO FORZATO
# ==========================================
df_omp = pd.read_csv('../risultati_avanzati.csv')
df_cuda = pd.read_csv('../risultati_cuda_stat.csv')

df_omp['Size'] = df_omp['M'].astype(str) + "x" + df_omp['N'].astype(str)
df_omp['Area'] = df_omp['M'] * df_omp['N']
df_omp['Cores'] = df_omp['MPI_Ranks'] * df_omp['OMP_Threads']

df_cuda['Size'] = df_cuda['M'].astype(str) + "x" + df_cuda['N'].astype(str)
df_cuda['Area'] = df_cuda['M'] * df_cuda['N']

df_omp['GFLOPS'] = np.where(df_omp['ParallelTime'] > 0,
                            (2.0 * df_omp['M'] * df_omp['N'] * df_omp['k']) / (df_omp['ParallelTime'] * 1e9), 0)

df_cuda['GFLOPS'] = np.where(df_cuda['ParallelTime'] > 0,
                             (2.0 * df_cuda['M'] * df_cuda['N'] * df_cuda['k']) / (df_cuda['ParallelTime'] * 1e9), 0)

# Ricalcolo Seriale e Speedup
serial = df_omp[df_omp['Mode'] == 'Serial'].groupby(['Size', 'k'])['ParallelTime'].mean().reset_index()
serial.rename(columns={'ParallelTime': 'SerialTime'}, inplace=True)

df_omp = df_omp.merge(serial, on=['Size', 'k'], how='left')
df_omp['Speedup'] = df_omp['SerialTime'] / df_omp['ParallelTime']
df_omp['Efficiency'] = df_omp['Speedup'] / df_omp['Cores']

sns.set_theme(style="whitegrid")

# ==========================================
# 2. NARRATIVA MPI: SpeedUp ed Efficienza
# ==========================================
plt.figure(figsize=(14, 6))
cols_to_agg = ['Speedup', 'Efficiency']
subset_mpi = df_omp[(df_omp['Size'] == '4000x4000') & (df_omp['k'] == 32) & (df_omp['Mode'].isin(['Opt_Hybrid', 'Opt_SMP']))].copy()
subset_mpi = subset_mpi.groupby('Cores')[cols_to_agg].max().reset_index().sort_values('Cores')

plt.subplot(1, 2, 1)
plt.plot(subset_mpi['Cores'], subset_mpi['Speedup'], marker='o', linewidth=3, markersize=8, color='#e74c3c', label='SpeedUp Reale')
plt.plot(subset_mpi['Cores'], subset_mpi['Cores'], linestyle='--', color='gray', label='SpeedUp Ideale')
plt.title("Scalabilità MPI+OMP: SpeedUp", fontsize=14, fontweight='bold')
plt.xlabel("Numero di Core Totali")
plt.ylabel("SpeedUp (Rispetto a Seriale)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(subset_mpi['Cores'], subset_mpi['Efficiency'] * 100, marker='s', linewidth=3, markersize=8, color='#3498db')
plt.axhline(100, linestyle='--', color='gray', label='Efficienza Ideale (100%)')
plt.title("Scalabilità MPI+OMP: Efficienza", fontsize=14, fontweight='bold')
plt.xlabel("Numero di Core Totali")
plt.ylabel("Efficienza (%)")
plt.ylim(0, 110)
plt.legend()

plt.tight_layout()
plt.savefig('1_MPI_Speedup_Efficiency.png', dpi=200)
plt.close()

# ==========================================
# 3. IL CONFRONTO VIOLENTO (CPU vs GPU)
# ==========================================
plt.figure(figsize=(10, 6))
k_target = 32
size_target = '4000x4000'

best_cpu = df_omp[(df_omp['Size'] == size_target) & (df_omp['k'] == k_target)]['GFLOPS'].max()
best_gpu = df_cuda[(df_cuda['Size'] == size_target) & (df_cuda['k'] == k_target)]['GFLOPS'].max()

bars = plt.bar(['Miglior CPU\n(MPI + OMP)', 'Miglior GPU\n(CUDA Float)'], [best_cpu, best_gpu], color=['#3498db', '#2ecc71'])
plt.title(f"CPU vs GPU: Throughput Massimo ({size_target}, k={k_target})", fontsize=16, fontweight='bold')
plt.ylabel("GFLOPS (Più alto è meglio)")

moltiplicatore = best_gpu / best_cpu if best_cpu > 0 else 0
plt.annotate(f"La GPU è {moltiplicatore:.1f}x\npiù veloce!",
             xy=(1, best_gpu), xytext=(0.5, best_gpu * 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=14, fontweight='bold', color='red', ha='center')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.0f} GFLOPS', va='bottom', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('2_Violento_CPU_vs_GPU.png', dpi=200)
plt.close()

# ==========================================
# 4. IL GRAFICO RIASSUNTIVO UNICO (Time vs Size)
# ==========================================
plt.figure(figsize=(12, 7))
k_summary = 32

line_serial = df_omp[(df_omp['Mode'] == 'Serial') & (df_omp['k'] == k_summary)].groupby(['Area', 'Size'], observed=True)['ParallelTime'].mean().reset_index().sort_values('Area')
line_cpu = df_omp[(df_omp['Mode'] != 'Serial') & (df_omp['k'] == k_summary)].groupby(['Area', 'Size'], observed=True)['ParallelTime'].min().reset_index().sort_values('Area')
line_gpu = df_cuda[df_cuda['k'] == k_summary].groupby(['Area', 'Size'], observed=True)['ParallelTime'].min().reset_index().sort_values('Area')

plt.plot(line_serial['Size'], line_serial['ParallelTime'], marker='x', color='gray', linestyle=':', linewidth=2, label='1. Seriale (Baseline)')
plt.plot(line_cpu['Size'], line_cpu['ParallelTime'], marker='s', color='#3498db', linestyle='--', linewidth=3, markersize=8, label='2. Miglior CPU (MPI + OMP)')
plt.plot(line_gpu['Size'], line_gpu['ParallelTime'], marker='o', color='#2ecc71', linestyle='-', linewidth=4, markersize=10, label='3. Miglior GPU (CUDA)')

plt.yscale('log')
plt.title("Il Riassunto: Tempo di Esecuzione (k=32)", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Taglia Matrice (M x N)", fontsize=14)
plt.ylabel("Tempo in Secondi (Scala Logaritmica)", fontsize=14)
plt.xticks(rotation=45)
plt.legend(fontsize=12, loc='upper left')

last_size = line_gpu['Size'].iloc[-1]
t_cpu = line_cpu['ParallelTime'].iloc[-1]
t_gpu = line_gpu['ParallelTime'].iloc[-1]

plt.annotate(f"CPU: {t_cpu:.3f}s", xy=(last_size, t_cpu), xytext=(-50, 15), textcoords='offset points', color='#3498db', fontweight='bold')
plt.annotate(f"GPU: {t_gpu:.4f}s", xy=(last_size, t_gpu), xytext=(-50, -25), textcoords='offset points', color='#2ecc71', fontweight='bold')

plt.tight_layout()
plt.savefig('3_Master_Summary_Slide.png', dpi=200)
print("I 3 grafici definitivi con i GFLOPS ricalcolati sono pronti.")