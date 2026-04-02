import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ==========================================
print("Caricamento dati da risultati_avanzati.csv...")
try:
    df = pd.read_csv('../risultatipd.csv')
except FileNotFoundError:
    print("ERRORE: File risultati_avanzati.csv non trovato!")
    exit()

# Creazione Etichette
df['Config'] = df['Mode'] + " (NP:" + df['MPI_Ranks'].astype(str) + ", Thr:" + df['OMP_Threads'].astype(str) + ")"
df['Size_Label'] = df['M'].astype(str) + "x" + df['N'].astype(str)
df['Total_Cores'] = df['MPI_Ranks'] * df['OMP_Threads']

# Ordinamento categorico per le matrici sull'asse X
size_order = ['2000x2000', '4000x4000', '8000x2000', '2000x8000']
df['Size_Label'] = pd.Categorical(df['Size_Label'], categories=size_order, ordered=True)

# ---------------------------------------------------------
# NOVITÀ: RICALCOLO ROBUSTO DELLO SPEEDUP
# Dato che le run 2 e 3 non fanno validazione e avrebbero Speedup = 0,
# lo calcoliamo noi in Python: (Tempo Medio Seriale) / (Tempo Parallelo della Run)
# ---------------------------------------------------------

# Trova il tempo medio della configurazione "Serial 1 1" per ogni matrice e k
serial_baseline = df[df['Mode'] == 'Serial'].groupby(['Size_Label', 'k'])['ParallelTime'].mean().reset_index()
serial_baseline.rename(columns={'ParallelTime': 'BaselineTime'}, inplace=True)

# Unisce questo tempo di riferimento a tutte le righe del dataset
df = df.merge(serial_baseline, on=['Size_Label', 'k'], how='left')

# Ora ricalcoliamo SpeedUp ed Efficienza per tutte le run!
df['SpeedUp'] = df['BaselineTime'] / df['ParallelTime']
df['Efficiency'] = df['SpeedUp'] / df['Total_Cores']

sns.set_theme(style="whitegrid")

MATRIX_TARGET = '4000x4000'
K_TARGET = 32

# FILTRO CRITICO: dataset senza Serial per i grafici GFLOPS
df_parallel = df[df['Mode'] != 'Serial'].copy()

# ==========================================
# 1.5 STATISTICHE GLOBALI (Stampa a terminale)
# ==========================================
print("\n=== STATISTICHE PRESTAZIONI (Media e Dev. Standard GFLOPS) ===")
stats = df.groupby(['Config','Size_Label','k'])['GFLOPS'].agg(['mean','std']).round(2)
print(stats.to_string())
print("=================================================================\n")


# ==========================================
# 2. TABELLA: RELATIVE ERROR
# ==========================================
print("\n" + "="*50)
print("TABELLA 1: ERRORE RELATIVO MASSIMO (Correttezza Numerica)")
print("="*50)
error_table = df[df['RelativeError'] > 0].groupby('Config')['RelativeError'].max().reset_index()
error_table['RelativeError'] = error_table['RelativeError'].apply(lambda x: f"{x:.2e}")
print(error_table.to_string(index=False))
print("="*50 + "\n")

# ==========================================
# GRAFICO 1: GFLOPS vs k (USIAMO IL DATASET FILTRATO)
# ==========================================
print("Generando grafico 1 (GFLOPS vs k)...")
plt.figure(figsize=(10, 6))
subset_1 = df_parallel[df_parallel['Size_Label'] == MATRIX_TARGET]
sns.lineplot(data=subset_1, x='k', y='GFLOPS', hue='Config', marker='o', linewidth=2, markersize=8)
plt.title(f"GFLOPS vs Dimensione k - Matrice {MATRIX_TARGET} (No Serial)", fontweight='bold')
plt.ylabel("Prestazioni (GFLOPS)")
plt.xlabel("Dimensione k (Riempimento registri SIMD)")
plt.xticks(sorted(df_parallel['k'].unique()))
plt.savefig('plot_1_GFLOPS_vs_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> Grafico 1 salvato")

# ==========================================
# GRAFICO 2: GFLOPS vs Taglia Matrice (USIAMO IL DATASET FILTRATO)
# ==========================================
print("Generando grafico 2 (GFLOPS vs Size)...")
plt.figure(figsize=(10, 6))
subset_2 = df_parallel[df_parallel['k'] == K_TARGET]
sns.lineplot(data=subset_2, x='Size_Label', y='GFLOPS', hue='Config', marker='s', linewidth=2, markersize=8)
plt.title(f"GFLOPS vs Taglia Matrice (k={K_TARGET}, No Serial)", fontweight='bold')
plt.ylabel("Prestazioni (GFLOPS)")
plt.xlabel("Dimensione Matrice")
plt.savefig('plot_2_GFLOPS_vs_Size.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> Grafico 2 salvato")

# ==========================================
# GRAFICO 3: SpeedUp vs MPI_Ranks
# ==========================================
print("Generando grafico 3 (SpeedUp vs MPI)...")
plt.figure(figsize=(10, 6))
subset_3 = df_parallel[(df_parallel['k'] == K_TARGET) & (df_parallel['Size_Label'] == MATRIX_TARGET)]

sns.lineplot(data=subset_3, x='MPI_Ranks', y='SpeedUp', hue='Config', marker='D', linewidth=2.5, markersize=9)

plt.title(f"SpeedUp vs Numero Processi MPI - Matrice {MATRIX_TARGET}, k={K_TARGET}", fontweight='bold')
plt.ylabel("SpeedUp (Rispetto al Seriale 1 Core)")
plt.xlabel("Numero Processi MPI (np)")
plt.xticks(sorted(df_parallel['MPI_Ranks'].unique()))
plt.legend()
plt.savefig('plot_3_SpeedUp_vs_MPI.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> Grafico 3 salvato")

# ==========================================
# GRAFICO 4: SpeedUp vs k
# ==========================================
print("Generando grafico 4 (SpeedUp vs k)...")
plt.figure(figsize=(10, 6))
# Qui usiamo subset_1 (già senza Serial) e tracciamo la baseline manualmente
sns.lineplot(data=subset_1, x='k', y='SpeedUp', hue='Config', marker='^', linewidth=2, markersize=8)
plt.axhline(y=1, color='black', linestyle='--', label='Baseline Seriale (1x)')
plt.title(f"SpeedUp vs k - Matrice {MATRIX_TARGET}", fontweight='bold')
plt.ylabel("SpeedUp")
plt.xlabel("Dimensione k")
plt.xticks(sorted(df_parallel['k'].unique()))
plt.legend()
plt.savefig('plot_4_SpeedUp_vs_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> Grafico 4 salvato")

# ==========================================
# GRAFICO 5: Efficienza Parallela
# ==========================================
print("Generando grafico 5 (Efficienza vs MPI)...")
plt.figure(figsize=(10, 6))
sns.lineplot(data=subset_3, x='MPI_Ranks', y='Efficiency', hue='Config', marker='v', linewidth=2, markersize=9)
plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.5, label='Efficienza Teorica (100%)')

plt.title(f"Efficienza Parallela vs Processi MPI - Matrice {MATRIX_TARGET}, k={K_TARGET}", fontweight='bold')
plt.ylabel("Efficienza (SpeedUp / Core Usati)")
plt.xlabel("Numero Processi MPI (np)")

max_eff = df_parallel['Efficiency'].max()
upper_limit = 1.1 if max_eff <= 1.0 else max_eff * 1.1
plt.ylim(0, upper_limit)

plt.xticks(sorted(df_parallel['MPI_Ranks'].unique()))
plt.legend()
plt.savefig('plot_5_Efficiency_vs_MPI.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> Grafico 5 salvato")

print("\nCOMPLETATO! Tutti i grafici sono pronti.")