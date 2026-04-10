import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione Report Generale HPC...")
df = pd.read_csv('risultati_avanzatisuca.csv')
df['Config'] = df['Mode'] + " (NP:" + df['MPI_Ranks'].astype(str) + ", Thr:" + df['OMP_Threads'].astype(str) + ")"
df['Size_Label'] = pd.Categorical(df['M'].astype(str) + "x" + df['N'].astype(str),
                                  categories=['2000x2000', '4000x4000', '8000x2000', '2000x8000'], ordered=True)
df['Total_Cores'] = df['MPI_Ranks'] * df['OMP_Threads']

# Ricalcolo SpeedUp
serial_means = df[(df['Mode'] == 'Serial') & (df['Run'] > 1)].groupby(
    ['Size_Label', 'k'])['ParallelTime'].mean().reset_index()

serial_means.rename(columns={'ParallelTime': 'BaselineTime'}, inplace=True)
df = df.merge(serial_means, on=['Size_Label', 'k'], how='left')
df['SpeedUp'] = df['BaselineTime'] / df['ParallelTime']
df['Efficiency'] = df['SpeedUp'] / df['Total_Cores']

df_parallel = df[df['Mode'] != 'Serial'].copy()
MATRIX_TARGET = '4000x4000'
K_TARGET = 32

sns.set_theme(style="whitegrid")

# GRAFICO 1: GFLOPS vs k (Linee)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_parallel[df_parallel['Size_Label'] == MATRIX_TARGET], x='k', y='GFLOPS', hue='Config', marker='o')
plt.title(f"GFLOPS vs k - Matrice {MATRIX_TARGET}")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Legenda fuori
plt.tight_layout()
plt.savefig('report_1_GFLOPS_vs_k.png', dpi=300)
plt.close()

# GRAFICO 2: SpeedUp Bar (Con Etichette Ruotate)
plt.figure(figsize=(12, 7))
subset_3 = df_parallel[(df_parallel['k'] == K_TARGET) & (df_parallel['Size_Label'] == MATRIX_TARGET)]
sns.barplot(data=subset_3, x='Config', y='SpeedUp', hue='Config', palette='viridis', legend=False)
plt.axhline(y=40, color='red', linestyle='--', label='Ideale (40x)')
plt.xticks(rotation=30, ha='right') # <-- FIX per la leggibilità
plt.title(f"SpeedUp - {MATRIX_TARGET}, k={K_TARGET}")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('report_2_SpeedUp.png', dpi=300)
plt.close()

# GRAFICO 3: Efficienza Bar (Con Etichette Ruotate)
plt.figure(figsize=(12, 7))
sns.barplot(data=subset_3, x='Config', y='Efficiency', hue='Config', palette='magma', legend=False)
plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.5, label='Efficienza Ideale (1.0)')
plt.ylim(0, 1.1)
plt.xticks(rotation=30, ha='right') # <-- FIX per la leggibilità
plt.title(f"Efficienza Parallela - {MATRIX_TARGET}, k={K_TARGET}")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('report_3_Efficiency.png', dpi=300)
plt.close()

print("✅ Report generale salvato!")