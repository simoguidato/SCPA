import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../risultati_avanzati.csv')
df['Size'] = df['M'].astype(str) + "x" + df['N'].astype(str)
df['Config'] = "NP:" + df['MPI_Ranks'].astype(str) + " T:" + df['OMP_Threads'].astype(str)
df_mean = df.groupby(['Mode', 'Size', 'k', 'Config'])[['GFLOPS', 'ParallelTime']].mean().reset_index()

sns.set_theme(style="whitegrid")

# GRAFICO 1: Prestazioni per Topologia (k=32)
plt.figure(figsize=(14, 7))
sns.barplot(data=df_mean[(df_mean['k'] == 3) & (df_mean['Mode'] == 'Opt_Hybrid')],
            x='Size', y='GFLOPS', hue='Config')
plt.title("OpenMP/MPI: Impatto della Topologia sulle Prestazioni (k=3)")
plt.savefig('omp_baseline_topology.png', dpi=200, bbox_inches='tight')

# GRAFICO 2: Prestazioni vs k (su matrice 4000x4000)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_mean[df_mean['Size'] == '8000x8000'], x='k', y='GFLOPS', hue='Config', marker='s')
plt.title("OpenMP/MPI: Scalabilità rispetto a k (Matrice 8000x8000)")
plt.savefig('omp_baseline_k_scaling.png', dpi=200, bbox_inches='tight')

print("Grafici OpenMP Baseline generati.")