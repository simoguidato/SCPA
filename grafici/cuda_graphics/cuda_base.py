import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento e pulizia
df = pd.read_csv('../risultati_cuda_stat.csv')
df['Size'] = df['M'].astype(str) + "x" + df['N'].astype(str)
df['Area'] = df['M'] * df['N']
df_mean = df.groupby(['Mode', 'Size', 'k', 'Area'])[['GFLOPS', 'ParallelTime']].mean().reset_index()
df_mean = df_mean.sort_values('Area')

sns.set_theme(style="whitegrid")

# GRAFICO 1: Throughput (GFLOPS) vs Taglia (per k=32)
plt.figure(figsize=(12, 6))
sns.barplot(data=df_mean[df_mean['k'] == 32], x='Size', y='GFLOPS', hue='Mode')
plt.title("CUDA Baseline: Throughput (GFLOPS) vs Taglia Matrice (k=32)")
plt.xticks(rotation=45)
plt.savefig('cuda_baseline_throughput.png', dpi=200, bbox_inches='tight')

# GRAFICO 2: Latenza (Tempo) vs Taglia (per k=32)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_mean[df_mean['k'] == 32], x='Size', y='ParallelTime', hue='Mode', marker='o')
plt.yscale('log')
plt.title("CUDA Baseline: Latenza (Secondi) vs Taglia Matrice (k=32)")
plt.ylabel("Tempo (s) - Scala Logaritmica")
plt.xticks(rotation=45)
plt.savefig('cuda_baseline_latency.png', dpi=200, bbox_inches='tight')

print("Grafici CUDA Baseline generati.")