import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Disattiviamo i warning per un output pulito da terminale
warnings.filterwarnings("ignore")

# Configurazione stile globale per tesi/relazioni
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# 1. Caricamento Dati e Preprocessing
print("Caricamento dati da risultati_cuda.csv...")
df = pd.read_csv("risultati_cuda.csv")

# Creiamo una colonna leggibile per la dimensione della matrice
df['Size'] = df['M'].astype(str) + "x" + df['N'].astype(str)

# Separiamo i dati della Run 1 per il corretto calcolo dello SpeedUp
df_run1 = df[df['Run'] == 1].copy()

# ==============================================================================
# 1. GRAFICO AL VARIARE DI k (Suddiviso per Dimensione Matrice)
# ==============================================================================
print("Generazione: 1_cuda_gflops_vs_k.png")
df_np1 = df[df['MPI_Ranks'] == 1]
# col_wrap=3 adatta perfettamente le 6 matrici in una griglia 2x3
g1 = sns.relplot(
    data=df_np1, x='k', y='GFLOPS', hue='Mode', col='Size', col_wrap=3,
    kind='line', marker='o', height=3.5, aspect=1.3, facet_kws={'sharey': False}
)
g1.set(xscale="symlog", xticks=[3, 6, 8, 20, 32, 64, 128])
g1.set_xticklabels([3, 6, 8, 20, 32, 64, 128])
g1.set_axis_labels("Larghezza Multivettore (k)", "Performance (GFLOPS)")
g1.set_titles("Matrice: {col_name}")
g1.fig.suptitle('GFLOPS al variare di k (1 MPI Rank)', y=1.05, fontweight='bold')
g1.savefig('1_cuda_gflops_vs_k.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 2. GRAFICO AL VARIARE DELLA DIMENSIONE (Suddiviso per k rappresentativi)
# ==============================================================================
print("Generazione: 2_cuda_gflops_vs_size.png")
df_k_sample = df_np1[df_np1['k'].isin([3, 8, 32, 128])]
g2 = sns.catplot(
    data=df_k_sample, x='Size', y='GFLOPS', hue='Mode', col='k', col_wrap=2,
    kind='bar', height=4, aspect=1.5, sharey=False, palette='muted'
)
g2.set_axis_labels("Dimensione Matrice (MxN)", "Performance (GFLOPS)")
g2.set_titles("Multivettore k={col_name}")
g2.fig.suptitle('GFLOPS al variare della Dimensione (1 MPI Rank)', y=1.05, fontweight='bold')
# Ruotiamo le etichette di 45 gradi per far leggere bene le 6 dimensioni
for ax in g2.axes.flat:
    ax.tick_params(axis='x', rotation=45)
g2.savefig('2_cuda_gflops_vs_size.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 3. GRAFICO AL VARIARE DI np (MPI Ranks da 1 a 8)
# ==============================================================================
print("Generazione: 3_cuda_gflops_vs_np.png")
g3 = sns.relplot(
    data=df, x='MPI_Ranks', y='GFLOPS', hue='Mode', col='Size', col_wrap=3,
    kind='line', marker='s', height=3.5, aspect=1.3, errorbar=None, facet_kws={'sharey': False}
)
# Aggiunto il valore 8 tra i tick dell'asse X
g3.set(xticks=[1, 2, 4, 8])
g3.set_axis_labels("Numero di Rank MPI (np)", "GFLOPS Medi")
g3.set_titles("Matrice: {col_name}")
g3.fig.suptitle('Scalabilità MPI sui GFLOPS', y=1.05, fontweight='bold')
g3.savefig('3_cuda_gflops_vs_np.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 4. GRAFICO DELLO SPEEDUP ASSOLUTO vs np (Run 1, Ranks da 1 a 8)
# ==============================================================================
print("Generazione: 4_cuda_speedup_vs_np.png")
g4 = sns.catplot(
    data=df_run1, x='MPI_Ranks', y='SpeedUp', hue='Mode', col='Size', col_wrap=3,
    kind='point', height=3.5, aspect=1.3, sharey=False
)
g4.set_axis_labels("Numero di Rank MPI (np)", "SpeedUp vs Sequenziale")
g4.set_titles("Matrice: {col_name}")
g4.fig.suptitle('SpeedUp in configurazione distribuita (Run 1)', y=1.05, fontweight='bold')
g4.savefig('4_cuda_speedup_vs_np.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 5. GRAFICI INCROCIATI (HEATMAPS)
# ==============================================================================
print("Generazione: 5_cuda_heatmap_k_vs_size.png")
best_kernel = df.groupby('Mode')['GFLOPS'].mean().idxmax()
df_best = df[(df['Mode'] == best_kernel) & (df['MPI_Ranks'] == 1)]

pivot_k_size = df_best.pivot_table(values='GFLOPS', index='Size', columns='k', aggfunc='mean')

plt.figure(figsize=(12, 7))
sns.heatmap(pivot_k_size, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5)
plt.title(f'Heatmap GFLOPS: Dimensione vs k (Kernel: {best_kernel})', fontsize=14, fontweight='bold')
plt.xlabel('Larghezza Multivettore (k)')
plt.ylabel('Dimensione Matrice (MxN)')
plt.savefig('5_cuda_heatmap_k_vs_size.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 6. HEATMAP INCROCIATA SULLA SCALABILITÀ DI RETE: np vs Dimensione
# ==============================================================================
print("Generazione: 6_cuda_heatmap_np_vs_size.png")
pivot_np_size = df_run1.pivot_table(values='SpeedUp', index='Size', columns='MPI_Ranks', aggfunc='mean')

plt.figure(figsize=(10, 7))
sns.heatmap(pivot_np_size, annot=True, fmt=".1f", cmap="rocket_r", linewidths=.5)
plt.title('Heatmap SpeedUp Medio: Dimensione vs Rank MPI', fontsize=14, fontweight='bold')
plt.xlabel('Numero di Rank MPI (np)')
plt.ylabel('Dimensione Matrice (MxN)')
plt.savefig('6_cuda_heatmap_np_vs_size.png', dpi=300, bbox_inches='tight')
plt.close()

print("Elaborazione completata. 6 file PNG generati nella cartella corrente.")