import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CARICAMENTO E PULIZIA DATI
# ==========================================
CSV_FILE = '../risultati_cuda_mpi_all.csv'

try:
    cols = ['Mode','M','N','k','MPI_Ranks','OMP_Threads','Run','ParallelTime','SequentialTime','RelativeError']
    df = pd.read_csv(CSV_FILE, names=cols, usecols=range(10), skiprows=1)
except FileNotFoundError:
    print(f"Errore: File '{CSV_FILE}' non trovato.")
    exit()

# Creazione label e metriche
df['Size'] = df['M'].astype(str) + "x" + df['N'].astype(str)
df['Area'] = df['M'] * df['N']
df['GFLOPS'] = np.where(df['ParallelTime'] > 0,
                        (2.0 * df['M'] * df['N'] * df['k']) / (df['ParallelTime'] * 1e9), 0)
df['Latency_ms'] = df['ParallelTime'] * 1000.0
df['MPI_Ranks'] = df['MPI_Ranks'].astype(int)

# da escludere la Run 1 se è usata come warmup/validation seriale
df_valid = df[df['Run'] > 1]
df_k32 = df_valid[df_valid['k'] == 32]

# Calcolo delle medie
df_mean = df_k32.groupby(['Mode', 'Size', 'Area', 'MPI_Ranks'])[['GFLOPS', 'Latency_ms']].mean().reset_index()

# Ordiniamo per area crescente della matrice per avere un asse X logico
df_mean = df_mean.sort_values(by=['Area', 'MPI_Ranks'])
df_mean['MPI_Ranks'] = df_mean['MPI_Ranks'].astype(str)

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE_MPI = sns.color_palette("rocket_r", len(df_mean['MPI_Ranks'].unique()))

# ==========================================
# GRAFICO 1: IL CROLLO DEI GFLOPS (Barplot per Taglia, facet per Kernel)
# Mostra i valori esatti del throughput che scendono
# ==========================================
print("Generazione Grafico 1: Crollo GFLOPS...")
g1 = sns.catplot(
    data=df_mean, kind="bar",
    x="Size", y="GFLOPS", hue="MPI_Ranks",
    col="Mode", col_wrap=2, sharey=False,
    palette=PALETTE_MPI, height=5, aspect=1.5, edgecolor='black'
)

g1.fig.suptitle("Degrado del Throughput (GFLOPS) all'aumentare dei Processi MPI (k=32)",
                y=1.05, fontsize=18, fontweight='bold')
g1.set_axis_labels("Taglia Matrice (M x N)", "Throughput Effettivo (GFLOPS)", fontweight='bold')
g1.set_titles("Kernel: {col_name}", size=14, fontweight='bold')
g1.legend.set_title("Processi MPI")

for ax in g1.axes.ravel():
    ax.tick_params(axis='x', rotation=45)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9, fontweight='bold', rotation=90)
    ax.margins(y=0.2)

plt.tight_layout()
plt.savefig('EXTREME_BAR_1_GFLOPS.png', dpi=200, bbox_inches='tight')
plt.close()

# ==========================================
# GRAFICO 2: L'ESPLOSIONE DELLA LATENZA (Scala Logaritmica)
# ==========================================
print("Generazione Grafico 2: Esplosione Latenza...")
g2 = sns.catplot(
    data=df_mean, kind="bar",
    x="Size", y="Latency_ms", hue="MPI_Ranks",
    col="Mode", col_wrap=2, sharey=False,
    palette="mako", height=5, aspect=1.5, edgecolor='black'
)

g2.fig.suptitle("Esplosione della Latenza (ms) all'aumentare dei Processi MPI (k=32) - Scala Logaritmica",
                y=1.05, fontsize=18, fontweight='bold')
g2.set_axis_labels("Taglia Matrice (M x N)", "Latenza (ms) [Log]", fontweight='bold')
g2.set_titles("Kernel: {col_name}", size=14, fontweight='bold')
g2.legend.set_title("Processi MPI")

for ax in g2.axes.ravel():
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9, fontweight='bold', rotation=90)
    ax.margins(y=0.3)

plt.tight_layout()
plt.savefig('EXTREME_BAR_2_Latency_Log.png', dpi=200, bbox_inches='tight')
plt.close()

# ==========================================
# GRAFICO 3: FOCUS SUL KERNEL MIGLIORE (CUDA_Tiled) SULLE TAGLIE GRANDI
# ==========================================
print("Generazione Grafico 3: Focus Kernel Ottimo...")
# Prendiamo solo le 3 taglie più grandi
large_sizes = df_mean['Size'].unique()[-3:]
df_focus = df_mean[(df_mean['Mode'] == 'CUDA_Tiled') & (df_mean['Size'].isin(large_sizes))]

fig, ax1 = plt.subplots(figsize=(12, 6))

sns.barplot(data=df_focus, x='Size', y='GFLOPS', hue='MPI_Ranks', palette=PALETTE_MPI, edgecolor='black', ax=ax1)

plt.title("Impatto dell'Oversubscription Estrema sul miglior Kernel (CUDA_Tiled, k=32)", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel("Taglia Matrice (Grandi Dimensioni)", fontweight='bold', fontsize=12)
ax1.set_ylabel("Throughput (GFLOPS)", fontweight='bold', fontsize=12)
ax1.legend(title="Rank MPI (Contesa GPU)", loc='upper right')

# Aggiungiamo le percentuali di degrado rispetto a MPI=1
for i, size in enumerate(df_focus['Size'].unique()):
    subset = df_focus[df_focus['Size'] == size]
    base_val = subset[subset['MPI_Ranks'] == '1']['GFLOPS'].values[0]

    # Calcoliamo le posizioni x per le barre di questo gruppo
    n_bars = len(subset)
    width = 0.8 / n_bars
    x_positions = np.linspace(i - 0.4 + width/2, i + 0.4 - width/2, n_bars)

    for j, (idx, row) in enumerate(subset.iterrows()):
        val = row['GFLOPS']
        drop_pct = ((base_val - val) / base_val) * 100
        ax1.text(x_positions[j], val + 10, f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        if drop_pct > 0:
            ax1.text(x_positions[j], val / 2, f'-{drop_pct:.0f}%', ha='center', va='center',
                     fontweight='bold', fontsize=11, color='white',
                     bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none'))

plt.ylim(0, df_focus['GFLOPS'].max() * 1.2)
plt.tight_layout()
plt.savefig('EXTREME_BAR_3_Focus_Degrado_Percentuale.png', dpi=200)
plt.close()

print("Finito! 3 nuovi grafici generati.")