import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione grafico di Scalabilità Topologica...")
df = pd.read_csv('risultati_avanzatisuca.csv')

df['Size_Label'] = pd.Categorical(df['M'].astype(str) + "x" + df['N'].astype(str),
                                  categories=['2000x2000', '4000x4000', '8000x2000', '2000x8000'], ordered=True)

# Prendiamo solo il kernel Ottimizzato per k=32 su tutte le topologie
df_topo = df[(df['Mode'].str.contains('Opt_')) & (df['k'] == 32)].copy()
df_topo['Topology'] = "NP:" + df_topo['MPI_Ranks'].astype(str) + " / Thr:" + df_topo['OMP_Threads'].astype(str)

sns.set_theme(style="whitegrid")

# Facet Grid per Topologia (4 grafici, uno per matrice)
g = sns.catplot(data=df_topo, kind="bar", x="Topology", y="GFLOPS",
                col="Size_Label", col_wrap=2, height=4, aspect=1.5, palette='viridis', hue="Topology", legend=False)

g.set_axis_labels("Configurazione", "Prestazioni (GFLOPS)")
g.set_titles("Matrice {col_name}")

# Ruotiamo le scritte di 30 gradi così non si scontrano
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=30)

g.fig.suptitle("Scalabilità Topologica - Kernel Ottimizzato (k=32)", y=1.05, fontweight='bold')
plt.savefig('confronto_2_Topologia_Leggibile.png', dpi=300, bbox_inches='tight')

print("✅ Grafico di scalabilità topologica salvato!")