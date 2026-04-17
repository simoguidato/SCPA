import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione Scalabilità Topologica — tutte le taglie e tutti i k...")
df = pd.read_csv('risultati_avanzati.csv')

size_order = ['1000x250', '4000x1000', '8000x2000',
              '1000x1000', '2000x2000', '4000x4000','8000x8000',
              '250x1000', '1000x4000', '2000x8000']
size_order = [s for s in size_order if s in (df['M'].astype(str) + "x" + df['N'].astype(str)).unique()]

df['Size_Label'] = pd.Categorical(df['M'].astype(str) + "x" + df['N'].astype(str),
                                  categories=size_order, ordered=True)

# Solo kernel ottimizzato
df_topo = df[df['Mode'].str.contains('Opt_')].copy()
df_topo['Topology'] = "NP:" + df_topo['MPI_Ranks'].astype(str) + " / Thr:" + df_topo['OMP_Threads'].astype(str)

sns.set_theme(style="whitegrid")

# ==========================================
# GRAFICO A: k=32 — FacetGrid per tutte le taglie
# ==========================================
print("Generando Grafico A (k=32, tutte le taglie)...")
df_k32 = df_topo[df_topo['k'] == 32]

g = sns.catplot(data=df_k32, kind="bar",
                x="Topology", y="GFLOPS",
                col="Size_Label", col_wrap=3,
                height=4, aspect=1.4,
                hue="Topology", palette='viridis', legend=False,
                sharey=False)
g.set_titles("Matrice {col_name}")
g.set_axis_labels("Configurazione", "GFLOPS")
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=30)
g.fig.suptitle("Scalabilità Topologica — Kernel Ottimizzato, k=32",
               fontweight='bold', y=1.02)
plt.savefig('confronto_topo_1_k32_ALL_sizes.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> confronto_topo_1_k32_ALL_sizes.png salvato")

# ==========================================
# GRAFICO B: Matrice 4000x4000 — FacetGrid per tutti i k
# ==========================================
print("Generando Grafico B (4000x4000, tutti i k)...")
matrix_target = '4000x4000' if '4000x4000' in df_topo['Size_Label'].cat.categories else \
                df_topo['Size_Label'].cat.categories[0]

df_mat = df_topo[df_topo['Size_Label'] == matrix_target]

g = sns.catplot(data=df_mat, kind="bar",
                x="Topology", y="GFLOPS",
                col="k", col_wrap=3,
                height=4, aspect=1.4,
                hue="Topology", palette='viridis', legend=False,
                sharey=False)
g.set_titles("k = {col_name}")
g.set_axis_labels("Configurazione", "GFLOPS")
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=30)
g.fig.suptitle(f"Scalabilità Topologica — Kernel Ottimizzato, Matrice {matrix_target}",
               fontweight='bold', y=1.02)
plt.savefig('confronto_topo_2_ALL_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> confronto_topo_2_ALL_k.png salvato")

print("\n✅ Grafici scalabilità topologica completati!")
