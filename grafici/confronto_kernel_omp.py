import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione confronto Naive vs Ottimizzato (Modalità Hybrid)...")
df = pd.read_csv('risultati_avanzatisuca.csv')

# 1. Filtriamo per la BATTAGLIA VERA: Hybrid a 2 processi (il picco dell'hardware)
df_cmp = df[(df['Mode'].isin(['Opt_Hybrid', 'Naive_Hybrid'])) & (df['MPI_Ranks'] == 2)].copy()

def assegna_nome_kernel(mode):
    return 'Base (Naive)' if 'Naive' in mode else 'Ottimizzato'

df_cmp['Kernel_Version'] = df_cmp['Mode'].apply(assegna_nome_kernel)
df_cmp['Size_Label'] = pd.Categorical(df_cmp['M'].astype(str) + "x" + df_cmp['N'].astype(str),
                                      categories=['2000x2000', '4000x4000', '8000x2000', '2000x8000'], ordered=True)

sns.set_theme(style="whitegrid")

# GRAFICO A: Barplot con Facet Grid (Grafico a Griglia Leggibile) per k=32
subset_k32 = df_cmp[df_cmp['k'] == 32].copy()
subset_k32['Topology'] = "NP:" + subset_k32['MPI_Ranks'].astype(str) + " / T:" + subset_k32['OMP_Threads'].astype(str)

g = sns.catplot(data=subset_k32, kind="bar", x="Topology", y="GFLOPS", hue="Kernel_Version",
                col="Size_Label", col_wrap=2, height=4, aspect=1.2, palette=['salmon', 'skyblue'])

g.set_axis_labels("Topologia", "Prestazioni (GFLOPS)")
g.set_titles("Matrice {col_name}")
g.fig.suptitle("Confronto Kernel (k=32) - Prestazioni Massime", y=1.05, fontweight='bold')
plt.savefig('confronto_1_Barre_Leggibili.png', dpi=300, bbox_inches='tight')

print("✅ Grafico di confronto (Kernel) salvato!")