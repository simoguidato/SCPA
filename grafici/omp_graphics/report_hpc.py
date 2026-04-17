import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione Report Generale HPC...")
df = pd.read_csv('risultati_avanzati.csv')

# ==========================================
# PREPARAZIONE DATI
# ==========================================
df['Config'] = df['Mode'] + " (NP:" + df['MPI_Ranks'].astype(str) + ", Thr:" + df['OMP_Threads'].astype(str) + ")"
df['Size_Label'] = df['M'].astype(str) + "x" + df['N'].astype(str)
df['Total_Cores'] = df['MPI_Ranks'] * df['OMP_Threads']

# ECCO LA RIGA CHE MANCAVA!
df['Total_Elements'] = df['M'] * df['N']

size_order = ['1000x250', '4000x1000', '8000x2000',   # tall
              '1000x1000', '2000x2000', '4000x4000','8000x8000',   # quadrate
              '250x1000', '1000x4000', '2000x8000']    # wide

# Tieni solo le taglie presenti nel CSV
size_order = [s for s in size_order if s in df['Size_Label'].unique()]
df['Size_Label'] = pd.Categorical(df['Size_Label'], categories=size_order, ordered=True)

# Ricalcolo SpeedUp robusto (observed=False per rimuovere il FutureWarning)
serial_means = df[(df['Mode'] == 'Serial') & (df['Run'] > 1)].groupby(
    ['Size_Label', 'k'], observed=False)['ParallelTime'].mean().reset_index()
serial_means.rename(columns={'ParallelTime': 'BaselineTime'}, inplace=True)
df = df.merge(serial_means, on=['Size_Label', 'k'], how='left')
df['SpeedUp'] = df['BaselineTime'] / df['ParallelTime']
df['Efficiency'] = df['SpeedUp'] / df['Total_Cores']

df_parallel = df[df['Mode'] != 'Serial'].copy()

sns.set_theme(style="whitegrid")

# ==========================================
# GRAFICO 1: GFLOPS vs k — FacetGrid per tutte le taglie
# ==========================================
print("Generando Grafico 1 (GFLOPS vs k, tutte le taglie)...")
g = sns.FacetGrid(df_parallel, col='Size_Label', col_wrap=3,
                  height=4, aspect=1.4, sharey=False)
g.map_dataframe(sns.lineplot, x='k', y='GFLOPS', hue='Config',
                marker='o', linewidth=2, markersize=7)
g.add_legend(title='Configurazione', bbox_to_anchor=(1.02, 0.5), loc='center left')
g.set_titles("Matrice {col_name}")
g.set_axis_labels("k", "GFLOPS")
g.fig.suptitle("GFLOPS vs k — Tutte le Taglie", fontweight='bold', y=1.02)
plt.savefig('report_1_GFLOPS_vs_k_ALL.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> report_1_GFLOPS_vs_k_ALL.png salvato")

# ==========================================
# GRAFICO 2: GFLOPS vs Taglia (Tutti i k) - SOLUZIONE FACETGRID
# ==========================================
print("Generando Grafico 2 (GFLOPS vs Taglia, Faceted)...")
def assegna_geometria(row):
    ratio = row['M'] / row['N']
    if abs(ratio - 1.0) < 0.1: return 'Quadrata (M = N)'
    elif ratio >= 2.0: return 'Tall (M >= 4N)'
    elif ratio <= 0.5: return 'Wide (N >= 4M)'
    else: return 'Altro'

df['Geometria'] = df.apply(assegna_geometria, axis=1)

# Filtriamo per la configurazione ottimale (Hybrid 2x20) per il confronto taglie
df_plot = df[(df['Mode'] == 'Opt_Hybrid') & (df['MPI_Ranks'] == 2)].copy()

# Usiamo col='Geometria' per separare i subplot ed evitare l'effetto "W"
g = sns.relplot(
    data=df_plot,
    x='Total_Elements', y='GFLOPS', hue='k',
    col='Geometria', kind='line', marker='o',
    palette='viridis', height=5, aspect=1.2,
    facet_kws={'sharex': False}
)

g.set_axis_labels("Elementi Totali (M x N) - Scala Log", "Prestazioni (GFLOPS)")
g.set_titles("{col_name}", fontweight='bold')

for ax in g.axes.flat:
    ax.set_xscale('log') # Fondamentale per la monotonicità

g.fig.suptitle("Analisi Scalabilità Taglia per Geometria e k", y=1.05, fontsize=16, fontweight='bold')
plt.savefig('report_2_GFLOPS_vs_Size_Faceted.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> report_2_GFLOPS_vs_Size_Faceted.png salvato")

# ==========================================
# GRAFICO 3: SpeedUp — FacetGrid per tutti i k (barre)
# ==========================================
print("Generando Grafico 3 (SpeedUp per tutti i k)...")
matrix_for_speedup = '4000x4000' if '4000x4000' in df_parallel['Size_Label'].cat.categories else \
    df_parallel['Size_Label'].cat.categories[0]

g = sns.FacetGrid(df_parallel[df_parallel['Size_Label'] == matrix_for_speedup],
                  col='k', col_wrap=3, height=4, aspect=1.4, sharey=False)
g.map_dataframe(sns.barplot, x='Config', y='SpeedUp',
                hue='Config', palette='viridis', legend=False)
g.set_titles("k = {col_name}")
g.set_axis_labels("Configurazione", "SpeedUp")
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=30)
g.fig.suptitle(f"SpeedUp per tutti i k — Matrice {matrix_for_speedup}", fontweight='bold', y=1.02)
plt.savefig('report_3_SpeedUp_ALL_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> report_3_SpeedUp_ALL_k.png salvato")

# ==========================================
# GRAFICO 4: Efficienza — FacetGrid per tutti i k (barre)
# ==========================================
print("Generando Grafico 4 (Efficienza per tutti i k)...")
g = sns.FacetGrid(df_parallel[df_parallel['Size_Label'] == matrix_for_speedup],
                  col='k', col_wrap=3, height=4, aspect=1.4, sharey=True)
g.map_dataframe(sns.barplot, x='Config', y='Efficiency',
                hue='Config', palette='magma', legend=False)

# Linea ideale su ogni pannello
for ax in g.axes.flat:
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.tick_params(axis='x', rotation=30)
    ax.set_ylim(0, 1.1)

g.set_titles("k = {col_name}")
g.set_axis_labels("Configurazione", "Efficienza (SpeedUp / Core)")
g.fig.suptitle(f"Efficienza Parallela per tutti i k — Matrice {matrix_for_speedup}",
               fontweight='bold', y=1.02)
plt.savefig('report_4_Efficiency_ALL_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> report_4_Efficiency_ALL_k.png salvato")

print("\n✅ Report generale completato!")