import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione Confronto Naive vs Ottimizzato — tutte le taglie e tutti i k...")
df = pd.read_csv('risultati_avanzati.csv')

size_order = ['1000x250', '4000x1000', '8000x2000',
              '1000x1000', '2000x2000', '4000x4000','8000x8000',
              '250x1000', '1000x4000', '2000x8000']
size_order = [s for s in size_order if s in (df['M'].astype(str) + "x" + df['N'].astype(str)).unique()]

df['Size_Label'] = pd.Categorical(df['M'].astype(str) + "x" + df['N'].astype(str),
                                  categories=size_order, ordered=True)

# Confronto su NP=2 (configurazione ottimale identificata)
df_cmp = df[(df['Mode'].isin(['Opt_Hybrid', 'Naive_Hybrid'])) & (df['MPI_Ranks'] == 2)].copy()
df_cmp['Kernel_Version'] = df_cmp['Mode'].apply(
    lambda m: 'Base (Naive)' if 'Naive' in m else 'Ottimizzato')

sns.set_theme(style="whitegrid")

# ==========================================
# GRAFICO A: k=32
# ==========================================
print("Generando Grafico A (k=32, tutte le taglie)...")
df_k32 = df_cmp[df_cmp['k'] == 32]

g = sns.catplot(data=df_k32, kind="bar",
                x="Kernel_Version", y="GFLOPS",
                col="Size_Label", col_wrap=3,
                height=4, aspect=1.2,
                hue="Kernel_Version",
                palette=['salmon', 'skyblue'],
                sharey=False)
g.set_titles("Matrice {col_name}")
g.set_axis_labels("Versione Kernel", "GFLOPS")
g.fig.suptitle("Confronto Kernel Naive vs Ottimizzato — k=32, NP:2/Thr:20",
               fontweight='bold', y=1.02)
plt.savefig('confronto_kernel_1_k32_ALL_sizes.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> confronto_kernel_1_k32_ALL_sizes.png salvato")

# ==========================================
# GRAFICO B: Matrice 4000x4000
# ==========================================
print("Generando Grafico B (4000x4000, tutti i k)...")
matrix_target = '4000x4000' if '4000x4000' in df_cmp['Size_Label'].cat.categories else \
                df_cmp['Size_Label'].cat.categories[0]

df_mat = df_cmp[df_cmp['Size_Label'] == matrix_target]

g = sns.catplot(data=df_mat, kind="bar",
                x="Kernel_Version", y="GFLOPS",
                col="k", col_wrap=3,
                height=4, aspect=1.2,
                hue="Kernel_Version",
                palette=['salmon', 'skyblue'],
                sharey=False)
g.set_titles("k = {col_name}")
g.set_axis_labels("Versione Kernel", "GFLOPS")
g.fig.suptitle(f"Confronto Kernel Naive vs Ottimizzato — Matrice {matrix_target}, NP:2/Thr:20",
               fontweight='bold', y=1.02)
plt.savefig('confronto_kernel_2_ALL_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> confronto_kernel_2_ALL_k.png salvato")

# ==========================================
# GRAFICO C:  per vedere dove il miglioramento è maggiore
# ==========================================
print("Generando Grafico C (Ratio Opt/Naive per geometria)...")
pivot = df_cmp.groupby(['Size_Label', 'k', 'Kernel_Version'])['GFLOPS'].mean().unstack('Kernel_Version')
pivot = pivot.reset_index()
pivot['Ratio'] = pivot['Ottimizzato'] / pivot['Base (Naive)']

plt.figure(figsize=(10, 6))
sns.lineplot(data=pivot, x='k', y='Ratio', hue='Size_Label', marker='o', linewidth=2)
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label='Pari prestazioni (1x)')
plt.title("Fattore di Miglioramento Ottimizzato/Naive — NP:2/Thr:20",
          fontweight='bold')
plt.xlabel("k")
plt.ylabel("Ratio GFLOPS (Ottimizzato / Naive)")
plt.legend(title='Taglia', bbox_to_anchor=(1.02, 0.5), loc='center left')
plt.tight_layout()
plt.savefig('confronto_kernel_3_Ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("--> confronto_kernel_3_Ratio.png salvato")

print("\n✅ Grafici confronto kernel completati!")
