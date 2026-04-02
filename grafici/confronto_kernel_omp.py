import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
print("Caricamento dati per il confronto dei kernel...")
df = pd.read_csv('risultati_avanzati.csv')

# Creiamo una colonna che identifica solo il tipo di Kernel
# Cerchiamo le righe che contengono 'Magico_SMP' e 'Naive_SMP'
df_cmp = df[df['Mode'].str.contains('SMP')].copy()

# Puliamo i nomi per la legenda
df_cmp['Kernel_Version'] = df_cmp['Mode'].apply(lambda x: 'Ottimizzato (Magico)' if 'Magico' in x else 'Base (Naive)')

# Ordinamento matrici
size_order = ['2000x2000', '4000x4000', '8000x2000', '2000x8000']
df_cmp['Size_Label'] = pd.Categorical(
    df['M'].astype(str) + "x" + df['N'].astype(str),
    categories=size_order,
    ordered=True
)

sns.set_theme(style="whitegrid")
MATRIX_TARGET = '4000x4000'

# ==========================================
# 2. CALCOLO DELL'INCREMENTO (SPEEDUP FACTOR)
# ==========================================
# Calcoliamo quanto il Magico è più veloce del Naive (Ratio)
pivot = df_cmp.groupby(['Kernel_Version', 'k', 'Size_Label'])['GFLOPS'].mean().unstack(level=0)
pivot['Improvement_Factor'] = pivot['Ottimizzato (Magico)'] / pivot['Base (Naive)']

print("\n=== FATTORE DI MIGLIORAMENTO (Magico / Naive) ===")
print(pivot['Improvement_Factor'].unstack(level=0).round(2))
print("=================================================\n")

# ==========================================
# GRAFICO A: GFLOPS vs k (Confronto Diretto)
# ==========================================
print("Generando Grafico Confronto GFLOPS...")
plt.figure(figsize=(10, 6))
subset = df_cmp[df_cmp['Size_Label'] == MATRIX_TARGET]

sns.lineplot(data=subset, x='k', y='GFLOPS', hue='Kernel_Version',
             marker='o', linewidth=3, markersize=10, palette=['red', 'blue'])

plt.title(f"Impatto delle Ottimizzazioni Manuali - Matrice {MATRIX_TARGET}\n(Kernel Naive vs Kernel Magico)", fontsize=14, fontweight='bold')
plt.ylabel("Prestazioni (GFLOPS)")
plt.xlabel("Dimensione k")
plt.xticks([3, 6, 8, 20, 32])
plt.savefig('confronto_1_GFLOPS.png', dpi=300, bbox_inches='tight')

# ==========================================
# GRAFICO B: BAR CHART - IL "GAP" PRESTAZIONALE
# ==========================================
print("Generando Grafico a Barre del Gap...")
plt.figure(figsize=(10, 6))
# Prendiamo solo k=32 per il massimo impatto visivo
subset_k32 = df_cmp[df_cmp['k'] == 32]

sns.barplot(data=subset_k32, x='Size_Label', y='GFLOPS', hue='Kernel_Version', palette=['salmon', 'skyblue'])

plt.title("Confronto Prestazioni Massime (k=32) su diverse taglie", fontsize=14, fontweight='bold')
plt.ylabel("GFLOPS")
plt.xlabel("Dimensione Matrice")
plt.savefig('confronto_2_Barre.png', dpi=300, bbox_inches='tight')

print("✅ Grafici di confronto salvati con successo!")