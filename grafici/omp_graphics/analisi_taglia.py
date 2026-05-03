import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione Analisi Taglia (Corretta)...")
df = pd.read_csv('risultati_avanzati.csv')

# --- PREPARAZIONE DATI ---
df['Total_Elements'] = df['M'] * df['N']
df_ottimale = df[
    (df['Mode'] == 'Opt_Hybrid') &
    (df['MPI_Ranks'] == 2) &
    (df['k'] == 32)
    ].copy()

def assegna_geometria(row):
    ratio = row['M'] / row['N']
    if abs(ratio - 1.0) < 0.1: return 'Quadrata (M = N)'
    elif ratio >= 2.0: return 'Tall (M = 4N)'
    elif ratio <= 0.5: return 'Wide (N = 4M)'
    else: return 'Altro'

df_ottimale['Geometria'] = df_ottimale.apply(assegna_geometria, axis=1)
df_mean = df_ottimale.groupby(['Total_Elements', 'Geometria', 'M', 'N'])['GFLOPS'].mean().reset_index()
df_mean = df_mean.sort_values('Total_Elements')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(11, 6))

# --- GRAFICO: Crescita Taglia (X = Area Totale) ---
sns.lineplot(
    data=df_mean,
    x='Total_Elements', y='GFLOPS', hue='Geometria',
    marker='o', linewidth=2.5, markersize=9,
    palette=['#1f77b4', '#ff7f0e', '#2ca02c']
)

for i, row in df_mean.iterrows():
    plt.annotate(f"{int(row['M'])}x{int(row['N'])}",
                 xy=(row['Total_Elements'], row['GFLOPS']),
                 xytext=(0, 7), textcoords='offset points',
                 fontsize=8, ha='center')

plt.xscale('log') # Trasforma il grafico in una scala di crescita reale
plt.title("Stabilità del Kernel al crescere del carico (M x N, k=32)", fontsize=14, fontweight='bold')
plt.xlabel("Numero totale di elementi (Scala Logaritmica)", fontsize=12)
plt.ylabel("Prestazioni (GFLOPS)", fontsize=12)
plt.ylim(0, max(df_mean['GFLOPS']) + 30)

plt.tight_layout()
plt.savefig('report_5_Crescita_Taglia_Monotona.png', dpi=300)
plt.close()

print("✅ Grafico 10 (Monotono) salvato correttamente.")