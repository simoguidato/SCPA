import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Generazione Grafico di Riepilogo Esecutivo...")

# Caricamento dati
df = pd.read_csv('risultati_avanzati.csv')
configurazioni_chiave = [
    ('Naive_SMP', 1, 40, 'Base (1x40)\nMemory Bound'),
    ('Opt_SMP', 1, 40, 'Ottimizzato (1x40)\nSaturazione Singolo Socket'),
    ('Naive_Hybrid', 2, 20, 'Base Ibrido (2x20)\nSblocco NUMA'),
    ('Opt_Hybrid', 8, 5, 'Ottimizzato (8x5)\nOverhead Rete MPI'),
    ('Opt_Hybrid', 2, 20, 'Ottimizzato (2x20)\nPICCO ASSOLUTO')
]

dati_riepilogo = []
for mode, np, thr, label in configurazioni_chiave:
    subset = df[(df['Mode'] == mode) & (df['MPI_Ranks'] == np) & (df['OMP_Threads'] == thr)]
    if not subset.empty:
        max_gflops = subset['GFLOPS'].max()
        dati_riepilogo.append({'Configurazione': label, 'GFLOPS': max_gflops})

df_riepilogo = pd.DataFrame(dati_riepilogo)

# Creazione del grafico
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# Colori: grigio per le versioni lente, rosso per l'overhead MPI, verde brillante per la vincente
colori = ['#b0bec5', '#90a4ae', '#4db6ac', '#e57373', '#2e7d32']

ax = sns.barplot(data=df_riepilogo, x='Configurazione', y='GFLOPS', palette=colori)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points',
                fontweight='bold', fontsize=11)

plt.title("Riepilogo Esecutivo Prestazioni di Picco (CPU)", fontsize=16, fontweight='bold')
plt.xlabel("Tappe dell'Ottimizzazione", fontsize=12)
plt.ylabel("Prestazioni Massime (GFLOPS)", fontsize=12)
plt.ylim(0, df_riepilogo['GFLOPS'].max() * 1.15)

plt.tight_layout()
plt.savefig('report_6_Riepilogo_Esecutivo.png', dpi=300)
plt.close()

print("✅ Grafico salvato come 'report_6_Riepilogo_Esecutivo.png'")