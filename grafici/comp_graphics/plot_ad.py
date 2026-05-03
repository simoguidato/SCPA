import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_bubble_chart(filename, platform_name):
    df = pd.read_csv(filename)
    df['Area'] = df['M'] * df['N']
    df_plot = df.groupby(['Area', 'k'])['GFLOPS'].max().reset_index()

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(np.log10(df_plot['Area']), df_plot['k'],
                          s=df_plot['GFLOPS'] * 0.5,
                          c=df_plot['GFLOPS'],
                          cmap='viridis', alpha=0.6, edgecolors="white")

    plt.colorbar(scatter, label='GFLOPS (Throughput)')
    plt.xlabel("Log10(Area della Matrice M*N)")
    plt.ylabel("Valore di k (Intensità Aritmetica)")
    plt.title(f"Analisi 3 Variabili: Prestazioni vs Taglia e k ({platform_name})")

    # Annotazioni valori
    for i, row in df_plot.iterrows():
        plt.annotate(f"{row['GFLOPS']:.0f}", (np.log10(row['Area']), row['k']),
                     fontsize=8, ha='center', va='center', fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'3var_bubble_{platform_name}.png', dpi=200)
    plt.close()

generate_bubble_chart('../risultati_cuda_stat.csv', 'CUDA')
generate_bubble_chart('../risultati_avanzati.csv', 'OpenMP')
print("Grafici a 3 variabili generati.")