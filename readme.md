# SCPA — Prodotto Matrice Densa × Multivettore (MPI + OpenMP/CUDA)

Nucleo di calcolo per Y = A·X, con A matrice densa M×N e X, Y multivettori
a k colonne. Parallelizzato con MPI su griglia di processi 2D, con
parallelismo intra-nodo a scelta tra OpenMP (CPU) e CUDA (GPU).

## 1. Requisiti

- CMake ≥ 3.22
- Compilatore C11
- Implementazione MPI 
- Per la build OpenMP: libreria OpenMP 
- Per la build CUDA: CUDA Toolkit (nvcc) e una GPU NVIDIA
- Python 3 con `pandas`, `numpy`, `matplotlib` — solo per generare i grafici
  

Sul server di dipartimento, caricare prima i moduli necessari:
```bash
module load gnu/13.3.0
module load mpich/4.3.0
module load cuda/12.8   # solo se si compila la variante CUDA
```

## 2. Compilazione

Il progetto compila **una sola variante alla volta** (OpenMP oppure CUDA):
cambiare `USE_CUDA` e ricompilare da zero per passare dall'una all'altra.

### Variante OpenMP (default)
```bash
mkdir -p build_omp && cd build_omp
cmake .. -DUSE_CUDA=OFF
cmake --build .
```
Produce l'eseguibile `./SCPA` (ricezione MPI + calcolo OpenMP).

### Variante CUDA
```bash
mkdir -p build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON -DCUDA_KERNEL=2 -DSCPA_CUDA_ARCHITECTURES=80
cmake --build .
```
- `CUDA_KERNEL` seleziona quale kernel CUDA compilare (0–4, vedi § 5).
- `SCPA_CUDA_ARCHITECTURES` è la compute capability della GPU target

Ogni volta che si cambia `CUDA_KERNEL` bisogna rilanciare
`cmake --build .` (o ripulire la cartella di build) perché il kernel è
selezionato a tempo di compilazione, non a runtime.

## 3. Esecuzione di un singolo run

```bash
mpirun -np <NP> ./SCPA <M> <N> <k> [warmup] [validate] [kernel_type] [grid_rows] [grid_cols]
```

| Argomento | Obbligatorio | Descrizione |
|---|---|---|
| `M`, `N` | sì | dimensioni globali della matrice A |
| `k` | sì | numero di colonne del multivettore |
| `warmup` | no (default 1) | esegue una chiamata di riscaldamento prima della misura |
| `validate` | no (default 1) | esegue anche il confronto col riferimento seriale |
| `kernel_type` | no (default 0) | nella build OpenMP: 0 = ottimizzato, 1 = naive. Ignorato nella build CUDA (il kernel è già fissato a compile-time da `CUDA_KERNEL`) |
| `grid_rows`, `grid_cols` | no | forma esplicita della griglia di processi P×Q. Se omessi (o 0 0), la griglia è calcolata automaticamente con `MPI_Dims_create`. Se specificati, deve valere `grid_rows * grid_cols == NP` |

Esempio, griglia 2×4 su 8 processi, matrice 4000×2000, k=32:
```bash
mpirun -np 8 ./SCPA 4000 2000 32 1 1 0 2 4
```

L'output su stdout include una riga
```
DATA_CSV:ParallelTime,SequentialTime,RelativeError,GFLOPS,SpeedUp
```
e, se `validate=1`, una seconda riga
```
EXTRA_TIMES_CSV:DistribTime,TransferTime
```

## 4. Campagne di benchmark automatiche

Tutti gli script vanno lanciati dalla root del progetto.

### 4.1 Campagna CUDA (tutti e 5 i kernel, tutte le taglie/k/NP/topologie)
```bash
CUDA_ARCH=80 NUM_RUNS=3 K_SET=all ./run_cuda_benchmark.sh
```
Variabili principali:
- `CUDA_ARCH`: compute capability della GPU
- `NUM_RUNS`: ripetizioni per ogni configurazione
- `K_SET`: `small` (3,6,8,20,32) | `large` (64,128) | `all` 
- `MPI_PROCS_LIST` (default `"1 2 4 8"`): numeri di processi da testare
- `KERNEL_IDS` (default `"0 1 2 3 4"`): quali kernel CUDA includere
- `TEST_TOPOLOGY` (default 1): se 1, per ogni NP>1 testa tutte le
  fattorizzazioni P×Q possibili (es. per NP=8: 1×8, 2×4, 4×2, 8×1), non
  solo la griglia automatica

Output: `results/risultati_cuda.csv`.
### 4.2 Campagna OpenMP (tutte le config Naive/Opt × SMP/Hybrid + topologie)
```bash
NUM_RUNS=5 ./run_omp_benchmark.sh
```
- `NUM_RUNS` (default 3)
- `NPROC_NODE` (default: rilevato con `nproc`): numero di thread massimo
  disponibile sul nodo, usato per le configurazioni SMP/Hybrid

Output: `results/risultati_omp.csv`.

### 4.3 Mini-campagna tempi di distribuzione/trasferimento
```bash
CUDA_ARCH=80 NUM_RUNS=3 ./run_timing_benchmark.sh
```
Campagna volutamente ridotta (2 kernel CUDA rappresentativi + 1 OpenMP,
NP=1), pensata per isolare il tempo di distribuzione MPI di X e il
trasferimento H2D/D2H dal tempo di calcolo puro, misurati separatamente.

Output: `results/risultati_tempi.csv`.

## 5. I 5 kernel CUDA

| ID | Nome | Strategia |
|---|---|---|
| 0 | Naive | un thread per elemento, tutto in memoria globale |
| 1 | Opt2D | come 0, indici rimappati per accessi coalescenti |
| 2 | Tiled | tiling in Shared Memory + cache di sola lettura (`__ldg`) |
| 3 | WarpRow | un warp per riga di A, accumulo su registri, nessuna Shared Memory |
| 4 | Transposed | trasposizione di A su GPU + scrittura coalescente tramite scambio di indici in Shared Memory |

## 6. Generazione dei grafici

Richiede `pip install pandas numpy matplotlib --break-system-packages`

```bash
# tutti i grafici di performance (GFLOPS, speedup, scaling, topologia...)
python3 plot_results.py results/risultati_cuda.csv results/risultati_omp.csv \
    --outdir grafici_report --k-fixed 32

# solo CUDA o solo OMP
python3 plot_results.py results/risultati_cuda.csv --outdir grafici_cuda
python3 plot_results.py results/risultati_omp.csv --outdir grafici_omp

# breakdown dei tempi (distribuzione MPI / trasferimento H2D-D2H / calcolo)
python3 plot_timing.py results/risultati_tempi.csv --outdir grafici_report
```

Tutti i PNG vengono scritti nella cartella indicata da `--outdir`.

## 7. Struttura del progetto

```
core/           main.c, distribuzione MPI, validazione, utility comuni
                a tutti i backend
include/        header condivisi
cuda/           i 5 kernel CUDA (uno per file) + utility device
omp/            kernel OpenMP (naive e ottimizzato)
results/        CSV prodotti dalle campagne di benchmark
grafici_report/ PNG prodotti da plot_results.py / plot_timing.py
*.sh            script di benchmark (vedi § 4)
plot_*.py       script di generazione grafici (vedi § 6)
```