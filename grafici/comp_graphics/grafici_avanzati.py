import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.titleweight':'bold',
                     'figure.facecolor':'#F8F8F8','axes.facecolor':'#F0F0F0',
                     'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--',
                     'axes.spines.top':False,'axes.spines.right':False})

K_VALS = [3,6,8,20,32]
GEO_CLR = {'Tall':'#3498db','Square':'#e74c3c','Wide':'#27ae60'}
CUDA_PAL = {'CUDA_Naive':'#4b0082','CUDA_Opt2D':'#c0396b','CUDA_Tiled':'#f08030','CUDA_WarpRow':'#8b5cf6'}
CUDA_LBL = {'CUDA_Naive':'K0–1D Baseline','CUDA_Opt2D':'K1–2D Grid','CUDA_Tiled':'K2–Shared Mem','CUDA_WarpRow':'K3–Warp-Row'}

# ── Caricamento ──────────────────────────────────────
df_omp = pd.read_csv('../risultati_avanzati.csv')
df_gpu = pd.read_csv('../risultati_cuda_stat.csv')

def enrich(df):
    df = df.copy()
    df['Size_Label'] = df['M'].astype(str)+'x'+df['N'].astype(str)
    df['Area']       = df['M']*df['N']
    df['Latency_ms'] = df['ParallelTime']*1000
    df['Bytes_GB']   = (df['M']*df['N']+df['N']*df['k']+df['M']*df['k'])*4/1e9
    df['BW_GBs']     = df['Bytes_GB']/df['ParallelTime'].clip(1e-9)
    df['AI']         = 2.0*df['M']*df['N']*df['k']/(df['Bytes_GB'].clip(1e-12)*1e9)
    df['Geo']        = df.apply(lambda r: 'Tall' if r.M/r.N>1.6 else ('Wide' if r.M/r.N<0.6 else 'Square'),axis=1)
    return df

df_omp = enrich(df_omp)
df_gpu = enrich(df_gpu)

def size_order(df):
    s = df[['Size_Label','Area','Geo']].drop_duplicates()
    s['gk'] = s['Geo'].map({'Tall':0,'Square':1,'Wide':2})
    return s.sort_values(['Area','gk'])['Size_Label'].tolist()

SO_OMP  = size_order(df_omp)
SO_CUDA = size_order(df_gpu)

def avg(df, gcols):
    return df[df['Run']>1].groupby(gcols,observed=True).agg(
        GFLOPS=('GFLOPS','mean'), GFLOPS_std=('GFLOPS','std'),
        Latency_ms=('Latency_ms','mean'), Latency_std=('Latency_ms','std'),
        BW_GBs=('BW_GBs','mean'), AI=('AI','mean')).reset_index()

omp_avg  = avg(df_omp,['Mode','MPI_Ranks','OMP_Threads','Size_Label','M','N','k','Geo','Area'])
cuda_avg = avg(df_gpu,['Mode','Size_Label','M','N','k','Geo','Area'])

serial_base = omp_avg[(omp_avg['Mode']=='Serial')&(omp_avg['MPI_Ranks']==1)&(omp_avg['OMP_Threads']==1)]\
              .set_index(['Size_Label','k'])['GFLOPS'].rename('Serial_GFLOPS')

def geo_yticks(ax, index, df_ref):
    for i,s in enumerate(index):
        rows = df_ref[df_ref['Size_Label']==s]
        if not rows.empty:
            ax.get_yticklabels()[i].set_color(GEO_CLR[rows['Geo'].iloc[0]])

# ═══════════════════════════════════════════════════════════════
# CLASS A1 — Heatmap GFLOPS per config OMP
# ═══════════════════════════════════════════════════════════════
print("A1...")
configs_omp = [('Serial',1,1,'Serial'),('Opt_Hybrid',2,20,'Opt 2×20 ★'),
               ('Opt_Hybrid',4,10,'Opt 4×10'),('Opt_Hybrid',8,5,'Opt 8×5'),
               ('Naive_Hybrid',2,20,'Naive 2×20'),('Opt_SMP',1,40,'Opt SMP 1×40')]
fig,axes = plt.subplots(2,3,figsize=(18,10))
fig.suptitle('CLASS A1 — Heatmap GFLOPS (Taglia × k) per Configurazione OMP',fontsize=14,fontweight='bold')
vmax = omp_avg['GFLOPS'].max()
for idx,(mode,np_,thr,title) in enumerate(configs_omp):
    ax = axes.flatten()[idx]
    sub = omp_avg[(omp_avg['Mode']==mode)&(omp_avg['MPI_Ranks']==np_)&(omp_avg['OMP_Threads']==thr)]
    pivot = sub.pivot_table(index='Size_Label',columns='k',values='GFLOPS').reindex(SO_OMP).dropna(how='all')
    im = ax.imshow(pivot.values,cmap='RdYlGn',aspect='auto',vmin=0,vmax=vmax)
    plt.colorbar(im,ax=ax,shrink=0.75,label='GFLOPS')
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v=pivot.values[r,c]
            if not np.isnan(v):
                ax.text(c,r,f'{v:.0f}',ha='center',va='center',fontsize=8,fontweight='bold',
                        color='white' if v<vmax*0.45 else 'black')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,fontsize=9)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index,fontsize=8)
    geo_yticks(ax,pivot.index,df_omp)
    ax.set_xlabel('k'); ax.set_title(title,fontsize=12)
plt.tight_layout(); plt.savefig('A1_heatmap_omp.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ A1_heatmap_omp.png")

# CLASS A2 — Heatmap CUDA
print("A2...")
fig,axes = plt.subplots(1,4,figsize=(20,6))
fig.suptitle('CLASS A2 — Heatmap GFLOPS (Taglia × k) per Kernel CUDA',fontsize=14,fontweight='bold')
vmax_c = cuda_avg['GFLOPS'].max()
for idx,mode in enumerate(['CUDA_Naive','CUDA_Opt2D','CUDA_Tiled','CUDA_WarpRow']):
    ax=axes[idx]
    sub=cuda_avg[cuda_avg['Mode']==mode]
    pivot=sub.pivot_table(index='Size_Label',columns='k',values='GFLOPS').reindex(SO_CUDA).dropna(how='all')
    im=ax.imshow(pivot.values,cmap='plasma',aspect='auto',vmin=0,vmax=vmax_c)
    plt.colorbar(im,ax=ax,shrink=0.75,label='GFLOPS')
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v=pivot.values[r,c]
            if not np.isnan(v):
                ax.text(c,r,f'{v:.0f}',ha='center',va='center',fontsize=8.5,fontweight='bold',
                        color='white' if v<vmax_c*0.55 else 'black')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,fontsize=9)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index,fontsize=8)
    geo_yticks(ax,pivot.index,df_gpu)
    ax.set_xlabel('k'); ax.set_title(CUDA_LBL[mode],fontsize=11,color=CUDA_PAL[mode])
plt.tight_layout(); plt.savefig('A2_heatmap_cuda.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ A2_heatmap_cuda.png")

# ═══════════════════════════════════════════════════════════════
# CLASS B1 — Violin plot latenza OMP per geometria
# ═══════════════════════════════════════════════════════════════
print("B1...")
fig,axes = plt.subplots(1,3,figsize=(18,6),sharey=False)
fig.suptitle('CLASS B1 — Distribuzione Latenza (ms) per Geometria e Config OMP',fontsize=13,fontweight='bold')
df_v = df_omp[(df_omp['Run']>1)&(df_omp['Mode']!='Serial')].copy()
df_v['Config'] = df_v.apply(lambda r: f"{'N' if 'Naive' in r['Mode'] else 'O'}{'H' if 'Hybrid' in r['Mode'] else 'S'} {r['MPI_Ranks']}×{r['OMP_Threads']}",axis=1)
cfg_order = ['NH 2×20','NS 1×40','OH 2×20','OH 4×10','OH 8×5','OS 1×40']
cfg_colors = ['#6666CC','#AAAAEE','#E63946','#FF8800','#FFBB44','#F4A460']
for gi,geo in enumerate(['Tall','Square','Wide']):
    ax=axes[gi]
    sub=df_v[df_v['Geo']==geo]
    present=[c for c in cfg_order if c in sub['Config'].unique()]
    data_list=[sub[sub['Config']==c]['Latency_ms'].dropna().values for c in present]
    data_list=[d for d in data_list if len(d)>0]
    if not data_list: ax.set_visible(False); continue
    parts=ax.violinplot(data_list,positions=range(len(present)),showmedians=True,showextrema=True)
    for pc,color in zip(parts['bodies'],cfg_colors[:len(present)]):
        pc.set_facecolor(color); pc.set_alpha(0.72)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present,rotation=25,ha='right',fontsize=9)
    ax.set_ylabel('Latenza (ms)'); ax.set_title(f'Geometria: {geo}',color=GEO_CLR[geo],fontsize=12)
    ax.set_yscale('log')
plt.tight_layout(); plt.savefig('B1_violin_latency_omp.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ B1_violin_latency_omp.png")

# CLASS B2 — Latenza ms vs taglia, OMP vs CUDA per ogni k
print("B2...")
common = [s for s in SO_CUDA if s in SO_OMP]
fig,axes = plt.subplots(1,len(K_VALS),figsize=(22,5),sharey=False)
fig.suptitle('CLASS B2 — Latenza (ms) vs Taglia per k: OMP Best vs CUDA',fontsize=13,fontweight='bold')
bar_w=0.17
for ki,k in enumerate(K_VALS):
    ax=axes[ki]
    x=np.arange(len(common))
    omp_lat=omp_avg[(omp_avg['Mode']=='Opt_Hybrid')&(omp_avg['MPI_Ranks']==2)&(omp_avg['OMP_Threads']==20)&(omp_avg['k']==k)]\
            .set_index('Size_Label').reindex(common)['Latency_ms']
    ax.bar(x-0.35,omp_lat.values,bar_w,color='#E63946',alpha=0.85,label='OMP 2×20',zorder=3)
    offsets=[-0.17,0.0,0.17]
    for oi,mode in enumerate(['CUDA_Opt2D','CUDA_Tiled','CUDA_WarpRow']):
        lat=cuda_avg[(cuda_avg['Mode']==mode)&(cuda_avg['k']==k)]\
            .set_index('Size_Label').reindex(common)['Latency_ms']
        ax.bar(x+offsets[oi],lat.values,bar_w,color=CUDA_PAL[mode],alpha=0.85,label=CUDA_LBL[mode],zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(common,rotation=45,ha='right',fontsize=7.5)
    ax.set_title(f'k = {k}',fontsize=11); ax.set_ylabel('ms' if ki==0 else '')
    ax.set_yscale('log')
    if ki==0: ax.legend(fontsize=7,loc='upper left')
plt.tight_layout(); plt.savefig('B2_latency_vs_taglia.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ B2_latency_vs_taglia.png")

# CLASS B3 — CDF latenza
print("B3...")
fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle('CLASS B3 — CDF della Latenza: OMP vs CUDA',fontsize=13,fontweight='bold')
omp_groups=[('Opt 2×20','#E63946',df_omp[(df_omp['Run']>1)&(df_omp['Mode']=='Opt_Hybrid')&(df_omp['MPI_Ranks']==2)]),
            ('Opt 4×10','#FF8800',df_omp[(df_omp['Run']>1)&(df_omp['Mode']=='Opt_Hybrid')&(df_omp['MPI_Ranks']==4)]),
            ('Opt 8×5','#FFBB44',df_omp[(df_omp['Run']>1)&(df_omp['Mode']=='Opt_Hybrid')&(df_omp['MPI_Ranks']==8)]),
            ('Naive 2×20','#6666CC',df_omp[(df_omp['Run']>1)&(df_omp['Mode']=='Naive_Hybrid')&(df_omp['MPI_Ranks']==2)])]
for label,color,grp in omp_groups:
    v=np.sort(grp['Latency_ms'].values); axes[0].plot(v,np.arange(1,len(v)+1)/len(v),color=color,lw=2,label=label)
axes[0].set_xscale('log'); axes[0].set_xlabel('Latenza (ms)'); axes[0].set_ylabel('CDF')
axes[0].set_title('OMP — CDF Latenza'); axes[0].legend(fontsize=8)
for mode in CUDA_PAL:
    grp=df_gpu[(df_gpu['Run']>1)&(df_gpu['Mode']==mode)]
    v=np.sort(grp['Latency_ms'].values); axes[1].plot(v,np.arange(1,len(v)+1)/len(v),color=CUDA_PAL[mode],lw=2,label=CUDA_LBL[mode])
axes[1].set_xscale('log'); axes[1].set_xlabel('Latenza (ms)'); axes[1].set_ylabel('CDF')
axes[1].set_title('CUDA — CDF Latenza'); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('B3_cdf_latency.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ B3_cdf_latency.png")

# ═══════════════════════════════════════════════════════════════
# CLASS C1 — SpeedUp vs config MPI per ogni k
# ---------------------------------------------------------------
print("C1...")
sq_sizes=[s for s in SO_OMP if df_omp[df_omp['Size_Label']==s]['Geo'].iloc[0]=='Square']
np_cfgs=[(2,20,'2x20'),(4,10,'4x10'),(8,5,'8x5')]
fig,axes=plt.subplots(1,len(K_VALS),figsize=(22,5))
fig.suptitle('CLASS C1 — SpeedUp vs Config MPI-OMP per ogni k (kernel Opt_Hybrid)',fontsize=13,fontweight='bold')

import matplotlib.cm as cm
colors = cm.get_cmap('tab10', len(sq_sizes))
# -------------------

for ki,k in enumerate(K_VALS):
    ax=axes[ki]
    for si, size in enumerate(sq_sizes):
        try: sg=serial_base.loc[(size,k)]
        except: continue
        speedups=[]
        for np_,thr,_ in np_cfgs:
            row=omp_avg[(omp_avg['Mode']=='Opt_Hybrid')&(omp_avg['MPI_Ranks']==np_)&
                        (omp_avg['OMP_Threads']==thr)&(omp_avg['Size_Label']==size)&(omp_avg['k']==k)]
            speedups.append(row['GFLOPS'].values[0]/sg if not row.empty else np.nan)

        line_color = colors(si)

        ax.plot(range(len(np_cfgs)),speedups,marker='o',lw=2,ms=8,color=line_color,label=size)
        for xi,sp in enumerate(speedups):
            if not np.isnan(sp):
                ax.annotate(f'{sp:.0f}x',xy=(xi,sp),xytext=(0,5),textcoords='offset points',
                            ha='center',fontsize=7.5,color=line_color)

    ax.set_xticks(range(len(np_cfgs))); ax.set_xticklabels([c for _,_,c in np_cfgs])
    ax.set_title(f'k = {k}',fontsize=11); ax.set_ylabel('SpeedUp vs Serial' if ki==0 else '')
    if ki==0: ax.legend(fontsize=7.5,loc='upper left')

plt.tight_layout(); plt.savefig('C1_speedup_mpi.png',dpi=150,bbox_inches='tight'); plt.close()
print("  > C1_speedup_mpi.png generato")
# CLASS C2 — Efficiency heatmap
print("C2...")
fig,axes=plt.subplots(1,len(sq_sizes),figsize=(6*len(sq_sizes),5))
if len(sq_sizes)==1: axes=[axes]
fig.suptitle('CLASS C2 — Efficienza Parallela (SpeedUp/nCore) — Opt_Hybrid',fontsize=13,fontweight='bold')
for si,size in enumerate(sq_sizes):
    ax=axes[si]; rows=[]
    for np_,thr,lbl in np_cfgs:
        for k in K_VALS:
            try: sg=serial_base.loc[(size,k)]
            except: continue
            row=omp_avg[(omp_avg['Mode']=='Opt_Hybrid')&(omp_avg['MPI_Ranks']==np_)&
                        (omp_avg['OMP_Threads']==thr)&(omp_avg['Size_Label']==size)&(omp_avg['k']==k)]
            if not row.empty:
                eff=(row['GFLOPS'].values[0]/sg)/(np_*thr)
                rows.append({'Config':lbl,'k':k,'Efficiency':eff})
    if not rows: ax.set_visible(False); continue
    pivot=pd.DataFrame(rows).pivot_table(index='Config',columns='k',values='Efficiency')
    pivot=pivot.reindex([c for _,_,c in np_cfgs])
    im=ax.imshow(pivot.values,cmap='RdYlGn',aspect='auto',vmin=0,vmax=1)
    plt.colorbar(im,ax=ax,label='Eff.')
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v=pivot.values[r,c]
            if not np.isnan(v):
                ax.text(c,r,f'{v:.2f}',ha='center',va='center',fontsize=9,fontweight='bold',
                        color='white' if v<0.45 else 'black')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,fontsize=9)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index,fontsize=9)
    ax.set_xlabel('k'); ax.set_title(size,fontsize=11)
plt.tight_layout(); plt.savefig('C2_efficiency_heatmap.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ C2_efficiency_heatmap.png")

# ═══════════════════════════════════════════════════════════════
# CLASS D — 3D Surfaces
# ═══════════════════════════════════════════════════════════════
print("D1...")
def surf3d(ax,data,y_col='k',z_col='GFLOPS',cmap='viridis',zlabel='GFLOPS',title=''):
    X=np.log10(data['Area'].values.astype(float))
    Y=data[y_col].values.astype(float)
    Z=data[z_col].values.astype(float)
    xi=np.linspace(X.min(),X.max(),30); yi=np.linspace(Y.min(),Y.max(),30)
    XI,YI=np.meshgrid(xi,yi)
    ZI=griddata((X,Y),Z,(XI,YI),method='linear')
    surf=ax.plot_surface(XI,YI,ZI,cmap=cmap,alpha=0.82,linewidth=0,antialiased=True)
    ax.scatter(X,Y,Z,c=Z,cmap=cmap,s=45,zorder=10,edgecolors='white',linewidth=0.4)
    ax.set_xlabel('log₁₀(Area)',labelpad=6); ax.set_ylabel('k',labelpad=6); ax.set_zlabel(zlabel,labelpad=6)
    ax.set_title(title,fontsize=10,fontweight='bold')
    return surf

d_omp3 = omp_avg[(omp_avg['Mode']=='Opt_Hybrid')&(omp_avg['MPI_Ranks']==2)&(omp_avg['OMP_Threads']==20)]
d_c1   = cuda_avg[cuda_avg['Mode']=='CUDA_Opt2D']
d_cwr  = cuda_avg[cuda_avg['Mode']=='CUDA_WarpRow']

fig=plt.figure(figsize=(21,7)); fig.suptitle('CLASS D1 — 3D GFLOPS(Area, k)',fontsize=14,fontweight='bold')
ax1=fig.add_subplot(1,3,1,projection='3d'); s1=surf3d(ax1,d_omp3,title='OMP Best (2×20)',cmap='Reds'); fig.colorbar(s1,ax=ax1,shrink=0.5,label='GFLOPS')
ax2=fig.add_subplot(1,3,2,projection='3d'); s2=surf3d(ax2,d_c1,title='CUDA K1–2D Grid',cmap='Blues'); fig.colorbar(s2,ax=ax2,shrink=0.5,label='GFLOPS')
ax3=fig.add_subplot(1,3,3,projection='3d'); s3=surf3d(ax3,d_cwr,title='CUDA K3–Warp-Row',cmap='Purples'); fig.colorbar(s3,ax=ax3,shrink=0.5,label='GFLOPS')
plt.tight_layout(); plt.savefig('D1_3d_gflops.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ D1_3d_gflops.png")

print("D2...")
fig=plt.figure(figsize=(21,7)); fig.suptitle('CLASS D2 — 3D Latenza/ms(Area, k)',fontsize=14,fontweight='bold')
ax1=fig.add_subplot(1,3,1,projection='3d'); surf3d(ax1,d_omp3,z_col='Latency_ms',title='OMP Best – Latenza',cmap='OrRd',zlabel='ms')
ax2=fig.add_subplot(1,3,2,projection='3d'); surf3d(ax2,d_c1,z_col='Latency_ms',title='CUDA K1 – Latenza',cmap='Blues',zlabel='ms')
merged=d_omp3[['Size_Label','k','Area','Latency_ms']].merge(d_c1[['Size_Label','k','Latency_ms']],on=['Size_Label','k'],suffixes=('_omp','_cuda'))
merged['Diff_ms']=merged['Latency_ms_omp']-merged['Latency_ms_cuda']
ax3=fig.add_subplot(1,3,3,projection='3d')
if not merged.empty: surf3d(ax3,merged,z_col='Diff_ms',title='ΔLatenza OMP−CUDA (ms)',cmap='RdYlGn',zlabel='Δms')
plt.tight_layout(); plt.savefig('D2_3d_latency.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ D2_3d_latency.png")

print("D3...")
fig=plt.figure(figsize=(14,6)); fig.suptitle('CLASS D3 — 3D Bandwidth (GB/s)(Area, k)',fontsize=13,fontweight='bold')
ax1=fig.add_subplot(1,2,1,projection='3d'); surf3d(ax1,d_omp3,z_col='BW_GBs',title='OMP Best – BW',cmap='hot',zlabel='GB/s')
ax2=fig.add_subplot(1,2,2,projection='3d'); surf3d(ax2,d_c1,z_col='BW_GBs',title='CUDA K1 – BW',cmap='cool',zlabel='GB/s')
plt.tight_layout(); plt.savefig('D3_3d_bandwidth.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ D3_3d_bandwidth.png")

# ═══════════════════════════════════════════════════════════════
# CLASS E1 — Roofline
# ═══════════════════════════════════════════════════════════════
print("E1...")
fig,axes=plt.subplots(1,2,figsize=(16,6))
fig.suptitle('CLASS E1 — Roofline Model: Intensità Aritmetica vs GFLOPS',fontsize=13,fontweight='bold')
for ax_i,(ax,peak,bw,arch,is_gpu) in enumerate([
    (axes[0],250,80,'OMP',False),
    (axes[1],15000,650,'CUDA',True)]):
    ai=np.logspace(-1,2,200)
    ax.loglog(ai,np.minimum(ai*bw,peak),'k-',lw=2.5,zorder=5,label='Roofline teorico')
    ax.axhline(peak,color='gray',ls=':',lw=1.5,alpha=0.7)
    ax.text(90,peak*1.05,f'{peak} GFLOPS',ha='right',fontsize=8,color='gray')
    ridge=peak/bw
    ax.axvline(ridge,color='orange',ls='--',lw=1.5,alpha=0.7)
    ax.text(ridge*1.1,0.5,f'Ridge\n{ridge:.1f}',fontsize=7,color='orange')
    if not is_gpu:
        for (mode,np_,thr),(label,color) in [
            (('Opt_Hybrid',2,20),('Opt 2×20','#E63946')),
            (('Opt_Hybrid',4,10),('Opt 4×10','#FF8800')),
            (('Opt_Hybrid',8, 5),('Opt 8×5','#FFBB44')),
            (('Naive_Hybrid',2,20),('Naive 2×20','#6666CC'))]:
            sub=omp_avg[(omp_avg['Mode']==mode)&(omp_avg['MPI_Ranks']==np_)&(omp_avg['OMP_Threads']==thr)]
            ax.scatter(sub['AI'],sub['GFLOPS'],c=color,s=60,label=label,alpha=0.8,edgecolors='white',lw=0.5,zorder=6)
    else:
        for mode in CUDA_PAL:
            sub=cuda_avg[cuda_avg['Mode']==mode]
            ax.scatter(sub['AI'],sub['GFLOPS'],c=CUDA_PAL[mode],s=60,label=CUDA_LBL[mode],alpha=0.8,edgecolors='white',lw=0.5,zorder=6)
    ax.set_xlabel('Intensità Aritmetica (FLOP/Byte)',fontsize=10)
    ax.set_ylabel('GFLOPS',fontsize=10)
    ax.set_title(arch,fontsize=11)
    ax.legend(fontsize=8,loc='lower right')
plt.tight_layout(); plt.savefig('E1_roofline.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ E1_roofline.png")

# CLASS E2 — Bubble chart
print("E2...")
fig,axes=plt.subplots(1,2,figsize=(16,7))
fig.suptitle('CLASS E2 — Bubble Chart: Area × k × GFLOPS',fontsize=13,fontweight='bold')
for ax_i,(ax,data,filt,title) in enumerate([
    (axes[0],omp_avg,lambda d:d[(d['Mode']=='Opt_Hybrid')&(d['MPI_Ranks']==2)&(d['OMP_Threads']==20)],'OMP Best (2×20)'),
    (axes[1],cuda_avg,lambda d:d[d['Mode']=='CUDA_Opt2D'],'CUDA K1–2D Grid')]):
    sub=filt(data)
    sc=ax.scatter(np.log10(sub['Area']),sub['k'],s=sub['GFLOPS']*1.5,
                  c=sub['GFLOPS'],cmap='RdYlGn',alpha=0.76,edgecolors='white',lw=0.8)
    plt.colorbar(sc,ax=ax,label='GFLOPS')
    for _,row in sub.iterrows():
        ax.annotate(f"{row['GFLOPS']:.0f}",xy=(np.log10(row['Area']),row['k']),
                    ha='center',va='center',fontsize=7.5,fontweight='bold',color='black')
    ax.set_xlabel('log₁₀(M×N)',fontsize=10); ax.set_ylabel('k',fontsize=10)
    ax.set_yticks(K_VALS)
    xt=sorted(sub['Area'].unique())
    ax.set_xticks([np.log10(a) for a in xt])
    ax.set_xticklabels([f'{a:.1e}' for a in xt],rotation=30,ha='right',fontsize=8)
    ax.set_title(title,fontsize=11)
    for geo,c in GEO_CLR.items():
        if not sub[sub['Geo']==geo].empty: ax.scatter([],[],c=c,s=60,label=geo,alpha=0.8)
    ax.legend(fontsize=8,loc='upper left',title='Geometria')
plt.tight_layout(); plt.savefig('E2_bubble_chart.png',dpi=150,bbox_inches='tight'); plt.close()
print("  ✅ E2_bubble_chart.png")

print("\n"+"="*55)
print("✅ TUTTE LE 12 FIGURE GENERATE")
print("  CLASS A: A1_heatmap_omp.png, A2_heatmap_cuda.png")
print("  CLASS B: B1_violin_latency_omp.png, B2_latency_vs_taglia.png, B3_cdf_latency.png")
print("  CLASS C: C1_speedup_mpi.png, C2_efficiency_heatmap.png")
print("  CLASS D: D1_3d_gflops.png, D2_3d_latency.png, D3_3d_bandwidth.png")
print("  CLASS E: E1_roofline.png, E2_bubble_chart.png")
print("="*55)
