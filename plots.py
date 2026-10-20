"""Grafik Matplotlib untuk ekspor PNG: tidak memerlukan browser/Chrome."""
from __future__ import annotations
from io import BytesIO
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from stat_core import Result

NAVY='#163746';TEAL='#138C81';ORANGE='#E0A24A';GRAY='#566A74'


def _png(fig, title,source,dpi=200,margins=None):
    fig.suptitle(title,fontsize=15,fontweight='bold',color=NAVY,y=.98,wrap=True)
    fig.text(.5,.015,source,ha='center',fontsize=8,color=GRAY)
    fig.subplots_adjust(**(margins or dict(top=.86,bottom=.14,left=.11,right=.95)))
    b=BytesIO();fig.savefig(b,format='png',dpi=dpi,facecolor='white');plt.close(fig)
    return b.getvalue()


def correlation_png(result: Result,title,source,dpi=200):
    mat=result.meta.get('matrix')
    if mat is None or mat.empty:raise ValueError('Matriks korelasi kosong.')
    n=len(mat); fig,ax=plt.subplots(figsize=(max(9,n*1.55),max(6.8,n*1.2)))
    def wrap_label(text):
        return '\n'.join(textwrap.wrap(str(text).replace('_',' '),width=17,break_long_words=True))
    labels=[wrap_label(x) for x in mat.columns]
    m=mat.to_numpy(float)
    im=ax.imshow(np.ma.masked_invalid(m),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(n),labels,rotation=22,ha='right',fontsize=8)
    ax.set_yticks(range(n),[wrap_label(x) for x in mat.index],fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isfinite(m[i,j]):ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=9,
                                          color='white' if abs(m[i,j])>.65 else NAVY,fontweight='bold')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Koefisien korelasi')
    return _png(fig,title,source,dpi,dict(top=.83,bottom=.30,left=.26,right=.86))


def scatter_png(data,x,y,title,source,dpi=200):
    xy=data[[x,y]].dropna();fig,ax=plt.subplots(figsize=(9,5.5))
    ax.scatter(xy[x],xy[y],s=45,alpha=.76,c=TEAL,edgecolors='white',linewidths=.5)
    if xy[x].nunique()>1 and len(xy)>=3:
        coef=np.polyfit(xy[x].astype(float),xy[y].astype(float),1)
        grid=np.linspace(xy[x].min(),xy[x].max(),70)
        ax.plot(grid,np.polyval(coef,grid),color=ORANGE,lw=2,label='Garis tren linear (ilustratif)')
        ax.legend(frameon=False)
    ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.18)
    return _png(fig,title,source,dpi)


def regression_png(result,title,source,dpi=200):
    df=result.plot_data
    if df.empty:raise ValueError('Data regresi kosong.')
    fig,axes=plt.subplots(1,2,figsize=(12,5.3))
    axes[0].scatter(df['Prediksi'],df[result.meta['y']],s=35,c=TEAL,alpha=.75)
    mn=min(df['Prediksi'].min(),df[result.meta['y']].min());mx=max(df['Prediksi'].max(),df[result.meta['y']].max())
    axes[0].plot([mn,mx],[mn,mx],color=ORANGE,linestyle='--',lw=1.5)
    axes[0].set_xlabel('Nilai prediksi');axes[0].set_ylabel('Nilai observasi')
    axes[0].set_title('Observasi vs Prediksi')
    axes[1].scatter(df['Prediksi'],df['Residual'],s=35,c=TEAL,alpha=.75)
    axes[1].axhline(0,color=ORANGE,ls='--')
    axes[1].set_xlabel('Nilai prediksi');axes[1].set_ylabel('Residual')
    axes[1].set_title('Residual vs Prediksi')
    for ax in axes:ax.grid(alpha=.2)
    return _png(fig,title,source,dpi)


def qq_png(result,title,source,dpi=200):
    residual=result.plot_data['Residual'].dropna().to_numpy(float)
    fig,ax=plt.subplots(figsize=(7.7,5.3))
    stats.probplot(residual,dist='norm',plot=ax)
    ax.set_title('Q-Q Plot Residual'); ax.grid(alpha=.17)
    return _png(fig,title,source,dpi)


def difference_png(result,title,source,dpi=200):
    df=result.plot_data
    fig,ax=plt.subplots(figsize=(9,5.5))
    if 'Kelompok' in df.columns:
        keys=[str(k) for k in pd.unique(df['Kelompok'])]
        arrays=[df.loc[df['Kelompok'].astype(str)==k,'Nilai'].dropna().to_numpy(float) for k in keys]
        ax.boxplot(arrays,tick_labels=keys,showmeans=True,meanline=False,patch_artist=True,
                   boxprops=dict(facecolor='#A9D9D4',edgecolor=TEAL),medianprops=dict(color=ORANGE,lw=2))
        rng=np.random.default_rng(301)
        for i,v in enumerate(arrays):
            jit=rng.uniform(-.08,.08,len(v));ax.scatter(np.repeat(i+1,len(v))+jit,v,color=TEAL,s=21,alpha=.56)
        ax.set_ylabel(result.meta['response'])
    else:
        x=df['Kelompok 1'].to_numpy(float);y=df['Kelompok 2'].to_numpy(float)
        for i,(vx,vy) in enumerate(zip(x,y)):
            ax.plot([1,2],[vx,vy],color='#ABC9CB',lw=1,alpha=.75)
        row=result.tables['Hasil_Uji'].iloc[0]
        first=str(row['Kelompok 1']);second=str(row['Kelompok 2'])
        ax.scatter(np.ones(len(x)),x,color=TEAL,s=26,label=first,zorder=3)
        ax.scatter(np.ones(len(y))*2,y,color=ORANGE,s=26,label=second,zorder=3)
        ax.set_xticks([1,2],[first,second])
        ax.legend(frameon=False)
        ax.set_ylabel(result.meta['response'])
    ax.grid(axis='y',alpha=.18)
    return _png(fig,title,source,dpi)
