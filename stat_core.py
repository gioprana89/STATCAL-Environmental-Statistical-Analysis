"""Perhitungan statistik STATCAL – independen dari Streamlit.

Fungsi menerima DataFrame dan menghasilkan tabel sehingga dapat diuji secara terpisah.
Hanya hasil yang benar-benar dapat dihitung yang ditampilkan; kondisi gagal memakai exception.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.stattools import durbin_watson

@dataclass
class Result:
    title: str
    tables: dict[str, pd.DataFrame]
    note: list[str] = field(default_factory=list)
    plot_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict = field(default_factory=dict)


def label_group(key):
    if key == ('Semua data',): return 'Semua data'
    if not isinstance(key, tuple): key=(key,)
    return ' | '.join(str(v) for v in key)


def grouped(df: pd.DataFrame, columns: list[str]):
    if not columns: return [(('Semua data',), df.copy())]
    for col in columns:
        if col not in df: raise ValueError(f'Kolom split tidak tersedia: {col}')
    out=[]
    for key, block in df.groupby(columns, dropna=False, sort=False, observed=True):
        if not isinstance(key,tuple): key=(key,)
        out.append((key,block.copy()))
    return out


def check_numeric(df, col):
    if col not in df: raise ValueError(f'Variabel {col} tidak ditemukan')
    if not pd.api.types.is_numeric_dtype(df[col]): raise ValueError(f'{col} bukan variabel numerik')


def corr_pair(df, x, y, method='pearson', confidence=0.95):
    check_numeric(df,x); check_numeric(df,y)
    if x == y: raise ValueError('Pilih dua variabel yang berbeda.')
    xy=df[[x,y]].replace([np.inf,-np.inf],np.nan).dropna()
    n=len(xy)
    if n<3: raise ValueError('Diperlukan minimal 3 pasangan pengamatan lengkap.')
    if xy[x].nunique()<2 or xy[y].nunique()<2: raise ValueError('Variabel konstan; korelasi tidak dapat dihitung.')
    if method=='pearson': test=stats.pearsonr(xy[x],xy[y]); r,p=float(test.statistic),float(test.pvalue)
    elif method=='spearman': test=stats.spearmanr(xy[x],xy[y]); r,p=float(test.statistic),float(test.pvalue)
    elif method=='kendall': test=stats.kendalltau(xy[x],xy[y]); r,p=float(test.statistic),float(test.pvalue)
    else: raise ValueError('Metode korelasi tidak tersedia.')
    ci_low=ci_high=np.nan
    if method=='pearson' and n>3 and np.abs(r)<1:
        z=np.arctanh(r); se=1/np.sqrt(n-3); zcrit=stats.norm.ppf((1+confidence)/2)
        ci_low=float(np.tanh(z-zcrit*se)); ci_high=float(np.tanh(z+zcrit*se))
    elif method=='pearson' and np.abs(r)==1:
        ci_low=ci_high=r
    row={'X':x,'Y':y,'Metode':method.title(),'N pasangan':n,'Koefisien':r,
         'p-value':p,f'CI {confidence:.0%} bawah':ci_low,f'CI {confidence:.0%} atas':ci_high}
    return row,xy


def corr_matrix(df, cols, method='pearson', confidence=0.95):
    if len(cols)<2: raise ValueError('Pilih paling sedikit 2 variabel numerik.')
    rows=[]
    for x,y in combinations(cols,2):
        try: row,_=corr_pair(df,x,y,method,confidence); rows.append(row)
        except ValueError as exc:
            rows.append({'X':x,'Y':y,'Metode':method.title(),'N pasangan':len(df[[x,y]].dropna()),
                         'Koefisien':np.nan,'p-value':np.nan,'Catatan':str(exc)})
    pair=pd.DataFrame(rows)
    matrix=pd.DataFrame(np.eye(len(cols)),index=cols,columns=cols,dtype=float)
    nmatrix=pd.DataFrame(index=cols,columns=cols,dtype=object)
    for c in cols: nmatrix.loc[c,c]=int(df[c].notna().sum())
    for row in rows:
        x,y=row['X'],row['Y']; matrix.loc[x,y]=matrix.loc[y,x]=row['Koefisien'];
        nmatrix.loc[x,y]=nmatrix.loc[y,x]=row['N pasangan']
    return Result('Analisis Korelasi',{'Pasangan':pair,'Matriks_Korelasi':matrix.reset_index(names='Variabel'),
                                    'N_Per_Pasangan':nmatrix.reset_index(names='Variabel')},
                  ['N pada setiap pasangan dapat berbeda karena missing values (pairwise deletion).',
                   'Interval kepercayaan Fisher-z hanya ditampilkan untuk Pearson; pendekatan memerlukan asumsi tertentu.'],
                  df[cols].copy(),{'method':method,'columns':cols,'matrix':matrix})


def regression(df, y, predictors, confidence=0.95, robust=False):
    if not predictors: raise ValueError('Pilih minimal satu variabel X.')
    if y in predictors: raise ValueError('Variabel Y tidak boleh sekaligus menjadi X.')
    if len(set(predictors)) != len(predictors): raise ValueError('Variabel X harus unik.')
    for c in [y]+predictors: check_numeric(df,c)
    dat=df[[y]+predictors].replace([np.inf,-np.inf],np.nan).dropna().astype(float)
    n=len(dat); k=len(predictors)
    if n<=k+2: raise ValueError(f'Minimal {k+3} observasi lengkap diperlukan untuk {k} prediktor.')
    if any(dat[c].nunique()<2 for c in predictors): raise ValueError('Ada prediktor yang konstan.')
    if dat[y].nunique()<2: raise ValueError('Variabel Y konstan.')
    xx=add_constant(dat[predictors],has_constant='add')
    if np.linalg.matrix_rank(xx.to_numpy())<xx.shape[1]: raise ValueError('Prediktor tidak bebas linear (multikolinearitas sempurna).')
    ols=OLS(dat[y],xx).fit()
    fit=ols.get_robustcov_results(cov_type='HC3') if robust else ols
    names=list(xx.columns)
    crit=stats.t.ppf((1+confidence)/2,ols.df_resid)
    estimates=np.array(fit.params,dtype=float)
    ses=np.array(fit.bse,dtype=float)
    ci_low=estimates-crit*ses; ci_high=estimates+crit*ses
    coef=pd.DataFrame({'Parameter':names,'Koefisien':estimates,'Std. Error':ses,
                       't-statistic':np.array(fit.tvalues,dtype=float),
                       'p-value':np.array(fit.pvalues,dtype=float),
                       f'CI {confidence:.0%} bawah':ci_low, f'CI {confidence:.0%} atas':ci_high})
    residual=np.asarray(ols.resid); fitted=np.asarray(ols.fittedvalues)
    rmse=float(np.sqrt(np.mean(residual**2)))
    summ=pd.DataFrame([{'N valid':n,'Jumlah X':k,'R Square':float(ols.rsquared),
                        'Adjusted R Square':float(ols.rsquared_adj),'RMSE':rmse,
                        'F-statistic':float(fit.fvalue) if fit.fvalue is not None else np.nan,
                        'p-value F':float(fit.f_pvalue) if fit.f_pvalue is not None else np.nan,
                        'df residual':float(ols.df_resid),'SE': 'HC3 robust' if robust else 'OLS klasik'}])
    diagnostic=[]
    if 3<=n<=5000:
        sh=stats.shapiro(residual)
        diagnostic.append({'Uji':'Shapiro-Wilk (residual)','Statistik':float(sh.statistic),'p-value':float(sh.pvalue)})
    try:
        bp=het_breuschpagan(residual,xx.to_numpy())
        diagnostic.append({'Uji':'Breusch-Pagan (LM)','Statistik':float(bp[0]),'p-value':float(bp[1])})
    except (ValueError,RuntimeError,ZeroDivisionError):pass
    diagnostic.append({'Uji':'Durbin-Watson (residual berurutan)','Statistik':float(durbin_watson(residual)),'p-value':np.nan})
    vif=[]
    for j, name in enumerate(predictors, start=1):
        val=float(variance_inflation_factor(xx.to_numpy(),j))
        vif.append({'Variabel':name,'VIF':val,'Tolerance':(1/val if val>0 and np.isfinite(val) else np.nan)})
    # raw OLS ANOVA via centered SSR and SSE
    ana=pd.DataFrame([
        {'Sumber':'Regresi','Sum of Squares':float(ols.ess),'df':int(ols.df_model),
         'Mean Square':float(ols.ess/ols.df_model) if ols.df_model else np.nan},
        {'Sumber':'Residual','Sum of Squares':float(ols.ssr),'df':int(ols.df_resid),
         'Mean Square':float(ols.mse_resid)},
        {'Sumber':'Total','Sum of Squares':float(ols.centered_tss),'df':int(n-1),'Mean Square':np.nan},
    ])
    data=dat.copy();data['Prediksi']=fitted;data['Residual']=residual
    signs=[('+' if float(c)>=0 else '-')+' '+f'{abs(float(c)):.4f}×{name}' for name,c in zip(predictors,estimates[1:])]
    equation=f'{y} = {float(estimates[0]):.4f} '+ ' '.join(signs)
    notes=[f'Persamaan: {equation}',
           'R² dan tabel ANOVA adalah ringkasan OLS; HC3 (jika dipilih) digunakan untuk inferensi koefisien dan uji F robust.',
           'Durbin-Watson bergantung urutan baris; periksa urutan waktu dan desain sampel sebelum interpretasi.',
           'Shapiro-Wilk menguji normalitas residual, bukan normalitas masing-masing variabel.']
    return Result('Regresi Linear',{'Ringkasan_Model':summ,'Koefisien':coef,'ANOVA_OLS':ana,
                                   'Diagnostik':pd.DataFrame(diagnostic),'VIF':pd.DataFrame(vif)},
                  notes,data,{'y':y,'xs':predictors,'fitted':fitted,'residual':residual,'n':n})


def paired_values(df, group, response, level_a,level_b,id_col):
    for c in [response]:check_numeric(df,c)
    if id_col in (group,response):raise ValueError('ID pasangan harus berbeda dari variabel kelompok dan Y.')
    sub=df.loc[df[group].isin([level_a,level_b]),[id_col,group,response]].dropna()
    if sub.duplicated([id_col,group]).any():
        raise ValueError('Terdapat lebih dari satu nilai per ID pasangan dan kelompok. Agregasi data dahulu atau gunakan ID yang unik.')
    wide=sub.pivot(index=id_col,columns=group,values=response)
    if level_a not in wide or level_b not in wide:raise ValueError('Salah satu kelompok tidak memiliki data.')
    pairs=wide[[level_a,level_b]].dropna()
    if len(pairs)<2:raise ValueError('Minimal 2 pasangan lengkap.')
    p=pairs.reset_index();p.columns=[id_col,'Kelompok 1','Kelompok 2']
    return p


def _effect_g(a,b):
    n1,n2=len(a),len(b)
    if n1+n2<=2:return np.nan
    den=n1+n2-2
    sp=np.sqrt(((n1-1)*np.var(a,ddof=1)+(n2-1)*np.var(b,ddof=1))/den)
    d=(np.mean(a)-np.mean(b))/sp if sp>0 else np.nan
    correction=1-3/(4*den-1) if den>1 else np.nan
    return float(d*correction)


def _dunn_posthoc(groups):
    all_values=np.concatenate([np.array(v,dtype=float) for v in groups.values()])
    ranks=stats.rankdata(all_values)
    n=len(all_values); ties=np.unique(all_values,return_counts=True)[1]
    c=1-np.sum(ties**3-ties)/(n**3-n) if n>1 else np.nan
    if not np.isfinite(c) or c<=0:raise ValueError('Seluruh data identik sehingga post-hoc tidak dapat dihitung.')
    starts={};offset=0
    for name,a in groups.items():
        starts[name]=ranks[offset:offset+len(a)].mean();offset+=len(a)
    rows=[]; pvals=[]
    for i,j in combinations(groups,2):
        ni,nj=len(groups[i]),len(groups[j])
        se=np.sqrt((n*(n+1)/12)*c*(1/ni+1/nj))
        z=(starts[i]-starts[j])/se
        p=float(2*stats.norm.sf(abs(z)))
        rows.append({'Kelompok 1':i,'Kelompok 2':j,'Selisih mean rank':starts[i]-starts[j],
                     'z':z,'p-value (mentah)':p})
        pvals.append(p)
    adj=multipletests(pvals,alpha=.05,method='holm')[1]
    for row,p in zip(rows,adj): row['p-value (Holm)']=p
    return pd.DataFrame(rows)


def test_difference(df, method, response, group=None, a=None,b=None,
                    id_col=None, response_b=None, confidence=.95, posthoc=True):
    """method: paired_t, wilcoxon, independent_t, welch_t, mannwhitney, anova, welch_anova, kruskal.
    Paired format can use group+id OR response,response_b in wide format.
    """
    check_numeric(df,response)
    notes=[]; detail={}; plot_df=pd.DataFrame()
    if method in ('paired_t','wilcoxon'):
        if response_b:
            check_numeric(df,response_b)
            if response==response_b:raise ValueError('Dua variabel berpasangan harus berbeda.')
            pairs=df[[response,response_b]].replace([np.inf,-np.inf],np.nan).dropna().copy()
            pairs.columns=['Kelompok 1','Kelompok 2']
            pairs.insert(0,'ID Pasangan',np.arange(1,len(pairs)+1))
            a_name,b_name=response,response_b
        else:
            if not all([group,id_col]) or a is None or b is None:raise ValueError('Pilih kelompok, dua kategori, dan ID pasangan.')
            if a==b:raise ValueError('Dua kelompok harus berbeda.')
            pairs=paired_values(df,group,response,a,b,id_col)
            a_name,b_name=str(a),str(b)
        x=pairs['Kelompok 1'].to_numpy(float); y=pairs['Kelompok 2'].to_numpy(float); n=len(x)
        if n<2:raise ValueError('Diperlukan paling sedikit 2 pasangan lengkap.')
        diff=x-y
        if method=='paired_t':
            if np.std(diff,ddof=1)<=0:raise ValueError('Selisih pasangan konstan, uji t tidak dapat dihitung secara valid.')
            t=stats.ttest_rel(x,y)
            stat=float(t.statistic);p=float(t.pvalue); name='Paired Samples t-Test'
            effect=float(np.mean(diff)/np.std(diff,ddof=1)); effect_name='Cohen dz'
            se=stats.sem(diff); ci=stats.t.interval(confidence,n-1,loc=np.mean(diff),scale=se)
            detail['Interval Kepercayaan Selisih']=pd.DataFrame([{'CI bawah':ci[0],'CI atas':ci[1]}])
        else:
            if np.allclose(diff,0):raise ValueError('Semua selisih pasangan nol; Wilcoxon tidak dapat dihitung.')
            w=stats.wilcoxon(x,y,alternative='two-sided',zero_method='wilcox',method='auto')
            stat=float(w.statistic);p=float(w.pvalue);name='Wilcoxon Signed-Rank'
            valid=diff[diff!=0]; rank=stats.rankdata(np.abs(valid)); positive=np.sum(rank[valid>0]); negative=np.sum(rank[valid<0]);
            effect=float((positive-negative)/(positive+negative));effect_name='Rank-biserial berpasangan'
            notes.append('Wilcoxon menggunakan peringkat selisih tidak nol; periksa kesimetrisan selisih untuk interpretasi lokasi.')
        try:
            sw=stats.shapiro(diff) if 3<=len(diff)<=5000 else None
        except ValueError:sw=None
        detail['Asumsi']=pd.DataFrame([{'Uji':'Shapiro-Wilk (selisih pasangan)','Statistik':sw.statistic if sw else np.nan,
                                       'p-value':sw.pvalue if sw else np.nan}])
        detail['Data_Pasangan']=pairs
        plot_df=pairs.copy()
        summ={'Kelompok 1':a_name,'Kelompok 2':b_name,'N pasangan':n,
              'Mean kelompok 1':np.mean(x),'Mean kelompok 2':np.mean(y),
              'Mean selisih':np.mean(diff),'Statistik uji':stat,'p-value':p,
              'Nama effect size':effect_name,'Effect size':effect}
    elif method in ('independent_t','welch_t','mannwhitney','anova','welch_anova','kruskal'):
        if not group:raise ValueError('Pilih variabel kategori pembanding.')
        if group==response:raise ValueError('Variabel kelompok dan respons harus berbeda.')
        if group not in df:raise ValueError('Variabel kategori tidak ditemukan.')
        if method in ('independent_t','welch_t','mannwhitney'):
            if a is None or b is None or a==b:raise ValueError('Pilih dua kategori berbeda.')
            vals={str(a):df.loc[df[group]==a,response].dropna().to_numpy(float),
                  str(b):df.loc[df[group]==b,response].dropna().to_numpy(float)}
        else:
            vals={str(v):q[response].dropna().to_numpy(float) for v,q in df.groupby(group,dropna=True,observed=True)}
            vals={name:v for name,v in vals.items() if len(v)>0}
        if any(len(v)<2 for v in vals.values()):raise ValueError('Setiap kelompok membutuhkan minimal 2 observasi valid.')
        if len(vals)<(3 if method in ('anova','welch_anova','kruskal') else 2):
            raise ValueError('Jumlah kategori tidak cukup untuk uji yang dipilih.')
        g=list(vals.values());keys=list(vals)
        descript=pd.DataFrame([{'Kelompok':name,'N':len(v),'Mean':np.mean(v),'SD':np.std(v,ddof=1),
                                'Median':np.median(v)} for name,v in vals.items()])
        detail['Deskriptif_Kelompok']=descript
        try:
            lv=stats.levene(*g,center='median')
            lev={'Uji':'Levene (median)','Statistik':lv.statistic,'p-value':lv.pvalue}
        except ValueError:lev={'Uji':'Levene (median)','Statistik':np.nan,'p-value':np.nan}
        asrows=[lev]
        for name,v in vals.items():
            if 3<=len(v)<=5000:
                try:
                    s=stats.shapiro(v);asrows.append({'Uji':f'Shapiro-Wilk: {name}','Statistik':s.statistic,'p-value':s.pvalue})
                except ValueError:pass
        detail['Asumsi']=pd.DataFrame(asrows)
        plot_df=df.loc[df[group].astype(str).isin(keys),[group,response]].dropna().copy()
        plot_df.columns=['Kelompok','Nilai']
        if method in ('independent_t','welch_t'):
            equal=method=='independent_t'; result=stats.ttest_ind(g[0],g[1],equal_var=equal)
            stat,p=float(result.statistic),float(result.pvalue)
            name='Independent Samples t-Test (pooled)' if equal else 'Welch Independent t-Test'
            effect_name='Hedges g';effect=_effect_g(*g)
            # difference CI (Welch-Satterthwaite or pooled)
            v1=np.var(g[0],ddof=1);v2=np.var(g[1],ddof=1);n1=len(g[0]);n2=len(g[1]);
            if equal:
                sp2=((n1-1)*v1+(n2-1)*v2)/(n1+n2-2);se=np.sqrt(sp2*(1/n1+1/n2));dfree=n1+n2-2
            else:
                se=np.sqrt(v1/n1+v2/n2); dfree=(v1/n1+v2/n2)**2/((v1/n1)**2/(n1-1)+(v2/n2)**2/(n2-1)) if se>0 else np.nan
            crit=stats.t.ppf((1+confidence)/2,dfree)
            diff=np.mean(g[0])-np.mean(g[1])
            detail['Interval Kepercayaan Selisih']=pd.DataFrame([{'Mean selisih':diff,'df':dfree,
                                                       'CI bawah':diff-crit*se,'CI atas':diff+crit*se}])
        elif method=='mannwhitney':
            result=stats.mannwhitneyu(g[0],g[1],alternative='two-sided',method='auto')
            stat,p=float(result.statistic),float(result.pvalue);name='Mann–Whitney U'
            effect_name='Rank-biserial';effect=2*stat/(len(g[0])*len(g[1]))-1
            notes.append('Mann–Whitney membandingkan peringkat/distribusi; tidak otomatis menguji median jika bentuk distribusi berbeda.')
        elif method=='anova':
            result=stats.f_oneway(*g);stat,p=float(result.statistic),float(result.pvalue)
            name='One-Way ANOVA'; effect_name='Eta squared'
            grand=np.mean(np.concatenate(g));ssb=sum(len(v)*(np.mean(v)-grand)**2 for v in g)
            sst=np.sum((np.concatenate(g)-grand)**2)
            effect=ssb/sst if sst>0 else np.nan
            if posthoc and p<.05:
                vals_long=plot_df
                tukey=pairwise_tukeyhsd(vals_long['Nilai'],vals_long['Kelompok'],alpha=1-confidence)
                detail['Posthoc_Tukey']=pd.DataFrame(tukey._results_table.data[1:],columns=tukey._results_table.data[0])
        elif method=='welch_anova':
            result=anova_oneway(g,use_var='unequal',welch_correction=True)
            stat,p=float(result.statistic),float(result.pvalue);name='Welch ANOVA'
            effect_name='Eta squared (deskriptif)'
            grand=np.mean(np.concatenate(g));ssb=sum(len(v)*(np.mean(v)-grand)**2 for v in g)
            sst=np.sum((np.concatenate(g)-grand)**2)
            effect=ssb/sst if sst>0 else np.nan
            if posthoc and p<.05:
                try:
                    gh=stats.tukey_hsd(*g,equal_var=False)
                    cidx=list(combinations(range(len(g)),2));rows=[]
                    for i,j in cidx:
                        rows.append({'Kelompok 1':keys[i],'Kelompok 2':keys[j],
                                     'Selisih mean':np.mean(g[i])-np.mean(g[j]),
                                     'p-value (Games-Howell)':float(gh.pvalue[i,j])})
                    detail['Posthoc_Games_Howell']=pd.DataFrame(rows)
                except (AttributeError,ValueError):notes.append('Games–Howell tidak tersedia pada versi SciPy ini.')
        else:
            result=stats.kruskal(*g);stat,p=float(result.statistic),float(result.pvalue)
            name='Kruskal–Wallis H';effect_name='Epsilon squared'
            nall=sum(len(v) for v in g); effect=max(0.,(stat-len(g)+1)/(nall-len(g)))
            if posthoc and p<.05:detail['Posthoc_Dunn_Holm']=_dunn_posthoc(vals)
        summ={'Kelompok 1':keys[0], 'Kelompok 2':keys[1] if len(keys)==2 else f'{len(keys)} kelompok',
              'N total':sum(len(v) for v in g),'Jumlah kelompok':len(keys),
              'Statistik uji':stat,'p-value':p,'Nama effect size':effect_name,'Effect size':effect}
    else:raise ValueError('Jenis uji tidak dikenal.')
    rows=[{'Metode':name,**summ,'Alpha':1-confidence,
           'Keputusan': 'Tolak H0' if np.isfinite(p) and p<(1-confidence) else 'Gagal menolak H0'}]
    detail={'Hasil_Uji':pd.DataFrame(rows),**detail}
    notes += ['Nilai p dievaluasi berdasarkan desain dan asumsi; signifikansi bukan bukti sebab-akibat.',
              'Penting: pengamatan berulang menurut waktu dapat melanggar asumsi independensi antarunit.']
    return Result(name,detail,notes,plot_df,{'method':method,'group':group,'response':response,
                                                      'a':a,'b':b,'id_col':id_col})
