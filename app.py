"""STATCAL ONLINE — Environmental Statistical Analysis.

Jalankan: streamlit run app.py
Batas: analisis inferensial klasik untuk data yang memenuhi rancangan dan asumsi.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_io import apply_filters,category_cols,detect_header,normalize,numeric_cols,options,parse,prefer_numeric
from stat_core import Result,corr_matrix,grouped,regression,test_difference,label_group
from plots import correlation_png,scatter_png,regression_png,qq_png,difference_png
from report import report_bytes

HERE=Path(__file__).parent
DEMO_1=HERE/'data'/'Kasus_1_Sepeda_Motor_CO.csv'
DEMO_2=HERE/'data'/'Kasus_2_Kualitas_Air_Tiga_Lokasi.csv'

st.set_page_config(page_title='STATCAL ONLINE | Environmental Statistical Analysis',page_icon='📊',layout='wide',initial_sidebar_state='expanded')
st.markdown('''
<style>
.block-container{max-width:1580px;padding-top:1.5rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background-color:#F2F8F8}
[data-testid="stMetric"]{background:#F5FAF9;border:1px solid #D4E9E7;border-radius:11px;padding:11px 15px}
.stButton button,.stDownloadButton button{border-radius:8px}
.statcal-overline{font-size:.78rem;color:#0B867C;letter-spacing:.12em;font-weight:800}
.statcal-header{color:#153F4E;font-size:2rem;font-weight:800;line-height:1.25;margin:.35rem 0}
.statcal-desc{color:#5E7680;font-size:1rem;margin-bottom:1.1rem}
.statcal-band{padding:.7rem .9rem;background:#EBF6F4;border-left:4px solid #138C81;border-radius:6px;margin-bottom:1.2rem;color:#224B57}
</style>''',unsafe_allow_html=True)


def pretty(p):
    if p is None or not np.isfinite(p):return 'tidak tersedia'
    return 'p < 0,001' if p<.001 else f'p = {p:.3f}'.replace('.',',')


def readable(frame:pd.DataFrame):
    out=frame.copy()
    for c in out:
        if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_integer_dtype(out[c]):
            if 'p-value' in c.lower():out[c]=out[c].map(lambda v:'< 0,001' if pd.notna(v) and float(v)<.001 else (f'{v:.4f}' if pd.notna(v) else '—'))
            else:out[c]=out[c].map(lambda v:f'{v:.4f}' if pd.notna(v) else '—')
    return out


def selector(label,opts,wanted=None,key=None):
    if not opts:return None
    index=opts.index(wanted) if wanted in opts else 0
    return st.selectbox(label,opts,index=index,key=key)


def split_selector(df,catcols,tag,exclusions=()):
    opts=[c for c in catcols if c not in exclusions]
    st.markdown('**Split analisis (opsional)**')
    chosen=st.multiselect('Pisahkan analisis berdasarkan maksimal 2 kategori',opts,key=f'{tag}_split',max_selections=2,
           help='Contoh: Tahun; aplikasi menjalankan uji untuk setiap tahun secara terpisah. Jangan pilih variabel pembanding pada uji beda.')
    size=1
    for c in chosen:size*=df[c].nunique(dropna=False)
    if size>16:
        st.warning('Terlalu banyak kombinasi grup. Pilih kategori lain atau filter data terlebih dahulu (maks. 16).')
        return None
    return chosen


def execute_by_split(df,split,runner):
    success=[]; failures=[]
    if split is None:return [],['Konfigurasi split melebihi batas.']
    for key,sub in grouped(df,split):
        tag=label_group(key)
        try:
            success.append((tag,runner(sub)))
        except (ValueError,TypeError,ZeroDivisionError,IndexError, FloatingPointError) as err:
            failures.append(f'{tag}: {err}')
        except Exception as err:
            failures.append(f'{tag}: proses gagal ({type(err).__name__}: {err}).')
    return success,failures


def export_buttons(results,df,title,source,detail,kind,make_png=None):
    if not results:return
    try:
        xlsx=report_bytes(results,df,title,source,detail)
        col1,col2=st.columns(2)
        with col1:st.download_button('📗 Download semua tabel ke Excel',xlsx,
                          file_name=f'STATCAL_{kind}_Hasil.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                          key=f'{kind}_excel',use_container_width=True)
        if make_png is not None:
            with col2:
                # PNG dalam ZIP bila beberapa split, satu PNG bila tanpa split
                if len(results)==1:
                    buf=make_png(results[0][1],title+' — '+results[0][0]);suffix='.png'
                    mime='image/png'
                else:
                    bio=BytesIO()
                    with ZipFile(bio,'w',ZIP_DEFLATED) as zf:
                        for i,(name,res) in enumerate(results,1):
                            zf.writestr(f'{kind}_grup_{i:02d}.png',make_png(res,title+' — '+name))
                    buf=bio.getvalue();suffix='.zip';mime='application/zip'
                st.download_button('🖼️ Download PNG' if suffix=='.png' else '🗂️ Download semua PNG (.zip)',
                        buf,file_name=f'STATCAL_{kind}_Grafik{suffix}',mime=mime,
                        key=f'{kind}_png',use_container_width=True)
    except Exception as exc:
        st.error(f'Ekspor gagal: {exc}')


def result_panel(results,failures,label):
    if failures:
        with st.expander(f'⚠️ {len(failures)} grup tidak dapat dianalisis',expanded=True):
            for m in failures:st.warning(m)
    if not results:
        st.info('Belum ada grup yang memenuhi persyaratan analisis. Periksa pilihan variabel, grup, dan data hilang.')
        return None,None
    names=[n for n,_ in results]
    selected=selector('Lihat hasil untuk kelompok',names,key=f'{label}_group_select')
    res=dict(results)[selected]
    st.markdown(f'#### {res.title} — {selected}')
    for table_name,frame in res.tables.items():
        with st.expander(table_name.replace('_',' '),expanded=(table_name in ['Hasil_Uji','Ringkasan_Model','Koefisien','Pasangan','Matriks_Korelasi'])):
            st.dataframe(readable(frame),hide_index=True,use_container_width=True)
    if res.note:
        with st.expander('📘 Catatan hasil dan metodologi',expanded=False):
            for item in res.note:st.write('• '+item)
    return selected,res


@st.cache_data(show_spinner=False)
def load_data(raw,filename,sheet,header):return parse(raw,filename,sheet,header)


def source_control():
    st.sidebar.subheader('1. Sumber data')
    choice=st.sidebar.radio('Dataset', ['Contoh Kasus 1 — Motor & CO (48 baris)',
                                      'Contoh Kasus 2 — Kualitas air (3 lokasi)',
                                      'Upload Excel / CSV'],label_visibility='collapsed')
    if choice.startswith('Contoh Kasus 1'):
        raw=DEMO_1.read_bytes();filename=DEMO_1.name;source='Data simulasi Kasus 1 — sepeda motor dan CO'
    elif choice.startswith('Contoh Kasus 2'):
        raw=DEMO_2.read_bytes();filename=DEMO_2.name;source='Data simulasi Kasus 2 — kualitas air tiga lokasi'
    else:
        up=st.sidebar.file_uploader('Upload Excel (.xlsx) atau CSV',type=['xlsx','csv'])
        if up is None:st.info('Silakan upload Excel/CSV untuk memulai, atau pilih dataset contoh di sidebar.');st.stop()
        if up.size>30*1024*1024:st.error('Ukuran file maksimum 30 MB.');st.stop()
        raw=up.getvalue();filename=up.name;source=f'Data pengguna: {filename}'
    sheet='';header=1
    if filename.lower().endswith('.xlsx'):
        try:
            xl=pd.ExcelFile(BytesIO(raw),engine='openpyxl');sheets=xl.sheet_names
            index=next((i for i,s in enumerate(sheets) if s.lower()=='data'),0)
            sheet=st.sidebar.selectbox('Sheet Excel',sheets,index=index)
            auto=detect_header(raw,sheet)
            header=int(st.sidebar.number_input('Nomor baris header',1,1000,value=auto,
                       help='Jika file memiliki judul di baris awal, pilih baris tempat nama kolom berada.'))
        except Exception as exc:st.error(f'Gagal membaca sheet Excel: {exc}');st.stop()
    try:df=load_data(raw,filename,sheet,header)
    except Exception as exc:st.error(f'Gagal membaca data: {exc}');st.stop()
    if df.empty:st.error('Dataset tidak memuat pengamatan.');st.stop()
    if len(df)>200_000:st.error('Dataset lebih dari 200.000 baris. Silakan gunakan sampel lebih kecil.');st.stop()
    if len(df.columns)>150:st.error('Maksimum 150 kolom per file.');st.stop()
    return df,source,filename


def main():
    st.markdown('<div class="statcal-overline">STATCAL ONLINE • ENVIRONMENTAL STATISTICS LAB</div>',unsafe_allow_html=True)
    st.markdown('<div class="statcal-header">Environmental Statistical Analysis</div>',unsafe_allow_html=True)
    st.markdown('<div class="statcal-desc">Korelasi • Regresi sederhana & berganda • Uji beda parametrik & nonparametrik • Split kategori • Excel & PNG</div>',unsafe_allow_html=True)
    raw,source,filename=source_control()
    if 'simulasi' in source.lower():
        st.info('Dataset contoh adalah DATA SIMULASI untuk pembelajaran, bukan data hasil pengamatan lapangan.')
    with st.sidebar:
        st.divider();st.subheader('2. Filter data')
        categories=category_cols(raw)
        filter_cols=st.multiselect('Pilih kolom untuk filter',categories,max_selections=3)
        chosen={}
        for c in filter_cols:
            choices=options(raw,c)
            chosen[c]=st.multiselect(f'Nilai {c}',choices,default=choices)
        df=apply_filters(raw,chosen)
        st.caption(f'Observasi aktif: **{len(df):,}** dari **{len(raw):,}**')
        st.divider();st.subheader('3. Pengaturan analisis')
        alpha=st.select_slider('Taraf signifikansi (α)',options=[.01,.05,.10],value=.05)
        confidence=1-alpha
        dpi=st.select_slider('Resolusi PNG',options=[150,200,300],value=200,format_func=lambda v:f'{v} DPI')
        source_caption=st.text_input('Sumber yang dicetak pada grafik',value='Sumber: Data simulasi, diolah dengan STATCAL ONLINE.' if 'simulasi' in source.lower() else f'Sumber: {filename}')
    nums=numeric_cols(df);cats=category_cols(df)
    if not nums:st.error('Tidak ada variabel numerik yang valid. Periksa format angka pada file.');st.stop()
    if df.empty:st.error('Filter menghasilkan nol baris. Pilih kembali kategori filter.');st.stop()
    tabs=st.tabs(['📁 Data & Variabel','🔗 Korelasi','📈 Regresi','⚖️ Uji Beda'])
    with tabs[0]:
        st.subheader('Data aktif & struktur variabel')
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Observasi aktif',f'{len(df):,}')
        c2.metric('Jumlah variabel',len(df.columns))
        c3.metric('Variabel numerik',len(nums))
        c4.metric('Sel kosong',int(df.isna().sum().sum()))
        st.dataframe(df.head(200),hide_index=True,use_container_width=True,height=340)
        info=pd.DataFrame([{'Variabel':c,'Tipe':str(df[c].dtype),'Jenis':('Numerik' if c in nums else 'Kategorik / lainnya'),
                            'N valid':int(df[c].notna().sum()),'Missing':int(df[c].isna().sum()),
                            'Nilai unik':int(df[c].nunique(dropna=True))} for c in df])
        st.markdown('#### Kamus variabel otomatis')
        st.dataframe(info,use_container_width=True,hide_index=True)
        st.caption('ID, Tahun, Bulan, dan variabel numerik dengan kategori terbatas dapat dipakai sebagai kelompok. '
                  'Pastikan satuan, unit observasi, dan rancangan pengukuran sudah benar sebelum uji inferensial.')
    with tabs[1]:
        st.subheader('Analisis Korelasi')
        left,right=st.columns([1,1.25])
        with left:
            prefer=[prefer_numeric(nums,['motor']),prefer_numeric(nums,['co_'])]
            prefer=list(dict.fromkeys(x for x in prefer if x))
            cols=st.multiselect('Variabel numerik untuk matriks korelasi (min. 2)',nums,
                                default=prefer if len(prefer)>=2 else nums[:min(4,len(nums))],key='cor_cols')
            method=st.selectbox('Metode', ['pearson','spearman','kendall'],format_func=lambda x:x.title(),key='corr_method')
            corr_split=split_selector(df,cats,'corr')
        with right:
            st.markdown('<div class="statcal-band">Tabel pasangan menampilkan koefisien, N dan p-value. CI Fisher-z ditampilkan untuk Pearson jika memungkinkan. Matriks memakai pairwise deletion.</div>',unsafe_allow_html=True)
            title_c=st.text_input('Judul hasil korelasi','Tabel 4.1. Hubungan Variabel Lingkungan (Analisis Korelasi)',key='c_title')
        if len(cols)>=2:
            results,failed=execute_by_split(df,corr_split,lambda part:corr_matrix(part,cols,method,confidence))
            selected,res=result_panel(results,failed,'corr')
            if res:
                title_fig='Gambar 4.1. Heatmap Korelasi Variabel Lingkungan'
                st.image(correlation_png(res,title_fig+' — '+selected,source_caption,dpi),use_container_width=True)
                x=selector('X untuk scatterplot pendamping',cols,key='corr_plot_x')
                ycols=[c for c in cols if c!=x]
                y=selector('Y untuk scatterplot pendamping',ycols,key='corr_plot_y')
                pair=res.plot_data[[x,y]].dropna()
                if len(pair)>=3 and pair[x].nunique()>1:
                    fig=px.scatter(pair,x=x,y=y,trendline=None,template='plotly_white',title='Eksplorasi pasangan variabel')
                    st.plotly_chart(fig,use_container_width=True)
                    st.download_button('🖼️ Unduh PNG Scatterplot',scatter_png(pair,x,y,title_fig.replace('Heatmap Korelasi','Scatterplot'),source_caption,dpi),
                                       file_name='STATCAL_Correlation_Scatterplot.png',mime='image/png')
                export_buttons(results,df,title_c,source,{'Metode':method,'Alpha':alpha,'Split':str(corr_split)},
                               'Korelasi',lambda r,t:correlation_png(r,t,source_caption,dpi))
        else:st.info('Pilih minimal 2 variabel numerik untuk korelasi.')
    with tabs[2]:
        st.subheader('Regresi Linear Sederhana & Berganda')
        left,right=st.columns([1,1.15])
        with left:
            y_default=prefer_numeric(nums,['co_','bod','nilai'])
            y=selector('Variabel dependen (Y)',nums,y_default,'reg_y')
            xchoices=[c for c in nums if c!=y]
            xdefault=prefer_numeric(xchoices,['motor','curah','suhu'])
            xs=st.multiselect('Variabel independen (X) — satu atau beberapa',xchoices,
                              default=[xdefault] if xdefault else [],key='reg_x')
            robust=st.checkbox('Robust standard errors (HC3)',value=False,key='reg_robust',
                          help='Untuk inferensi yang lebih tahan terhadap heteroskedastisitas; tidak mengganti koefisien OLS.')
            reg_split=split_selector(df,cats,'reg')
        with right:
            st.markdown('<div class="statcal-band">Satu X → regresi sederhana. Dua atau lebih X → regresi berganda. Tabel diagnostik menampilkan Shapiro–Wilk residual, Breusch–Pagan, Durbin–Watson, dan VIF.</div>',unsafe_allow_html=True)
            title_r=st.text_input('Judul hasil regresi','Tabel 5.1. Hasil Regresi Linear Variabel Lingkungan',key='r_title')
        if xs:
            results,failed=execute_by_split(df,reg_split,lambda part:regression(part,y,xs,confidence,robust))
            selected,res=result_panel(results,failed,'reg')
            if res:
                summary=res.tables['Ringkasan_Model'].iloc[0]
                c1,c2,c3=st.columns(3);c1.metric('N valid',int(summary['N valid']))
                c2.metric('R Square',f"{summary['R Square']:.4f}")
                c3.metric('Adj R Square',f"{summary['Adjusted R Square']:.4f}")
                st.image(regression_png(res,'Gambar 5.1. Diagnostik Regresi — '+selected,source_caption,dpi),use_container_width=True)
                with st.expander('Q-Q Plot residual',expanded=False):
                    st.image(qq_png(res,'Gambar 5.2. Q-Q Plot Residual — '+selected,source_caption,dpi),use_container_width=True)
                    st.download_button('Download Q-Q plot PNG',qq_png(res,'Q-Q Plot Residual',source_caption,dpi),
                                       file_name='STATCAL_Regresi_QQ.png',mime='image/png')
                export_buttons(results,df,title_r,source,{'Model':'Regresi berganda' if len(xs)>1 else 'Regresi sederhana',
                               'Y':y,'X':', '.join(xs),'Robust HC3':robust,'Alpha':alpha,'Split':str(reg_split)},
                               'Regresi',lambda r,t:regression_png(r,t,source_caption,dpi))
        else:st.info('Pilih minimal satu variabel independen X.')
    with tabs[3]:
        st.subheader('Uji Beda — Parametrik & Nonparametrik')
        METHODS={
           'Paired Samples t-Test':'paired_t','Wilcoxon Signed-Rank':'wilcoxon',
           'Independent Samples t-Test (pooled)':'independent_t','Welch Independent t-Test':'welch_t',
           'Mann–Whitney U':'mannwhitney','One-Way ANOVA':'anova',
           'Welch ANOVA':'welch_anova','Kruskal–Wallis':'kruskal'}
        method_label=st.selectbox('Pilih jenis uji beda',list(METHODS),key='test_method')
        method=METHODS[method_label]
        paired=method in ('paired_t','wilcoxon');two=paired or method in ('independent_t','welch_t','mannwhitney')
        c1,c2=st.columns([1,1])
        with c1:
            response=selector('Variabel hasil numerik (Y)',nums,prefer_numeric(nums,['co_','bod','nilai']),'test_y')
            response_b=None; group=None; a=b=id_col=None
            if paired:
                fmt=st.radio('Bentuk data berpasangan',['Long (ID + kategori)','Wide (dua kolom numerik)'],horizontal=True,
                             help='Long untuk Kasus 1: Periode sebagai ID pasangan, Jenis_hari sebagai kelompok.')
                if fmt.startswith('Wide'):
                    response_b=selector('Variabel hasil pasangan kedua',[c for c in nums if c!=response],key='pair_b')
                else:
                    group=selector('Kolom kategori pembanding', [x for x in cats if x!=response],
                                   'Jenis_hari' if 'Jenis_hari' in cats else None,'pair_group')
                    id_opts=[c for c in df.columns if c not in (response,group)]
                    id_col=selector('ID pasangan — harus unik per kelompok',id_opts,
                                    'Periode' if 'Periode' in id_opts else None,'pair_id')
            else:
                group=selector('Variabel kategori pembanding',[x for x in cats if x!=response],
                               'Jenis_hari' if 'Jenis_hari' in cats else None,'test_group')
            if two and group is not None:
                levels=options(df,group)
                if len(levels)>=2:
                    a=selector('Kelompok 1',levels,'Hari kerja' if 'Hari kerja' in levels else None,'test_a')
                    other=[x for x in levels if x!=a]
                    b=selector('Kelompok 2',other,'Akhir pekan' if 'Akhir pekan' in other else None,'test_b')
            excludes=[group] if group else []
            if id_col:excludes.append(id_col)
            test_split=split_selector(df,cats,'test',exclusions=excludes)
            posthoc=st.checkbox('Post-hoc otomatis jika uji keseluruhan signifikan',value=True,
                              disabled=method not in ('anova','welch_anova','kruskal'))
        with c2:
            st.markdown('<div class="statcal-band">Untuk kasus sepeda motor dan CO, pilih <b>Paired Samples t-Test</b>, variabel <b>CO_rata2_ppm</b>, kategori <b>Jenis_hari</b>, dan ID pasangan <b>Periode</b>. Untuk membedakan tahun, gunakan <b>Split: Tahun</b>.</div>',unsafe_allow_html=True)
            st.info('Uji independen memerlukan kelompok independen; uji berpasangan memerlukan ID pasangan yang tepat. '
                    'ANOVA/Welch/Kruskal memerlukan ≥3 kelompok. Aplikasi tidak dapat memastikan desain sampling hanya dari nama kolom.')
            title_t=st.text_input('Judul hasil uji beda','Tabel 6.1. Hasil Uji Perbedaan Variabel Lingkungan',key='t_title')
        if group is None and not (paired and response_b):
            st.warning('Tidak tersedia kolom kategori yang sesuai. Gunakan format Wide jika data berupa dua kolom berpasangan.')
        else:
            results,failed=execute_by_split(df,test_split,lambda part:test_difference(part,method,response,
                                            group=group,a=a,b=b,id_col=id_col,response_b=response_b,
                                            confidence=confidence,posthoc=posthoc))
            selected,res=result_panel(results,failed,'test')
            if res:
                line=res.tables['Hasil_Uji'].iloc[0]
                p=float(line['p-value'])
                decision='terdapat bukti statistik untuk perbedaan antarkelompok' if p<alpha else 'bukti belum cukup untuk menyimpulkan perbedaan antarkelompok'
                st.markdown(f'**Ringkasan:** {pretty(p)} (α = {alpha:.2f}); {decision}. '
                            'Ini bukan otomatis bukti perbedaan kausal atau kepatuhan baku mutu.')
                st.image(difference_png(res,'Gambar 6.1. Perbandingan Kelompok — '+selected,source_caption,dpi),use_container_width=True)
                export_buttons(results,df,title_t,source,{'Metode':method_label,'Alpha':alpha,'Split':str(test_split),
                                                           'Y':response,'Kelompok':group or 'Wide',
                                                           'ID pasangan':id_col or 'Tidak digunakan'},
                               'Uji_Beda',lambda r,t:difference_png(r,t,source_caption,dpi))
    st.divider()
    st.caption('STATCAL ONLINE | Praktikum Statistika Lingkungan • Data contoh adalah simulasi. '
               'Untuk data panel, deret waktu, pengukuran berulang kompleks, atau desain cluster, gunakan model yang sesuai '
               'dan jangan menganggap semua baris sebagai observasi independen.')

if __name__=='__main__':main()
