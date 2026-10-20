"""Reader, tipe variabel, filter, dan lapisan kontrol kualitas input."""
from __future__ import annotations
from io import BytesIO
import re
import numpy as np
import pandas as pd

NUMERIC_MAX_UNIQUE=30


def detect_header(raw: bytes,sheet: str):
    preview=pd.read_excel(BytesIO(raw),sheet_name=sheet,header=None,nrows=25)
    for i,row in preview.iterrows():
        values=set(row.dropna().astype(str).str.strip().str.lower())
        if 'id_observasi' in values:return i+1
    for i,row in preview.iterrows():
        vals=row.dropna()
        if len(vals)>=2 and sum(isinstance(v,str) for v in vals)>=2:return i+1
    return 1


def normalize(df):
    df=df.dropna(axis=0,how='all').dropna(axis=1,how='all').copy()
    names=[];seen={}
    for col in df.columns:
        name=str(col).strip() or 'Kolom'
        if name.startswith('Unnamed:'):name='Kolom'
        if name in seen:
            seen[name]+=1; name=f'{name}_{seen[name]}'
        else:seen[name]=1
        names.append(name)
    df.columns=names
    for col in df:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):continue
        s=df[col].astype('string').str.strip()
        s=s.replace({'':pd.NA,'-':pd.NA,'NA':pd.NA,'nan':pd.NA})
        cleaned=s.str.replace(' ','',regex=False)
        # Indonesian commas only where evidence is strong and conversion almost complete
        decimal=cleaned.str.replace(',','.',regex=False)
        numeric=pd.to_numeric(decimal,errors='coerce')
        if len(s.dropna()) and numeric.notna().sum()>=max(2,int(.95*s.notna().sum())):
            # Prevent accidental conversion of identifiers, year-month strings, and leading-zero codes
            if ('id' not in col.casefold() and 'kode' not in col.casefold()
                    and 'periode' not in col.casefold()
                    and not (s.dropna().str.match(r'^0\d+$').any())):
                df[col]=numeric;continue
        df[col]=s.astype(object).where(s.notna(),np.nan)
    return df.reset_index(drop=True)


def parse(raw,filename,sheet='',header=1):
    if filename.lower().endswith('.csv'):
        errors=[]
        for enc in ['utf-8-sig','latin-1']:
            try:
                data=pd.read_csv(BytesIO(raw),sep=None,engine='python',encoding=enc)
                return normalize(data)
            except Exception as exc:errors.append(str(exc))
        raise ValueError('Gagal membaca file CSV: '+'; '.join(errors))
    if filename.lower().endswith('.xlsx'):
        return normalize(pd.read_excel(BytesIO(raw),sheet_name=sheet,header=header-1,engine='openpyxl'))
    raise ValueError('Format file hanya .xlsx atau .csv.')


def numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and pd.api.types.is_bool_dtype(df[c]) is False]


def category_cols(df,max_unique=NUMERIC_MAX_UNIQUE):
    return [c for c in df.columns if 1<df[c].nunique(dropna=True)<=max_unique]


def options(df,col):
    a=df[col].dropna().unique().tolist()
    try: return sorted(a)
    except TypeError:return sorted(a,key=str)


def apply_filters(df,chosen:dict):
    out=df
    for col,values in chosen.items():
        if col in df and values:out=out[out[col].isin(values)]
    return out.copy()


def prefer_numeric(cols,needle):
    for piece in needle:
        for c in cols:
            if piece in c.casefold():return c
    return cols[0] if cols else None
