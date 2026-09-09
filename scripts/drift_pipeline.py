"""
DRIFT Pipeline v5.0 — Digital Risk & Financial Trust Index
==========================================================
Autor : Ceyner D. Llontop Herrera | ORCID: 0009-0007-6058-7711

Fix v5: clave de merge YYYYMM (numérica) — independiente del formato
        de texto del BCRP ("Ene18" o "Ene.2018").
        Mapeo de columnas BCRP por posición verificada con rango esperado.
"""

import requests, pandas as pd, numpy as np, json, sys, os
from datetime import datetime, date
from pathlib import Path
from io import StringIO

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

INICIO    = "2018-1"
BCRP_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
FRED_API  = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY  = os.environ.get('FRED_API_KEY', '')
IPD_OFICIAL_2025 = 15_628.0

MESES = {'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,
         'Jul':7,'Ago':8,'Sep':9,'Oct':10,'Nov':11,'Dic':12}

def a_fecha(p):
    """'Ene18' o 'Ene.2018' → Timestamp."""
    try:
        p = str(p).strip()
        mes = MESES[p[:3].capitalize()]
        yr  = int(p[3:].lstrip('.').strip())
        if yr < 100: yr += 2000
        return pd.Timestamp(yr, mes, 1)
    except:
        return pd.NaT

def a_yyyymm(p):
    """'Ene18' o 'Ene.2018' → 201801 (int). Clave de merge robusta."""
    f = a_fecha(p)
    return int(f.strftime('%Y%m')) if pd.notna(f) else None

# ── FETCH BCRP (una serie a la vez, sin problemas de orden) ──
def fetch_serie(codigo: str, ini: str, fin: str) -> pd.DataFrame:
    """Extrae UNA serie. Sin ambigüedad de columnas."""
    url = f"{BCRP_BASE}/{codigo}/json/{ini}/{fin}/esp"
    try:
        r = requests.get(url, timeout=40, headers={'User-Agent':'DRIFT/5.0'})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    {codigo} ERROR: {e}"); return pd.DataFrame()

    if 'periods' not in data:
        print(f"    {codigo}: sin datos"); return pd.DataFrame()

    records = []
    for p in data['periods']:
        val = p.get('values', [None])[0]
        try:   v = float(val) if val not in ['n.d.','',None] else np.nan
        except: v = np.nan
        records.append({'yyyymm': a_yyyymm(p['name']),
                        'periodo': p['name'],
                        codigo: v})
    df = pd.DataFrame(records).dropna(subset=['yyyymm'])
    df['yyyymm'] = df['yyyymm'].astype(int)
    df = df.sort_values('yyyymm').reset_index(drop=True)
    ok = df[codigo].notna().sum()
    print(f"    {codigo}: {len(df)} períodos ({ok} con dato) "
          f"| rango: {df[codigo].min():.2f}–{df[codigo].max():.2f}"
          if ok > 0 else f"    {codigo}: {len(df)} períodos (sin dato)")
    return df

# ── FETCH FRED ────────────────────────────────────────────────
def fetch_dxy() -> pd.DataFrame:
    if not FRED_KEY:
        print("  Sin FRED_API_KEY"); return pd.DataFrame()
    try:
        r = requests.get(FRED_API, params={
            'series_id':'TWEXBGSMTH','api_key':FRED_KEY,
            'file_type':'json','observation_start':'2018-01-01',
            'frequency':'m'}, timeout=30)
        r.raise_for_status()
        obs = r.json().get('observations', [])
        records = [{'yyyymm': int(o['date'][:4]+o['date'][5:7]),
                    'DXY': float(o['value'])}
                   for o in obs if o['value'] != '.']
        df = pd.DataFrame(records)
        print(f"  DXY: {len(df)} meses | último={df['DXY'].iloc[-1]:.1f}")
        return df
    except Exception as e:
        print(f"  FRED ERROR: {e}"); return pd.DataFrame()

# ── DIMPD ─────────────────────────────────────────────────────
DENOM = ['PN09398SM','PN09399SM','PN09400SM','PN09403SM','PN09406SM',
         'PN09408SM','PN09409SM','PN09411SM','PN09414SM','PN09416SM']

def fetch_dimpd(ini, fin):
    """Extrae 10 series una a una y calcula DimPD."""
    base = None
    for cod in DENOM:
        ds = fetch_serie(cod, ini, fin)
        if ds.empty: continue
        if base is None:
            base = ds[['yyyymm', cod]]
        else:
            base = base.merge(ds[['yyyymm', cod]], on='yyyymm', how='outer')
    if base is None or base.empty:
        return pd.DataFrame()
    base = base.sort_values('yyyymm').reset_index(drop=True)
    # Proyectar series incompletas
    for col in DENOM:
        if col not in base.columns: continue
        nas = base[col].isna().sum()
        if nas == 0: continue
        ok = base[col].dropna()
        if len(ok) < 4: base[col] = base[col].ffill(); continue
        t  = np.arange(len(ok))
        cf = np.polyfit(t, ok.values, 1)
        tp = np.arange(len(ok), len(ok)+nas)
        pv = np.maximum(np.polyval(cf,tp), ok.iloc[-1]*0.85)
        base.loc[base[col].isna(), col] = pv
    cols_ok = [c for c in DENOM if c in base.columns]
    base['total'] = base[cols_ok].sum(axis=1)
    base['DimPD'] = base['PN09416SM'] / base['total']
    print(f"  DimPD: {base['DimPD'].min():.4f}–{base['DimPD'].max():.4f} "
          f"({base['DimPD'].notna().sum()} válidos)")
    return base[['yyyymm','DimPD']]

# ── IPD PROYECCIÓN ────────────────────────────────────────────
def proyectar_ipd(df):
    ok  = df[df['IPD'].notna() & (df['IPD'] > 1)]  # solo valores reales >1M
    nas = df[~df.index.isin(ok.index)]
    df['IPD_proy'] = False
    if ok.empty: return df
    ultima_yyyymm = ok['yyyymm'].max()
    nas_future = nas[nas['yyyymm'] > ultima_yyyymm]
    if nas_future.empty: return df
    print(f"  IPD datos reales hasta {ultima_yyyymm} | "
          f"proyectando {len(nas_future)} meses")
    t_base  = ok['yyyymm'].min()
    t_known = (ok['yyyymm'] - t_base).values
    log_y   = np.log(ok['IPD'].values + 1)
    coef    = np.polyfit(t_known, log_y, 1)
    t_proj  = (nas_future['yyyymm'] - t_base).values
    y_proj  = np.exp(np.polyval(coef, t_proj)) - 1
    # Anclar 2025
    nas25 = nas_future[nas_future['yyyymm'] // 100 == 2025]
    if len(nas25) > 0:
        tot25 = ok[ok['yyyymm'] // 100 == 2025]['IPD'].sum()
        rest  = max(0, IPD_OFICIAL_2025 - tot25)
        if rest > 0:
            pw = np.linspace(1.0,1.08,len(nas25)); pw /= pw.sum()
            for i, idx in enumerate(nas25.index):
                pi = list(nas_future.index).index(idx)
                y_proj[pi] = (rest*pw)[i]
    df = df.copy()
    df.loc[nas_future.index,'IPD'] = y_proj
    df.loc[nas_future.index,'IPD_proy'] = True
    return df

# ── DRIFT ─────────────────────────────────────────────────────
def calcular_drift(df):
    def mm(s, inv=False):
        s2=pd.Series(s,dtype=float).ffill().bfill()
        mn,mx=s2.min(),s2.max()
        if mx==mn: return pd.Series([50.]*len(s2))
        n=(s2-mn)/(mx-mn)*100
        return 100-n if inv else n
    DPS=mm(df['DimPD']); DVS=mm(df['IPD']); MCS=mm(df['Dolarizacion'],inv=True)
    T=0.40*DPS+0.40*MCS+0.20*DVS
    CRS=mm(df['EMBI']); DSS=mm(df['DXY']); PIS=mm(df['IPC'].abs())
    IGS=mm(df['BVL'],inv=True)
    R=0.35*CRS+0.25*DSS+0.20*PIS+0.20*IGS
    DRIFT=T-R
    def reg(v):
        if pd.isna(v): return 'Sin datos'
        if v>=20: return 'Confianza Sólida'
        if v>=5:  return 'Expansión'
        if v>=-5: return 'Equilibrio'
        if v>=-20:return 'Tensión Alta'
        return 'Crisis Severa'
    df=df.copy()
    df['DPS']=DPS.values; df['MCS']=MCS.values; df['DVS']=DVS.values
    df['CRS']=CRS.values; df['DSS']=DSS.values; df['PIS']=PIS.values
    df['IGS']=IGS.values; df['DRIFT_T']=T.values; df['DRIFT_R']=R.values
    df['DRIFT']=DRIFT.values; df['Régimen']=DRIFT.apply(reg)
    df['Año']=df['yyyymm']//100
    return df

# ── MAIN ──────────────────────────────────────────────────────
def run():
    print("="*60)
    print(f"DRIFT Pipeline v5.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"FRED key: {'SÍ' if FRED_KEY else 'NO'}")
    print("="*60); sys.stdout.flush()

    hoy=date.today()
    fin_mo=hoy.month-1 if hoy.month>1 else 12
    fin_yr=hoy.year   if hoy.month>1 else hoy.year-1
    fin=f"{fin_yr}-{fin_mo}"
    print(f"Período: {INICIO} → {fin}\n")

    # 1. Series principales (una por una — sin confusión de columnas)
    print("[1] Series BCRP principales (una por una)...")
    MAIN = {
        'Dolarizacion': ('PN00531MM', (20, 40)),    # % dolarización
        'IPC':          ('PN01271PM', (-5, 5)),      # var% mensual
        'BVL':          ('PN01158MM', (0, 10000)),   # millones S/
        'IPD':          ('PN39971SM', (0, 5000)),    # M operaciones
        'EMBI':         ('PN01129XM', (50, 500)),    # bps
        'Adopcion':     ('PN09416SM', (0, 500000)),  # millones S/
    }
    # Base con yyyymm
    df = None
    for nombre, (codigo, rango_esperado) in MAIN.items():
        ds = fetch_serie(codigo, INICIO, fin)
        if ds.empty:
            print(f"  ADVERTENCIA: {codigo} sin datos")
            continue
        ds = ds.rename(columns={codigo: nombre})
        # Verificar rango
        vals = ds[nombre].dropna()
        if len(vals) > 0:
            ok = rango_esperado[0] <= vals.median() <= rango_esperado[1]
            print(f"  {nombre}: mediana={vals.median():.2f} "
                  f"{'✓' if ok else '✗ RANGO INESPERADO'}")
        if df is None:
            df = ds[['yyyymm','periodo',nombre]]
        else:
            df = df.merge(ds[['yyyymm',nombre]], on='yyyymm', how='outer')
    if df is None or df.empty:
        print("CRÍTICO: Sin datos"); sys.exit(1)
    df = df.sort_values('yyyymm').reset_index(drop=True)
    df['fecha'] = df['yyyymm'].apply(
        lambda x: pd.Timestamp(x//100, x%100, 1))
    print(f"  Dataset base: {len(df)} períodos")
    sys.stdout.flush()

    # 2. DimPD
    print("\n[2] DimPD (10 series, una por una)...")
    df_dimpd = fetch_dimpd(INICIO, fin)
    if not df_dimpd.empty:
        df = df.merge(df_dimpd, on='yyyymm', how='left')
        print(f"  DimPD: {df['DimPD'].notna().sum()}/{len(df)} válidos")
    else:
        df['DimPD'] = np.nan
    sys.stdout.flush()

    # 3. DXY
    print("\n[3] DXY desde FRED API...")
    df_dxy = fetch_dxy()
    if not df_dxy.empty:
        df = df.merge(df_dxy, on='yyyymm', how='left')
        df['DXY'] = df['DXY'].ffill()
        print(f"  DXY: {df['DXY'].notna().sum()}/{len(df)} válidos")
    else:
        if 'DXY' not in df.columns: df['DXY'] = np.nan
    sys.stdout.flush()

    # 4. IPD
    print("\n[4] IPD proyección...")
    df = proyectar_ipd(df)
    sys.stdout.flush()

    # 5. DRIFT
    print("\n[5] Calculando DRIFT...")
    for col in ['DimPD','Dolarizacion','IPD','EMBI','DXY','IPC','BVL']:
        if col not in df.columns: continue
        n = df[col].notna().sum()
        v = df[col].dropna()
        print(f"  {col}: {n}/{len(df)} | "
              f"último={v.iloc[-1]:.2f}" if len(v) else
              f"  {col}: 0/{len(df)}")
    df = calcular_drift(df)
    df_ok = df[df['DRIFT'].notna()]
    if len(df_ok) > 0:
        last = df_ok.iloc[-1]
        fav=0
        for i in range(len(df_ok)-1,-1,-1):
            if df_ok['DRIFT'].iloc[i]>=5: fav+=1
            else: break
        corr=np.corrcoef(df_ok['DRIFT'],df_ok['Dolarizacion'])[0,1]
        print(f"\n  N={len(df)} | DRIFT válidos={len(df_ok)}")
        print(f"  Último: {last['periodo']} DRIFT={last['DRIFT']:+.1f} "
              f"| {last['Régimen']}")
        print(f"  Dolarización={last['Dolarizacion']:.2f}% "
              f"DimPD={last['DimPD']*100:.2f}%")
        print(f"  r(DRIFT,Dol)={corr:.3f} | Meses favorables={fav}")
    sys.stdout.flush()

    # 6. Outputs
    print("\n[6] Guardando outputs...")
    cols=['periodo','fecha','Año','DimPD','Dolarizacion','IPD','IPD_proy',
          'EMBI','DXY','IPC','BVL','Adopcion',
          'DPS','MCS','DVS','CRS','DSS','PIS','IGS',
          'DRIFT_T','DRIFT_R','DRIFT','Régimen']
    cols=[c for c in cols if c in df.columns]
    df[cols].to_csv(DATA_DIR/"drift_serie.csv",index=False,encoding='utf-8')
    print(f"  drift_serie.csv — {len(df)} filas")

    def s(v,d=1): return round(float(v),d) if pd.notna(v) else None
    records=[{
        'periodo': str(r.periodo) if hasattr(r,'periodo') else '',
        'year':    int(r.Año) if pd.notna(r.Año) else None,
        'drift':   s(getattr(r,'DRIFT',None)),
        'drift_t': s(getattr(r,'DRIFT_T',None)),
        'drift_r': s(getattr(r,'DRIFT_R',None)),
        'dps':     s(getattr(r,'DPS',None)),
        'mcs':     s(getattr(r,'MCS',None)),
        'dvs':     s(getattr(r,'DVS',None)),
        'crs':     s(getattr(r,'CRS',None)),
        'dss':     s(getattr(r,'DSS',None)),
        'pis':     s(getattr(r,'PIS',None)),
        'igs':     s(getattr(r,'IGS',None)),
        'dim_pd':  s(getattr(r,'DimPD',None),4),
        'dol':     s(getattr(r,'Dolarizacion',None),2),
        'embi':    s(getattr(r,'EMBI',None),0),
        'dxy':     s(getattr(r,'DXY',None),1),
        'ipd':     s(getattr(r,'IPD',None),1),
        'ipd_proy':bool(getattr(r,'IPD_proy',False)),
        'regime':  str(getattr(r,'Régimen','')),
    } for r in df.itertuples()]

    last_r = df_ok.iloc[-1] if len(df_ok)>0 else df.iloc[-1]
    meta={
        'generado':       datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
        'N':              len(df),
        'inicio':         str(df['periodo'].iloc[0]) if 'periodo' in df else '',
        'fin':            str(df['periodo'].iloc[-1]) if 'periodo' in df else '',
        'ultimo_drift':   s(getattr(last_r,'DRIFT',None)),
        'ultimo_regimen': str(getattr(last_r,'Régimen','')),
        'meses_favorables': int(fav) if len(df_ok)>0 else 0,
        'corr_drift_dol': round(float(corr),3) if len(df_ok)>10 else None,
    }
    with open(DATA_DIR/"drift_latest.json",'w',encoding='utf-8') as f:
        json.dump({'meta':meta,'series':records},
                  f,separators=(',',':'),ensure_ascii=False)
    print(f"  drift_latest.json — {len(records)} registros")
    print("\n[✓] Pipeline v5.0 completado.")

if __name__ == '__main__':
    run()
