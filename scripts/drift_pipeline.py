"""
DRIFT Pipeline v3.0 — Digital Risk & Financial Trust Index
==========================================================
Autor : Ceyner D. Llontop Herrera | ORCID: 0009-0007-6058-7711

Estrategia: llamadas en lote (rápido) + mapeo por config.series (correcto)
"""

import requests, pandas as pd, numpy as np, json, sys
from datetime import datetime, date
from pathlib import Path
from io import StringIO

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

INICIO    = "2018-1"
BCRP_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
IPD_OFICIAL_2025 = 15_628.0  # M ops totales 2025 (Reporte BCRP mar-2026)

MESES = {'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,
         'Jul':7,'Ago':8,'Sep':9,'Oct':10,'Nov':11,'Dic':12}

def a_fecha(p):
    try: return pd.Timestamp(2000+int(p[3:]), MESES[p[:3]], 1)
    except: return pd.NaT

# ── FETCH BCRP (lote, mapeo por config) ──────────────────────
def fetch_bcrp(codigos, ini, fin):
    """
    Extrae hasta 10 series en una sola llamada.
    Usa config.series de la respuesta para mapear columnas — no el índice.
    """
    url = f"{BCRP_BASE}/{'-'.join(codigos[:10])}/json/{ini}/{fin}/esp"
    print(f"  GET {url[-80:]}")
    try:
        r = requests.get(url, timeout=45,
                        headers={'User-Agent':'DRIFT/3.0'})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ERROR: {e}"); return pd.DataFrame()

    if 'periods' not in data:
        print("  Sin datos"); return pd.DataFrame()

    # Mapeo correcto: usar shortname del config, no el índice
    cfg_series = data.get('config', {}).get('series', [])
    # shortname puede ser el código o una descripción — buscamos el código
    col_names = []
    for i, s in enumerate(cfg_series):
        sn = s.get('shortname', '').strip()
        # Si el shortname tiene formato de código BCRP (ej: PN00531MM), usarlo
        # Si no, usar el código que pedimos por índice
        if len(sn) >= 6 and sn[:2].isalpha():
            col_names.append(sn)
        elif i < len(codigos):
            col_names.append(codigos[i])
        else:
            col_names.append(f'col_{i}')

    print(f"  Columnas detectadas: {col_names}")

    records = []
    for period in data['periods']:
        row = {'periodo': period['name'], 'fecha': a_fecha(period['name'])}
        for i, val in enumerate(period.get('values', [])):
            if i < len(col_names):
                try: row[col_names[i]] = float(val) if val not in ['n.d.','',None] else np.nan
                except: row[col_names[i]] = np.nan
        records.append(row)

    df = pd.DataFrame(records).sort_values('fecha').reset_index(drop=True)
    print(f"  OK: {len(df)} períodos x {len(col_names)} series")
    return df

# ── FETCH FRED ────────────────────────────────────────────────
def fetch_dxy():
    print("  GET FRED TWEXBGSMTH...")
    for url in [
        f"{FRED_BASE}?id=TWEXBGSMTH",
        f"{FRED_BASE}?id=TWEXBGSMTH&file_type=csv",
    ]:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), header=0)
            print(f"  Columnas FRED: {list(df.columns)}")
            # Primera col = fecha, segunda = valor — sin importar el nombre
            df.columns = ['fecha_str', 'DXY']
            df['fecha'] = pd.to_datetime(df['fecha_str'], errors='coerce')
            df['DXY']   = pd.to_numeric(df['DXY'], errors='coerce')
            df = df.dropna(subset=['fecha','DXY'])
            df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
            df = df.groupby('fecha')['DXY'].mean().reset_index()
            print(f"  DXY OK: {len(df)} períodos, último={df['DXY'].iloc[-1]:.1f}")
            return df
        except Exception as e:
            print(f"  Intento fallido: {e}")
    return pd.DataFrame()

# ── DIMPD ─────────────────────────────────────────────────────
DENOM = ['PN09398SM','PN09399SM','PN09400SM','PN09403SM','PN09406SM',
         'PN09408SM','PN09409SM','PN09411SM','PN09414SM','PN09416SM']

def calcular_dimpd(df_denom):
    """DimPD = PN09416SM / suma(10 series). Proyecta series incompletas."""
    for col in DENOM:
        if col not in df_denom.columns: continue
        nas = df_denom[col].isna().sum()
        if nas == 0: continue
        ok = df_denom[col].dropna()
        if len(ok) < 4: df_denom[col] = df_denom[col].ffill(); continue
        t  = np.arange(len(ok))
        cf = np.polyfit(t, ok.values, 1)
        tp = np.arange(len(ok), len(ok)+nas)
        pv = np.maximum(np.polyval(cf,tp), ok.iloc[-1]*0.85)
        df_denom.loc[df_denom[col].isna(), col] = pv
        print(f"  {col}: {nas} meses proyectados")

    cols_ok = [c for c in DENOM if c in df_denom.columns]
    df_denom['total'] = df_denom[cols_ok].sum(axis=1)
    dimpd = df_denom['PN09416SM'] / df_denom['total']
    print(f"  DimPD: rango {dimpd.min():.4f}–{dimpd.max():.4f}")
    return dimpd

# ── IPD PROYECCIÓN ────────────────────────────────────────────
def proyectar_ipd(df):
    ok  = df[df['IPD'].notna()]
    nas = df[df['IPD'].isna()]
    if nas.empty:
        df['IPD_proy'] = False; return df

    ultima = ok['fecha'].max()
    print(f"  IPD datos hasta {ultima.strftime('%b-%Y')}, proyectando {len(nas)} meses")

    t_base  = ok['fecha'].min()
    t_known = (ok['fecha'] - t_base).dt.days.values
    log_y   = np.log(ok['IPD'].values + 1)
    coef    = np.polyfit(t_known, log_y, 1)
    t_proj  = (nas['fecha'] - t_base).dt.days.values
    y_proj  = np.exp(np.polyval(coef, t_proj)) - 1

    # Anclar 2025 al total oficial
    nas25 = nas[nas['fecha'].dt.year == 2025]
    if len(nas25) > 0:
        tot_conocido = ok[ok['fecha'].dt.year==2025]['IPD'].sum()
        restante = max(0, IPD_OFICIAL_2025 - tot_conocido)
        if restante > 0:
            pesos = np.linspace(1.0, 1.08, len(nas25))
            pesos /= pesos.sum()
            vals25 = restante * pesos
            for i, idx in enumerate(nas25.index):
                pi = list(nas.index).index(idx)
                y_proj[pi] = vals25[i]

    df = df.copy()
    df.loc[nas.index, 'IPD'] = y_proj
    df['IPD_proy'] = False
    df.loc[df['fecha'] > ultima, 'IPD_proy'] = True
    return df

# ── DRIFT ─────────────────────────────────────────────────────
def calcular_drift(df):
    def mm(s, inv=False):
        s2 = pd.Series(s, dtype=float).ffill().bfill()
        mn, mx = s2.min(), s2.max()
        if mx==mn: return pd.Series([50.]*len(s2))
        n = (s2-mn)/(mx-mn)*100
        return 100-n if inv else n

    DPS = mm(df['DimPD'])
    DVS = mm(df['IPD'])
    MCS = mm(df['Dolarizacion'], inv=True)
    T   = 0.40*DPS + 0.40*MCS + 0.20*DVS

    CRS = mm(df['EMBI'])
    DSS = mm(df['DXY'])
    PIS = mm(df['IPC'].abs())
    IGS = mm(df['BVL'], inv=True)
    R   = 0.35*CRS + 0.25*DSS + 0.20*PIS + 0.20*IGS

    DRIFT = T - R

    def reg(v):
        if pd.isna(v): return 'Sin datos'
        if v>=20: return 'Confianza Sólida'
        if v>=5:  return 'Expansión'
        if v>=-5: return 'Equilibrio'
        if v>=-20:return 'Tensión Alta'
        return 'Crisis Severa'

    df = df.copy()
    df['DPS']=DPS.values; df['MCS']=MCS.values; df['DVS']=DVS.values
    df['CRS']=CRS.values; df['DSS']=DSS.values; df['PIS']=PIS.values
    df['IGS']=IGS.values; df['DRIFT_T']=T.values; df['DRIFT_R']=R.values
    df['DRIFT']=DRIFT.values; df['Régimen']=DRIFT.apply(reg)
    df['Año']=df['fecha'].dt.year
    return df

# ── MAIN ─────────────────────────────────────────────────────
def run():
    print("="*60)
    print(f"DRIFT Pipeline v3.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    sys.stdout.flush()

    hoy = date.today()
    fin_mo = hoy.month-1 if hoy.month>1 else 12
    fin_yr = hoy.year   if hoy.month>1 else hoy.year-1
    fin = f"{fin_yr}-{fin_mo}"
    print(f"Período: {INICIO} → {fin}\n")

    # 1. Series principales (1 llamada, 6 series)
    print("[1] Series BCRP principales...")
    MAIN = {'Dolarizacion':'PN00531MM','IPC':'PN01271PM','BVL':'PN01158MM',
            'IPD':'PN39971SM','EMBI':'PN01129XM','Adopcion':'PN09416SM'}
    df_raw = fetch_bcrp(list(MAIN.values()), INICIO, fin)
    if df_raw.empty:
        print("CRÍTICO: Sin datos BCRP"); sys.exit(1)

    # Renombrar usando los códigos que pedimos (mapeo explícito)
    # Primero identificar qué columnas llegaron con código correcto
    df = df_raw[['periodo','fecha']].copy()
    for nombre, codigo in MAIN.items():
        if codigo in df_raw.columns:
            df[nombre] = df_raw[codigo]
        else:
            # Buscar por nombre alternativo si el API devolvió shortname diferente
            print(f"  AVISO: {codigo} no encontrado directamente")
            df[nombre] = np.nan

    # Verificar dolarización
    dol = df['Dolarizacion'].dropna()
    print(f"  Dolarización: {dol.min():.2f}%–{dol.max():.2f}% "
          f"(último: {dol.iloc[-1]:.2f}%)" if len(dol)>0 else "  Sin dato dolarización")
    sys.stdout.flush()

    # 2. Denominador DimPD (1 llamada, 10 series)
    print("\n[2] Denominador DimPD (10 series)...")
    df_denom = fetch_bcrp(DENOM, INICIO, fin)
    if not df_denom.empty:
        # Renombrar columnas usando los códigos que pedimos
        df_denom_clean = df_denom[['fecha']].copy()
        for cod in DENOM:
            if cod in df_denom.columns:
                df_denom_clean[cod] = df_denom[cod]
        df_denom_clean['DimPD'] = calcular_dimpd(df_denom_clean)
        df = df.merge(df_denom_clean[['fecha','DimPD']], on='fecha', how='left')
    else:
        df['DimPD'] = np.nan
    sys.stdout.flush()

    # 3. DXY
    print("\n[3] DXY (FRED)...")
    df_dxy = fetch_dxy()
    if not df_dxy.empty:
        df = df.merge(df_dxy, on='fecha', how='left')
        df['DXY'] = df['DXY'].ffill()
    else:
        df['DXY'] = np.nan
        print("  DXY no disponible — componente DSS será NaN")
    sys.stdout.flush()

    # 4. IPD
    print("\n[4] IPD proyección...")
    df = proyectar_ipd(df)
    sys.stdout.flush()

    # 5. DRIFT
    print("\n[5] Calculando DRIFT...")
    for col in ['DimPD','Dolarizacion','IPD','EMBI','DXY','IPC','BVL']:
        n = df[col].notna().sum() if col in df.columns else 0
        print(f"  {col}: {n}/{len(df)} válidos")
    df = calcular_drift(df)

    df_ok = df[df['DRIFT'].notna()]
    if len(df_ok) > 0:
        last = df_ok.iloc[-1]
        fav  = sum(1 for _ in
                   [i for i in range(len(df_ok)-1,-1,-1)
                    if df_ok['DRIFT'].iloc[i]>=5][::-1]
                   if True)
        # cuenta meses consecutivos desde el final
        fav = 0
        for i in range(len(df_ok)-1,-1,-1):
            if df_ok['DRIFT'].iloc[i]>=5: fav+=1
            else: break
        corr = np.corrcoef(df_ok['DRIFT'],df_ok['Dolarizacion'])[0,1]
        print(f"\n  Último: {last['periodo']} | "
              f"T={last['DRIFT_T']:.1f} R={last['DRIFT_R']:.1f} "
              f"DRIFT={last['DRIFT']:+.1f} | {last['Régimen']}")
        print(f"  Dolarización: {last['Dolarizacion']:.2f}% | "
              f"DimPD: {last['DimPD']*100:.2f}%")
        print(f"  Correlación: r={corr:.3f} | Meses favorables: {fav}")

    # 6. Outputs
    print("\n[6] Guardando...")
    cols = ['periodo','fecha','Año','DimPD','Dolarizacion','IPD','IPD_proy',
            'EMBI','DXY','IPC','BVL','Adopcion',
            'DPS','MCS','DVS','CRS','DSS','PIS','IGS',
            'DRIFT_T','DRIFT_R','DRIFT','Régimen']
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(DATA_DIR/"drift_serie.csv", index=False, encoding='utf-8')
    print(f"  drift_serie.csv — {len(df)} filas")

    def s(v,d=1): return round(float(v),d) if pd.notna(v) else None
    records = [{'periodo':str(r.periodo),'year':int(r.Año) if pd.notna(r.Año) else None,
                'drift':s(r.DRIFT),'drift_t':s(r.DRIFT_T),'drift_r':s(r.DRIFT_R),
                'dps':s(r.DPS),'mcs':s(r.MCS),'dvs':s(r.DVS),
                'crs':s(r.CRS),'dss':s(r.DSS),'pis':s(r.PIS),'igs':s(r.IGS),
                'dim_pd':s(getattr(r,'DimPD',None),4),
                'dol':s(r.Dolarizacion,2),'embi':s(r.EMBI,0),
                'dxy':s(r.DXY,1),'ipd':s(r.IPD,1),
                'ipd_proy':bool(getattr(r,'IPD_proy',False)),
                'regime':str(r.Régimen)}
               for r in df.itertuples()]
    last_r = df_ok.iloc[-1] if len(df_ok)>0 else df.iloc[-1]
    meta = {'generado':datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
            'N':len(df),'inicio':str(df['periodo'].iloc[0]),
            'fin':str(df['periodo'].iloc[-1]),
            'ultimo_drift':s(last_r.DRIFT),
            'ultimo_regimen':str(last_r.Régimen)}
    with open(DATA_DIR/"drift_latest.json",'w',encoding='utf-8') as f:
        json.dump({'meta':meta,'series':records},f,
                  separators=(',',':'),ensure_ascii=False)
    print(f"  drift_latest.json — {len(records)} registros")
    print("\n[✓] Pipeline v3.0 completado.")

if __name__ == '__main__':
    run()
