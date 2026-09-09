"""
DRIFT Pipeline v4.1 — Digital Risk & Financial Trust Index
==========================================================
Autor : Ceyner D. Llontop Herrera | ORCID: 0009-0007-6058-7711

Fix v4.1:
  - a_fecha maneja "Ene.2018" y "Ene18"
  - Merge por 'periodo' (string) — sin NaT cartesiano
  - FRED API oficial con clave (FRED_API_KEY en GitHub Secrets)
  - Mapeo de columnas BCRP por código explícito
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
MESES_INV = {v:k for k,v in MESES.items()}

def a_fecha(p):
    """Parsea 'Ene18', 'Ene.2018' → Timestamp."""
    try:
        p = str(p).strip()
        mes_str = p[:3].capitalize()
        if mes_str not in MESES:
            return pd.NaT
        mes    = MESES[mes_str]
        yr_str = p[3:].lstrip('.').strip()
        yr     = int(yr_str)
        if yr < 100:
            yr = 2000 + yr
        return pd.Timestamp(yr, mes, 1)
    except:
        return pd.NaT

def fecha_a_periodo(f):
    """Timestamp → 'Ene18' (para merge con datos BCRP)."""
    try:
        return f"{MESES_INV[f.month]}{str(f.year)[2:]}"
    except:
        return None

# ── FETCH BCRP ────────────────────────────────────────────────
def fetch_bcrp(codigos: list, ini: str, fin: str) -> pd.DataFrame:
    url = f"{BCRP_BASE}/{'-'.join(codigos[:10])}/json/{ini}/{fin}/esp"
    print(f"  GET {url[-75:]}")
    try:
        r = requests.get(url, timeout=45, headers={'User-Agent':'DRIFT/4.1'})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ERROR BCRP: {e}"); return pd.DataFrame()

    if 'periods' not in data:
        print("  Sin datos"); return pd.DataFrame()

    # Mapear columnas desde config.series
    cfg = data.get('config', {}).get('series', [])
    col_names = []
    for i, s in enumerate(cfg):
        sn = s.get('shortname', '').strip()
        # Código BCRP: empieza con 2 letras, luego dígitos (ej: PN00531MM)
        if len(sn) >= 8 and sn[:2].isalpha() and sn[2:8].isdigit():
            col_names.append(sn)
        elif i < len(codigos):
            col_names.append(codigos[i])
        else:
            col_names.append(f'col_{i}')

    print(f"  Config ({len(cfg)} series): {col_names}")

    records = []
    for period in data['periods']:
        nombre = period['name']
        row = {'periodo': nombre, 'fecha': a_fecha(nombre)}
        for i, val in enumerate(period.get('values', [])):
            if i < len(col_names):
                try:
                    row[col_names[i]] = (float(val)
                                         if val not in ['n.d.','',None]
                                         else np.nan)
                except:
                    row[col_names[i]] = np.nan
        records.append(row)

    df = pd.DataFrame(records)
    nat = df['fecha'].isna().sum()
    print(f"  {len(df)} períodos | "
          f"ejemplo: '{df['periodo'].iloc[0]}' → {df['fecha'].iloc[0]} "
          f"| NaT: {nat}")
    if nat == len(df):
        print("  ADVERTENCIA: Todas las fechas son NaT")
    return df.sort_values('fecha').reset_index(drop=True)

# ── FETCH FRED API OFICIAL ────────────────────────────────────
def fetch_dxy() -> pd.DataFrame:
    if not FRED_KEY:
        print("  FRED_API_KEY no configurada — DXY omitido")
        return pd.DataFrame()
    print(f"  GET FRED TWEXBGSMTH (API key: ...{FRED_KEY[-6:]})")
    try:
        params = {
            'series_id':         'TWEXBGSMTH',
            'api_key':           FRED_KEY,
            'file_type':         'json',
            'observation_start': '2018-01-01',
            'frequency':         'm',  # mensual
        }
        r = requests.get(FRED_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        obs = data.get('observations', [])
        if not obs:
            print("  Sin observaciones FRED"); return pd.DataFrame()
        records = []
        for o in obs:
            try:
                val = float(o['value'])
                fecha = pd.Timestamp(o['date'])
                records.append({'fecha': fecha, 'DXY': val})
            except:
                pass
        df = pd.DataFrame(records)
        df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
        df = df.groupby('fecha')['DXY'].mean().reset_index()
        # Crear clave periodo para merge ("Ene18", "Feb18", ...)
        df['periodo'] = df['fecha'].apply(fecha_a_periodo)
        print(f"  DXY OK: {len(df)} meses | "
              f"último: {df['fecha'].iloc[-1].strftime('%b-%Y')} "
              f"= {df['DXY'].iloc[-1]:.1f}")
        return df[['periodo','DXY']]
    except Exception as e:
        print(f"  ERROR FRED: {e}"); return pd.DataFrame()

# ── DIMPD ─────────────────────────────────────────────────────
DENOM = ['PN09398SM','PN09399SM','PN09400SM','PN09403SM','PN09406SM',
         'PN09408SM','PN09409SM','PN09411SM','PN09414SM','PN09416SM']

def calcular_dimpd(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    for col in DENOM:
        if col not in df.columns: continue
        nas = df[col].isna().sum()
        if nas == 0: continue
        ok = df[col].dropna()
        if len(ok) < 4:
            df[col] = df[col].ffill(); continue
        t  = np.arange(len(ok))
        cf = np.polyfit(t, ok.values, 1)
        tp = np.arange(len(ok), len(ok)+nas)
        pv = np.maximum(np.polyval(cf,tp), ok.iloc[-1]*0.85)
        df.loc[df[col].isna(), col] = pv
        print(f"    {col}: {nas} meses proyectados")
    cols_ok = [c for c in DENOM if c in df.columns]
    total   = df[cols_ok].sum(axis=1)
    dimpd   = df['PN09416SM'] / total
    print(f"  DimPD: {dimpd.min():.4f}–{dimpd.max():.4f} "
          f"({dimpd.notna().sum()} válidos)")
    return dimpd

# ── IPD PROYECCIÓN ────────────────────────────────────────────
def proyectar_ipd(df: pd.DataFrame) -> pd.DataFrame:
    ok   = df[df['IPD'].notna()]
    nas  = df[df['IPD'].isna()]
    df['IPD_proy'] = False
    if nas.empty:
        return df
    ultima = ok['fecha'].max()
    print(f"  IPD hasta {ultima.strftime('%b-%Y')}, "
          f"proyectando {len(nas)} meses")
    t_base  = ok['fecha'].min()
    t_known = (ok['fecha'] - t_base).dt.days.values
    log_y   = np.log(ok['IPD'].values + 1)
    coef    = np.polyfit(t_known, log_y, 1)
    t_proj  = (nas['fecha'] - t_base).dt.days.values
    y_proj  = np.exp(np.polyval(coef, t_proj)) - 1
    # Anclar 2025 al total oficial
    nas25 = nas[nas['fecha'].dt.year == 2025]
    if len(nas25) > 0:
        tot25 = ok[ok['fecha'].dt.year==2025]['IPD'].sum()
        rest  = max(0, IPD_OFICIAL_2025 - tot25)
        if rest > 0:
            pw = np.linspace(1.0,1.08,len(nas25)); pw /= pw.sum()
            for i, idx in enumerate(nas25.index):
                pi = list(nas.index).index(idx)
                y_proj[pi] = (rest*pw)[i]
    df = df.copy()
    df.loc[nas.index, 'IPD'] = y_proj
    df.loc[df['fecha'] > ultima, 'IPD_proy'] = True
    return df

# ── DRIFT ─────────────────────────────────────────────────────
def calcular_drift(df: pd.DataFrame) -> pd.DataFrame:
    def mm(s, inv=False):
        s2 = pd.Series(s, dtype=float).ffill().bfill()
        mn, mx = s2.min(), s2.max()
        if mx == mn: return pd.Series([50.]*len(s2))
        n = (s2-mn)/(mx-mn)*100
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
    df['Año']=df['fecha'].dt.year
    return df

# ── MAIN ──────────────────────────────────────────────────────
def run():
    print("="*60)
    print(f"DRIFT Pipeline v4.1 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"FRED_API_KEY configurada: {'SÍ' if FRED_KEY else 'NO'}")
    print("="*60); sys.stdout.flush()

    hoy=date.today()
    fin_mo=hoy.month-1 if hoy.month>1 else 12
    fin_yr=hoy.year   if hoy.month>1 else hoy.year-1
    fin=f"{fin_yr}-{fin_mo}"
    print(f"Período: {INICIO} → {fin}\n")

    # 1. Series principales
    print("[1] Series BCRP principales...")
    MAIN = {'Dolarizacion':'PN00531MM','IPC':'PN01271PM','BVL':'PN01158MM',
            'IPD':'PN39971SM','EMBI':'PN01129XM','Adopcion':'PN09416SM'}
    df_raw = fetch_bcrp(list(MAIN.values()), INICIO, fin)
    if df_raw is None or df_raw.empty:
        print("CRÍTICO: Sin datos BCRP"); sys.exit(1)

    df = df_raw[['periodo','fecha']].copy()
    for nombre, codigo in MAIN.items():
        if codigo in df_raw.columns:
            df[nombre] = df_raw[codigo]
            vals = df_raw[codigo].dropna()
            print(f"  ✓ {nombre} ({codigo}): "
                  f"{vals.min():.2f} – {vals.max():.2f} "
                  f"| último={vals.iloc[-1]:.2f}")
        else:
            print(f"  ✗ {codigo} no encontrado → NaN")
            df[nombre] = np.nan
    sys.stdout.flush()

    # 2. DimPD
    print("\n[2] DimPD (10 series denominador)...")
    df_denom = fetch_bcrp(DENOM, INICIO, fin)
    if not df_denom.empty:
        df_denom['DimPD'] = calcular_dimpd(df_denom)
        # Merge por 'periodo' — sin NaT cartesiano
        df = df.merge(df_denom[['periodo','DimPD']],
                      on='periodo', how='left')
        print(f"  DimPD: {df['DimPD'].notna().sum()}/{len(df)} válidos")
    else:
        df['DimPD'] = np.nan
    sys.stdout.flush()

    # 3. DXY (FRED API oficial)
    print("\n[3] DXY desde FRED API...")
    df_dxy = fetch_dxy()
    if not df_dxy.empty:
        df = df.merge(df_dxy, on='periodo', how='left')
        df['DXY'] = df['DXY'].ffill()
        print(f"  DXY: {df['DXY'].notna().sum()}/{len(df)} válidos")
    else:
        if 'DXY' not in df.columns:
            df['DXY'] = np.nan
    sys.stdout.flush()

    # 4. IPD proyección
    print("\n[4] IPD proyección...")
    df = proyectar_ipd(df)
    sys.stdout.flush()

    # 5. DRIFT
    print("\n[5] Calculando DRIFT...")
    for col in ['DimPD','Dolarizacion','IPD','EMBI','DXY','IPC','BVL']:
        n = df[col].notna().sum() if col in df.columns else 0
        v = df[col].dropna()
        ult = f"{v.iloc[-1]:.2f}" if len(v)>0 else "N/A"
        print(f"  {col}: {n}/{len(df)} válidos | último={ult}")

    df = calcular_drift(df)
    df_ok = df[df['DRIFT'].notna()]

    if len(df_ok) > 0:
        last = df_ok.iloc[-1]
        fav = 0
        for i in range(len(df_ok)-1,-1,-1):
            if df_ok['DRIFT'].iloc[i] >= 5: fav += 1
            else: break
        corr = (np.corrcoef(df_ok['DRIFT'],df_ok['Dolarizacion'])[0,1]
                if len(df_ok)>10 else np.nan)
        print(f"\n  N={len(df)} | Períodos con DRIFT: {len(df_ok)}")
        print(f"  Último: {last['periodo']}")
        print(f"  DRIFT-T={last['DRIFT_T']:.1f} "
              f"DRIFT-R={last['DRIFT_R']:.1f} "
              f"DRIFT={last['DRIFT']:+.1f} | {last['Régimen']}")
        print(f"  Dolarización={last['Dolarizacion']:.2f}% | "
              f"DimPD={last['DimPD']*100:.2f}%")
        print(f"  r(DRIFT,Dol)={corr:.3f} | Meses favorables={fav}")
    sys.stdout.flush()

    # 6. Guardar
    print("\n[6] Guardando outputs...")
    cols = ['periodo','fecha','Año','DimPD','Dolarizacion','IPD','IPD_proy',
            'EMBI','DXY','IPC','BVL','Adopcion',
            'DPS','MCS','DVS','CRS','DSS','PIS','IGS',
            'DRIFT_T','DRIFT_R','DRIFT','Régimen']
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(DATA_DIR/"drift_serie.csv", index=False, encoding='utf-8')
    print(f"  drift_serie.csv — {len(df)} filas, {len(cols)} columnas")

    def s(v,d=1):
        return round(float(v),d) if pd.notna(v) else None

    records = []
    for r in df.itertuples():
        records.append({
            'periodo':  str(r.periodo),
            'year':     int(r.Año) if pd.notna(r.Año) else None,
            'drift':    s(getattr(r,'DRIFT',None)),
            'drift_t':  s(getattr(r,'DRIFT_T',None)),
            'drift_r':  s(getattr(r,'DRIFT_R',None)),
            'dps':      s(getattr(r,'DPS',None)),
            'mcs':      s(getattr(r,'MCS',None)),
            'dvs':      s(getattr(r,'DVS',None)),
            'crs':      s(getattr(r,'CRS',None)),
            'dss':      s(getattr(r,'DSS',None)),
            'pis':      s(getattr(r,'PIS',None)),
            'igs':      s(getattr(r,'IGS',None)),
            'dim_pd':   s(getattr(r,'DimPD',None),4),
            'dol':      s(getattr(r,'Dolarizacion',None),2),
            'embi':     s(getattr(r,'EMBI',None),0),
            'dxy':      s(getattr(r,'DXY',None),1),
            'ipd':      s(getattr(r,'IPD',None),1),
            'ipd_proy': bool(getattr(r,'IPD_proy',False)),
            'regime':   str(getattr(r,'Régimen','')),
        })

    last_r = df_ok.iloc[-1] if len(df_ok)>0 else df.iloc[-1]
    meta = {
        'generado':       datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
        'N':              len(df),
        'inicio':         str(df['periodo'].iloc[0]),
        'fin':            str(df['periodo'].iloc[-1]),
        'ultimo_drift':   s(getattr(last_r,'DRIFT',None)),
        'ultimo_regimen': str(getattr(last_r,'Régimen','')),
        'meses_favorables': int(fav) if len(df_ok)>0 else 0,
        'corr_drift_dol': round(corr,3) if len(df_ok)>10 else None,
    }
    with open(DATA_DIR/"drift_latest.json",'w',encoding='utf-8') as f:
        json.dump({'meta':meta,'series':records},
                  f,separators=(',',':'),ensure_ascii=False)
    print(f"  drift_latest.json — {len(records)} registros")
    print("\n[✓] Pipeline v4.1 completado.")

if __name__ == '__main__':
    run()
