"""
DRIFT Pipeline v2.0 — Digital Risk & Financial Trust Index
==========================================================
Autor : Ceyner D. Llontop Herrera | ORCID: 0009-0007-6058-7711
Versión: 2.0 | Fix: extracción individual por serie (sin problemas de orden)

Cambios v2.0:
  - Cada serie BCRP se extrae individualmente → sin errores de columna
  - FRED DXY con detección robusta de columnas
  - DimPD preservado correctamente en todos los merges
  - Outputs con nombres fijos (drift_serie.csv, drift_latest.json)
"""

import requests
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime, date
from pathlib import Path
from io import StringIO

# ── CONFIGURACIÓN ─────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

INICIO    = "2018-1"
BCRP_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Dato oficial Reporte BCRP marzo 2026
# 665 pagos/adulto × 23.5M adultos = 15,628M ops totales en 2025
IPD_OFICIAL_2025 = 15_628.0

# ── SERIES ────────────────────────────────────────────────────
SERIES_MAIN = {
    'Dolarizacion': 'PN00531MM',
    'IPC':          'PN01271PM',
    'BVL':          'PN01158MM',
    'IPD':          'PN39971SM',
    'EMBI':         'PN01129XM',
    'Adopcion':     'PN09416SM',
}

SERIES_DIMPD = [
    'PN09398SM', 'PN09399SM', 'PN09400SM', 'PN09403SM',
    'PN09406SM', 'PN09408SM', 'PN09409SM', 'PN09411SM',
    'PN09414SM', 'PN09416SM',
]

MESES_ES = {
    'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,
    'Jul':7,'Ago':8,'Sep':9,'Oct':10,'Nov':11,'Dic':12
}

def bcrp_a_fecha(periodo: str) -> pd.Timestamp:
    try:
        mes = MESES_ES[periodo[:3]]
        yr  = 2000 + int(periodo[3:])
        return pd.Timestamp(yr, mes, 1)
    except:
        return pd.NaT

# ── EXTRACCIÓN BCRP (UNA SERIE A LA VEZ) ─────────────────────
def fetch_serie_bcrp(codigo: str, ini: str, fin: str) -> pd.DataFrame:
    """
    Extrae UNA sola serie del BCRP API.
    Sin problemas de orden — siempre solo una columna de valores.
    """
    url = f"{BCRP_BASE}/{codigo}/json/{ini}/{fin}/esp"
    try:
        r = requests.get(url, timeout=30,
                        headers={'User-Agent': 'DRIFT-Index/2.0'})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    ERROR {codigo}: {e}")
        return pd.DataFrame()

    if 'periods' not in data:
        print(f"    Sin datos: {codigo}")
        return pd.DataFrame()

    records = []
    for period in data['periods']:
        val = period.get('values', [None])[0]
        try:
            valor = float(val) if val not in ['n.d.', '', None] else np.nan
        except:
            valor = np.nan
        records.append({
            'periodo': period['name'],
            'fecha':   bcrp_a_fecha(period['name']),
            codigo:    valor
        })

    df = pd.DataFrame(records).sort_values('fecha').reset_index(drop=True)
    n_ok = df[codigo].notna().sum()
    print(f"    {codigo}: {len(df)} períodos ({n_ok} con dato)")
    return df


def fetch_todas_bcrp(series_dict: dict, ini: str, fin: str) -> pd.DataFrame:
    """
    Extrae múltiples series una por una y las une por fecha.
    Garantiza que cada columna tiene el valor correcto.
    """
    df_base = None
    for nombre, codigo in series_dict.items():
        df_serie = fetch_serie_bcrp(codigo, ini, fin)
        if df_serie.empty:
            continue
        df_serie = df_serie.rename(columns={codigo: nombre})
        if df_base is None:
            df_base = df_serie[['periodo', 'fecha', nombre]]
        else:
            df_base = df_base.merge(
                df_serie[['fecha', nombre]], on='fecha', how='outer')

    if df_base is not None:
        df_base = df_base.sort_values('fecha').reset_index(drop=True)
    return df_base if df_base is not None else pd.DataFrame()


def fetch_dimpd_bcrp(ini: str, fin: str) -> pd.DataFrame:
    """
    Extrae las 10 series del denominador de DimPD una por una.
    Calcula DimPD = PN09416SM / suma(10 series).
    Proyecta series incompletas con tendencia lineal.
    """
    df_base = None
    for codigo in SERIES_DIMPD:
        df_s = fetch_serie_bcrp(codigo, ini, fin)
        if df_s.empty:
            continue
        if df_base is None:
            df_base = df_s[['fecha', codigo]]
        else:
            df_base = df_base.merge(df_s[['fecha', codigo]],
                                    on='fecha', how='outer')

    if df_base is None or df_base.empty:
        return pd.DataFrame()

    df_base = df_base.sort_values('fecha').reset_index(drop=True)

    # Proyectar series incompletas
    for col in SERIES_DIMPD:
        if col not in df_base.columns:
            continue
        faltantes = df_base[col].isna().sum()
        if faltantes == 0:
            continue
        conocidos = df_base[col].dropna()
        if len(conocidos) < 6:
            df_base[col] = df_base[col].ffill()
            continue
        t_known = np.arange(len(conocidos))
        coef    = np.polyfit(t_known, conocidos.values, 1)
        t_proj  = np.arange(len(conocidos),
                            len(conocidos) + faltantes)
        proyect = np.polyval(coef, t_proj)
        proyect = np.maximum(proyect, conocidos.iloc[-1] * 0.85)
        df_base.loc[df_base[col].isna(), col] = proyect
        print(f"    {col}: {faltantes} meses proyectados")

    # Calcular DimPD
    cols_disponibles = [c for c in SERIES_DIMPD if c in df_base.columns]
    df_base['total_pagos'] = df_base[cols_disponibles].sum(axis=1)
    df_base['DimPD'] = df_base['PN09416SM'] / df_base['total_pagos']
    rng = f"{df_base['DimPD'].min():.4f} – {df_base['DimPD'].max():.4f}"
    print(f"    DimPD calculado | rango: {rng}")
    return df_base[['fecha', 'DimPD', 'total_pagos']]


# ── EXTRACCIÓN FRED (DXY) ────────────────────────────────────
def fetch_dxy(ini_date: str = '2018-01-01') -> pd.DataFrame:
    """
    Extrae TWEXBGSMTH desde FRED con detección robusta de columnas.
    """
    print(f"    FRED TWEXBGSMTH...")
    try:
        params = {'id': 'TWEXBGSMTH',
                  'observation_start': ini_date,
                  'file_type': 'csv'}
        r = requests.get(FRED_BASE, params=params, timeout=30)
        r.raise_for_status()

        # Leer sin asumir nombre de columna
        df = pd.read_csv(StringIO(r.text), header=0)
        print(f"    Columnas FRED: {list(df.columns)}")

        # Primera columna = fecha, segunda = valor
        df.columns = ['fecha_str', 'DXY']
        df['fecha'] = pd.to_datetime(df['fecha_str'], errors='coerce')
        df['DXY']   = pd.to_numeric(df['DXY'], errors='coerce')
        df = df.dropna(subset=['fecha', 'DXY'])

        # Agrupar por mes (FRED puede dar semanal o mensual)
        df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
        df = df.groupby('fecha')['DXY'].mean().reset_index()
        print(f"    DXY: {len(df)} períodos | "
              f"último: {df['DXY'].iloc[-1]:.1f}")
        return df

    except Exception as e:
        print(f"    ERROR FRED: {e}")
        return pd.DataFrame()


# ── PROYECCIÓN IPD ────────────────────────────────────────────
def proyectar_ipd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proyecta IPD para meses sin dato usando tendencia exponencial.
    Ancla 2025 al dato oficial: 15,628M ops anuales.
    """
    conocidos      = df[df['IPD'].notna()].copy()
    sin_dato       = df[df['IPD'].isna()].copy()
    ultima_fecha   = conocidos['fecha'].max()

    if sin_dato.empty:
        df['IPD_proyectado'] = False
        return df

    print(f"    IPD disponible hasta: {ultima_fecha.strftime('%b-%Y')}")
    print(f"    Proyectando {len(sin_dato)} meses...")

    # Tendencia exponencial sobre log(IPD)
    t_base  = conocidos['fecha'].min()
    t_known = (conocidos['fecha'] - t_base).dt.days.values
    log_y   = np.log(conocidos['IPD'].values + 1)
    coef    = np.polyfit(t_known, log_y, 1)
    t_proj  = (sin_dato['fecha'] - t_base).dt.days.values
    y_proj  = np.exp(np.polyval(coef, t_proj)) - 1

    # Anclar 2025 al total oficial
    sin_dato_2025 = sin_dato[sin_dato['fecha'].dt.year == 2025]
    if len(sin_dato_2025) > 0:
        total_conocido_2025 = conocidos[
            conocidos['fecha'].dt.year == 2025]['IPD'].sum()
        restante = IPD_OFICIAL_2025 - total_conocido_2025
        if restante > 0:
            n_meses = len(sin_dato_2025)
            pesos   = np.linspace(1.0, 1.08, n_meses)
            pesos  /= pesos.sum()
            vals_2025 = restante * pesos
            for i, idx in enumerate(sin_dato_2025.index):
                proj_idx = list(sin_dato.index).index(idx)
                y_proj[proj_idx] = vals_2025[i]

    df = df.copy()
    df.loc[sin_dato.index, 'IPD'] = y_proj
    df['IPD_proyectado'] = False
    df.loc[df['fecha'] > ultima_fecha, 'IPD_proyectado'] = True
    print(f"    IPD proyectado hasta {df['fecha'].max().strftime('%b-%Y')}")
    return df


# ── CÁLCULO DRIFT ─────────────────────────────────────────────
def calcular_drift(df: pd.DataFrame) -> pd.DataFrame:
    def mm(s, invert=False):
        s2 = pd.Series(s, dtype=float).ffill().bfill()
        mn, mx = s2.min(), s2.max()
        if mx == mn:
            return pd.Series([50.0] * len(s2))
        norm = (s2 - mn) / (mx - mn) * 100
        return 100 - norm if invert else norm

    DPS     = mm(df['DimPD'])
    DVS     = mm(df['IPD'])
    MCS     = mm(df['Dolarizacion'], invert=True)
    DRIFT_T = 0.40*DPS + 0.40*MCS + 0.20*DVS

    CRS     = mm(df['EMBI'])
    DSS     = mm(df['DXY'])
    PIS     = mm(df['IPC'].abs())
    IGS     = mm(df['BVL'], invert=True)
    DRIFT_R = 0.35*CRS + 0.25*DSS + 0.20*PIS + 0.20*IGS

    DRIFT   = DRIFT_T - DRIFT_R

    def regime(v):
        if pd.isna(v):    return 'Sin datos'
        if v >= 20:       return 'Confianza Sólida'
        if v >= 5:        return 'Expansión'
        if v >= -5:       return 'Equilibrio'
        if v >= -20:      return 'Tensión Alta'
        return 'Crisis Severa'

    df = df.copy()
    df['DPS']     = DPS.values;     df['MCS']     = MCS.values
    df['DVS']     = DVS.values;     df['CRS']     = CRS.values
    df['DSS']     = DSS.values;     df['PIS']     = PIS.values
    df['IGS']     = IGS.values;     df['DRIFT_T'] = DRIFT_T.values
    df['DRIFT_R'] = DRIFT_R.values; df['DRIFT']   = DRIFT.values
    df['Régimen'] = DRIFT.apply(regime)
    df['Año']     = df['fecha'].dt.year
    return df


# ── PIPELINE PRINCIPAL ────────────────────────────────────────
def run():
    print("=" * 60)
    print(f"DRIFT Pipeline v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Período: hasta el mes anterior al actual
    hoy   = date.today()
    fin_mo = hoy.month - 1 if hoy.month > 1 else 12
    fin_yr = hoy.year if hoy.month > 1 else hoy.year - 1
    fin   = f"{fin_yr}-{fin_mo}"
    print(f"\nPeríodo: {INICIO} → {fin}")

    # ── 1. Series BCRP principales ──────────────────────────
    print("\n[1] Extrayendo series BCRP principales (una por una)...")
    df = fetch_todas_bcrp(SERIES_MAIN, INICIO, fin)
    if df is None or df.empty:
        print("ERROR CRÍTICO: Sin datos BCRP principales.")
        sys.exit(1)
    print(f"  Dataset base: {len(df)} períodos × "
          f"{len([c for c in SERIES_MAIN if c in df.columns])} series")

    # Verificar dolarización
    dol_ok = df['Dolarizacion'].dropna()
    if len(dol_ok) > 0:
        print(f"  Dolarización: min={dol_ok.min():.2f}% "
              f"max={dol_ok.max():.2f}% último={dol_ok.iloc[-1]:.2f}%")

    # ── 2. DimPD ────────────────────────────────────────────
    print("\n[2] Construyendo DimPD (denominador 10 series)...")
    df_dimpd = fetch_dimpd_bcrp(INICIO, fin)
    if not df_dimpd.empty:
        df = df.merge(df_dimpd, on='fecha', how='left')
        print(f"  DimPD incorporado | "
              f"válidos: {df['DimPD'].notna().sum()}/{len(df)}")
    else:
        df['DimPD'] = np.nan
        print("  ADVERTENCIA: DimPD no disponible")

    # ── 3. DXY desde FRED ───────────────────────────────────
    print("\n[3] Extrayendo DXY desde FRED...")
    df_dxy = fetch_dxy()
    if not df_dxy.empty:
        df = df.merge(df_dxy, on='fecha', how='left')
        df['DXY'] = df['DXY'].ffill()
        print(f"  DXY incorporado | "
              f"válidos: {df['DXY'].notna().sum()}/{len(df)}")
    else:
        # Intentar con URL alternativa
        print("  Intentando URL alternativa FRED...")
        try:
            alt_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TWEXBGSMTH"
            r2 = requests.get(alt_url, timeout=30)
            df2 = pd.read_csv(StringIO(r2.text), header=0)
            df2.columns = ['fecha_str', 'DXY']
            df2['fecha'] = pd.to_datetime(df2['fecha_str'], errors='coerce')
            df2['DXY']   = pd.to_numeric(df2['DXY'], errors='coerce')
            df2 = df2.dropna(subset=['fecha'])
            df2['fecha'] = df2['fecha'].dt.to_period('M').dt.to_timestamp()
            df2 = df2.groupby('fecha')['DXY'].mean().reset_index()
            df = df.merge(df2, on='fecha', how='left')
            df['DXY'] = df['DXY'].ffill()
            print(f"  DXY OK (alt) | último: {df['DXY'].dropna().iloc[-1]:.1f}")
        except Exception as e2:
            print(f"  ADVERTENCIA: DXY no disponible ({e2})")
            df['DXY'] = np.nan

    # ── 4. Proyectar IPD ────────────────────────────────────
    print("\n[4] Verificando y proyectando IPD...")
    df = proyectar_ipd(df)

    # ── 5. Calcular DRIFT ────────────────────────────────────
    print("\n[5] Calculando DRIFT...")

    # Verificar columnas críticas
    for col in ['DimPD', 'Dolarizacion', 'IPD', 'EMBI', 'DXY', 'IPC', 'BVL']:
        n_ok = df[col].notna().sum() if col in df.columns else 0
        print(f"  {col}: {n_ok}/{len(df)} válidos")

    df = calcular_drift(df)

    # Resumen del último período con dato completo
    df_ok = df[df['DRIFT'].notna()]
    if len(df_ok) > 0:
        last  = df_ok.iloc[-1]
        fav   = 0
        for i in range(len(df_ok)-1, -1, -1):
            if df_ok['DRIFT'].iloc[i] >= 5: fav += 1
            else: break
        corr  = np.corrcoef(df_ok['DRIFT'],
                            df_ok['Dolarizacion'])[0,1] if len(df_ok) > 10 else np.nan
        print(f"\n  Último período completo: {last['periodo']}")
        print(f"  DRIFT-T: {last['DRIFT_T']:.1f} | "
              f"DRIFT-R: {last['DRIFT_R']:.1f}")
        print(f"  DRIFT neto: {last['DRIFT']:+.1f} | {last['Régimen']}")
        print(f"  Dolarización: {last['Dolarizacion']:.2f}%")
        print(f"  DimPD: {last['DimPD']:.4f} = {last['DimPD']*100:.2f}%")
        print(f"  Correlación DRIFT-Dolarización: {corr:.3f}")
        print(f"  Meses consecutivos favorables: {fav}")
    else:
        print("  ADVERTENCIA: Sin períodos con DRIFT completo")
        last = df.iloc[-1]

    # ── 6. Guardar outputs ───────────────────────────────────
    print("\n[6] Guardando outputs...")

    # CSV — nombre fijo
    cols_csv = ['periodo','fecha','Año','DimPD','Dolarizacion','IPD',
                'IPD_proyectado','EMBI','DXY','IPC','BVL','Adopcion',
                'DPS','MCS','DVS','CRS','DSS','PIS','IGS',
                'DRIFT_T','DRIFT_R','DRIFT','Régimen']
    cols_csv = [c for c in cols_csv if c in df.columns]
    csv_path = DATA_DIR / "drift_serie.csv"
    df[cols_csv].to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  drift_serie.csv — {len(df)} filas, {len(cols_csv)} columnas")

    # JSON — nombre fijo
    ultima_ipd = df.loc[df['IPD_proyectado'] == False,
                        'fecha'].max() if 'IPD_proyectado' in df.columns else ''

    records = []
    for _, row in df.iterrows():
        def safe(v, dec=1):
            return round(float(v), dec) if pd.notna(v) else None
        records.append({
            'periodo':  str(row.get('periodo', '')),
            'year':     int(row['Año']) if pd.notna(row['Año']) else None,
            'drift':    safe(row.get('DRIFT')),
            'drift_t':  safe(row.get('DRIFT_T')),
            'drift_r':  safe(row.get('DRIFT_R')),
            'dps':      safe(row.get('DPS')),
            'mcs':      safe(row.get('MCS')),
            'dvs':      safe(row.get('DVS')),
            'crs':      safe(row.get('CRS')),
            'dss':      safe(row.get('DSS')),
            'pis':      safe(row.get('PIS')),
            'igs':      safe(row.get('IGS')),
            'dim_pd':   safe(row.get('DimPD'), 4),
            'dol':      safe(row.get('Dolarizacion'), 2),
            'embi':     safe(row.get('EMBI'), 0),
            'dxy':      safe(row.get('DXY'), 1),
            'ipd':      safe(row.get('IPD'), 1),
            'ipd_proy': bool(row.get('IPD_proyectado', False)),
            'regime':   str(row.get('Régimen', '')),
        })

    meta = {
        'generado':     datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
        'N':            len(df),
        'inicio':       str(df['periodo'].iloc[0]) if len(df) > 0 else '',
        'fin':          str(df['periodo'].iloc[-1]) if len(df) > 0 else '',
        'ipd_datos_reales_hasta': str(ultima_ipd)[:7] if ultima_ipd else '',
        'ultimo_drift': safe(last.get('DRIFT')),
        'ultimo_regimen': str(last.get('Régimen', '')),
    }

    json_path = DATA_DIR / "drift_latest.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'series': records},
                  f, separators=(',',':'), ensure_ascii=False)
    print(f"  drift_latest.json — {len(records)} registros")

    print("\n[✓] Pipeline v2.0 completado.")


if __name__ == '__main__':
    run()
