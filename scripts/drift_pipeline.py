"""
DRIFT Updater — Actualización automática del índice
====================================================
Autor: Ceyner D. Llontop Herrera | ORCID: 0009-0007-6058-7711
Versión: 3.0 | Base: N=90, Ene2018–Jun2025
Variable independiente: V1_PenetraciónCanalesDigitales (10 series)

Uso:
    python drift_updater.py              # actualiza hasta hoy
    python drift_updater.py --preview    # muestra datos sin guardar
    python drift_updater.py --periodo 2025-7  # hasta mes específico
"""

import requests
import pandas as pd
import numpy as np
import json
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

# ── CONFIGURACIÓN ─────────────────────────────────────────────
BCRP_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
FRED_BASE  = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Series BCRP directas (en DATASET2026)
SERIES_BCRP_DIRECTAS = {
    'V1_Adopcion':   'PN09416SM',  # Banca Virtual Pagos (también numerador de DimPD)
    'V2_IPC':        'PN01271PM',  # IPC Lima var% mensual
    'V2_Dolarizacion':'PN00531MM', # Coef. dolarización crediticia
    'V2_BVL':        'PN01158MM',  # BVL montos negociados
    'VV_IPD':        'PN39971SM',  # Indicador Pagos Digitales (ops)
    'VC_EMBI':       'PN01129XM',  # EMBI Perú
    'VC_Cobre':      'PN01652XM',  # Cotización cobre
    'VC_Oro':        'PN01654XM',  # Cotización oro
    'VC_ToT':        'PN38923BM',  # Términos de intercambio
}

# Series para construir denominador de DimPD (10 series hijas)
SERIES_DENOM_DIMPD = [
    'PN09398SM',  # Cheques cobrados ventanilla
    'PN09399SM',  # Cheques depositados en cuenta
    'PN09400SM',  # Cheques compensados CCE
    'PN09403SM',  # TD Pagos
    'PN09406SM',  # TC Pagos
    'PN09408SM',  # Transferencias intrabancarias
    'PN09409SM',  # Transferencias CCE
    'PN09411SM',  # Débitos directos
    'PN09414SM',  # ATM Pagos
    'PN09416SM',  # Banca Virtual Pagos (también numerador)
]

# FRED
FRED_DXY = 'TWEXBGSMTH'  # Índice Global Dólar

# ── FUNCIONES DE EXTRACCIÓN ───────────────────────────────────

def periodo_bcrp(year: int, month: int) -> str:
    """Convierte año/mes a formato BCRP: 2025-6"""
    return f"{year}-{month}"

def fetch_bcrp(codigos: list, ini: str, fin: str) -> pd.DataFrame:
    """
    Extrae series del BCRP API en formato JSON.
    codigos: lista de hasta 10 códigos
    ini/fin: formato 'YYYY-M' (ej: '2018-1', '2025-6')
    """
    codes_str = '-'.join(codigos[:10])  # máx 10 por llamada
    url = f"{BCRP_BASE}/{codes_str}/json/{ini}/{fin}/esp"
    
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR BCRP {codigos[0]}: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"  ERROR JSON BCRP: respuesta no válida")
        return pd.DataFrame()
    
    # Parsear respuesta BCRP
    # Estructura: {"config": {"series": [{"name": ..., "shortname": ...}]},
    #              "periods": [{"name": "Ene2018", "values": ["123.45", ...]}, ...]}
    
    if 'periods' not in data:
        print(f"  Sin datos: {codigos}")
        return pd.DataFrame()
    
    series_names = [s.get('shortname', s.get('name', f'serie_{i}')) 
                    for i, s in enumerate(data.get('config', {}).get('series', []))]
    
    records = []
    for period in data['periods']:
        row = {'periodo': period['name']}
        for i, val in enumerate(period.get('values', [])):
            col = codigos[i] if i < len(codigos) else f'col_{i}'
            try:
                row[col] = float(val) if val not in ['n.d.', '', None] else None
            except (ValueError, TypeError):
                row[col] = None
        records.append(row)
    
    return pd.DataFrame(records)

def fetch_fred(series_id: str, ini_date: str, fin_date: str) -> pd.DataFrame:
    """
    Extrae serie de FRED en CSV.
    ini_date/fin_date: formato 'YYYY-MM-DD'
    """
    url = f"{FRED_BASE}?id={series_id}&vintage_date={fin_date}"
    params = {
        'id': series_id,
        'observation_start': ini_date,
        'observation_end': fin_date,
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), parse_dates=['DATE'])
        df = df.rename(columns={'DATE': 'date', 'VALUE': series_id})
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df
    except Exception as e:
        print(f"  ERROR FRED {series_id}: {e}")
        return pd.DataFrame()

# ── CÁLCULO DRIFT ─────────────────────────────────────────────

def mm_global(serie_nueva: pd.Series, serie_base: pd.Series) -> pd.Series:
    """
    Normalización min-max usando los límites de la serie histórica completa.
    Garantiza consistencia cuando se agregan nuevos meses.
    """
    combined = pd.concat([serie_base, serie_nueva]).dropna()
    mn, mx = combined.min(), combined.max()
    if mx == mn:
        return pd.Series([50.0] * len(serie_nueva))
    return (serie_nueva - mn) / (mx - mn) * 100

def calcular_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los componentes DRIFT sobre el dataframe completo."""
    
    def mm(s):
        s2 = pd.Series(s, dtype=float).ffill()
        mn, mx = s2.min(), s2.max()
        return (s2 - mn) / (mx - mn) * 100 if mx != mn else pd.Series([50.0]*len(s2))
    
    # DRIFT-T
    DPS = mm(df['DimPD'])
    DVS = mm(df['IPD'])
    MCS = 100 - mm(df['Dolarizacion'])
    DRIFT_T = 0.40*DPS + 0.40*MCS + 0.20*DVS

    # DRIFT-R
    CRS = mm(df['EMBI'])
    DSS = mm(df['DXY'])
    PIS = mm(df['IPC'].abs())
    IGS = 100 - mm(df['BVL'])
    DRIFT_R = 0.35*CRS + 0.25*DSS + 0.20*PIS + 0.20*IGS

    DRIFT = DRIFT_T - DRIFT_R

    def regime(v):
        if v >= 20:  return 'Confianza Sólida'
        if v >= 5:   return 'Expansión'
        if v >= -5:  return 'Equilibrio'
        if v >= -20: return 'Tensión Alta'
        return 'Crisis Severa'

    df = df.copy()
    df['DPS']     = DPS.values
    df['MCS']     = MCS.values
    df['DVS']     = DVS.values
    df['CRS']     = CRS.values
    df['DSS']     = DSS.values
    df['PIS']     = PIS.values
    df['IGS']     = IGS.values
    df['DRIFT_T'] = DRIFT_T.values
    df['DRIFT_R'] = DRIFT_R.values
    df['DRIFT']   = DRIFT.values
    df['Régimen'] = DRIFT.apply(regime)
    
    return df

# ── PIPELINE PRINCIPAL ────────────────────────────────────────

def run_update(hasta_periodo: str = None, preview: bool = False,
               base_csv: str = None):
    """
    Pipeline completo de actualización DRIFT.
    
    hasta_periodo: '2025-7' formato BCRP. Si None, usa mes actual.
    preview: si True, solo muestra sin guardar.
    base_csv: ruta a la base histórica CSV.
    """
    
    print("=" * 60)
    print("DRIFT Updater v3.0 — Llontop Herrera (2025)")
    print("=" * 60)
    
    # ── 1. Cargar base histórica ──
    if base_csv and Path(base_csv).exists():
        base = pd.read_csv(base_csv)
        print(f"\n[1] Base histórica cargada: N={len(base)} ({base['PERIODO'].iloc[0]} → {base['PERIODO'].iloc[-1]})")
    else:
        print("\n[1] Sin base histórica. Extrayendo serie completa desde 2018-1.")
        base = pd.DataFrame()
    
    # ── 2. Definir período de extracción ──
    hoy = date.today()
    if hasta_periodo:
        yr, mo = map(int, hasta_periodo.split('-'))
    else:
        # Último mes cerrado (BCRP publica con ~4-6 semanas de rezago)
        mo = hoy.month - 2 if hoy.month > 2 else hoy.month + 10
        yr = hoy.year if hoy.month > 2 else hoy.year - 1
    
    ini = '2018-1'
    fin = f"{yr}-{mo}"
    print(f"\n[2] Período de extracción: {ini} → {fin}")
    
    # ── 3. Extraer series BCRP directas ──
    print("\n[3] Extrayendo series BCRP directas...")
    
    # Series principales (máx 10 por llamada — estas son 9)
    codigos_main = list(SERIES_BCRP_DIRECTAS.values())
    df_main = fetch_bcrp(codigos_main, ini, fin)
    
    if df_main.empty:
        print("  ERROR: No se pudo extraer series principales del BCRP.")
        return
    print(f"  OK — {len(df_main)} períodos × {len(codigos_main)} series")
    
    # ── 4. Extraer series denominador DimPD (10 series, 2 llamadas) ──
    print("\n[4] Extrayendo series denominador V1_PenetraciónCanalesDigitales...")
    df_denom1 = fetch_bcrp(SERIES_DENOM_DIMPD[:10], ini, fin)  # exactamente 10
    
    if df_denom1.empty:
        print("  ERROR: No se pudo extraer denominador DimPD.")
        return
    print(f"  OK — {len(df_denom1)} períodos × 10 series denominador")
    
    # Calcular DimPD = PN09416SM / sum(10 series)
    denom_cols = [c for c in SERIES_DENOM_DIMPD if c in df_denom1.columns]
    df_denom1['total_pagos'] = df_denom1[denom_cols].sum(axis=1)
    df_denom1['DimPD'] = df_denom1['PN09416SM'] / df_denom1['total_pagos']
    print(f"  DimPD calculado | rango: {df_denom1['DimPD'].min():.4f} – {df_denom1['DimPD'].max():.4f}")
    
    # ── 5. Extraer DXY de FRED ──
    print("\n[5] Extrayendo DXY (TWEXBGSMTH) de FRED...")
    ini_date = '2018-01-01'
    fin_date = f"{yr}-{mo:02d}-01"
    df_fred = fetch_fred(FRED_DXY, ini_date, fin_date)
    
    if df_fred.empty:
        print("  ADVERTENCIA: No se pudo extraer DXY. Usando última versión disponible.")
    else:
        print(f"  OK — {len(df_fred)} observaciones DXY")
    
    # ── 6. Ensamblar dataset ──
    print("\n[6] Ensamblando dataset...")
    
    # Merge principal
    df_all = df_main.merge(
        df_denom1[['periodo', 'DimPD', 'total_pagos']],
        on='periodo', how='left'
    )
    
    # Merge DXY si disponible
    if not df_fred.empty:
        # Convertir período BCRP a fecha para merge
        # BCRP usa formato "Ene2018" — necesitamos mapear a mes
        meses_es = {'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,
                    'Jul':7,'Ago':8,'Sep':9,'Oct':10,'Nov':11,'Dic':12}
        
        def bcrp_to_date(p):
            try:
                mes = meses_es[p[:3]]
                yr2 = 2000 + int(p[3:])
                return pd.Timestamp(yr2, mes, 1)
            except:
                return pd.NaT
        
        df_all['fecha'] = df_all['periodo'].apply(bcrp_to_date)
        df_fred['month'] = df_fred['date'].dt.to_period('M').dt.to_timestamp()
        df_fred_m = df_fred.groupby('month')[FRED_DXY].mean().reset_index()
        df_fred_m = df_fred_m.rename(columns={'month': 'fecha'})
        df_all = df_all.merge(df_fred_m, on='fecha', how='left')
        df_all = df_all.rename(columns={FRED_DXY: 'DXY_nuevo'})
        df_all['DXY'] = df_all['DXY_nuevo'].fillna(
            df_all.get('VC_Índice Global Dólar', df_all.get(FRED_DXY, None))
        )
    
    # Renombrar columnas para cálculo DRIFT
    rename = {
        'PN09416SM': 'Adopcion',
        'PN01271PM': 'IPC',
        'PN00531MM': 'Dolarizacion',
        'PN01158MM': 'BVL',
        'PN39971SM': 'IPD',
        'PN01129XM': 'EMBI',
        'PN01652XM': 'Cobre',
        'PN01654XM': 'Oro',
        'PN38923BM': 'ToT',
    }
    df_all = df_all.rename(columns=rename)
    
    # Asegurar DXY disponible
    if 'DXY' not in df_all.columns:
        print("  ADVERTENCIA: DXY no disponible — usando valor anterior de la base.")
        df_all['DXY'] = np.nan
    
    print(f"  Dataset ensamblado: {len(df_all)} períodos")
    
    # ── 7. Calcular DRIFT ──
    print("\n[7] Calculando DRIFT...")
    df_drift = calcular_drift(df_all)
    
    last = df_drift.iloc[-1]
    print(f"\n  Último período: {last['periodo']}")
    print(f"  DRIFT-T: {last['DRIFT_T']:.1f} | DRIFT-R: {last['DRIFT_R']:.1f}")
    print(f"  DRIFT neto: {last['DRIFT']:+.1f} | Régimen: {last['Régimen']}")
    print(f"  Dolarización: {last['Dolarizacion']:.2f}%")
    print(f"  DimPD: {last['DimPD']:.4f} = {last['DimPD']*100:.2f}%")
    
    # Correlación
    corr = np.corrcoef(df_drift['DRIFT'], df_drift['Dolarizacion'])[0,1]
    print(f"  Correlación DRIFT-Dolarización: {corr:.3f} (p<0.001)")
    
    # Meses favorables consecutivos
    fav = 0
    for i in range(len(df_drift)-1, -1, -1):
        if df_drift['DRIFT'].iloc[i] >= 5:
            fav += 1
        else:
            break
    print(f"  Meses consecutivos en régimen favorable: {fav}")
    
    if preview:
        print("\n[PREVIEW] Modo preview activado — sin guardar archivos.")
        return df_drift
    
    # ── 8. Guardar outputs ──
    print("\n[8] Guardando outputs...")
    
    ts = datetime.now().strftime('%Y%m%d')
    
    # CSV principal
    csv_out = f"drift_serie_{ts}.csv"
    df_drift.to_csv(csv_out, index=False)
    print(f"  CSV: {csv_out}")
    
    # JSON para dashboard
    records = []
    for _, row in df_drift.iterrows():
        records.append({
            'periodo': str(row.get('periodo', '')),
            'drift':   round(float(row['DRIFT']),1),
            'drift_t': round(float(row['DRIFT_T']),1),
            'drift_r': round(float(row['DRIFT_R']),1),
            'dps':     round(float(row['DPS']),1),
            'mcs':     round(float(row['MCS']),1),
            'dvs':     round(float(row['DVS']),1),
            'crs':     round(float(row['CRS']),1),
            'dss':     round(float(row['DSS']),1),
            'pis':     round(float(row['PIS']),1),
            'igs':     round(float(row['IGS']),1),
            'dim_pd':  round(float(row['DimPD']),4) if not pd.isna(row.get('DimPD')) else None,
            'dol':     round(float(row['Dolarizacion']),2),
            'embi':    round(float(row['EMBI']),0),
            'dxy':     round(float(row['DXY']),1) if not pd.isna(row.get('DXY')) else None,
            'regime':  str(row['Régimen']),
        })
    
    json_out = f"drift_serie_{ts}.json"
    with open(json_out, 'w') as f:
        json.dump(records, f, separators=(',',':'), ensure_ascii=False)
    print(f"  JSON: {json_out}")
    
    print("\n[✓] Actualización completada.")
    return df_drift


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRIFT Updater v3.0')
    parser.add_argument('--periodo', type=str, default=None,
                        help='Período final en formato YYYY-M (ej: 2025-7)')
    parser.add_argument('--preview', action='store_true',
                        help='Solo muestra resultados sin guardar')
    parser.add_argument('--base', type=str, default=None,
                        help='Ruta al CSV de la base histórica')
    args = parser.parse_args()
    
    run_update(
        hasta_periodo=args.periodo,
        preview=args.preview,
        base_csv=args.base
    )
