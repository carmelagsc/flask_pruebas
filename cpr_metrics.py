#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------
# Helpers
# --------------------------

def _detect_time_column(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Detecta la columna de tiempo en el DataFrame y la convierte a segundos desde el inicio de la sesión.
    """
    cols = {c.lower(): c for c in df.columns}
    if "timestamp_s" in cols:
        s = pd.to_numeric(df[cols["timestamp_s"]], errors="coerce")
        if s.isna().any():
            raise ValueError("Found 'timestamp_s' but contains non-numeric values.")
        return s.astype(float), cols["timestamp_s"]

    for cand in ["timestamp", "time"]:
        if cand in cols:
            ts = pd.to_datetime(df[cols[cand]], errors="coerce", utc=True)
            if ts.isna().any():
                raise ValueError(f"Column '{cols[cand]}' has unparsable datetimes.")
            s = (ts - ts.iloc[0]).dt.total_seconds()
            return s.astype(float), cols[cand]

    raise ValueError("No valid time column found. Provide either 'timestamp_s' or 'timestamp'/'time' in CSV.")

def _require_cpm(df: pd.DataFrame) -> pd.Series:

# Garantiza que exista una columna cpm (compresiones por minuto) y la devuelve como numérica.
    cols = {c.lower(): c for c in df.columns}
    if "cpm" not in cols:
        raise ValueError("CSV must contain a 'cpm' column (compressions per minute).")
    s = pd.to_numeric(df[cols["cpm"]], errors="coerce")
    if s.isna().any():
        raise ValueError("Column 'cpm' contains non-numeric values.")
    return s.astype(float)

def _segments_by_condition(times_s: np.ndarray, values: np.ndarray, predicate) -> List[Tuple[float, float]]:
    """
Construye segmentos de tiempo contiguos donde se cumple una condición
booleana sobre el valor (por ejemplo: “CPM == 0” ó “CPM > 130”).
Asume hold por tramos (cada muestra vale hasta la siguiente)..
    """
    n = len(values)
    if n < 2:
        return []
    segs = []
    in_seg = False
    seg_start = None
    for i in range(n-1):
        v = values[i]
        dt = times_s[i+1] - times_s[i]
        good = bool(predicate(v)) and dt > 0
        if good and not in_seg:
            in_seg = True
            seg_start = times_s[i]
        if not good and in_seg:
            in_seg = False
            segs.append((seg_start, times_s[i]))
            seg_start = None
    if in_seg:
        segs.append((seg_start, times_s[-1]))
    return segs # Lista de tuplas (start_s, end_s) para cada tramo donde la condición se cumple.

def _time_weighted_percent(times_s: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    """
    Calcula el porcentaje de tiempo en el que el valor estuvo dentro del rango [lo, hi], 
    ponderando por la duración entre muestras (modelo “step-hold”: cada muestra vale hasta la próxima marca de tiempo).
    """
    n = len(values)
    if n < 2:
        return 0.0
    total = float(times_s[-1] - times_s[0])
    if total <= 0:
        return 0.0
    acc = 0.0
    for i in range(n-1):
        dt = times_s[i+1] - times_s[i]
        if dt <= 0:
            continue
        v = values[i]
        if lo <= v <= hi:
            acc += dt
    return 100.0 * acc / total

def _time_weighted_mean(times_s: np.ndarray, values: np.ndarray) -> float:
    # Calcula la media ponderada por tiempo de la señal (otra vez, step-hold entre muestras)
    n = len(values)
    if n < 2:
        return float(values.iloc[0]) if n == 1 else float("nan")
    total = float(times_s[-1] - times_s[0])
    if total <= 0:
        return float("nan")
    acc = 0.0
    for i in range(n-1):
        dt = times_s[i+1] - times_s[i]
        if dt <= 0:
            continue
        acc += values.iloc[i] * dt
    return acc / total

def _time_weighted_std(times_s: np.ndarray, values: pd.Series) -> float:
    """
    Calcula el desvío estándar poblacional ponderado
    por tiempo (respecto de la media ponderada anterior).
    """
    mu = _time_weighted_mean(times_s, values)
    n = len(values)
    if n < 2 or not np.isfinite(mu):
        return float("nan")
    total = float(times_s[-1] - times_s[0])
    if total <= 0:
        return float("nan")
    var_acc = 0.0
    for i in range(n-1):
        dt = times_s[i+1] - times_s[i]
        if dt <= 0:
            continue
        diff = values.iloc[i] - mu
        var_acc += (diff * diff) * dt
    var = var_acc / total
    return float(np.sqrt(var))

def _contiguous_zero_pauses(times_s: np.ndarray, values: np.ndarray, min_pause_s: float) -> List[Tuple[float,float]]:
    #Detecta pausas donde CPM == 0 de forma contigua y devuelve solo las que duran al menos min_pause_s (10s).
    segs = _segments_by_condition(times_s, values, predicate=lambda v: v == 0)
    long_segs = [(a,b) for (a,b) in segs if (b-a) >= min_pause_s]
    return long_segs

def _first_time_condition(times_s: np.ndarray, values: np.ndarray, predicate) -> Optional[float]:
    """ Encuentra el primer instante (en segundos relativos al inicio, es decir, t - t0) 
    en el que se cumple una condición sobre el valor (ej.: “primera compresión” ⇒ CPM>0).
    """
    n = len(values)
    if n < 2:
        return None
    for i in range(n-1):
        if predicate(values[i]):
            return float(times_s[i] - times_s[0])
    return None

def _sustained_over_threshold(times_s: np.ndarray, values: np.ndarray, thr: float, min_duration_s: float) -> List[Tuple[float,float]]: #""" Detecta segmentos donde el valor está estrictamente por encima de un umbral thr durante al menos min_duration_s. --> pq? En el script se usa para “frecuencia sostenida alta"""
    segs = _segments_by_condition(times_s, values, predicate=lambda v: v > thr)
    long = [(a,b) for (a,b) in segs if (b-a) >= min_duration_s]
    return long

# --------------------------
# Main computation
# --------------------------

def compute_metrics_from_cpm(df: pd.DataFrame,
                             pause_threshold_s: float = 10.0,
                             sustained_high_thr: float = 130.0,
                             sustained_high_min_s: float = 10.0) -> Dict[str, Any]:
    ts_s, detected_time_col = _detect_time_column(df)
    cpm = _require_cpm(df)
    #ignorar los primeros 9s que no calcula todavia cpm
    #mask = ts_s - ts_s.iloc[0] >= 9.0
   # ts_s = ts_s[mask].reset_index(drop=True)
    #cpm = cpm[mask].reset_index(drop=True)

    # Ensure monotonic time
    order = np.argsort(ts_s.values)
    ts_s = ts_s.iloc[order].reset_index(drop=True)
    cpm = cpm.iloc[order].reset_index(drop=True)

    n = len(cpm)
    duration_s = float(ts_s.iloc[-1]) if n >= 2 else 0.0
    
    #duration_s=n/100
    # Time-weighted stats (preferred when sampling step not uniform)
    mean_cpm = _time_weighted_mean(ts_s.values, cpm)
    std_cpm = _time_weighted_std(ts_s.values, cpm)
    # Median using sample values (not time-weighted) as a robust quick stat
    median_cpm = float(np.median(cpm.values)) if n > 0 else float("nan")

    in_target_pct = _time_weighted_percent(ts_s.values, cpm.values, 100.0, 120.0)
    time_with_comp_s = 0.0
    if n >= 2:
        for i in range(n-1):
            dt = ts_s.iloc[i+1]/2 - ts_s.iloc[i]/2  #me marca como si la frecuencia estuviera en 50 - no se porque por eso divido por dos
            if dt > 0 and cpm.iloc[i] > 0:
                time_with_comp_s += dt
    compression_fraction_pct = 100.0 * time_with_comp_s / duration_s if duration_s > 0 else 0.0
    hands_off_time_s = duration_s - time_with_comp_s if duration_s > 0 else 0.0

    # Pauses (CPM == 0) longer than threshold
    long_pauses = _contiguous_zero_pauses(ts_s.values, cpm.values, min_pause_s=pause_threshold_s)
    long_pauses_durations = [round(b-a, 3) for (a,b) in long_pauses]
    # Top-k (3) by duration
    topk = sorted(long_pauses, key=lambda ab: (ab[1]-ab[0]), reverse=True)[:3]
    topk_list = [{"start_s": round(a - ts_s.iloc[0], 3), "end_s": round(b - ts_s.iloc[0], 3), "duration_s": round(b-a, 3)} for (a,b) in topk]

    # Time to first compression (first CPM > 0)
    t_first_comp = _first_time_condition(ts_s.values, cpm.values, predicate=lambda v: v > 0)

    # Sustained high rate > 130 CPM for >= 10 s
    high_rate_segs = _sustained_over_threshold(ts_s.values, cpm.values, thr=sustained_high_thr, min_duration_s=sustained_high_min_s)
    high_rate_list = [{"start_s": round(a - ts_s.iloc[0], 3), "end_s": round(b - ts_s.iloc[0], 3), "duration_s": round(b-a, 3)} for (a,b) in high_rate_segs]

    results = {
        "session": {
            "samples": int(n),
            "time_column": detected_time_col,
            "duration_s": round(duration_s, 3),
            "sampling_note": "Assume step-hold between samples; time-weighted stats used."
        },
        "cpr_quality": {
            "rate_bpm": {
                "mean": round(mean_cpm, 3) if np.isfinite(mean_cpm) else None,
                "median": round(median_cpm, 3) if np.isfinite(median_cpm) else None,
                "std": round(std_cpm, 3) if np.isfinite(std_cpm) else None,
                "in_target_pct_100_120": round(in_target_pct, 3)
            },
            "compression_fraction_pct": round(compression_fraction_pct, 3),
            "hands_off_time_s": round(hands_off_time_s, 3),
            "pauses_over_threshold_s": {
                "threshold_s": pause_threshold_s,
                "count": len(long_pauses),
                "top3_by_duration": topk_list
            },
            "time_to_first_compression_s": t_first_comp
        },
        "alarms": {
            "sustained_high_rate": {
                "threshold_cpm": sustained_high_thr,
                "min_duration_s": sustained_high_min_s,
                "segments": high_rate_list
            }
        },
        "not_computable_from_cpm_only": [
            "depth_metrics",
            "recoil_complete_pct",
            "ventilation_rate_or_ratio",
            "time_to_first_shock"
        ],
        "reference_ranges_adult": {
            "rate_bpm_target": "100-120/min",
            "compression_fraction_target_pct": "> 60% (orientativo)"
        }
    }
    return results

def write_markdown_summary(results: Dict[str, Any]) -> str:
    s = results["session"]
    q = results["cpr_quality"]
    alarms = results["alarms"]

    md = []
    md.append("# Resumen de calidad de RCP (CPM-only)")
    md.append("")
    md.append(f"- Duración: **{s['duration_s']} s** (muestras: {s['samples']})")
    md.append(f"- Franja objetivo (adulto): **100–120/min**")
    md.append("")
    rbpm = q["rate_bpm"]
    md.append("## Frecuencia de compresiones (bpm)")
    md.append(f"- Media: **{rbpm['mean']}**")
    md.append(f"- Mediana: **{rbpm['median']}**")
    md.append(f"- Desvío: **{rbpm['std']}**")
    md.append(f"- % dentro de 100–120: **{rbpm['in_target_pct_100_120']}%**")
    md.append("")
    md.append("## Fractions & Pausas")
    md.append(f"- Compression Fraction: **{q['compression_fraction_pct']}%** (objetivo > 60%)")
    md.append(f"- Hands-off total: **{q['hands_off_time_s']} s**")
    md.append(f"- Pausas ≥ {q['pauses_over_threshold_s']['threshold_s']} s: **{q['pauses_over_threshold_s']['count']}**")
    if q['pauses_over_threshold_s']['top3_by_duration']:
        md.append("  - Top 3 pausas (inicio–fin, dur):")
        for p in q['pauses_over_threshold_s']['top3_by_duration']:
            md.append(f"    - {p['start_s']}–{p['end_s']} s, dur **{p['duration_s']} s**")
    md.append("")
    md.append("## Hitos")
    t1 = q["time_to_first_compression_s"]
    md.append(f"- Tiempo a primera compresión: **{t1 if t1 is not None else 'N/D'} s**")
    md.append("")
    md.append("## Alarmas")
    segs = alarms["sustained_high_rate"]["segments"]
    if segs:
        md.append(f"- Frecuencia sostenida > {alarms['sustained_high_rate']['threshold_cpm']} cpm por ≥ {alarms['sustained_high_rate']['min_duration_s']} s: **{len(segs)} segmentos**")
        for s in segs:
            md.append(f"  - {s['start_s']}–{s['end_s']} s, dur **{s['duration_s']} s**")
    else:
        md.append("- Sin segmentos de frecuencia sostenida alta detectados.")
    md.append("")
    md.append("## No calculable con CPM solamente")
    md.append("- Profundidad, recoil, ventilaciones, primer choque: **no disponibles** en este flujo.")
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="Compute CPR metrics from CPM-only CSV.")
    parser.add_argument("--csv", required=True, help="Ruta al CSV con columnas: cpm y timestamp_s OR timestamp/time.")
    parser.add_argument("--out", required=False, default="/mnt/data/metrics.json", help="Ruta de salida del JSON.")
    parser.add_argument("--md", required=False, default="/mnt/data/metrics_summary.md", help="Ruta de salida del resumen Markdown.")
    parser.add_argument("--pause_threshold_s", type=float, default=10.0, help="Umbral de pausa (s) para conteo/top-k.")
    parser.add_argument("--sustained_high_thr", type=float, default=130.0, help="CPM para alarma de sostenido alto.")
    parser.add_argument("--sustained_high_min_s", type=float, default=10.0, help="Duración mínima (s) de sostenido alto.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    results = compute_metrics_from_cpm(
        df,
        pause_threshold_s=args.pause_threshold_s,
        sustained_high_thr=args.sustained_high_thr,
        sustained_high_min_s=args.sustained_high_min_s,
    )
    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Write Markdown summary
    md_path = Path(args.md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(write_markdown_summary(results))

    print(f"Wrote JSON to: {out_path}")
    print(f"Wrote Markdown to: {md_path}")

if __name__ == "__main__":
    main()
