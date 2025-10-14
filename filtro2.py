
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences
from flask import Flask, Blueprint

# =============================================================
# Configuración general
# =============================================================
app = Flask(__name__)

# Frecuencia de muestreo
fs = 100  # Hz

# Ventaneo para el cómputo de compresiones
WIN_S = 10            # tamaño de ventana (segundos)
HOP_S = 2             # avance/actualización (segundos)
WIN_N = WIN_S * fs
HOP_N = HOP_S * fs

# Conversión (aprox) de amplitud a profundidad (cm por m/s^2) – ajustar a tu calibración
CAL_CM_PER_MPS2 = 2.5

# Estado interno para actualizar cada 2 s (sin recalcular en cada llamado)
_last_hop = -1
_last_result = (0, 0.0, 0.0)  # (n_comp, cpm, profundidad_cm)

# =============================================================
# Filtro Pan–Tompkins adaptado para picos "hacia arriba"
# =============================================================
def bandpass(signal, fs, f1=0.5, f2=8.0, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_avg(x, w):
    # ventana simétrica para no desplazar fase
    k = int(max(1, w))
    win = np.ones(k)/k
    return np.convolve(x, win, mode='same')

def pan_tompkins_like(z, fs,
                      bp_lo=0.5, bp_hi=8.0,
                      der_win=1,              # derivada simple (opcional suavizada)
                      integ_win_sec=0.15,     # 120–200 ms típico
                      refractory_sec=0.30,    # evita dobles picos (CPR ~0.5–0.6 s periodo)
                      height_q=65,            # percentil para umbral inicial
                      prominence=0.0):
    """
    z: señal 1D (aceleración Z)
    fs: frecuencia de muestreo (Hz)
    Devuelve: (idx_peaks, traces)
      - idx_peaks: índices (enteros) de picos confirmados en z
      - traces: dict con etapas intermedias (zf, dz, s2, yi, thr)
    """
    z = np.asarray(z, dtype=float)

    # 1) Pasa-banda
    zf = bandpass(z, fs, f1=bp_lo, f2=bp_hi, order=3)

    # 2) Derivada (centrada para no desplazar)
    dz = np.gradient(zf) if der_win == 1 else np.gradient(moving_avg(zf, der_win))

    # 3) Cuadrado
    s2 = dz**2

    # 4) Integración por ventana móvil
    integ_win = int(np.round(integ_win_sec*fs))
    yi = moving_avg(s2, max(1, integ_win))

    # 5) Umbral adaptativo (EMA)
    base = np.percentile(yi, height_q)
    alpha = 0.01
    thr_trace = np.empty_like(yi)
    thr = base
    for i, v in enumerate(yi):
        thr = (1-alpha)*thr + alpha*v
        thr_trace[i] = max(thr, base)

    # Candidatos a pico en la señal integrada
    distance = int(refractory_sec*fs)
    cand, _ = find_peaks(yi, height=thr_trace, distance=max(1, distance), prominence=prominence)

    # Confirmar como picos "hacia arriba" en zf: máximo local en vecindario
    confirmed = []
    rad = max(1, int(0.05*fs))  # ±50 ms vecindario
    for c in cand:
        i0 = max(0, c-rad); i1 = min(len(zf), c+rad+1)
        loc = i0 + np.argmax(zf[i0:i1])
        if len(confirmed) == 0 or (loc - confirmed[-1]) >= distance:
            confirmed.append(loc)

    peaks = np.array(confirmed, dtype=int)
    traces = {"zf": zf, "dz": dz, "s2": s2, "yi": yi, "thr": thr_trace}
    return peaks, traces

# =============================================================
# Utilitarios de ventaneo (10 s con salto de 2 s)
# =============================================================
def _get_window_indices(total_len):
    """
    Devuelve (start, end, hop_idx) para la *última* ventana completa al ritmo de HOP_N.
    Si no hay ventana completa disponible, devuelve (None, None, -1).
    """
    if total_len < WIN_N:
        return None, None, -1
    hop_idx = (total_len - WIN_N) // HOP_N
    end = WIN_N + hop_idx * HOP_N
    start = end - WIN_N
    return start, end, hop_idx

# =============================================================
# API pública para la app (compatibles con tu interfaz)
# =============================================================
def compute_counts_depth(arr):
    """
    Entrada:
      - arr: lista/np.array con TODA la serie hasta el momento (aceleración Z).
    Salida:
      - (n_comp, cpm, profundidad_cm) evaluados en la última ventana de 10 s.
        * El primer resultado aparece recién cuando hay 10 s.
        * Luego se actualiza cada 2 s (si no toca actualización, repite el último resultado).
    """
    global _last_hop, _last_result

    z = np.asarray(arr, dtype=float)
    start, end, hop_idx = _get_window_indices(len(z))

    # Aún no hay ventana completa
    if hop_idx == -1:
        return _last_result if _last_hop != -1 else (0, 0.0, 0.0)

    # Si no cambió el hop, devolvemos lo último (evita recomputar cada llamada)
    if hop_idx == _last_hop:
        return _last_result

    # Ventana nueva -> recalcular
    win = z[start:end]

    peaks, traces = pan_tompkins_like(
        win, fs=fs,
        bp_lo=0.5, bp_hi=8.0,
        integ_win_sec=0.15,
        refractory_sec=0.30,
        height_q=65
    )

    n_comp = int(peaks.size)
    cpm = float((n_comp / WIN_S) * 60.0)

    # Profundidad aproximada: prominencia de picos en zf
    profundidad_cm = 0.0
    if n_comp > 0:
        zf = traces["zf"]
        prom, _, _ = peak_prominences(zf, peaks)
        if prom.size > 0:
            k = min(3, prom.size)
            prom_media = float(np.mean(prom[-k:]))
            profundidad_cm = max(0.0, prom_media * CAL_CM_PER_MPS2)
            # limitamos a un rango razonable (opcional)
            profundidad_cm = float(min(profundidad_cm, 8.0))

    _last_hop = hop_idx
    _last_result = (n_comp, round(cpm, 1), round(profundidad_cm, 1))
    return _last_result

def compute_counts(arr):
    """Wrapper para compatibilidad: devuelve solo (n_comp, cpm)."""
    n, cpm, _ = compute_counts_depth(arr)
    return n, cpm

# Mantengo el blueprint si lo estás importando en tu app
stats_bp = Blueprint('stats_bp', __name__)
