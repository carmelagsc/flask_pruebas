import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
from flask import Flask, Blueprint, jsonify

app = Flask(__name__)

# Parámetros
# === Parámetros base ===
fs = 100                    # Hz de muestreo
WIN_S = 10                  # ventana acumulada (segundos)
window_size = WIN_S * fs

# Banda típica compresión (ajustable)
LOWCUT, HIGHCUT = 0.5, 5.0
ORDER = 2

# Anti-rebote y robustez
MIN_DIST_S = 0.35           # no contar > ~171 CPM
SMOOTH_N  = 3               # suavizado ligero
PROM_K    = 0.6             # prominencia = PROM_K * std local

# Calibración de profundidad (m/s² -> cm)
CAL_CM_PER_MPS2 = 2.5

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band')

def _prep_window(arr):
    L = len(arr)
    if L < window_size:
        return None
    w = np.array(arr[-window_size:], dtype=float)
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, order=ORDER)
    sig = filtfilt(b, a, w)                 # zero-phase
    sig = uniform_filter1d(sig, size=SMOOTH_N)
    return sig

def compute_counts_depth(arr):
    sig = _prep_window(arr)
    if sig is None:
        return 0, 0.0, 0.0

    # Invertimos para buscar mínimos como picos positivos
    sig_inv = -sig

    # Umbrales/anti-rebote
    min_dist = int(MIN_DIST_S * fs)
    prom = PROM_K * np.std(sig_inv)  # robusto al nivel local
    if prom <= 0:
        prom = 1e-6

    peaks, props = find_peaks(sig_inv, distance=min_dist, prominence=prom)
    n_comp = int(len(peaks))

    # CPM acumulado en WIN_S
    cpm = (n_comp / WIN_S) * 60.0

    # Profundidad: convertimos prominencias a cm (media de últimos 1-3 picos)
    profundidad_cm = 0.0
    if n_comp > 0 and "prominences" in props:
        prominences = props["prominences"]
        # tomar últimos hasta 3 picos
        k = min(3, len(prominences))
        prom_media = np.mean(prominences[-k:])
        profundidad_cm = max(0.0, float(prom_media) * CAL_CM_PER_MPS2)
        # recorte razonable (opcional)
        profundidad_cm = min(profundidad_cm, 8.0)

    return n_comp, round(cpm, 1), round(profundidad_cm, 1)

# --- wrapper de compatibilidad con app.py ---
def compute_counts(arr):
    """
    Wrapper para mantener compatibilidad:
    devuelve (n_comp, cpm) tomando de compute_counts_depth
    """
    n, cpm, _prof = compute_counts_depth(arr)
    return n, cpm

# Mantengo el blueprint si lo querés usar
stats_bp = Blueprint('stats_bp', __name__)