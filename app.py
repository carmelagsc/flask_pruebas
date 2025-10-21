from flask import Flask, request, render_template, jsonify, send_file
import csv
import os
import io
import filtro3 as ft
import pandas as pd
from datetime import datetime, timezone
import time

from filtro3 import stats_bp
from cpr_metrics import compute_metrics_from_cpm  
import json

app = Flask(__name__)
app.register_blueprint(stats_bp)

datos_z = []
archivo_csv = "datos.csv"
guardando = False  # Estado de adquisición
comentario_actual = ""

# Crear el CSV si no existe (con las 3 columnas que luego escribes)
if not os.path.exists(archivo_csv):
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "timestamp_s", "Aceleración", "cpm"])

@app.route("/")
def mostrar_grafico():
    # Renderiza plantilla con pestañas
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global guardando, datos_z, comentario_actual
    datos_z = []
    guardando = True
    comentario_actual = request.json.get("comentario", "").strip()

    # ⬇️ Reiniciar estados del filtro/streaming/calibración
    try:
        ft.reset_stream()
    except Exception:
        pass

    # Reiniciar CSV con encabezados
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "timestamp_s", "Aceleración", "cpm"])

    return "Guardado iniciado"

@app.route("/stop", methods=["POST"])
def stop_recording():
    global guardando
    guardando = False  # 1) frena la captura

    try:
        results = compute_metrics_from_cpm(
            df = pd.read_csv( archivo_csv, encoding="latin1"),
            pause_threshold_s=10.0,
            sustained_high_thr=130.0,
            sustained_high_min_s=10.0,
        )

        return jsonify({"ok": True, "samples": results["session"]["samples"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/datos")
def datos_json():
    return jsonify({
        "valores": datos_z,
        "indices": list(range(len(datos_z)))
    })

@app.route("/esp32", methods=["POST"])
def recibir_datos():
    global datos_z, guardando, comentario_actual

    contenido = request.data.decode("utf-8").strip()
    try:
        nuevas_medidas = [float(linea) for linea in contenido.splitlines() if linea]
    except ValueError:
        return "CPM:0,PROF:0.0"

    # Agregamos primero al buffer en memoria
    start_index = len(datos_z)
    datos_z.extend(nuevas_medidas)

    # 2) ⬇️ Procesar SOLO este lote con el filtro causal (estado interno + calibración 5 cm)
    try:
        m = ft.update_stream(nuevas_medidas)
        n_comp = m["n_comp"]
        cpm    = m["cpm"]
        prof_cm = m["depth_cm"]        # ¡Ojo! ahora es relativo al baseline=5 cm
        # Extras disponibles si querés: m["pct_target"], m["hit_5cm"], m["calibrated"],
        #                               m["no_rcp_active"], m["no_rcp_time_s"]
    except Exception:
        n_comp, cpm, prof_cm = 0, 0.0, 0.0

    # 3) Guardado opcional a CSV (con timestamp y cpm)
    if guardando:
        fs_local = getattr(ft, "fs", 100)
        with open(archivo_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            for i, valor in enumerate(nuevas_medidas):
                indice = start_index + i
                timestamp_s = indice / fs_local
                writer.writerow([indice, f"{timestamp_s:.2f}", valor, f"{cpm:.1f}"])

    # 4) Limitar memoria RAM
    if len(datos_z) > 2000:
        datos_z = datos_z[-2000:]

    # 5) Responder a la ESP32 (protocolo intacto)
    return f"CPM:{int(round(cpm))},PROF:{prof_cm:.1f}"


@app.route("/descargar")
def descargar_csv():
    global datos_z
    global guardando
    guardando = False

    # Leer contenido actual del archivo CSV
    with open(archivo_csv, "r") as f:
        contenido = f.read()

    # Vaciar la variable en memoria
    datos_z = []

    # Reiniciar CSV con encabezados
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "timestamp_s", "Aceleración", "cpm"])


    # Devolver como archivo descargable
    return send_file(
        io.BytesIO(contenido.encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="datos_pruebas_profundidad.csv"
    )

@app.route("/cal_status")
def cal_status():
    try:
        m = ft.get_last_metrics()
        return jsonify({
            "calibrated": m.get("calibrated", False),
            "pct_target": m.get("pct_target", 0.0),     # 100% == 5 cm
            "hit_5cm":    m.get("hit_5cm", False),
            "depth_cm_rel": m.get("depth_cm", 0.0),     # relativo al baseline
            "no_rcp_active": m.get("no_rcp_active", False),
            "no_rcp_time_s": round(m.get("no_rcp_time_s", 0.0), 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats')
def stats():
    arr = datos_z
    m = ft.get_last_metrics()   # ya está procesando en /esp32
    return jsonify({'n_comp': m["n_comp"], 'cpm': m["cpm"]})


@app.route("/metrics")
def metrics_json():
    """Devuelve el JSON de métricas CPM-only computado desde datos.csv (salta 9 s iniciales)."""
    try:
        results = compute_metrics_from_cpm(df = pd.read_csv( archivo_csv, encoding="latin1"), pause_threshold_s=10.0,
                                           sustained_high_thr=130.0,
                                           sustained_high_min_s=10.0)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/reporte")
def reporte_html():
    """
    Renderiza el informe clásico en una pestaña HTML.
    Internamente lee /metrics y pasa 'metrics' al template.
    """
    try:
        results = compute_metrics_from_cpm(df = pd.read_csv( archivo_csv, encoding="latin1"), pause_threshold_s=10.0,
                                           sustained_high_thr=130.0,
                                           sustained_high_min_s=10.0)
        print(results)
        return render_template("reporte.html", metrics=results)
       

    except Exception as e:
        # Muestra el error en la UI si algo falla
        return render_template("reporte.html", metrics=None, error=str(e))

class State:
    device_name = "CP.Ar– v1.0"
    serial = "CP.Ar-00001"
    firmware = "1.3.2"

    # Sensores
    acc_ok = True

    # Sesión
    session_active = False
    session_start_ts = None
    session_compressions = 0
    session_cpm = None

    # Conexión/sync
    last_sync_ts = None

state = State()

def iso_utc(ts):
    if not ts:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()

def mmss_from_seconds(seconds):
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

# ---- Página de la pestaña "Información del equipo" ----
@app.route("/info")
def info_equipo():
    return render_template("info_equipo.html")

# ---- API sencilla que consume el HTML para poblar datos ----
@app.route("/api/device-info")
def api_device_info():
    if state.session_active and state.session_start_ts:
        elapsed = time.time() - state.session_start_ts
        sess_status = "Grabando"
    else:
        elapsed = None
        sess_status = "Detenido"

    # IP del servidor (LAN)
    server_ip = request.host.split(":")[0] if request.host else "—"

    payload = {
        "device_name": state.device_name,
        "serial": state.serial,
        "firmware": state.firmware,
        "sensors": [
            {"name": "Acelerómetro", "status": "OK" if state.acc_ok else "Error"}
        ],
        "session": {
            "status": sess_status,
            "duration": mmss_from_seconds(elapsed),
            "compressions": state.session_compressions,
            "cpm": state.session_cpm if state.session_cpm is not None else "—"
        },
        "connection": {
            "connected": True,   # si el server responde, mostrá conectado
            "ip": server_ip
        },
        "contact": "cp_ar@itba.edu"
    }
    return jsonify(payload)


@app.route("/guia")
def guia_html():
      return render_template("guia.html")




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)