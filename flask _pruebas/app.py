from flask import Flask, request, render_template, jsonify, send_file
import csv
import os
import io
import filtro as ft

from filtro import stats_bp

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
        writer.writerow(["Índice", "Aceleracion_Z", "Comentario"])

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
    return "Guardado iniciado"

@app.route("/stop")
def stop():
    global guardando
    guardando = False
    return "Guardado detenido"

@app.route("/datos")
def datos_json():
    return jsonify({
        "valores": datos_z,
        "indices": list(range(len(datos_z)))
    })

# ¡OJO! Evitar el decorador duplicado de /esp32
@app.route("/esp32", methods=["POST"])
def recibir_datos():
    global datos_z, guardando, comentario_actual

    contenido = request.data.decode("utf-8").strip()
    try:
        nuevas_medidas = [float(linea) for linea in contenido.splitlines() if linea]
    except ValueError:
        return "CPM:0,PROF:0.0"

    datos_z.extend(nuevas_medidas)

    # Guardado opcional a CSV
    if guardando:
        import csv
        with open(archivo_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            start_index = len(datos_z) - len(nuevas_medidas)
            for i, valor in enumerate(nuevas_medidas):
                writer.writerow([start_index + i, valor, comentario_actual])

    # Limitar memoria (últimos 10 s como mínimo: 1000 muestras). Podés dejar 2000.
    if len(datos_z) > 2000:
        datos_z = datos_z[-2000:]

    # === NUEVO: calcular métricas acumuladas con tu filtro mejorado ===
    try:
        n_comp, cpm, prof_cm = ft.compute_counts_depth(datos_z)
    except Exception as e:
        # ante cualquier error del filtro, no rompas el protocolo
        n_comp, cpm, prof_cm = 0, 0.0, 0.0

    # Responder en el formato que espera la ESP32
    return f"CPM:{int(round(cpm))},PROF:{prof_cm:.1f}"

@app.route("/descargar")
def descargar_csv():
    global datos_z

    # Leer contenido actual del archivo CSV
    with open(archivo_csv, "r") as f:
        contenido = f.read()

    # Vaciar la variable en memoria
    datos_z = []

    # Reiniciar CSV con encabezados
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Índice", "Aceleracion_Z", "Comentario"])

    # Devolver como archivo descargable
    return send_file(
        io.BytesIO(contenido.encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="datos_pruebas_profundidad.csv"
    )

@app.route('/stats')
def stats():
    arr = datos_z
    print(f"[DEBUG stats] len(datos_z) = {len(arr)}, últimos 5 = {arr[-5:]}")
    n_comp, cpm = ft.compute_counts(arr)
    print(f"[DEBUG stats] detectados = {n_comp}, cpm = {cpm}")
    return jsonify({'n_comp': n_comp, 'cpm': cpm})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

