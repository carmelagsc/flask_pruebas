import serial
import time
import numpy as np
import csv
import os
from datetime import datetime, timezone
import filtro3 as ft # Importa tu lógica de procesamiento

# ==========================================================
# --- CONFIGURACIÓN (AJUSTAR) ---
# ==========================================================

# *** ¡IMPORTANTE! Reemplaza 'COM3' con el puerto al que se conecta el ESP32.
# En Linux/Mac: '/dev/ttyACM0', '/dev/ttyUSB0', etc.
# En Windows: 'COM3', 'COM4', etc.
SERIAL_PORT = 'COM7' 
BAUDRATE = 115200 # Debe coincidir con Cfg.UART_BAUDRATE en el ESP32
TERMINATOR = b'\n' # Terminador de línea (salto de línea)

# --- Estado de la Sesión y Archivo ---
archivo_csv = "datos_serial.csv"
guardando = True  

# Crear el CSV si no existe
if not os.path.exists(archivo_csv):
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "Aceleración", "cpm", "prof_cm"])
        
# Reiniciar el flujo de procesamiento de la lógica de filtro3.py
try:
    ft.reset_stream()
except Exception:
    pass

# ==========================================================
# --- Bucle Principal del Servidor Serie ---
# ==========================================================

def run_serial_server():
    """Bucle principal que escucha el puerto serie, procesa y responde."""
    
    print(f"Iniciando Servidor Serial en {SERIAL_PORT}@{BAUDRATE}...")
    
    try:
        # Inicializar el puerto serial
        ser = serial.Serial(
            port=SERIAL_PORT, 
            baudrate=BAUDRATE, 
            timeout=0.1,  # Timeout bajo para lectura
            write_timeout=0.1 # Timeout bajo para escritura
        )
        ser.reset_input_buffer()
        print("✅ Puerto serial inicializado. Esperando datos del ESP32...")
        
    except serial.SerialException as e:
        print(f"❌ ERROR: No se pudo abrir el puerto serial {SERIAL_PORT}.")
        print(f"Verifica la conexión USB y que el puerto no esté en uso.")
        print(f"Detalle del error: {e}")
        return

    # Tiempo de referencia para calcular el timestamp_s
    start_time = time.time() 
    
    while True:
        try:
            # 1. Leer datos hasta encontrar el terminador (b'\n')
            if ser.in_waiting > 0:
                # Leer hasta el terminador
                line = ser.read_until(TERMINATOR) 
                
                # 2. Decodificar y convertir a lista de floats
                data_str = line.decode('utf-8').strip()
                data_list = data_str.split(',')
                data_z_float = [float(x) for x in data_list if x] 
                
                if not data_z_float:
                    continue 
                
                # 3. Procesamiento de Datos (Llama a filtro3.py)
                arr_z = np.array(data_z_float)
                
                # Obtener métricas
                metrics = ft.update_stream(arr_z)
                
                # 4. Preparar Respuesta para el ESP32
                cpm = metrics.get("cpm", 0)
                profundidad_cm = metrics.get("depth_cm", 0.0)
                
                # Formato: "CPM:120,PROF:4.5\n"
                response_text = f"CPM:{cpm:.1f},PROF:{profundidad_cm:.1f}\n"
                
                # 5. Enviar Respuesta al ESP32
                ser.write(response_text.encode('utf-8'))
                
                print(f"-> Proceso OK. CPM:{cpm:.1f}, PROF:{profundidad_cm:.1f}")
                
                # 6. Guardar en CSV
                if guardando:
                    with open(archivo_csv, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        current_time = time.time()
                        timestamp_s_end = current_time - start_time
                        
                        # Asumiendo que la frecuencia de muestreo es 100 Hz (0.01s)
                        sample_rate_s = 0.01 
                        
                        for i, z in enumerate(data_z_float):
                            # Calcular el timestamp de cada muestra
                            ts_muestra = timestamp_s_end - (len(data_z_float) - 1 - i) * sample_rate_s 
                            writer.writerow([f"{ts_muestra:.3f}", f"{z:.3f}", f"{cpm:.1f}", f"{profundidad_cm:.1f}"])
                
        except serial.SerialTimeoutException:
            pass # Continúa el bucle si no hay datos
        except Exception as e:
            # Manejo de errores graves de procesamiento o lectura
            print(f"\n❌ Error durante el bucle: {e}")
            ser.reset_input_buffer()
            time.sleep(1) 

        # Pequeña pausa
        time.sleep(0.001)


if __name__ == "__main__":
    try:
        run_serial_server()
    except KeyboardInterrupt:
        print("\nServidor Serial detenido por el usuario.")
    finally:
        # Asegurarse de que el puerto se cierre
        try:
            # Esta línea se ejecuta al salir del bucle (Ctrl+C)
            pass 
        except:
            pass