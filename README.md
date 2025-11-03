                    ┌─────────────────────────────┐
                    │        Dispositivo ESP32     │
                    │  - Aceleración (muestras)    │
                    │  - Calcula CPM/PROF          │
                    │  - Wi-Fi HTTP POST           │
                    └──────────────┬───────────────┘
                                   │  POST /esp32 (muestras)
                                   │  ← respuesta: "CPM,PROF"
                                   ▼
        ┌──────────────────────────────────────────────────────────┐
        │                   Servidor Flask (PC)                    │
        │──────────────────────────────────────────────────────────│
        │  Rutas:                                                  │
        │   /start(POST)  /stop(POST)  /esp32(POST)                │
        │   /metrics(JSON) /stats(JSON) /datos(JSON)               │
        │   /reporte(HTML) /descargar(CSV) /info /guia             │
        │   /api/device-info (estado equipo)                       │
        │                                                          │
        │  Submódulos internos:                                    │
        │   ┌────────────────────┐   ┌──────────────────────────┐  │
        │   │ Session/State      │   │  Buffer + Almacenamiento │  │
        │   │ - flags de sesión  │   │ - datos_z (cap ~2000)    │  │
        │   │ - duración, cpm    │   │ - fs = 100 Hz            │  │
        │   │ - device info API  │   │ - CSV: indice,t,acc,cpm  │  │
        │   └────────────────────┘   └──────────────────────────┘  │
        │           │                         │                    │
        │           │                         ▼                    │
        │   ┌────────────────────┐   ┌──────────────────────────┐  │
        │   │ Filtro (ft)        │   │ Analyzer (cpr_metrics)   │  │
        │   │ - compute_counts   │   │ - compresiones totales   │  │
        │   │ - compute_depth    │   │ - CPM medio/mediana      │  │
        │   │ - robustez errores │   │ - pausas, CCF, % en rango│  │
        │   └────────────────────┘   └──────────────────────────┘  │
        │                                                          │
        │  Plantillas y estáticos: /templates (HTML), /static (JS,CSS)
        └──────────────┬───────────────────────────────────────────┘
                       │
         (HTTP GET / AJAX desde GUI)                               
                       ▼
          ┌────────────────────────────────────────┐
          │        Interfaz Web (Navegador)        │
          │  - (/) Tabs: Data / Reporte / Info     │
          │  - AJAX: /metrics y /stats             │
          │  - Botones: POST /start, POST /stop    │
          │  - Ver /reporte y bajar /descargar     │
          └────────────────────────────────────────┘
