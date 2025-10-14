// === chart (igual que antes) ===
const ctx = document.getElementById('grafico').getContext('2d');
const config = {
  type: 'line',
  data: { labels: [], datasets: [{ label:'Aceleración Z (m/s²)', data:[], borderColor:'#6b63bf', fill:false, tension:0.1 }] },
  options: {
    responsive:true, animation:false,
    scales:{ y:{ min:-40,max:40, title:{display:true,text:'m/s²'} },
             x:{ min:0, max:1000, title:{display:true,text:'Índice de muestra'} } }
  }
};
const grafico = new Chart(ctx, config);

async function actualizarDatos(){
  const r = await fetch('/datos'); const j = await r.json();
  config.data.labels = j.indices;
  config.data.datasets[0].data = j.valores;
  grafico.update();
}
setInterval(actualizarDatos, 200);

// === stats (iguales, pero pintamos métricas y resumen) ===
async function actualizarStats(){
  const r = await fetch('/stats'); const j = await r.json();
  document.getElementById('n_comp').textContent = j.n_comp;
  document.getElementById('cpm').textContent    = j.cpm.toFixed(1);

  // Métricas grandes
  document.getElementById('metric-cpm').textContent = j.cpm.toFixed(0);
  // Si más adelante calculás profundidad/tiempos reales, actualizá aquí:
  // document.getElementById('metric-prof').textContent = `${profundidad.toFixed(1)} cm`;
}
setInterval(actualizarStats, 500);

// === controles ===
async function iniciar(){
  const comentario = document.getElementById("comentario").value;
  await fetch('/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ comentario })
  });
  startTimerRCP({reset:true});
  toggleDescarga(false);
}
async function detener(){
  await fetch('/stop');
  stopTimerRCP();
  toggleDescarga(true);
}
function toggleDescarga(enable){
  const btn = document.getElementById('btn-descargar');
  if(enable){ btn.classList.remove('disabled'); }
  else{ btn.classList.add('disabled'); }
}

// ==== TIMER RCP ====
let rcpTimerInterval = null;
let rcpStartTs = null;     // timestamp cuando empezó/continuó
let rcpAccumulated = 0;    // ms acumulados (para reanudar si hiciera falta)

function fmtMMSS(ms){
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function renderRcpTime(ms){
  const el = document.getElementById('metric-t_rcp');
  if (el) el.textContent = fmtMMSS(ms);
}

function startTimerRCP({reset=true} = {}){
  if (reset){ rcpAccumulated = 0; }
  rcpStartTs = Date.now();
  clearInterval(rcpTimerInterval);
  rcpTimerInterval = setInterval(() => {
    const elapsed = rcpAccumulated + (Date.now() - rcpStartTs);
    renderRcpTime(elapsed);
  }, 200);
}

function stopTimerRCP(){
  // acumulo lo transcurrido y freno el intervalo
  if (rcpStartTs){
    rcpAccumulated += (Date.now() - rcpStartTs);
    rcpStartTs = null;
  }
  clearInterval(rcpTimerInterval);
  rcpTimerInterval = null;
  renderRcpTime(rcpAccumulated); // pintar valor final
}
