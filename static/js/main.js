const ctx = document.getElementById('grafico').getContext('2d');
const config = {
  type: 'line',
  data: { labels: [], datasets: [{ label:'Aceleración Z (m/s²)', data:[], borderColor:'#6b63bf', fill:false, tension:0.1 }] },
  options: {
    responsive:true, animation:false,
    scales:{ y:{ min:-20,max:20, title:{display:true,text:'m/s²'} },
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
setInterval(actualizarDatos, 100);

// === stats (iguales, pero pintamos métricas y resumen) ===
async function actualizarStats(){
  const r = await fetch('/stats'); const j = await r.json();
  document.getElementById('n_comp').textContent = j.n_comp;
  document.getElementById('cpm').textContent    = j.cpm.toFixed(1);
  document.getElementById('metric-cpm').textContent = j.cpm.toFixed(0);
  document.getElementById('prof_cm').textContent = j.prof_cm.toFixed(1);
  document.getElementById('metric-prof').textContent = j.prof_cm.toFixed(1);


  // Si más adelante calculás profundidad/tiempos reales, actualizá aquí:
  // document.getElementById('metric-prof').textContent = `${profundidad.toFixed(1)} cm`;
}
setInterval(actualizarStats, 300);
const reportBtn    = document.getElementById('btn-ver-reporte'); 
const reportTabBtn = document.getElementById('tab-reporte-tab'); 
const reportFrame  = document.getElementById('report-frame');    

function reloadReportFrame(){
  if (!reportFrame) return;
  if (reportFrame.contentWindow) reportFrame.contentWindow.location.reload();
  else reportFrame.src = reportFrame.src;
}

function showReportTabAndReload(){
  if (typeof bootstrap !== 'undefined' && reportTabBtn) {
    const tab = new bootstrap.Tab(reportTabBtn);
    tab.show();
  }
  reloadReportFrame();
}


function toggleReporte(enable){
  if (!reportBtn) return;
  reportBtn.disabled = !enable;
  reportBtn.onclick = enable
    ? () => showReportTabAndReload()  
    : null;
}

// === controles ===
async function iniciar(){
  const comentario = document.getElementById("comentario").value;
  await fetch('/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ comentario })
  });
  startTimerRCP({reset:true});
  toggleDescarga(false);
  toggleReporte(false);
}
async function detener(){
  const stopBtn = document.querySelector('.btn.btn-soft[onclick="detener()"]') 
                  || document.getElementById('btn-stop'); // por si luego le ponés id
  try {
    if (stopBtn) { stopBtn.disabled = true; stopBtn.textContent = 'Deteniendo...'; }


    const res = await fetch('/stop', { method: 'POST' });
    const ct  = res.headers.get('content-type') || '';
    const data = ct.includes('application/json') ? await res.json() : null;

    stopTimerRCP();
    toggleDescarga(true);

    const success = res.ok && (!data || data.ok === true);
    if (!success) {
      const msg = (data && (data.error || data.message)) || res.statusText;
      alert('No se pudo generar el reporte.\n' + msg);
      return;
    }

 
    toggleReporte(true);
    

  } catch (err) {
    alert('Error de red al detener: ' + err.message);
  } finally {
    if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = '⏹ Stop'; }
  }
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
  if (rcpStartTs){
    rcpAccumulated += (Date.now() - rcpStartTs);
    rcpStartTs = null;
  }
  clearInterval(rcpTimerInterval);
  rcpTimerInterval = null;
  renderRcpTime(rcpAccumulated);
}


