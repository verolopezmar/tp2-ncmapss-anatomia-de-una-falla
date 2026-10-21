/* Anatomía de una Falla · Parte II — interacciones del informe.
   Basado en el script de compartir.html (Parte I). Todos los datos llegan en window.REPORT,
   generado por src/ncmapss/report/build.py a partir de outputs/<modo>/*.json. */
(() => {
'use strict';
/* ═══ 0 · DATOS Y UTILIDADES ═════════════════════════════════════════════════ */
const R = window.REPORT;
const $  = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d=0) => Number(v).toLocaleString('es-AR', {minimumFractionDigits:d, maximumFractionDigits:d});
const root = document.documentElement;
const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
const ALL = /[?&]all\b/.test(location.search);   // ?all = todo dibujado, sin animaciones (impresión/PDF)
const esc = s => String(s).replace(/"/g,'&quot;');

if (!R || !window.Plotly) {
  document.body.insertAdjacentHTML('afterbegin','<div class="missing">Faltan datos. Generá el informe con: python scripts/run_all.py --only report</div>');
  return;
}

/* ═══ 1 · CONFIGURACIÓN DE ILUSTRACIONES ═════════════════════════════════════ */
const COMP_VAR = {fan:'--c-fan', LPC:'--c-lpc', HPC:'--c-hpc', HPT:'--c-hpt', LPT:'--c-lpt'};
const COMP_ORDER = ['fan','LPC','HPC','HPT','LPT'];
const ENGINE_GEOM = [
  {c:'fan', step:'1', x:40,  y:40, w:70, h:150, r:12, label:'FAN'},
  {c:'LPC', step:'2', x:125, y:64, w:80, h:102, r:9,  label:'LPC'},
  {c:'HPC', step:'2', x:215, y:76, w:95, h:78,  r:9,  label:'HPC'},
  {c:'comb',step:'3', x:322, y:84, w:58, h:62,  r:28, label:'COMB.', burn:true},
  {c:'HPT', step:'4', x:392, y:78, w:70, h:74,  r:9,  label:'HPT'},
  {c:'LPT', step:'4', x:472, y:70, w:90, h:90,  r:9,  label:'LPT'},
];
const STATIONS = [
  {n:'2',  x:60,  name:'Entrada del fan', hot:false}, {n:'21', x:150, name:'Salida del fan, hacia el núcleo', hot:false},
  {n:'15', x:160, name:'Conducto de derivación (bypass)', hot:false, y:44}, {n:'24', x:295, name:'Salida del compresor de baja (LPC)', hot:false},
  {n:'30', x:435, name:'Salida del compresor de alta (HPC)', hot:false}, {n:'40', x:530, name:'Salida de la cámara de combustión', hot:true},
  {n:'48', x:625, name:'Salida de la turbina de alta (HPT)', hot:true}, {n:'50', x:760, name:'Salida de la turbina de baja (LPT)', hot:true},
];
const PLAN = [
  ['01','Dataset','Cómo está armado','#c1',false], ['02','Vuelo','Perfiles y clases','#c2',false],
  ['03','Contexto','Vuelo vs. desgaste','#c3',false], ['04','Desgaste','Qué se daña','#c4',true],
  ['05','Señales','Qué lo anticipa','#c5',true], ['06','Vida útil','Del EDA al modelo','#c6',true],
  ['07','Método','Sin hacer trampa','#c7',false], ['08','Resultados','Cuánto le queda','#c8',true],
  ['09','Incertidumbre','Con qué certeza','#c9',true], ['10','Líneas','Lo que viene','#c10',false],
];

function EngineSchematic(svg, {colored}) {
  const nacelle = `<ellipse cx="300" cy="226" rx="270" ry="8" fill="#0a2540" opacity=".2" filter="url(#blur6)"/>
    <path d="M20 40 Q40 22 92 26 L560 60 Q612 70 632 115 Q612 160 560 170 L92 204 Q40 208 20 190 Z" fill="url(#metal)" stroke="#a9b8c9" filter="url(#drop)"/>
    <path d="M32 38 Q50 30 92 32 L558 64" fill="none" stroke="#fff" stroke-width="2" opacity=".9"/>
    <path d="M10 115 H632" stroke="url(#gasPath)" stroke-width="2" class="flowdash" opacity=".7"/>`;
  const parts = ENGINE_GEOM.map(g => {
    const fill = g.burn ? 'url(#burn)' : colored ? `var(${COMP_VAR[g.c]})` : 'url(#metalPart)';
    const base = g.burn
      ? `<ellipse class="comp-base" cx="${g.x+g.w/2}" cy="${g.y+g.h/2}" rx="${g.w/2+6}" ry="${g.h/2}" fill="${fill}" filter="url(#blur6)"/>
         <ellipse cx="${g.x+g.w/2}" cy="${g.y+g.h/2}" rx="${g.w/3}" ry="${g.h/4}" fill="url(#burn)"/>`
      : `<rect class="comp-base" x="${g.x}" y="${g.y}" width="${g.w}" height="${g.h}" rx="${g.r}" fill="${fill}" stroke="rgba(10,37,64,.15)"/>
         <rect x="${g.x}" y="${g.y}" width="${g.w}" height="${g.h}" rx="${g.r}" fill="url(#sheen)" pointer-events="none"/>
         <rect x="${g.x+4}" y="${g.y+3}" width="${g.w-8}" height="2" rx="1" fill="#fff" opacity=".6" pointer-events="none"/>`;
    return `<g class="comp" data-c="${g.c}" data-s="${g.step}" style="--cf:var(${COMP_VAR[g.c]||'--amber'})">${base}
      <text class="comp-label" x="${g.x+g.w/2}" y="226" text-anchor="middle">${g.label}</text></g>`;
  }).join('');
  svg.innerHTML = nacelle + parts;
}

/* ═══ 2 · CONTENIDO GENERADO POR EL PIPELINE ═════════════════════════════════ */
function bindFacts() {
  $$('[data-kpi]').forEach(el => { const v = R.facts[el.dataset.kpi]; if (v !== undefined) el.textContent = v; });
  if ($('#genStamp')) $('#genStamp').textContent = `Generado por scripts/run_all.py el ${R.generated} · modo ${R.mode}`;
}
function renderKpis() {
  $('#kpis').innerHTML = R.kpis.map(k =>
    `<div class="kpi reveal" data-code="${k.code||''}"><div class="v" data-count="${k.value}" data-dec="${k.dec||0}"><span>0</span>${k.unit?`<small>${k.unit}</small>`:''}</div><div class="l">${k.label}</div><i class="bar"></i></div>`).join('');
}
function renderHierarchy() {
  const colors = ['var(--ti)','var(--hud)','var(--c-hpc)','var(--orange)','var(--red)'], sq = [1,9,24,48,80], px = [22,12,8,6,4];
  $('#hier').innerHTML = R.hierarchy.map((h,i) => `<div class="lvl" style="--lc:${colors[i]}">
    <div class="ix">NIVEL ${i+1}</div><div class="num">${h.n}</div><div class="nm">${h.name}</div><div class="ds">${h.desc}</div>
    <div class="sq">${Array.from({length:sq[i]}, () => `<i style="width:${px[i]}px;height:${px[i]}px"></i>`).join('')}</div></div>`).join('');
}
function renderExampleRow() {
  if (!R.example_row) return;
  const info = {A:['Auxiliares','#6a7e97'], W:['Condiciones de vuelo','#8d99ae'], X_s:['Sensores físicos','#0077b6'],
    X_v:['Sensores virtuales','#1d4ed8'], T:['Salud (θ)','#f3722c'], Y:['Objetivo','#d62828']};
  $('#exLabel').textContent = R.example_label;
  const by = {}; R.example_row.forEach(r => (by[r.block] = by[r.block] || []).push(r));
  $('#exRow').innerHTML = ['A','W','Y','X_s','X_v','T'].map(b => `<div class="row-block" style="--bc:${info[b][1]}">
    <div class="hd">${b} <span>· ${info[b][0]}</span></div>
    <table>${by[b].map(r => `<tr><td>${r.var}</td><td>${r.value}</td><td>${r.desc}</td></tr>`).join('')}</table></div>`).join('');
}
function renderVarGroups() {
  const accent = {W:'var(--ti)', X_s:'var(--hud)', X_v:'var(--c-hpc)', T:'var(--orange)', Y:'var(--red)', A:'var(--muted)'};
  const real = {W:true, X_s:true, X_v:false, T:false, Y:null, A:true};
  $('#vars').innerHTML = R.var_groups.map((g,i) => {
    const r = real[g.key];
    const tag = r === null ? '<span class="real no">OBJETIVO</span>' : r ? '<span class="real yes">EN UN AVIÓN REAL</span>' : '<span class="real no">SÓLO SIMULADOR</span>';
    return `<div class="glass var-group reveal d${i%3}" style="--accent:${accent[g.key]}"><h5>${g.key} · ${g.name}</h5>
      <div class="n">${g.cols.length}${tag}</div><p>${g.desc}</p>
      <div class="chips">${g.cols.map(c => `<span title="${esc(R.descriptions[c]||'')}">${c}</span>`).join('')}</div></div>`;
  }).join('');
}
function renderScenarios() {
  $('#scenTable').innerHTML = `<thead><tr><th>SUBCONJUNTO</th><th>COMPONENTES QUE SE DESGASTAN</th><th style="text-align:right">MOTORES</th><th style="text-align:right">VUELOS</th><th style="text-align:right">VIDA MEDIA</th><th style="text-align:right">REGISTROS</th></tr></thead><tbody>` +
    R.datasets.map(d => `<tr class="${d.ok?'':'bad'}"><td class="mono"><b>${d.name}</b></td>
      <td>${d.ok ? `<span class="mini-eng">${COMP_ORDER.map(c => `<i class="${d.components.includes(c)?'on':''}" style="--cf:var(${COMP_VAR[c]})" title="${c}"></i>`).join('')}</span><span class="comp-tags">${d.components.join(' + ')}</span>` : 'archivo dañado en la fuente'}</td>
      <td class="n">${d.ok?d.units:'—'}</td><td class="n">${d.ok?fmt(d.flights):'—'}</td><td class="n">${d.ok?d.eol+' vuelos':'—'}</td><td class="n">${d.rows_fmt}</td></tr>`).join('') + '</tbody>';
}
function renderCards(sel, list) {
  const el = $(sel); if (!el || !list) return;
  el.innerHTML = list.map((t,i) => `<div class="glass take reveal d${i%3}"><div class="k">${String(i+1).padStart(2,'0')}</div><h4>${t.title}</h4><p>${t.text}</p></div>`).join('');
}
function renderSignalNoise() {
  $('#snCards').innerHTML = (R.signal_noise||[]).map((c,i) => `<div class="glass sn-card reveal d${i}"><div class="k">${c.k}</div>
    <div class="big" style="color:${c.hot?'var(--orange)':'var(--hud)'}">${c.value}</div><div class="t">${c.text}</div></div>`).join('');
}
function renderChecks() {
  $('#checks').innerHTML = R.checks.map(c => `<div class="check ${c.ok?'':'bad'}"><div class="mk">${c.ok?'✓':'!'}</div><div><b>${c.name}</b><span>${c.detail}</span></div></div>`).join('');
}
function renderDecisions() {
  $('#decisions').innerHTML = R.decisions.map((d,i) => `<div class="glass decision reveal d${i%3}"><div class="f">${d.finding}</div><div class="ar">→</div><div class="d">${d.decision}</div></div>`).join('');
}
/* tabla genérica a partir de una lista de objetos; hl(row) marca filas destacadas */
function renderTable(sel, rows, {hl, cls} = {}) {
  const el = $(sel); if (!el || !rows || !rows.length) return;
  const cols = Object.keys(rows[0]);
  // decimales parejos por columna: los del valor más preciso de esa columna (máx. 3)
  const dec = Object.fromEntries(cols.map(c => [c, Math.min(3, Math.max(0, ...rows.map(r =>
    typeof r[c] === 'number' ? ((String(r[c]).split('.')[1] || '').length) : 0)))]));
  el.innerHTML = `<table class="tbl"><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map(r =>
    `<tr class="${hl && hl(r) ? 'hl' : ''}">${cols.map(c => { const v = r[c]; const k = cls ? cls(c, v, r) : '';
      return `<td class="${k}">${typeof v === 'number' ? fmt(v, dec[c]) : (v ?? '—')}</td>`; }).join('')}</tr>`).join('')}</tbody></table>`;
}
function renderModels() {
  const el = $('#models'); if (!el) return;
  const desc = {
    'CatBoost': 'Baseline. Gradient boosting sobre árboles con regresión multi-cuantil: cada vuelo es una fila con variables que sólo miran hacia atrás.',
    'Temporal Fusion Transformer': 'Modelo de la línea propuesta. Lee la secuencia de los últimos vuelos del motor (LSTM + atención), selecciona variables y predice cuantiles.',
  };
  el.innerHTML = Object.values(R.model_meta).map(m => {
    const hw = m.hardware || {};
    const rows = [['Dónde se entrenó', `${hw.where||'—'} · ${m.device}${m.device==='GPU' && hw.gpu ? ' ('+hw.gpu+')' : ''}`],
      ['Tiempo de entrenamiento', `${fmt(m.train_seconds,0)} s${m.search ? ' (búsqueda de '+m.search.length+' configuraciones)' : ''}`],
      ['Salida', `cuantiles ${m.quantiles.map(q=>fmt(q*100)+' %').join(' · ')}`]];
    if (m.best_iteration !== undefined) rows.push(['Árboles', fmt(m.best_iteration+1)]);
    if (m.n_params) rows.push(['Parámetros', fmt(m.n_params)]);
    if (m.config) rows.push(['Configuración elegida', Object.entries(m.config).map(([k,v])=>`${k} = ${v}`).join(' · ')]);
    if (m.epochs) rows.push(['Épocas', fmt(m.epochs)]);
    const chosen = R.models_chosen === m.model;
    return `<div class="glass model-card reveal"${chosen ? ' style="box-shadow:var(--lift-shadow)"' : ''}><div class="k">${m.model === 'CatBoost' ? 'BASELINE · MODELO TABULAR' : 'MODELO SECUENCIAL'}${chosen ? ' <span class="part-tag">ELEGIDO EN VALIDACIÓN</span>' : ''}</div><h4>${m.model}</h4>
      <p class="fig-sub">${desc[m.model]||''}</p><dl>${rows.map(([a,b])=>`<dt>${a}</dt><dd>${b}</dd>`).join('')}</dl></div>`;
  }).join('');
}
function renderParagraphs(sel, list) { const el = $(sel); if (el && list) el.innerHTML = list.map(t => `<p>${t}</p>`).join(''); }

/* ═══ 3 · INTERACCIONES ══════════════════════════════════════════════════════ */
function initHud() {
  const H = R.hero; if (!H || !H.alt) return;
  $('#hLabel').textContent = (R.flight_path && R.flight_path.label) || 'un vuelo del dataset';
  let i = 0;
  const tick = () => {
    $('#hAlt').textContent = fmt(H.alt[i]); $('#hMach').textContent = fmt(H.mach[i],2);
    $('#hTra').textContent = fmt(H.tra[i],1); $('#hT48').textContent = fmt(H.t48[i]); $('#hT').textContent = fmt(H.t[i]);
    i = (i+1) % H.alt.length;
  };
  tick(); if (!REDUCE) setInterval(tick, 90);
}
function initSteps() {
  const svg = $('#engSteps'); EngineSchematic(svg, {colored:true});
  const focus = s => {
    svg.classList.toggle('focus', !!s);
    $$('.comp', svg).forEach(p => p.classList.toggle('on', p.dataset.s === s));
    $$('.step').forEach(p => p.classList.toggle('on', p.dataset.s === s));
  };
  $$('.step').forEach(st => ['mouseenter','click'].forEach(ev => st.addEventListener(ev, () => focus(st.dataset.s))));
  $('#stepList').addEventListener('mouseleave', () => focus(null));
}
function initPlan() {
  const n = PLAN.length, xs = PLAN.map((_,i) => 50 + i*(900/(n-1)));
  const ys = PLAN.map((_,i) => 120 - Math.sin(Math.PI * i/(n-1)) * 80);
  $('#planSvg').innerHTML = `<path d="M${xs.map((x,i)=>`${x},${ys[i]}`).join(' L')}" fill="none" stroke="url(#gasPath)" stroke-width="2.5" stroke-dasharray="5 7" opacity=".7"/>` +
    PLAN.map((w,i) => `<g class="wp ${w[4]?'hot':''}" data-href="${w[3]}" tabindex="0" role="link" aria-label="${w[1]}">
      <circle cx="${xs[i]}" cy="${ys[i]}" r="10"/><text class="n" x="${xs[i]}" y="${ys[i]-19}" text-anchor="middle">${w[0]}</text>
      <text class="t" x="${xs[i]}" y="${ys[i]+30}" text-anchor="middle">${w[1]}</text><text class="d" x="${xs[i]}" y="${ys[i]+46}" text-anchor="middle">${w[2]}</text></g>`).join('');
  $$('#planSvg .wp').forEach(g => { const go = () => $(g.dataset.href).scrollIntoView({behavior:'smooth'});
    g.addEventListener('click', go); g.addEventListener('keydown', e => { if (e.key === 'Enter') go(); }); });
}
function initStations() {
  $('#stationBody').innerHTML = `<ellipse cx="460" cy="236" rx="420" ry="8" fill="#0a2540" opacity=".18" filter="url(#blur6)"/>
    <path d="M40 40 Q60 26 120 28 L800 64 Q850 72 870 100 L870 150 Q850 178 800 186 L120 222 Q60 224 40 210 Z" fill="url(#metal)" stroke="#a9b8c9" filter="url(#drop)"/>
    <path d="M200 80 L760 92 Q790 100 800 125 Q790 150 760 158 L200 170 Z" fill="url(#cavity)" opacity=".7"/>
    <path d="M200 80 L760 92 Q790 100 800 125 Q790 150 760 158 L200 170 Z" fill="url(#heatTint)"/>
    ${[[70,40,70,170,'#7cc6e6'],[200,80,90,90,'#2a9fd6'],[310,86,120,78,'#1d4ed8'],[550,88,70,74,'#d62828'],[640,80,110,90,'#f3722c']]
      .map(([x,y,w,h,c]) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${c}" opacity=".18" stroke="${c}" stroke-width="1.5"/>`).join('')}
    <ellipse cx="490" cy="125" rx="44" ry="28" fill="url(#burn)" filter="url(#blur6)"/>
    <g font-family="JetBrains Mono, monospace" font-size="10" text-anchor="middle" letter-spacing="1" font-weight="700">
      <text x="105" y="246" fill="#2a9fd6">FAN</text><text x="245" y="246" fill="#2a9fd6">LPC</text><text x="370" y="246" fill="#1d4ed8">HPC</text>
      <text x="490" y="246" fill="#c98a00">COMB.</text><text x="585" y="246" fill="#d62828">HPT</text><text x="695" y="246" fill="#f3722c">LPT</text></g>`;
  const sensorAt = n => Object.keys(R.descriptions).filter(c => { const m = c.match(/^(T|P|Ps|W)(\d+)$/); return m && m[2] === n; });
  $('#stationNodes').innerHTML = STATIONS.map((s,i) => { const y = s.y || 125;
    return `<g class="st ${s.hot?'hot':''}" data-i="${i}" tabindex="0" role="button" aria-label="Estación ${s.n}">
      <line x1="${s.x}" y1="${y}" x2="${s.x}" y2="${y < 100 ? 16 : 232}"/><circle cx="${s.x}" cy="${y}" r="12"/>
      <text x="${s.x}" y="${y+4}" text-anchor="middle">${s.n}</text></g>`; }).join('') +
    `<text x="900" y="120" text-anchor="middle" style="font-family:JetBrains Mono;font-size:10px;fill:var(--muted)">sin estación</text><text x="900" y="136" text-anchor="middle" style="font-family:JetBrains Mono;font-size:10px;fill:var(--muted)">Nf · Nc · Wf</text>`;
  const show = i => {
    const s = STATIONS[i], ss = sensorAt(s.n);
    $$('.st').forEach(g => g.classList.toggle('on', +g.dataset.i === i));
    const p = $('#stPanel'); p.classList.toggle('hot', s.hot);
    p.innerHTML = `<div class="num">${s.n}</div><div><div class="nm">Estación ${s.n} · ${s.name}</div>
      <div class="ss">${ss.map(c => `<span><b>${c}</b> · ${R.descriptions[c]}</span>`).join('') || '<span>—</span>'}</div></div>`;
  };
  $$('.st[data-i]').forEach(g => ['mouseenter','focus','click'].forEach(ev => g.addEventListener(ev, () => show(+g.dataset.i))));
  show(6);
}
const Flight = { len:0, pts:[] };
function initFlightPath() {
  const FP = R.flight_path; if (!FP) return;
  $('#fpLabel').textContent = FP.label;
  const W = 1000, Hh = 300, padT = 30, padB = 40, tMax = FP.t.at(-1), aMax = Math.max(...FP.alt) * 1.1;
  const X = t => 10 + t / tMax * (W - 20), Y = a => Hh - padB - a / aMax * (Hh - padT - padB);
  Flight.pts = FP.t.map((t,k) => [X(t), Y(FP.alt[k]), FP.alt[k], t]);
  const d = 'M' + Flight.pts.map(p => p[0].toFixed(1)+','+p[1].toFixed(1)).join(' L');
  $('#fpPath').setAttribute('d', d);
  $('#fpArea').setAttribute('d', d + ` L${Flight.pts.at(-1)[0]},${Hh-padB} L${Flight.pts[0][0]},${Hh-padB} Z`);
  let g = '';
  for (let a = 0; a <= aMax; a += 10000) g += `<line x1="0" x2="${W}" y1="${Y(a)}" y2="${Y(a)}" stroke="var(--grid)"/><text x="${W-4}" y="${Y(a)-4}" text-anchor="end" class="fp-phase">${fmt(a)} ft</text>`;
  $('#fpGrid').innerHTML = g;
  $('#fpPhases').innerHTML = (FP.phases||[]).map(p => `<text class="fp-phase" x="${X(p.t)}" y="${Hh-12}" text-anchor="middle">${p.name}</text>`).join('');
  Flight.len = $('#fpPath').getTotalLength();
  $('#fpPath').style.strokeDasharray = Flight.len;
}
function updateFlight() {
  if (!Flight.len) return;
  const box = $('#flightpath').getBoundingClientRect();
  const p = ALL ? 1 : Math.min(1, Math.max(0, (innerHeight*.85 - box.top) / (box.height + innerHeight*.35)));
  $('#fpPath').style.strokeDashoffset = Flight.len * (1 - p);
  const k = Math.min(Flight.pts.length-1, Math.round(p * (Flight.pts.length-1)));
  const [x,y,a,t] = Flight.pts[k], [x2,y2] = Flight.pts[Math.min(Flight.pts.length-1,k+1)];
  const ang = Math.atan2(y2-y, x2-x) * 180/Math.PI;
  $('#fpPlane').setAttribute('transform', `translate(${x},${y}) rotate(${isFinite(ang)?ang:0})`);
  $('#fpRead').textContent = `minuto ${fmt(t/60)} · ${fmt(a)} ft`;
}
function initAnatomy() {
  const svg = $('#anatSvg'); EngineSchematic(svg, {colored:false});
  const ds = R.datasets, pills = $('#dsPills');
  pills.innerHTML = ds.map((d,i)=>`<button data-i="${i}" aria-pressed="false" ${d.ok?'':'disabled title="Archivo dañado en la fuente"'}>${d.name}</button>`).join('');
  const select = i => {
    const d = ds[i];
    $$('button',pills).forEach((b,j)=>{b.classList.toggle('on',i===j);b.setAttribute('aria-pressed',i===j)});
    $$('.comp', svg).forEach(el => el.classList.toggle('fail', d.components.includes(el.dataset.c)));
    const mx = Math.max(...Object.values(d.loss||{0:1}), 1e-9);
    $('#dsCard').innerHTML = `<div class="meta">${d.units} MOTORES · ${fmt(d.flights)} VUELOS · ${d.rows_fmt} REGISTROS</div>
      <h4>${d.name}: <em>${d.components.length ? d.components.join(' + ') : '—'}</em></h4>
      <ul>${d.bullets.map(b=>`<li>${b}</li>`).join('')}</ul>
      <div class="bars">${Object.entries(d.loss||{}).map(([k,v]) => `<div><span>${k}</span><i style="width:${v/mx*100}%;--cf:var(${COMP_VAR[k.split('_')[0]]||'--ti'})"></i><span>${fmt(v,2)} %</span></div>`).join('')}</div>`;
  };
  pills.addEventListener('click', e => { const b = e.target.closest('button'); if (b && !b.disabled) select(+b.dataset.i); });
  select(Math.max(0, ds.findIndex(d => d.ok)));
}

/* ═══ 4 · GRÁFICOS (Plotly con la paleta de la página) ═══════════════════════ */
const Charts = {
  resolve(o) {   // los colores llegan como var(--token) desde el pipeline
    if (typeof o === 'string') return o.replace(/var\((--[\w-]+)\)/g, (_,n) => css(n));
    if (Array.isArray(o)) return o.map(x => Charts.resolve(x));
    if (o && typeof o === 'object') { const r = {}; for (const k in o) r[k] = Charts.resolve(o[k]); return r; }
    return o;
  },
  layout(layout) {
    const ink = css('--ink-2'), grid = css('--grid'), axis = css('--axis'), muted = css('--muted');
    const L = structuredClone(layout);
    Object.assign(L, {
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', separators:',.',
      font:{family:'Inter, system-ui, sans-serif', color:ink, size:12.5},
      hoverlabel:{bgcolor:'rgba(255,255,255,.96)', bordercolor:css('--line-2'), font:{color:css('--ink'), family:'Inter, sans-serif', size:12.5}},
      margin: Object.assign({l:64,r:24,t:34,b:52}, layout.margin||{}),
    });
    L.legend = Object.assign({orientation:'h', y:1.12, x:0, bgcolor:'rgba(0,0,0,0)', font:{color:ink, size:12}}, layout.legend||{});
    Object.keys(L).filter(k => /^[xy]axis\d*$/.test(k)).forEach(k => {
      const t = typeof L[k].title === 'string' ? {text:L[k].title} : (L[k].title||{});
      L[k] = Object.assign({gridcolor:grid, linecolor:axis, zerolinecolor:axis, tickfont:{color:muted, family:'Inter, system-ui, sans-serif', size:11}}, L[k], {title:Object.assign({}, t, {font:{color:ink, size:12}})});
    });
    if (L.updatemenus) L.updatemenus = L.updatemenus.map(u => Object.assign({}, u, {bgcolor:'#fff', bordercolor:css('--line-2'), font:{color:css('--ink'), family:'Inter, sans-serif', size:12}}));
    return L;
  },
  draw(card) {
    const id = card.dataset.fig, spec = R.figures[id], el = $('.plot', card);
    if (!spec) { el.innerHTML = `<div class="missing">Falta el gráfico “${id}”: corré la etapa correspondiente de run_all.py</div>`; return; }
    document.fonts.ready.then(() => Plotly.newPlot(el, Charts.resolve(spec.data), Charts.layout(Charts.resolve(spec.layout||{})),
      {responsive:true, displaylogo:false, modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']}));
    if (card.hasAttribute('data-short')) return;     // versión corta: sin lectura larga ni tabla
    const ins = R.insights[id];
    if (ins) card.insertAdjacentHTML('beforeend', `<div class="insight"><div class="ic">LECTURA</div><div>${ins}</div></div>`);
    const tbl = R.tables && R.tables[id];
    if (tbl) card.insertAdjacentHTML('beforeend', `<details class="table"><summary>▸ VER LOS DATOS EN TABLA</summary><table><thead><tr>${tbl.columns.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${tbl.rows.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></details>`);
  },
};

/* ═══ 5 · SCROLL: ENTRADAS, CONTADORES, BARRA Y ALTÍMETRO ════════════════════ */
function countUp(el) {
  const end = +el.dataset.count, dec = +el.dataset.dec, span = el.querySelector('span');
  if (REDUCE || ALL) { span.textContent = fmt(end, dec); return; }
  const t0 = performance.now(), dur = 1600;
  const step = now => { const p = Math.min(1,(now-t0)/dur), e = 1-Math.pow(1-p,4); span.textContent = fmt(end*e, dec); if (p<1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
function activate(t) {
  t.classList.add('in');
  if (t.matches('.figure') && !t.dataset.drawn) { t.dataset.drawn = 1; Charts.draw(t); }
  if (t.matches('.kpi')) countUp($('.v', t));
}
function initReveal() {
  $$('.split').forEach(h => { h.innerHTML = h.textContent.split(' ').map((w,i)=>`<span class="split-word" style="transition-delay:${i*50}ms">${w}</span>`).join(' '); });
  const targets = $$('.reveal, .chap-head .split');
  if (ALL) { root.classList.add('static'); targets.forEach(activate); return; }
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { activate(e.target); io.unobserve(e.target); } }),
    {threshold:.1, rootMargin:'0px 0px -5% 0px'});
  targets.forEach(el => io.observe(el));
  addEventListener('beforeprint', () => $$('.reveal').forEach(activate));
}
function initScrollChrome() {
  const tape = $('#tape');
  [0,.25,.5,.75,1].forEach(f => tape.insertAdjacentHTML('beforeend', `<i class="tick" style="top:${f*100}%"></i>`));
  const PH = [[.06,'TAXI'],[.18,'ASCENSO'],[.66,'CRUCERO'],[.92,'DESCENSO'],[1.01,'ATERRIZAJE']];
  const altAt = p => p < .06 ? 0 : p < .18 ? (p-.06)/.12*35000 : p < .66 ? 35000 : p < .97 ? 35000*(1-(p-.66)/.31) : 0;
  const links = $$('#chapnav a'), chapters = $$('.chapter');
  const onScroll = () => {
    const h = document.documentElement, p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
    root.style.setProperty('--heat-a', Math.min(1, Math.max(0, (p-.2)/.7)).toFixed(3));
    $('#topbar').style.setProperty('--p', p.toFixed(4));
    const past = h.scrollTop > innerHeight*.6;
    $('#topbar').classList.toggle('show', past); $('#altimeter').classList.toggle('show', past);
    const a = altAt(p);
    $('#bug').style.top = (100 - a/35000*100) + '%';
    $('#altRead').textContent = fmt(Math.round(a/100)*100);
    $('#phaseRead').textContent = PH.find(x => p < x[0])[1];
    let cur = null; chapters.forEach(c => { if (c.getBoundingClientRect().top < innerHeight*.4) cur = c.id; });
    links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#'+cur));
    updateFlight();
  };
  addEventListener('scroll', onScroll, {passive:true}); onScroll();
}

/* ═══ 6 · PARTÍCULAS DE LA PORTADA (flujo de aire frío → caliente) ═══════════ */
function initParticles() {
  const cv = $('#particles'), ctx = cv.getContext('2d');
  let W, H, pts = [];
  const size = () => { W = cv.width = cv.offsetWidth*devicePixelRatio; H = cv.height = cv.offsetHeight*devicePixelRatio;
    pts = Array.from({length: Math.min(110, Math.floor(W/16))}, () => ({x:Math.random()*W, y:Math.random()*H, v:.5+Math.random()*2.4, l:20+Math.random()*90})); };
  const tick = () => {
    ctx.clearRect(0,0,W,H);
    const cold = css('--hud'), hot = css('--orange');
    pts.forEach(p => {
      p.x += p.v*devicePixelRatio*1.5; if (p.x - p.l > W) { p.x = -p.l; p.y = Math.random()*H; }
      const g = ctx.createLinearGradient(p.x-p.l, 0, p.x, 0); g.addColorStop(0,'rgba(0,0,0,0)'); g.addColorStop(1, p.x/W > .6 ? hot : cold);
      ctx.strokeStyle = g; ctx.globalAlpha = .08 + p.v/18; ctx.lineWidth = devicePixelRatio;
      ctx.beginPath(); ctx.moveTo(p.x-p.l,p.y); ctx.lineTo(p.x,p.y); ctx.stroke();
    });
    ctx.globalAlpha = 1;
    if (!REDUCE) requestAnimationFrame(tick);
  };
  size(); tick(); addEventListener('resize', size);
}

/* ═══ 7 · INICIO ═════════════════════════════════════════════════════════════ */
// El mismo script sirve para la versión corta y la completa: cada parte se dibuja sólo si su elemento existe.
const when = (sel, fn) => { if ($(sel)) fn(); };
const methodHl = r => /ACI/.test(r.intervalo || '') && r.modelo === R.facts.r_main_model;
when('#kpis', renderKpis); bindFacts(); when('#hier', renderHierarchy); when('#exRow', renderExampleRow);
when('#vars', renderVarGroups); when('#scenTable', renderScenarios);
renderCards('#takeaways', R.takeaways); renderCards('#conclusions', R.conclusions); renderCards('#closing', R.closing);
when('#snCards', renderSignalNoise); when('#checks', renderChecks); when('#decisions', renderDecisions);
renderTable('#splitTable', R.split_summary);
renderTable('#metricsTable', R.metrics_table, {hl: methodHl});
renderTable('#modelCompare', (R.metrics_table||[]).filter(r => /ACI/.test(r.intervalo)).map(r => ({modelo: r.modelo, RMSE: r.RMSE, 'score NASA': r['score NASA'],
  'cobertura [%]': r['cobertura [%]'], 'ancho [vuelos]': r['ancho medio [vuelos]'], '% real < cota inferior': r['% real < cota inferior']})), {hl: r => r.modelo === R.facts.r_main_model});
renderTable('#line2Table', R.line2_table, {hl: r => /^C/.test(r.variante)});
renderTable('#line3Table', R.line3_table, {hl: r => /CUSUM/.test(r['método'])});
renderTable('#searchTable', R.tft_search);
when('#models', renderModels); renderParagraphs('#discussion', R.discussion); renderParagraphs('#futureText', R.future_discussion);
when('#hAlt', initHud); when('#engSteps', initSteps); when('#planSvg', initPlan); when('#stationSvg', initStations);
when('#flightpath', initFlightPath); when('#anatSvg', initAnatomy);
initReveal(); initScrollChrome(); when('#particles', initParticles);
})();
