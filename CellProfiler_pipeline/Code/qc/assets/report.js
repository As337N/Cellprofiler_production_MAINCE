// ─────────────────────────────────────────────────────────────────────────────
// report.js — QC report client logic
//
// Data is injected by the Python builder as a single object on window.__QC__
// (see qc/report.py). This preamble unpacks it into the module-level names the
// rest of this file expects, so the logic below is untouched by the refactor.
// ─────────────────────────────────────────────────────────────────────────────
const Q = window.__QC__;

const DATA           = Q.data;
const COUNT_RANGES   = Q.countRanges;
const MFI_DATA       = Q.mfi.data;
const MFI_CHANNELS   = Q.mfi.channels;
const MFI_COLORS     = Q.mfi.colors;
const MFI_IMG        = Q.mfi.img;
const RADIUS_DATA    = Q.radius;
const PLATES         = Object.keys(DATA);

const SLOPE_SPECS    = Q.specs.slope;
const PCT_MAX_SPECS  = Q.specs.pctMax;
const PCT_MIN_SPECS  = Q.specs.pctMin;
const MEDIANINT_SPECS = Q.specs.medianInt;
const MAD_SCORES     = Q.madScores;
const RED_COUNTS     = Q.redCounts;
const METRIC_GROUPS = [
  { label: 'PowerLogLogSlope',        specs: SLOPE_SPECS  },
  { label: 'PercentMaximal',   specs: PCT_MAX_SPECS  },
  { label: 'PercentMinimal',   specs: PCT_MIN_SPECS  },
  { label: 'MedianIntensity', specs: MEDIANINT_SPECS },
];
 
// ROWS: A at top (index 0) -> H at bottom (index 7)
// Plotly heatmap y-axis goes bottom->top, so we reverse for display.
const ROW_LABELS  = ['A','B','C','D','E','F','G','H'];
const ROWS_PLOTLY = ['H','G','F','E','D','C','B','A'];  // reversed for Plotly
const COLS = Array.from({length:12}, (_,i) => String(i+1).padStart(2,'0'));
const PLATE_ROWS = 8, PLATE_COLS = 12;
 
// MFI_COLORS and MFI_DATA/MFI_CHANNELS injected above from Python
function _chColor(ch) {
  // First try exact key match in injected MFI_COLORS, then substring fallback
  if (MFI_COLORS[ch]) return MFI_COLORS[ch];
  const fallbacks = {
    'DNA':'#6495ed','Hoechst':'#6495ed','DAPI':'#6495ed',
    'Syto':'#4bc870','Golgi':'#ffc04d','ER':'#b06aff',
    'Mito':'#ff5555','BF':'#909090',
  };
  for (const [k,v] of Object.entries({...MFI_COLORS,...fallbacks}))
    if (ch.toLowerCase().includes(k.toLowerCase())) return v;
  return '#8ab0d0';
}
 
// ── Helpers ───────────────────────────────────────────────────────────────────
function badge(pct) {
  const cls = pct >= 80 ? 'pass' : pct >= 50 ? 'warn' : 'fail';
  return `<span class="badge badge-${cls}">${pct}%</span>`;
}
function fmt3(v) { return v != null ? v.toFixed(3) : '—'; }
function fmt5(v) { return v != null ? v.toFixed(5) : '—'; }
 
// Median of array (ignoring nulls)
function arrMedian(arr) {
  const a = arr.filter(x => x != null && !isNaN(x)).sort((a,b)=>a-b);
  if (!a.length) return null;
  const m = Math.floor(a.length/2);
  return a.length%2 ? a[m] : (a[m-1]+a[m])/2;
}

// ── 1. Summary ────────────────────────────────────────────────────────────────
(function() {
  const tbody = document.getElementById('summary-tbody');
  PLATES.forEach(p => {
    const d = DATA[p];
    // Compute median MFI Δ for Hoechst/DNA and Syto across wells
    const snrSummary = () => {
      if (!MFI_CHANNELS.length) return '<td style="color:var(--muted)">—</td>';

      const warnCh = [], badCh = [];
      MFI_CHANNELS.forEach(ch => {
        const plateMfiData = (MFI_DATA[p]||{})[ch]||{};
        const objVals = Object.values(plateMfiData).flat();
        const objMed  = objVals.length ? arrMedian(objVals) : null;
        const imgVals = Object.values((MFI_IMG[p]||{}))
                          .map(w => w[ch]).filter(v => v != null);
        const imgMed  = imgVals.length ? arrMedian(imgVals) : null;
        if (objMed == null || imgMed == null || imgMed === 0) return;
        const snr = (objMed - imgMed) / imgMed;
        if      (snr < 0.2) badCh.push(ch);
        else if (snr < 0.5) warnCh.push(ch);
      });

      if (!badCh.length && !warnCh.length) {
        return `<td style="text-align:center;">
          <span style="color:#4bd760;font-weight:700;">Good</span>
        </td>`;
      }
      let content = '';
      if (badCh.length)  content += `<span style="color:#ff4444;font-weight:700;">Bad</span><br>
        <span style="color:#ff4444;font-size:0.75rem;">${badCh.join(', ')}</span>`;
      if (warnCh.length) content += `${badCh.length?'<br>':''}
        <span style="color:#ffbe00;font-weight:700;">Warning</span><br>
        <span style="color:#ffbe00;font-size:0.75rem;">${warnCh.join(', ')}</span>`;
      return `<td style="text-align:center;">${content}</td>`;
    };

    const positionalSummary = () => {
      if (!MFI_CHANNELS.length) return '<td style="color:var(--muted)">—</td>';

      // Calcular baseline Hoechst (igual que renderMFI)
      let baseRow = null, baseCol = null;
      if ((MFI_DATA[p]||{})['Hoechst']) {
        const hData = MFI_DATA[p]['Hoechst'];
        const hRowG = ROW_LABELS.map(r => COLS.flatMap(c => hData[r+c]||[]));
        const hColG = COLS.map(c => ROW_LABELS.flatMap(r => hData[r+c]||[]));
        baseRow = mfiAnova(hRowG);
        baseCol = mfiAnova(hColG);
        if (baseRow!=null && baseCol!=null) {
          const maxH = Math.max(baseRow, baseCol);
          if (Math.abs(baseRow-baseCol)/maxH < 0.20) {
            const mean = (baseRow+baseCol)/2;
            baseRow = mean; baseCol = mean;
          }
        }
      }

      const warnCh = [], badCh = [];
      MFI_CHANNELS.forEach(ch => {
        const wellData = (MFI_DATA[p]||{})[ch]||{};
        const rowGroups = ROW_LABELS.map(r => COLS.flatMap(c => wellData[r+c]||[]));
        const colGroups = COLS.map(c => ROW_LABELS.flatMap(r => wellData[r+c]||[]));
        const e2r   = mfiAnova(rowGroups);
        const e2c   = mfiAnova(colGroups);
        const worst = Math.max(e2r??0, e2c??0);
        const isHoechst = ch === 'Hoechst';

        let level = 'good';
        if (isHoechst) {
          if      (worst >= 0.14) level = 'bad';
          else if (worst >= 0.06) level = 'warn';
        } else {
          if (worst >= 0.20) {
            level = 'bad';
          } else if (baseRow != null) {
            const ratio = worst / Math.max(baseRow, baseCol, 0.001);
            if      (worst >= 0.2)                                  level = 'bad';
            else if (worst >= 0.14 && ratio >= 4)                   level = 'bad';
            else if (worst >= 0.06 && ratio >= 2 || worst >= 0.14)  level = 'warn';
          } else {
            if      (worst >= 0.14) level = 'bad';
            else if (worst >= 0.06) level = 'warn';
          }
        }

        if      (level === 'bad')  badCh.push(ch);
        else if (level === 'warn') warnCh.push(ch);
      });

      if (!badCh.length && !warnCh.length) {
        return `<td style="text-align:center;">
          <span style="color:#4bd760;font-weight:700;">Good</span>
        </td>`;
      }
      let content = '';
      if (badCh.length)  content += `<span style="color:#ff4444;font-weight:700;">Bad</span><br>
        <span style="color:#ff4444;font-size:0.75rem;">${badCh.join(', ')}</span>`;
      if (warnCh.length) content += `${badCh.length?'<br>':''}
        <span style="color:#ffbe00;font-weight:700;">Warning</span><br>
        <span style="color:#ffbe00;font-size:0.75rem;">${warnCh.join(', ')}</span>`;
      return `<td style="text-align:center;">${content}</td>`;
    };

    tbody.insertAdjacentHTML('beforeend', `<tr>
      <td><strong>${p}</strong></td><td>${d.n_wells}</td>
      <td style="text-align:center;font-family:monospace;">
        ${(RED_COUNTS.slope[p]||0) > 0
            ? `<span style="color:#ff4444;font-weight:700;">${RED_COUNTS.slope[p]}</span>`
            : `<span style="color:#4bd760;">0</span>`}
      </td>
      <td style="text-align:center;font-family:monospace;">
        ${(RED_COUNTS.pctmax[p]||0) > 0
            ? `<span style="color:#ff4444;font-weight:700;">${RED_COUNTS.pctmax[p]}</span>`
            : `<span style="color:#4bd760;">0</span>`}
      </td>
      <td style="text-align:center;font-family:monospace;">
        ${(RED_COUNTS.pctmin[p]||0) > 0
            ? `<span style="color:#ff4444;font-weight:700;">${RED_COUNTS.pctmin[p]}</span>`
            : `<span style="color:#4bd760;">0</span>`}
      </td>
      <td style="text-align:center;font-family:monospace;">
        ${(RED_COUNTS.medint[p]||0) > 0
            ? `<span style="color:#ff4444;font-weight:700;">${RED_COUNTS.medint[p]}</span>`
            : `<span style="color:#4bd760;">0</span>`}
      </td>
      ${snrSummary()}
      ${positionalSummary()}
    </tr>`);
  });
})();
 
// ── 2. Plate browser ──────────────────────────────────────────────────────────
const slider   = document.getElementById('plate-slider');
const pSelect  = document.getElementById('plate-select');
const nameDisp = document.getElementById('plate-name-display');
 
slider.max = PLATES.length - 1;
PLATES.forEach((p, i) =>
  pSelect.insertAdjacentHTML('beforeend', `<option value="${i}">${p}</option>`)
);
 
let currentPlate = null;
let selectedWell = null;
let selectedSite = null;
 
function renderPlate(idx) {
  const name    = PLATES[idx];
  currentPlate  = name;
  selectedWell  = null;
  nameDisp.textContent = name;
  slider.value  = idx;
  pSelect.value = idx;
 
  const pd = DATA[name];
  document.getElementById('overview-img').src =
    `data:image/jpeg;base64,${pd.overview_b64}`;
  buildOverlaySVG(name);
 
  document.getElementById('well-info').innerHTML =
    '<p class="no-well">Click a well in the overview below to inspect it.</p>';
  document.getElementById('well-montage').innerHTML =
    '<span class="no-img">Select a well to see its site montage.</span>';
  selectedSite = null;
  clearSiteInfo();
  hideSiteTables();
 
  renderMetrics(name);
  renderCounts(name);
  renderMFI();
}
 
// Evalúa el color semántico de un valor según su colorscale (igual que renderWellInfo)
function specColor(v, spec) {
  if (v == null) return null;
  const t = Math.max(0, Math.min(1, (v - spec.cmin) / (spec.cmax - spec.cmin)));
  const cs = spec.cs;
  for (let i = 0; i < cs.length - 1; i++) {
    if (t >= cs[i][0] && t <= cs[i+1][0]) {
      if (cs[i][1] === cs[i+1][1]) return cs[i][1];
      if (cs[i][0] === cs[i+1][0]) return cs[i+1][1];
      return cs[i][1];
    }
  }
  return cs[cs.length-1][1];
}

// Prioridad de colores: rojo > amarillo > verde > null
const COLOR_PRIORITY = ['#ff4444','#ffbe00','#4bd760'];
function worstColor(colors) {
  for (const c of COLOR_PRIORITY) {
    if (colors.includes(c)) return c;
  }
  return null;
}

function buildOverlaySVG(plateName) {
  const pd  = DATA[plateName];
  const cw  = pd.overview_cw;
  const ch  = pd.overview_ch;
  const svg = document.getElementById('overview-svg');

  const imgW = cw * PLATE_COLS;
  const imgH = ch * PLATE_ROWS;
  svg.setAttribute('viewBox', `0 0 ${imgW} ${imgH}`);
  svg.innerHTML = '';

  ROW_LABELS.forEach((r, ri) => {
    COLS.forEach((c, ci) => {
      const well = r + c;
      const m    = pd.wells[well] || {};
      const x0   = ci * cw, y0 = ri * ch;

      // Calcular el peor color entre todas las métricas del pozo
      const colors = ALL_METRIC_SPECS
        .map(spec => specColor(m[spec.col], spec))
        .filter(Boolean);
      const border = worstColor(colors);

      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('class', 'well-cell');
      rect.setAttribute('id', `sv-${well}`);
      rect.setAttribute('x', x0 + 1); rect.setAttribute('y', y0 + 1);
      rect.setAttribute('width', cw - 2); rect.setAttribute('height', ch - 2);
      rect.setAttribute('data-well', well);
      if (border) {
        rect.setAttribute('stroke', border);
        rect.setAttribute('stroke-width', '3');
        rect.setAttribute('fill', border + '18');  // 18 = ~10% opacity
      }
      rect.onclick = () => selectWell(plateName, well);
      svg.appendChild(rect);
    });
  });
}
 
// ── Cross-platemap well highlight ─────────────────────────────────────────────
// Registro de gráficos Plotly (heatmaps) para poder resaltar la misma celda en
// todos. Cada entrada: divId de un heatmap con ejes x:COLS, y:ROWS_PLOTLY.
const HEATMAP_DIVS = new Set();

// Engancha el click de sincronización a un heatmap. IMPORTANTE: Plotly.react()
// devuelve una promesa y solo expone gd.on() DESPUÉS de resolverla, así que el
// binding debe hacerse en el .then — hacerlo síncrono falla ("gd.on is not a
// function") y por eso los clicks en los platemaps no se registraban.
function registerHeatmap(divIdOrPromise) {
  // Compatibilidad: acepta tanto la promesa de Plotly.react como un divId string.
  const p = (divIdOrPromise && typeof divIdOrPromise.then === 'function')
    ? divIdOrPromise
    : Promise.resolve(document.getElementById(divIdOrPromise));
  p.then(gd => {
    if (!gd || !gd.id) return;
    HEATMAP_DIVS.add(gd.id);
    if (gd._syncBound) return;
    gd._syncBound = true;
    // plotly_click → well. Usamos pointIndex (índices en la matriz z), que es
    // robusto y NO depende de si el eje es categórico o del formato de point.x.
    // z se construye como ROWS_PLOTLY.map(r => COLS.map(c => …)), por lo que
    // pointIndex = [fila_en_ROWS_PLOTLY, col_en_COLS].
    gd.on('plotly_click', ev => {
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      let well = null;
      if (Array.isArray(pt.pointIndex) && pt.pointIndex.length === 2) {
        const [yi, xi] = pt.pointIndex;
        if (ROWS_PLOTLY[yi] && COLS[xi]) well = ROWS_PLOTLY[yi] + COLS[xi];
      }
      // Respaldo: normalizar x/y a etiqueta de columna/fila si no hubo pointIndex.
      if (!well) {
        const row = String(pt.y);
        const colNum = parseInt(pt.x, 10);
        const col = Number.isFinite(colNum) ? String(colNum).padStart(2, '0') : String(pt.x);
        well = row + col;
      }
      if (currentPlate && DATA[currentPlate]?.wells?.[well]) {
        selectWell(currentPlate, well);
      }
    });
  });
}

// Rectángulo de resaltado sobre una celda categórica (índice ± 0.5)
function heatmapHighlightShape(well) {
  const row = well[0], col = well.slice(1);
  const xi = COLS.indexOf(col);
  const yi = ROWS_PLOTLY.indexOf(row);
  if (xi < 0 || yi < 0) return [];
  return [{
    type: 'rect', xref: 'x', yref: 'y',
    x0: xi - 0.5, x1: xi + 0.5, y0: yi - 0.5, y1: yi + 0.5,
    line: { color: '#ffffff', width: 3 }, fillcolor: 'rgba(0,0,0,0)',
    layer: 'above',
  }];
}

function highlightWellEverywhere(well) {
  // 1) MFI grids (divs): limpiar y marcar
  document.querySelectorAll('.mfi-well-cell.well-selected')
    .forEach(el => el.classList.remove('well-selected'));
  MFI_CHANNELS.forEach(ch => {
    const cell = document.getElementById(`mfi-cell-${ch}-${well}`);
    if (cell) cell.classList.add('well-selected');
  });

  // 2) Plotly heatmaps: relayout con la shape (no re-renderiza los datos)
  const shape = heatmapHighlightShape(well);
  HEATMAP_DIVS.forEach(divId => {
    const div = document.getElementById(divId);
    if (div && div.data) Plotly.relayout(div, { shapes: shape });
  });
}

function selectWell(plateName, well) {
  if (selectedWell) {
    const prev = document.getElementById(`sv-${selectedWell}`);
    if (prev) prev.classList.remove('selected');
  }
  selectedWell = well;
  const rect = document.getElementById(`sv-${well}`);
  if (rect) rect.classList.add('selected');

  highlightWellEverywhere(well);
  renderWellInfo(plateName, well);
  renderWellMontage(plateName, well);
  renderSiteTables(plateName, well);
}
 
// ── Well info panel ───────────────────────────────────────────────────────────
const ALL_METRIC_SPECS = [...SLOPE_SPECS, ...PCT_MAX_SPECS, ...PCT_MIN_SPECS, ...MEDIANINT_SPECS];
 
function renderWellInfo(plateName, well) {
  const pd   = DATA[plateName];
  const m    = pd.wells[well] || {};
  const cmpd = m.compound || '—';

  // Evalúa el color de un valor según la colorscale del spec (igual que el heatmap)
  // Interpola la colorscale para obtener el color semántico: verde/amarillo/rojo
  function specColor(v, spec) {
    if (v == null) return '';
    const t = Math.max(0, Math.min(1, (v - spec.cmin) / (spec.cmax - spec.cmin)));
    const cs = spec.cs;
    // Buscar el segmento de la colorscale donde cae t
    for (let i = 0; i < cs.length - 1; i++) {
      if (t >= cs[i][0] && t <= cs[i+1][0]) {
        // Si el par de colores es igual, devolver ese color directamente
        if (cs[i][1] === cs[i+1][1]) return cs[i][1];
        // Si el par forma un escalón (mismo t), es la transición de un segmento al siguiente
        if (cs[i][0] === cs[i+1][0]) return cs[i+1][1];
        return cs[i][1];
      }
    }
    return cs[cs.length-1][1];
  }

  let html = `<h3>${well}</h3>
    <div class="well-compound">${cmpd}</div>`;

  html += `<div class="section-label">Cell counts</div>`;
  [['Cells','Count_Cells'],['Nuclei','Count_Nuclei'],
   ['Raw nuclei','Count_Raw_nuclei'],['Artifacts','Count_Illum_artifacts_filtered']].forEach(([lbl,col]) => {
    const v = m[col];
    html += `<div class="metric-row">
      <span class="metric-label">${lbl}</span>
      <span class="metric-val">${v != null ? Math.round(v) : '—'}</span>
    </div>`;
  });

  ALL_METRIC_SPECS.forEach(spec => {
    const v = m[spec.col];
    if (v == null) return;
    const color = specColor(v, spec);
    html += `<div class="metric-row">
      <span class="metric-label">${spec.title}</span>
      <span class="metric-val" style="color:${color}">${fmt3(v)}</span>
    </div>`;
  });

  document.getElementById('well-info').innerHTML = html;
}
 
function renderWellMontage(plateName, well) {
  const pd  = DATA[plateName];
  const b64 = pd.flagged_b64[well];
  const el  = document.getElementById('well-montage');
  selectedSite = null;
  clearSiteInfo();

  if (!b64) {
    const flags = pd.well_flags[well];
    el.innerHTML = flags
      ? '<span class="no-img">Flagged well — montage not pre-generated.</span>'
      : '<span class="no-img">Well passes all QC thresholds — no image preloaded.</span>';
    return;
  }

  // Montage is a square 3×3 grid of sites (build_well_montage). Overlay a
  // matching 3×3 SVG of clickable cells so each site can be inspected.
  const siteData = (pd.site_data || {})[well] || {};
  const nGrid    = 3;  // ceil(sqrt(9)); montage is padded to a full 3×3
  el.innerHTML = `
    <div class="montage-wrap">
      <img src="data:image/jpeg;base64,${b64}" alt="Well ${well} montage">
      <svg class="montage-overlay" viewBox="0 0 ${nGrid} ${nGrid}"
           preserveAspectRatio="none"></svg>
    </div>`;

  const svg = el.querySelector('.montage-overlay');
  for (let s = 1; s <= nGrid * nGrid; s++) {
    const gr = Math.floor((s - 1) / nGrid);
    const gc = (s - 1) % nGrid;
    const hasData = Object.prototype.hasOwnProperty.call(siteData, String(s));

    // Peor color entre todas las métricas del sitio (rojo > amarillo > verde)
    let border = null;
    if (hasData) {
      const m = siteData[String(s)];
      const colors = ALL_METRIC_SPECS
        .map(spec => specColor(m[spec.col], spec))
        .filter(Boolean);
      border = worstColor(colors);
    }

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('class', 'site-cell' + (hasData ? '' : ' site-empty'));
    rect.setAttribute('id', `site-${s}`);
    rect.setAttribute('x', gc + 0.02);
    rect.setAttribute('y', gr + 0.02);
    rect.setAttribute('width', 0.96);
    rect.setAttribute('height', 0.96);
    // style.* (no setAttribute) para ganarle a la regla CSS .site-cell
    if (border) {
      rect.style.stroke = border;
      rect.style.strokeWidth = '0.02';
    }
    if (hasData) rect.onclick = () => selectSite(plateName, well, s);
    svg.appendChild(rect);
  }
}

function selectSite(plateName, well, site) {
  if (selectedSite != null) {
    const prev = document.getElementById(`site-${selectedSite}`);
    if (prev) prev.classList.remove('selected');
  }
  selectedSite = site;
  const rect = document.getElementById(`site-${site}`);
  if (rect) rect.classList.add('selected');
  renderSiteInfo(plateName, well, site);
}

function clearSiteInfo() {
  const el = document.getElementById('site-info');
  if (el) el.innerHTML =
    '<p class="no-well">Click a site tile in the montage to inspect it.</p>';
}

function renderSiteInfo(plateName, well, site) {
  const pd = DATA[plateName];
  const m  = ((pd.site_data || {})[well] || {})[String(site)] || {};

  // Same colour logic as the well-info panel (interpolates the spec colorscale).
  function siteSpecColor(v, spec) {
    if (v == null) return '';
    const t = Math.max(0, Math.min(1, (v - spec.cmin) / (spec.cmax - spec.cmin)));
    const cs = spec.cs;
    for (let i = 0; i < cs.length - 1; i++) {
      if (t >= cs[i][0] && t <= cs[i+1][0]) {
        if (cs[i][1] === cs[i+1][1]) return cs[i][1];
        if (cs[i][0] === cs[i+1][0]) return cs[i+1][1];
        return cs[i][1];
      }
    }
    return cs[cs.length-1][1];
  }

  let html = `<h3>${well} · site ${site}</h3>
    <div class="well-compound">${(pd.wells[well]||{}).compound || '—'}</div>`;

  html += `<div class="section-label">Cell counts</div>`;
  [['Cells','Count_Cells'],['Nuclei','Count_Nuclei'],
   ['Raw nuclei','Count_Raw_nuclei'],['Artifacts','Count_Illum_artifacts_filtered']]
   .forEach(([lbl,col]) => {
    const v = m[col];
    html += `<div class="metric-row">
      <span class="metric-label">${lbl}</span>
      <span class="metric-val">${v != null ? Math.round(v) : '—'}</span>
    </div>`;
  });

  ALL_METRIC_SPECS.forEach(spec => {
    const v = m[spec.col];
    if (v == null) return;
    const color = siteSpecColor(v, spec);
    html += `<div class="metric-row">
      <span class="metric-label">${spec.title}</span>
      <span class="metric-val" style="color:${color}">${fmt3(v)}</span>
    </div>`;
  });

  document.getElementById('site-info').innerHTML = html;
}

// ── Site-level metric tables (QC Metrics / Cell Counts) ───────────────────────
// Canales QC derivados del sufijo de columna de los slope specs, en el orden en
// que Python los emitió (config.CHANNELS). Cada spec.col = ImageQuality_<m>_<ch>.
const QC_CHANNELS = SLOPE_SPECS.map(s => s.col.split('_').pop());

// Nº de sitios a mostrar: el máximo field observado en cualquier well (default 9).
function _maxSites() {
  let n = 0;
  PLATES.forEach(p => {
    const sd = DATA[p].site_data || {};
    Object.values(sd).forEach(fields => {
      Object.keys(fields).forEach(f => { const k = parseInt(f, 10); if (k > n) n = k; });
    });
  });
  return n || 9;
}
const N_SITES = _maxSites();
const SITE_IDS = Array.from({length: N_SITES}, (_, i) => i + 1);

// Filas de la tabla QC para un canal dado: [{label, specGroup}] → col por canal.
const QC_ROW_DEFS = [
  { label: 'PowerLogLogSlope', specs: SLOPE_SPECS   },
  { label: 'PercentMaximal',   specs: PCT_MAX_SPECS },
  { label: 'PercentMinimal',   specs: PCT_MIN_SPECS },
  { label: 'MedianIntensity',  specs: MEDIANINT_SPECS },
];

let qcSiteTableChannel = QC_CHANNELS[0] || null;

function _siteVal(plateName, well, site, col) {
  const fields = (DATA[plateName].site_data || {})[well] || {};
  const f = fields[String(site)];
  return f ? f[col] : undefined;
}

// Valor agregado por well (ya calculado en Python: media para métricas/%, suma
// para counts — collapse_sites_to_wells). Aquí solo se lee.
function _wellVal(plateName, well, col) {
  const m = (DATA[plateName].wells || {})[well];
  return m ? m[col] : undefined;
}

function _buildSiteTableHTML(headerCells, rows, hasWellCol) {
  // headerCells: ['Metric', 'Site 1', …, 'Well']; rows: [{label, cells:[html]}]
  // Si hasWellCol, la última columna se destaca como agregado del well.
  const lastIdx = headerCells.length - 1;
  let h = '<table class="site-table"><thead><tr>';
  headerCells.forEach((c, i) => {
    let cls = i === 0 ? 'site-th-metric' : 'site-th-site';
    if (hasWellCol && i === lastIdx) cls = 'site-th-well';
    h += `<th class="${cls}">${c}</th>`;
  });
  h += '</tr></thead><tbody>';
  rows.forEach(r => {
    h += `<tr><td class="site-td-metric">${r.label}</td>`;
    r.cells.forEach((c, i) => {
      const cls = (hasWellCol && i === r.cells.length - 1) ? 'site-td-well' : 'site-td-val';
      h += `<td class="${cls}">${c}</td>`;
    });
    h += '</tr>';
  });
  return h + '</tbody></table>';
}

function renderQCSiteTable(plateName, well) {
  const wrap = document.getElementById('qc-site-table-wrap');
  if (!well || !DATA[plateName].site_data || !DATA[plateName].site_data[well]) {
    wrap.style.display = 'none';
    return;
  }
  wrap.style.display = 'block';
  document.getElementById('qc-site-table-title').textContent =
    `Site-level metrics — ${well}`;

  // Pestañas de canal (una vez por well render; resalta la activa)
  const tabsEl = document.getElementById('qc-site-table-tabs');
  tabsEl.innerHTML = '';
  if (!QC_CHANNELS.includes(qcSiteTableChannel)) qcSiteTableChannel = QC_CHANNELS[0];
  QC_CHANNELS.forEach(ch => {
    const b = document.createElement('button');
    b.className = 'site-tab' + (ch === qcSiteTableChannel ? ' active' : '');
    b.textContent = ch;
    b.onclick = () => { qcSiteTableChannel = ch; renderQCSiteTable(plateName, well); };
    tabsEl.appendChild(b);
  });

  const ch = qcSiteTableChannel;
  const header = ['Metric', ...SITE_IDS.map(s => `Site ${s}`), 'Well'];
  const rows = QC_ROW_DEFS.map(def => {
    // spec del canal activo dentro de este grupo de métrica
    const spec = def.specs.find(s => s.col.endsWith('_' + ch));
    const col  = spec ? spec.col : null;
    const fmtCell = v => {
      if (v == null) return '<span class="site-na">—</span>';
      const color = spec ? specColor(v, spec) : null;
      return `<span style="color:${color || 'var(--text)'}">${fmt3(v)}</span>`;
    };
    const cells = SITE_IDS.map(s => fmtCell(col ? _siteVal(plateName, well, s, col) : undefined));
    cells.push(fmtCell(col ? _wellVal(plateName, well, col) : undefined));  // agregado por well
    return { label: def.label, cells };
  });

  document.getElementById('qc-site-table').innerHTML =
    _buildSiteTableHTML(header, rows, true);
}

function renderCountsSiteTable(plateName, well) {
  const wrap = document.getElementById('counts-site-table-wrap');
  if (!well || !DATA[plateName].site_data || !DATA[plateName].site_data[well]) {
    wrap.style.display = 'none';
    return;
  }
  wrap.style.display = 'block';
  document.getElementById('counts-site-table-title').textContent =
    `Site-level counts — ${well}`;

  const header = ['Count', ...SITE_IDS.map(s => `Site ${s}`), 'Well'];
  const COUNT_ROWS = [
    ['Cells',      'Count_Cells'],
    ['Nuclei',     'Count_Nuclei'],
    ['Raw nuclei', 'Count_Raw_nuclei'],
    ['Artifacts',  'Count_Illum_artifacts_filtered'],
  ];
  const fmtCount = v => v == null ? '<span class="site-na">—</span>' : String(Math.round(v));
  const rows = COUNT_ROWS.map(([label, col]) => {
    const cells = SITE_IDS.map(s => fmtCount(_siteVal(plateName, well, s, col)));
    cells.push(fmtCount(_wellVal(plateName, well, col)));  // suma por well (Python)
    return { label, cells };
  });

  document.getElementById('counts-site-table').innerHTML =
    _buildSiteTableHTML(header, rows, true);
}

function renderSiteTables(plateName, well) {
  renderQCSiteTable(plateName, well);
  renderCountsSiteTable(plateName, well);
}

function hideSiteTables() {
  const q = document.getElementById('qc-site-table-wrap');
  const c = document.getElementById('counts-site-table-wrap');
  if (q) q.style.display = 'none';
  if (c) c.style.display = 'none';
}

slider.oninput   = () => renderPlate(+slider.value);
pSelect.onchange = () => renderPlate(+pSelect.value);
 
// ── Compound filter state ────────────────────────────────────────────────────
let activeCompounds = new Set();
let hideAllMode     = false;
function compoundVisible(cmpd) {
  if (hideAllMode)              return false;
  if (activeCompounds.size===0) return true;
  return activeCompounds.has(cmpd);
}
 
// ── 3. QC Metrics ─────────────────────────────────────────────────────────────
function wellMatrix(plateData, colName) {
  return ROWS_PLOTLY.map(r => COLS.map(c => {
    const w    = r + c;
    const well = plateData.wells[w];
    if (!well) return null;
    if (!compoundVisible(well.compound || '')) return null;
    return well[colName] ?? null;
  }));
}
 
function isFiltered() {
  return hideAllMode || activeCompounds.size > 0;
}
 
function makeHeatmap(plateData, spec) {
  const z    = wellMatrix(plateData, spec.col);
  const plateName = Object.keys(DATA).find(p => DATA[p] === plateData) || '';
  const text = ROWS_PLOTLY.map(r => COLS.map(c => {
    const w    = r + c;
    const v    = plateData.wells[w]?.[spec.col];
    const cmpd = plateData.wells[w]?.compound || '';
    const vis  = compoundVisible(cmpd);
    if (!vis) return `<b>${w}</b><br>${cmpd}<br>(hidden)`;
    const isMedianInt = spec.col.startsWith('ImageQuality_MedianIntensity_');
    const sc   = isMedianInt ? (MAD_SCORES[spec.col]||{})[plateName]?.[w] : null;
    const zpl  = sc ? (sc.z_pl>=0?'+':'') + sc.z_pl + '× MAD (plate)'  : null;
    const zco  = sc ? (sc.z_co>=0?'+':'') + sc.z_co + '× MAD (cohort)' : null;
    const extraLine = (zpl && zco)
      ? `Plate MAD-score: ${zpl}<br>Cohort MAD-score: ${zco}`
      : '';
    const cmpdColor = cmpd === 'DMSO' ? '#90c870' : cmpd ? '#a8c8ff' : 'var(--muted)';
    return `<b style="font-size:12px">${w}</b> <span style="color:${cmpdColor}">${cmpd||'—'}</span><br>`+
           `${spec.title}: <b>${v!=null?v.toFixed(4):'N/A'}</b><br>`+
           `${extraLine}`;
  }));
  return {
    type:'heatmap', z, text, hoverinfo:'text',
    x:COLS, y:ROWS_PLOTLY, colorscale: spec.cs,
    zmin: spec.cmin, zmax: spec.cmax,
    showscale: false,
    xgap:4, ygap:4,
  };
}
 
function heatmapLayout(title, extraY) {
  const filtered = isFiltered();
  const gridcolor = filtered ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
  return {
    paper_bgcolor:'#090b14', plot_bgcolor:'#090b14',
    margin:{t:32,b:42,l:42,r:8}, height:340, width:520,
    title:{text:title, font:{size:11,color:'#8ab0e0'}, x:0.5},
    xaxis:{ type:'category', tickfont:{size:9}, showgrid:!filtered, gridcolor, zeroline:false,
             tickvals:COLS, ticktext:COLS.map(c=>parseInt(c)) },
    yaxis:{ type:'category', tickfont:{size:9}, showgrid:!filtered, gridcolor, zeroline:false,
             title: extraY ? {text:'Row', font:{size:9}} : undefined },
    hoverlabel:{
      bgcolor:'#141c34', bordercolor:'#304080',
      font:{size:12, color:'#d0e0ff', family:'monospace'},
    },
  };
}
 
const tabsEl     = document.getElementById('metrics-tabs');
const contentsEl = document.getElementById('metrics-contents');
 
function renderMetrics(plateName) {
  tabsEl.innerHTML = ''; contentsEl.innerHTML = '';
  const pd = DATA[plateName];
  let firstGroup = true;
  METRIC_GROUPS.forEach((grp, gi) => {
    const isFirst = firstGroup;
    tabsEl.insertAdjacentHTML('beforeend',
      `<div class="tab${isFirst?' active':''}" data-group="${gi}">${grp.label}</div>`);
    contentsEl.insertAdjacentHTML('beforeend',
      `<div class="tab-content${isFirst?' active':''}" id="grp-${gi}">
         <div class="channel-grid" id="chgrid-${gi}"></div></div>`);
    setTimeout(() => {
      const grid = document.getElementById(`chgrid-${gi}`);
      if (!grid) return;
      grp.specs.forEach((s, si) => {
        const cid  = `ch-${gi}-${si}`;
        const card = document.createElement('div');
        card.className = 'channel-card';
        card.innerHTML = `<div id="${cid}"></div>`;
        grid.appendChild(card);
        registerHeatmap(Plotly.react(cid, [makeHeatmap(pd, s)],
          heatmapLayout(s.title, si % 3 === 0),
          {responsive:true, displayModeBar:false}));
      });
    }, 0);
    if (isFirst) firstGroup = false;
  });
  tabsEl.querySelectorAll('.tab').forEach(tab => {
    tab.onclick = () => {
      const gi = tab.dataset.group;
      tabsEl.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      contentsEl.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      document.querySelector(`.tab[data-group="${gi}"]`).classList.add('active');
      document.getElementById(`grp-${gi}`).classList.add('active');
      document.querySelectorAll(`#chgrid-${gi} [id^="ch-"]`).forEach(el => Plotly.Plots.resize(el));
    };
  });
}
 
// ── 4. Cell counts — Cells heatmap + Cells/Nuclei ratio ──────────────────────
function renderCounts(plateName) {
  const grid = document.getElementById('counts-grid');
  grid.innerHTML = '';
  const pd = DATA[plateName];
 
// Card 1: Cells count heatmap con borde rojo para pozos bajo el umbral
  (function() {
    const cid  = `cnt-cells`;
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${cid}"></div>`;
    grid.appendChild(card);

    function drawCellsHeatmap() {
      const threshold = 700;
      const filtered  = isFiltered();
      const gridcolor = filtered ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
      const cRange    = COUNT_RANGES['Count_Cells'] || [null,null];

      const z    = ROWS_PLOTLY.map(r => COLS.map(c => {
        const w = r+c, well = pd.wells[w];
        if (!well || !compoundVisible(well.compound||'')) return null;
        return well['Count_Cells'] ?? null;
      }));
      const text = ROWS_PLOTLY.map(r => COLS.map(c => {
        const w = r+c, v = pd.wells[w]?.['Count_Cells'];
        const below = v != null && v < threshold;
        return `<b>${w}</b><br>${pd.wells[w]?.compound||''}<br>Cells: ${v!=null?Math.round(v):'N/A'}` +
               (below ? `<br><span style="color:#ff4444">⚠ below threshold (${threshold})</span>` : '');
      }));

      return Plotly.react(cid,
        [{type:'heatmap',z,text,hoverinfo:'text',x:COLS,y:ROWS_PLOTLY,colorscale:'Viridis',
           zmin:cRange[0],zmax:cRange[1],xgap:4,ygap:4,
           colorbar:{thickness:14,len:0.85,tickfont:{size:10},x:1.02,xanchor:'left'}}],
        {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#0a0c18',
          font:{color:'#c8d8f0',size:11},margin:{t:40,b:50,l:50,r:70},
          height:340, width:520,
          title:{text:`Cells (umbral: ${threshold})`,font:{size:13,color:'#a8c8ff'},x:0.5},
          xaxis:{type:'category',title:'Column',tickfont:{size:10},tickvals:COLS,ticktext:COLS.map(c=>parseInt(c)),
                  showgrid:!filtered,gridcolor,zeroline:false,fixedrange:true,automargin:false},
          yaxis:{type:'category',title:'Row',tickfont:{size:10},showgrid:!filtered,gridcolor,zeroline:false,fixedrange:true,automargin:false},
          hoverlabel:{bgcolor:'#141c34',bordercolor:'#304080',font:{size:12,color:'#d0e0ff',family:'monospace'}},
        },{responsive:false,displayModeBar:false});
    }

    registerHeatmap(drawCellsHeatmap());

  })();
 
  // Card 2: Cells/Nuclei ratio with custom colour scale (1=green, 0.99-095=yellow, <0.95=red)
  (function() {
    const cid  = `cnt-ratio`;
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${cid}"></div>`;
    grid.appendChild(card);
 
    const ratioCS = [
      [0,     '#ff4444'],
      [0.286, '#ff4444'],   // 0.95 = (0.95-0.93)/0.07
      [0.286, '#ffbe00'],
      [0.571, '#ffbe00'],   // 0.97 = (0.97-0.93)/0.07
      [0.571, '#4bd760'],
      [1.0,   '#4bd760'],
    ];
 
    const z    = ROWS_PLOTLY.map(r => COLS.map(c => {
      const w = r+c, well = pd.wells[w];
      if (!well || !compoundVisible(well.compound||'')) return null;
      const cells  = well['Count_Cells'];
      const nuclei = well['Count_Nuclei'];
      if (cells == null || nuclei == null || nuclei === 0) return null;
      return Math.min(1, cells / nuclei);
    }));
    const text = ROWS_PLOTLY.map(r => COLS.map(c => {
      const w    = r+c, well = pd.wells[w];
      const cells  = well?.['Count_Cells'];
      const nuclei = well?.['Count_Nuclei'];
      const ratio  = (cells != null && nuclei != null && nuclei > 0)
                     ? (cells/nuclei).toFixed(4) : 'N/A';
      return `<b>${w}</b><br>${well?.compound||''}<br>` +
             `Ratio: ${ratio}<br>Cells: ${cells!=null?Math.round(cells):'—'} · ` +
             `Nuclei: ${nuclei!=null?Math.round(nuclei):'—'}`;
    }));
    const filtered = isFiltered();
    const gridcolor = filtered ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
    registerHeatmap(Plotly.react(cid,
      [{type:'heatmap',z,text,hoverinfo:'text',x:COLS,y:ROWS_PLOTLY,
         colorscale:ratioCS, zmin:0.93, zmax:1.0, xgap:4,ygap:4,
         colorbar:{thickness:14,len:0.85,tickfont:{size:10},
                    tickvals:[0.93, 0.95, 0.97, 1.0],
                    ticktext:['<0.95','0.95','0.97','1.00']}}],
       {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#0a0c18',
        font:{color:'#c8d8f0',size:11},margin:{t:40,b:50,l:50,r:20},height:340, width:520,
        title:{text:'Cells / Nuclei ratio',font:{size:13,color:'#a8c8ff'},x:0.5},
        xaxis:{type:'category',title:'Column',tickfont:{size:10},tickvals:COLS,ticktext:COLS.map(c=>parseInt(c)),
                showgrid:!filtered,gridcolor,zeroline:false},
        yaxis:{type:'category',title:'Row',tickfont:{size:10},showgrid:!filtered,gridcolor,zeroline:false},
        hoverlabel:{bgcolor:'#141c34',bordercolor:'#304080',font:{size:12,color:'#d0e0ff',family:'monospace'}},
      },{responsive:true,displayModeBar:false}));
  })();
 
  // Card 3: Illumination Artifacts count heatmap (white=0, dark red=100+)
  (function() {
    const cid  = `cnt-artifacts`;
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${cid}"></div>`;
    grid.appendChild(card);
 
    // White (0 artifacts) -> dark red (100+ artifacts)
    const artifactCS = [
      [0,   '#ffffff'],
      [0.1, '#ffcccc'],
      [0.3, '#ff6666'],
      [0.6, '#cc2222'],
      [1.0, '#660000'],
    ];
 
    const z    = ROWS_PLOTLY.map(r => COLS.map(c => {
      const w = r+c, well = pd.wells[w];
      if (!well || !compoundVisible(well.compound||'')) return null;
      return well['Count_Illum_artifacts_filtered'] ?? null;
    }));
    const text = ROWS_PLOTLY.map(r => COLS.map(c => {
      const w    = r+c, well = pd.wells[w];
      const n    = well?.['Count_Illum_artifacts_filtered'];
      const cmpd = well?.compound || '';
      return `<b>${w}</b><br>${cmpd}<br>Illum. Artifacts: ${n!=null?Math.round(n):'N/A'}`;
    }));
    const filtered2 = isFiltered();
    const gridcolor2 = filtered2 ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
    registerHeatmap(Plotly.react(cid,
      [{type:'heatmap',z,text,hoverinfo:'text',x:COLS,y:ROWS_PLOTLY,
         colorscale:artifactCS, zmin:0, zmax:100, xgap:4,ygap:4,
         colorbar:{thickness:14,len:0.85,tickfont:{size:10},
                    tickvals:[0,100,250,100],ticktext:['0','100','250','≥100']}}],
       {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#0a0c18',
        font:{color:'#c8d8f0',size:11},margin:{t:40,b:50,l:50,r:20},height:340, width:520,
        title:{text:'Illum. Artifacts',font:{size:13,color:'#a8c8ff'},x:0.5},
        xaxis:{type:'category',title:'Column',tickfont:{size:10},tickvals:COLS,ticktext:COLS.map(c=>parseInt(c)),
                showgrid:!filtered2,gridcolor:gridcolor2,zeroline:false},
        yaxis:{type:'category',title:'Row',tickfont:{size:10},showgrid:!filtered2,gridcolor:gridcolor2,zeroline:false},
        hoverlabel:{bgcolor:'#141c34',bordercolor:'#304080',font:{size:12,color:'#d0e0ff',family:'monospace'}},
      },{responsive:true,displayModeBar:false}));
  })();
  // ── Histograms row ────────────────────────────────────────────────────────
  const histRow = document.getElementById('counts-hist-row');
  histRow.innerHTML = '';

  // Datos compartidos: recolectar todos los valores no-null por métrica
  const allCells    = [];
  const allRatios   = [];
  const allArtifacts = [];
  ROW_LABELS.forEach(r => COLS.forEach(c => {
    const w = r + c, well = pd.wells[w];
    if (!well || !compoundVisible(well.compound || '')) return;
    const cells    = well['Count_Cells'];
    const nuclei   = well['Count_Nuclei'];
    const arts     = well['Count_Illum_artifacts_filtered'];
    if (cells    != null) allCells.push(Math.round(cells));
    if (cells != null && nuclei != null && nuclei > 0)
      allRatios.push(cells / nuclei);
    if (arts != null) allArtifacts.push(Math.round(arts));
  }));

  // Helper: crear una card de histograma y agregarlo al histRow
  function addHistCard(cidHist, values, title, color, xLabel, threshold=null) {
    const card = document.createElement('div');
    card.className = 'count-hist-card';
    histRow.appendChild(card);
    if (!values.length) return;

    // Mediana
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) / 2;

    const medLabel = median < 10 ? median.toFixed(2) : Math.round(median);
    const thrLabel = threshold != null
      ? (threshold < 10 ? threshold.toFixed(2) : threshold)
      : null;

    card.innerHTML = `
      <div class="count-hist-header">
        <span class="count-hist-title">${title}</span>
        <span class="count-hist-stat">
          <span class="count-hist-line" style="background:#ff4444;border-top:2px dashed #ff4444;height:0;"></span>
          <span style="color:#ff4444;">Median: ${medLabel}</span>
        </span>
        ${thrLabel != null ? `
        <span class="count-hist-stat">
          <span class="count-hist-line" style="border-top:2px dotted #ffbe00;height:0;"></span>
          <span style="color:#ffbe00;">Bad: ${thrLabel}</span>
        </span>` : ''}
      </div>
      <div id="${cidHist}"></div>`;

    const nbins30 = 30
    const minV30 = Math.min(...values), maxV30 = Math.max(...values);
    const binW30 = (maxV30 - minV30) / nbins30;
    const bins30 = new Array(nbins30).fill(0);
    values.forEach(v => { const i = Math.min(Math.floor((v - minV30) / binW30), nbins30 - 1); bins30[i]++; });
    const ymax = Math.ceil(Math.max(...bins30) * 1.15);

    Plotly.react(cidHist,
          [
            {
              type: 'histogram', x: values,
              nbinsx: 30,
              marker: { color: color, opacity: 0.85, line: { color: 'rgba(0,0,0,0.3)', width: 0.5 } },
              name: xLabel,
              hovertemplate: `${xLabel}: %{x}<br>Wells: %{y}<extra></extra>`,
            },
            {
              type: 'scatter', mode: 'lines',
              x: [median, median], y: [0, ymax],
              line: { color: '#ff4444', width: 2, dash: 'dash' },
              hoverinfo: 'skip', showlegend: false,
            },
            {
              type: 'scatter', mode: 'lines',
              x: [threshold, threshold], y: [0, ymax],
              line: { color: '#ffbe00', width: 2, dash: 'dot' },
              hoverinfo: 'skip', showlegend: false,
              visible: threshold != null ? true : false,
            }
          ],
          {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#0a0c18',
            font: { color: '#c8d8f0', size: 11 },
            margin: { t: 40, b: 50, l: 50, r: 20 },
            height: 320,
            title: { text: '', },
            xaxis: { title: xLabel, tickfont: { size: 10 }, zeroline: false, gridcolor: 'rgba(80,90,120,0.4)' },
            yaxis: { title: 'Wells', tickfont: { size: 10 }, zeroline: false, gridcolor: 'rgba(80,90,120,0.4)', range: [0, ymax] },
            showlegend: false,
            bargap: 0.05,
            hoverlabel: { bgcolor: '#141c34', bordercolor: '#304080', font: { size: 12, color: '#d0e0ff', family: 'monospace' } },
          },
          { responsive: true, displayModeBar: false }
        );
      }

  function addRadiusHistCard(cidHist, values, title, color, xLabel) {
    const card = document.createElement('div');
    card.className = 'count-hist-card';
    histRow2.appendChild(card);
    if (!values.length) return;
    const sorted = [...values].sort((a, b) => a - b);
    const mid    = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid-1] + sorted[mid]) / 2;
    const medLabel = median.toFixed(2);
    card.innerHTML = `
      <div class="count-hist-header">
        <span class="count-hist-title">${title}</span>
        <span class="count-hist-stat">
          <span class="count-hist-line" style="border-top:2px dashed #ff4444;height:0;"></span>
          <span style="color:#ff4444;">Median: ${medLabel}</span>
        </span>
      </div>
      <div id="${cidHist}"></div>`;
    
    const nbins30=30
    const minV30 = Math.min(...values), maxV30 = Math.max(...values);
    const binW30 = (maxV30 - minV30) / nbins30;
    const bins30 = new Array(nbins30).fill(0);
    values.forEach(v => { const i = Math.min(Math.floor((v -minV30) / binW30), nbins30 - 1); bins30[i]++; });
    const ymax = Math.ceil(Math.max(...bins30) * 1.15)
    
    Plotly.react(cidHist,
      [
        {type:'histogram', x:values, nbinsx:30,
          marker:{color:color, opacity:0.85, line:{color:'rgba(0,0,0,0.3)', width:0.5}},
          hovertemplate:`${xLabel}: %{x}<br>Wells: %{y}<extra></extra>`,
        },
        {type:'scatter', mode:'lines',
          x:[median, median], y:[0, ymax],
          line:{color:'#ff4444', width:2, dash:'dash'},
          hoverinfo:'skip', showlegend:false,
        }
      ],
      {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#0a0c18',
        font:{color:'#c8d8f0', size:11},
        margin:{t:10, b:50, l:50, r:20},
        height:280, showlegend:false,
        xaxis:{title:xLabel, tickfont:{size:10}, zeroline:false, gridcolor:'rgba(80,90,120,0.4)'},
        yaxis:{title:'Wells', tickfont:{size:10}, zeroline:false, gridcolor:'rgba(80,90,120,0.4)', range: [0, ymax] },
        bargap:0.05,
        hoverlabel:{bgcolor:'#141c34', bordercolor:'#304080', font:{size:12, color:'#d0e0ff', family:'monospace'}},
      },
      {responsive:true, displayModeBar:false}
    );
  }

  const cellThreshold = parseInt(document.getElementById('cells-threshold-slider')?.value || 700);
  addHistCard('hist-cells',     allCells,     'Cells — distribution',        '#4b8fd7', 'Count_Cells',   cellThreshold);
  addHistCard('hist-ratio',     allRatios,    'Cells/Nuclei — distribution', '#4bd760', 'Ratio',         0.95);
  addHistCard('hist-artifacts', allArtifacts, 'Artifacts — distribution',    '#ff6666', 'Count_Artifacts', null);

  // ── Median Radius section ─────────────────────────────────────────────────
  const radiusGrid    = document.getElementById('radius-grid');
  const radiusHistRow = document.getElementById('radius-hist-row');
  const histRow2 = radiusHistRow;
  radiusGrid.innerHTML = '';
  radiusHistRow.innerHTML = '';

  const radiusData = RADIUS_DATA[plateName] || {};

  // Cohort-wide range para colorscale compartida
  const allRadiusVals = Object.values(RADIUS_DATA)
    .flatMap(p => ['Cells','Nuclei'].flatMap(src =>
      Object.values(p[src] || {})
    ));
  const rSorted  = [...allRadiusVals].filter(v => v != null).sort((a,b) => a-b);
  const rMin = rSorted.length ? rSorted[Math.floor(rSorted.length * 0.02)] : 0;
  const rMax = rSorted.length ? rSorted[Math.floor(rSorted.length * 0.98)] : 1;

  ['Cells', 'Nuclei'].forEach((src, si) => {
    const srcData = radiusData[src] || {};
    const cid     = `radius-${src.toLowerCase()}`;
    const color   = src === 'Cells' ? '#FFB43C' : '#64A0FF';

    // ── Platemap ──────────────────────────────────────────────────────────
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${cid}"></div>`;
    radiusGrid.appendChild(card);

    const z    = ROWS_PLOTLY.map(r => COLS.map(c => {
      const w = r + c;
      if (!pd.wells[w] || !compoundVisible(pd.wells[w].compound || '')) return null;
      return srcData[w] ?? null;
    }));
    const text = ROWS_PLOTLY.map(r => COLS.map(c => {
      const w   = r + c;
      const v   = srcData[w];
      const cmp = pd.wells[w]?.compound || '';
      return `<b>${w}</b><br>${cmp}<br>Median Radius (${src}): ${v != null ? v.toFixed(2) : 'N/A'}`;
    }));

    const filtered3  = isFiltered();
    const gridcolor3 = filtered3 ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
    registerHeatmap(Plotly.react(cid,
      [{type:'heatmap', z, text, hoverinfo:'text', x:COLS, y:ROWS_PLOTLY,
         colorscale:'Viridis', zmin:rMin, zmax:rMax, xgap:4, ygap:4,
         colorbar:{thickness:14, len:0.85, tickfont:{size:10}}}],
      {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#0a0c18',
        font:{color:'#c8d8f0', size:11},
        margin:{t:40, b:50, l:50, r:70},
        height:340, width:520,
        title:{text:`Median Radius — ${src}`, font:{size:13, color:'#a8c8ff'}, x:0.5},
        xaxis:{title:'Column', tickfont:{size:10}, tickvals:COLS,
                ticktext:COLS.map(c => parseInt(c)), showgrid:!filtered3,
                gridcolor:gridcolor3, zeroline:false, type:'category'},
        yaxis:{title:'Row', tickfont:{size:10}, showgrid:!filtered3,
                gridcolor:gridcolor3, zeroline:false, type:'category'},
        hoverlabel:{bgcolor:'#141c34', bordercolor:'#304080',
                     font:{size:12, color:'#d0e0ff', family:'monospace'}},
      },
      {responsive:true, displayModeBar:false}
    ));

    // ── Histograma ────────────────────────────────────────────────────────
    const histVals = Object.entries(srcData)
      .filter(([w]) => pd.wells[w] && compoundVisible(pd.wells[w].compound || ''))
      .map(([, v]) => v)
      .filter(v => v != null);

    addRadiusHistCard(`rhist-${src.toLowerCase()}`, histVals,
                      `Median Radius — ${src} distribution`, color, 'Median Radius (px)');
  });

  // Placeholder vacío para mantener el grid de 3 columnas balanceado
  const emptyCard = document.createElement('div');
  radiusGrid.appendChild(emptyCard);
  const emptyHist = document.createElement('div');
  radiusHistRow.appendChild(emptyHist);
}
 
// ── 5. MFI section ────────────────────────────────────────────────────────────
 
function mfiQuantile(sorted, q) {
  const pos = q * (sorted.length - 1);
  const lo  = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function mfiBoxStats(vals) {
  if (!vals || !vals.length) return null;
  const s    = [...vals].sort((a,b) => a-b);
  const q1   = mfiQuantile(s, 0.25), med = mfiQuantile(s, 0.50), q3 = mfiQuantile(s, 0.75);
  const iqr  = q3 - q1;
  return { q1, median:med, q3,
            whislo: s.find(v => v >= q1 - 1.5*iqr) ?? s[0],
            whishi: [...s].reverse().find(v => v <= q3 + 1.5*iqr) ?? s[s.length-1] };
}
function mfiHexAlpha(hex, alpha) {
  const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}
 
function mfiRenderBoxplot(divId, plateName, channel, groupBy) {
  const wellData   = (MFI_DATA[plateName]||{})[channel]||{};
  const color      = MFI_COLORS[channel]||'#8ab0d0';
  const categories = groupBy==='row' ? ROW_LABELS : COLS.map(String);
  const wellCmpd   = DATA[plateName]?.wells||{};
  const allVals    = Object.entries(wellData)
    .filter(([w]) => compoundVisible(wellCmpd[w]?.compound||''))
    .flatMap(([,v]) => v);
  const allSorted  = [...allVals].sort((a,b)=>a-b);
  const globalMed  = allSorted.length ? mfiQuantile(allSorted, 0.5) : null;
 
  const boxTraces=[], ptX=[], ptY=[];
  categories.forEach(cat => {
    const wells = Object.keys(wellData).filter(w => {
      const match = groupBy==='row' ? w[0]===cat : w.slice(1)===cat.padStart(2,'0');
      return match && compoundVisible(wellCmpd[w]?.compound||'');
    });
    let vals=[]; wells.forEach(w => vals=vals.concat(wellData[w]||[]));
    const stats = mfiBoxStats(vals);
    if (!stats) return;
    boxTraces.push({
      type:'box', name:String(cat), x:[String(cat)],
      q1:[stats.q1], median:[stats.median],
      q3:[stats.q3],
      marker:{color,size:3,opacity:0.7}, line:{color,width:1.5},
      fillcolor:mfiHexAlpha(color,0.18), showlegend:false,
    });
    const sample = vals.length>150 ? [...vals].sort(()=>Math.random()-0.5).slice(0,150) : vals;
    sample.forEach(v => { ptX.push(String(cat)); ptY.push(v); });
  });
 
    const imgVals = Object.entries(wellData)
    .filter(([w]) => compoundVisible(wellCmpd[w]?.compound||''))
    .map(([w]) => (MFI_IMG[plateName]||{})[w]?.[channel])
    .filter(v => v!=null);
    const imgMed = imgVals.length ? mfiQuantile([...imgVals].sort((a,b)=>a-b), 0.5) : null;

    const allYVals = [...ptY, globalMed??0, imgMed??0].filter(v=>v!=null);
    const yMin = allYVals.length ? Math.min(...allYVals)*0.97 : 0;
    const yMax = allYVals.length ? Math.max(...allYVals)*1.03 : 1;

    const shapes = [];
    if (globalMed!=null) shapes.push({type:'line',xref:'paper',yref:'y',x0:0,x1:1,
    y0:globalMed,y1:globalMed,line:{color:'#FF4040',width:2.5,dash:'dot'}});
    if (imgMed!=null) shapes.push({type:'line',xref:'paper',yref:'y',x0:0,x1:1,
    y0:imgMed,y1:imgMed,line:{color:'#FF9900',width:1.5,dash:'dash'}});

    Plotly.react(divId,
    [...boxTraces, {type:'scatter',mode:'markers',x:ptX,y:ptY,
        marker:{color:'#000',size:3,opacity:0.45},showlegend:false,hoverinfo:'none'}],
    { paper_bgcolor:'#090b14', plot_bgcolor:'#090b14',
        margin:{t:28,b:34,l:50,r:14},
        title:{text:`${channel} — by ${groupBy==='row'?'Row':'Column'}`,
                font:{color:'#8a9ab8',size:10},x:0.03,xanchor:'left'},
        xaxis:{type:'category',color:'#5a6a88',tickfont:{size:9,color:'#7a8aaa'},
                gridcolor:'#161c2c',tickmode:'array',tickvals:categories.map(String)},
        yaxis:{title:{text:'MFI',font:{color:'#5a6a88',size:9},standoff:4},
                color:'#5a6a88',tickfont:{size:9,color:'#7a8aaa'},
                gridcolor:'#161c2c',zeroline:false,range:[yMin,yMax]},
        shapes,
    }, {responsive:true,displayModeBar:false});
    }
    
function mfiRenderPlatemap(channel, plateName) {
  const grid     = document.getElementById(`mfi-grid-${channel}`);
  const wellData = (MFI_DATA[plateName]||{})[channel]||{};
  const color    = MFI_COLORS[channel]||'#8ab0d0';
  const wellCmpd = DATA[plateName]?.wells||{};
  grid.innerHTML = '';
 
  const wellMeds={};
  Object.entries(wellData).forEach(([w,vals]) => {
    const s=[...vals].sort((a,b)=>a-b);
    wellMeds[w] = s.length ? mfiQuantile(s,0.5) : 0;
  });
  const medVals=Object.values(wellMeds);
  const minMed=medVals.length?Math.min(...medVals):0, maxMed=medVals.length?Math.max(...medVals):1;
  const opacityFor = w => {
    const m=wellMeds[w];
    if (m===undefined) return 0.1;
    if (maxMed===minMed) return 0.6;
    return 0.18+0.82*(m-minMed)/(maxMed-minMed);
  };
 
  grid.appendChild(document.createElement('div')); // corner spacer
  COLS.forEach(c => {
    const lbl=document.createElement('div'); lbl.className='mfi-grid-label';
    lbl.textContent=parseInt(c); grid.appendChild(lbl);
  });
  ROW_LABELS.forEach(row => {
    const rowLbl=document.createElement('div'); rowLbl.className='mfi-grid-label';
    rowLbl.textContent=row; grid.appendChild(rowLbl);
    COLS.forEach(col => {
      const well=row+col, compound=wellCmpd[well]?.compound||'';
      const visible=compoundVisible(compound);
      const cell=document.createElement('div');
      cell.className='mfi-well-cell'+(visible?'':' dimmed');
      cell.id=`mfi-cell-${channel}-${well}`;
      cell.style.background = visible
        ? mfiHexAlpha(color, opacityFor(well))
        : 'rgba(16,20,34,1)';
 
      cell.addEventListener('mouseenter', e => {
        const vals=wellData[well]||[], s=[...vals].sort((a,b)=>a-b);
        const med=s.length?mfiQuantile(s,0.5):null;
        const q1=s.length?mfiQuantile(s,0.25):null, q3=s.length?mfiQuantile(s,0.75):null;
        const fmt=v=>v!=null?v.toFixed(4):'—';
        document.getElementById('mfi-tooltip').innerHTML =
          `<b style="font-size:12px">${well}</b> <span style="color:#90c870">${compound}</span><br>`+
          `Channel: <span style="color:${color}">${channel}</span><br>`+
          `Median: <b>${fmt(med)}</b><br>Q1: ${fmt(q1)} · Q3: ${fmt(q3)}<br>IQR: ${fmt(q3-q1)}<br>n sites: ${vals.length}`;
        const tt=document.getElementById('mfi-tooltip');
        tt.style.display='block'; tt.style.left=(e.clientX+15)+'px'; tt.style.top=(e.clientY-12)+'px';
      });
      cell.addEventListener('mousemove', e => {
        const tt=document.getElementById('mfi-tooltip');
        tt.style.left=(e.clientX+15)+'px'; tt.style.top=(e.clientY-12)+'px';
      });
      cell.addEventListener('mouseleave', () => document.getElementById('mfi-tooltip').style.display='none');
      if (visible) cell.addEventListener('click', () => {
        if (currentPlate) selectWell(currentPlate, well);
      });
      grid.appendChild(cell);
    });
  });
}
 
function mfiAnova(groups) {
  const allVals = groups.flat();
  if (allVals.length < 2) return null;
  const grandMean = allVals.reduce((a,b)=>a+b,0) / allVals.length;
  const ssBetween = groups.reduce((s,g) => {
    if (!g.length) return s;
    const gMean = g.reduce((a,b)=>a+b,0)/g.length;
    return s + g.length*(gMean-grandMean)**2;
  }, 0);
  const ssTotal = allVals.reduce((s,v) => s+(v-grandMean)**2, 0);
  if (ssTotal === 0) return null;
  return ssBetween / ssTotal;
}

function eta2Badge(eta2, label, baseline) {
  if (eta2 == null) return '';
  let color, interpretation;
  if (baseline == null) {
    // Hoechst — criterio Cohen absoluto
    if      (eta2 < 0.01) { color='#4bd760'; interpretation='negligible'; }
    else if (eta2 < 0.06) { color='#ffbe00'; interpretation='small';      }
    else if (eta2 < 0.14) { color='#ffbe00'; interpretation='medium';     }
    else                  { color='#ff4444'; interpretation='large';      }
  } else {
        const ratio = eta2 / baseline;
        if (eta2 >= 0.20) {
        color='#ff4444'; interpretation=`${ratio.toFixed(1)}× baseline (absolute ceiling)`;
        } else if (ratio < 2 || eta2 < 0.06) {
        // Verde si ratio bajo O si eta2 es pequeño en términos absolutos (Cohen small)
        color='#4bd760'; interpretation=`${ratio.toFixed(1)}× baseline`;
        } else if (ratio < 4 || eta2 < 0.14) {
        // Amarillo si ratio medio O si eta2 es medium en Cohen
        color='#ffbe00'; interpretation=`${ratio.toFixed(1)}× baseline`;
        } else {
        color='#ff4444'; interpretation=`${ratio.toFixed(1)}× baseline`;
        }
  }
  return `<span title="η²=${eta2.toFixed(4)} — ${interpretation}"
    style="font-size:0.72rem;margin-left:6px;padding:1px 7px;border-radius:10px;
           background:${color}22;color:${color};border:1px solid ${color}55;cursor:help;">
    η²<sub>${label}</sub>: ${eta2.toFixed(3)}
  </span>`;
}

function renderMFI() {
  const plateName = currentPlate;
  if (!plateName) return;
  const container = document.getElementById('mfi-content');
  container.innerHTML = '';

  if (!MFI_CHANNELS.length) {
    container.innerHTML='<p style="color:var(--muted);padding:16px;">No MFI data found.</p>';
    return;
  }

  // Baseline de Hoechst para criterio relativo
  let hoechstEta2Row = null, hoechstEta2Col = null;
  if ((MFI_DATA[plateName]||{})['Hoechst']) {
    const hData = MFI_DATA[plateName]['Hoechst'];
    const hRowGroups = ROW_LABELS.map(r => COLS.flatMap(c => hData[r+c]||[]));
    const hColGroups = COLS.map(c => ROW_LABELS.flatMap(r => hData[r+c]||[]));
    hoechstEta2Row = mfiAnova(hRowGroups);
    hoechstEta2Col = mfiAnova(hColGroups);
    if (hoechstEta2Row!=null && hoechstEta2Col!=null) {
      const maxH = Math.max(hoechstEta2Row, hoechstEta2Col);
      const diff = Math.abs(hoechstEta2Row - hoechstEta2Col) / maxH;
      if (diff < 0.20) {
        const mean = (hoechstEta2Row + hoechstEta2Col) / 2;
        hoechstEta2Row = mean;
        hoechstEta2Col = mean;
      }
    }
  }

  MFI_CHANNELS.forEach(ch => {
    const color    = MFI_COLORS[ch]||'#8ab0d0';
    const wellData = (MFI_DATA[plateName]||{})[ch]||{};
    if (!Object.keys(wellData).length) return;
    const allVals  = Object.values(wellData).flat();
    const plateMed = allVals.length ? mfiQuantile([...allVals].sort((a,b)=>a-b),0.5) : null;

    const rowGroups = ROW_LABELS.map(r => COLS.flatMap(c => wellData[r+c]||[]));
    const colGroups = COLS.map(c => ROW_LABELS.flatMap(r => wellData[r+c]||[]));
    const eta2Row   = mfiAnova(rowGroups);
    const eta2Col   = mfiAnova(colGroups);
    const isHoechst = ch === 'Hoechst';
    const baseRow   = isHoechst ? null : hoechstEta2Row;
    const baseCol   = isHoechst ? null : hoechstEta2Col;
    const badgeRow  = eta2Badge(eta2Row, 'row', baseRow);
    const badgeCol  = eta2Badge(eta2Col, 'col', baseCol);

    const sec=document.createElement('div'); sec.className='mfi-channel-section';
    sec.innerHTML=`
      <div class="mfi-channel-header">
        <span class="mfi-ch-dot" style="background:${color}"></span>
        <h3>${ch} <span style="color:var(--muted);font-size:0.78rem;font-weight:normal;">
        — obj median: ${plateMed!=null?plateMed.toFixed(5):'—'}
        ${(() => {
          const imgVals = Object.entries((MFI_IMG[plateName]||{}))
            .map(([w,chs]) => chs[ch]).filter(v=>v!=null);
          const imgMed = imgVals.length ? mfiQuantile([...imgVals].sort((a,b)=>a-b), 0.5) : null;
          const delta  = (plateMed!=null && imgMed!=null) ? plateMed - imgMed : null;
          const snr    = (delta!=null && imgMed>0) ? delta / imgMed : null;
          if (imgMed==null) return '';
          let snrColor, snrLabel;
          if      (snr==null)  { snrColor='var(--muted)'; snrLabel='N/A'; }
          else if (snr >= 0.5) { snrColor='#4bd760';      snrLabel=`${snr.toFixed(2)}×`; }
          else if (snr >= 0.2) { snrColor='#ffbe00';      snrLabel=`${snr.toFixed(2)}×`; }
          else                 { snrColor='#ff4444';      snrLabel=`${snr.toFixed(2)}×`; }
          return `· img: ${imgMed.toFixed(5)} · <b>Δ: ${delta>=0?'+':''}${delta.toFixed(5)}</b>`
            + ` · <span title="SNR proxy = Δ / MFI_img — fold-change of objects over background"
                style="color:${snrColor};cursor:help;">SNR: ${snrLabel}</span>`;
        })()}
        </span></h3>
        ${badgeRow}${badgeCol}
        ${isHoechst ? '<span style="font-size:0.7rem;color:var(--muted);margin-left:8px;">(Cohen 1988 — baseline reference)</span>' : ''}
      </div>
      <div style="font-size:0.72rem;color:var(--muted);padding:4px 14px 6px;background:#0d0f1e;border-bottom:1px solid var(--border);">
        <span style="color:#FF4040;">— — —</span> obj median &nbsp;
        <span style="color:#FF9900;">- - -</span> img median &nbsp;·&nbsp;
        Δ = MFI<sub>obj</sub> − MFI<sub>img</sub>
      </div>
      <div class="mfi-body">
        <div class="mfi-boxplot-col">
          <div class="mfi-plot-box" id="mfi-box-row-${ch}"></div>
          <div class="mfi-plot-divider"></div>
          <div class="mfi-plot-box" id="mfi-box-col-${ch}"></div>
        </div>
        <div class="mfi-platemap-col">
          <div class="mfi-platemap-title">Platemap — median MFI</div>
          <div class="mfi-platemap-grid" id="mfi-grid-${ch}"></div>
        </div>
      </div>`;
    container.appendChild(sec);
    setTimeout(()=>mfiRenderBoxplot(`mfi-box-row-${ch}`,plateName,ch,'row'),0);
    setTimeout(()=>mfiRenderBoxplot(`mfi-box-col-${ch}`,plateName,ch,'col'),0);
    setTimeout(()=>mfiRenderPlatemap(ch,plateName),0);
  });
}

// ── 6. Flagged wells ──────────────────────────────────────────────────────────
(function buildCompoundToolbar() {
  const toolbar  = document.getElementById('compound-toolbar');
  const allCmpds = new Set();
  PLATES.forEach(p => {
    Object.values(DATA[p].wells).forEach(m => {
      if (m.compound) allCmpds.add(m.compound);
    });
  });
  [...allCmpds].sort().forEach(cmpd => {
    const btn = document.createElement('button');
    btn.className = 'cmpd-btn'; btn.textContent = cmpd; btn.dataset.cmpd = cmpd;
    btn.onclick = () => {
      hideAllMode = false;
      btn.classList.toggle('active');
      btn.classList.contains('active') ? activeCompounds.add(cmpd) : activeCompounds.delete(cmpd);
      applyGlobalFilter();
    };
    toolbar.appendChild(btn);
  });
  document.getElementById('btn-show-all').onclick = () => {
    hideAllMode = false; activeCompounds.clear();
    toolbar.querySelectorAll('.cmpd-btn:not(.ctrl)').forEach(b => b.classList.remove('active'));
    applyGlobalFilter();
  };
  document.getElementById('btn-hide-all').onclick = () => {
    hideAllMode = true; activeCompounds.clear();
    toolbar.querySelectorAll('.cmpd-btn:not(.ctrl)').forEach(b => b.classList.remove('active'));
    applyGlobalFilter();
  };
})();
 
function applyGlobalFilter() {
  if (!currentPlate) return;
  renderMetrics(currentPlate);
  renderCounts(currentPlate);
  renderMFI();
  // Los gráficos se recrean en setTimeout(…,0); re-aplicar el resaltado del well
  // seleccionado una vez que las celdas/heatmaps existan de nuevo.
  if (selectedWell) setTimeout(() => {
    highlightWellEverywhere(selectedWell);
    renderSiteTables(currentPlate, selectedWell);
  }, 0);
}

 
// ── Init ──────────────────────────────────────────────────────────────────────
renderPlate(0);
// ═════════════════════════════════════════════════════════════════════════════
// IMAGE VIEWER (optional layer) — real TIFF images per well·site·channel.
// Files are user-provided (file picker), decoded locally with UTIF.js. Nothing
// is uploaded. Intensity windowing is visualization-only; the histogram shows
// the untouched 16-bit values so quantitative judgement stays on real data.
// ═════════════════════════════════════════════════════════════════════════════
(function initImageViewer() {
  const IMG_CHANNELS = ['Hoechst', 'Syto', 'Golgi', 'ER', 'Mito', 'Brightfield'];
  const COMPRESSION_NAMES = {1:'Uncompressed',5:'LZW',6:'JPEG (old)',7:'JPEG',
    8:'Deflate (Adobe)',32773:'PackBits',32946:'Deflate'};
  const SAMPLEFORMAT_NAMES = {1:'unsigned int',2:'signed int',3:'float'};
  const PHOTOMETRIC_NAMES = {0:'WhiteIsZero',1:'BlackIsZero',2:'RGB',3:'Palette'};

  // Estado del visor
  const S = {
    imgIndex: null, fileMap: null, rawRows: null,
    plate: null, well: null, field: null, channel: 'Hoechst',
    meta: null, hist: null, win: {lo:0, hi:65535},
    brightness: 1, contrast: 1,
    regexPlate: '', regexName: '',
    // zoom/pan: escala y offset del canvas mostrado
    zoom: 1, panX: 0, panY: 0,
    histRegion: false,   // histograma sobre región visible vs imagen completa
  };

  // Extrae {plate?, well?, field?} vía regex; grupos por nombre o por orden.
  // Para el nombre: si hay 3 grupos numéricos (row,col,field) se arma el well.
  function applyPlateRegex(pathStr, pattern) {
    if (!pattern) return null;
    try {
      const m = new RegExp(pattern).exec(pathStr || '');
      if (!m) return null;
      // grupo nombrado 'plate' o primer grupo de captura
      const g = m.groups && m.groups.plate != null ? m.groups.plate : m[1];
      return g != null ? String(g).trim() : null;
    } catch { return null; }
  }
  function applyNameRegex(fileName, pattern) {
    if (!pattern) return null;
    try {
      const m = new RegExp(pattern).exec(fileName || '');
      if (!m) return null;
      if (m.groups && (m.groups.well || m.groups.row)) {
        if (m.groups.well) {
          return { well: String(m.groups.well).trim().toUpperCase(),
                   field: m.groups.field != null ? String(m.groups.field).trim() : null };
        }
        // row+col numéricos → letra+padded
        const row = parseInt(m.groups.row, 10), col = parseInt(m.groups.col, 10);
        const well = String.fromCharCode(64 + row) + String(col).padStart(2, '0');
        return { well, field: m.groups.field != null ? String(m.groups.field).trim() : null };
      }
      // por orden: (row)(col)(field) numéricos
      if (m.length >= 4) {
        const row = parseInt(m[1], 10), col = parseInt(m[2], 10);
        const well = String.fromCharCode(64 + row) + String(col).padStart(2, '0');
        return { well, field: String(m[3]).trim() };
      }
      return null;
    } catch { return null; }
  }

  // ── CSV mínimo (el Image.txt reducido: sin comas internas) ──────────────────
  function parseCSV(text) {
    const lines = text.replace(/\r\n/g, '\n').split('\n').filter(l => l.length);
    if (!lines.length) return [];
    const split = line => {
      // soporta campos entrecomillados por robustez, aunque el CSV no los use
      const out = []; let cur = '', q = false;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (q) { if (c === '"') q = false; else cur += c; }
        else if (c === '"') q = true;
        else if (c === ',') { out.push(cur); cur = ''; }
        else cur += c;
      }
      out.push(cur); return out;
    };
    const header = split(lines[0]);
    return lines.slice(1).map(l => {
      const cells = split(l), row = {};
      header.forEach((h, i) => row[h.trim()] = (cells[i] ?? '').trim());
      return row;
    });
  }

  function buildImageIndex(rows) {
    const index = {};
    for (const row of rows) {
      // plate: regex sobre el path (si se dio) o Metadata_Plate del CSV
      let plate = String(row.Metadata_Plate ?? '').trim();
      if (S.regexPlate) {
        // usar el path de cualquier canal disponible
        let anyPath = '';
        for (const ch of IMG_CHANNELS) { if (row[`Image_PathName_${ch}`]) { anyPath = row[`Image_PathName_${ch}`]; break; } }
        const p = applyPlateRegex(anyPath, S.regexPlate);
        if (p != null) plate = p;
      }
      // well/field: regex sobre el file name (si se dio) o Metadata_* del CSV
      let well  = String(row.Metadata_Well  ?? '').trim().toUpperCase();
      let field = String(row.Metadata_Field ?? '').trim();
      if (S.regexName) {
        let anyName = '';
        for (const ch of IMG_CHANNELS) { if (row[`Image_FileName_${ch}`]) { anyName = row[`Image_FileName_${ch}`]; break; } }
        const r = applyNameRegex(anyName, S.regexName);
        if (r) { if (r.well) well = r.well; if (r.field != null) field = r.field; }
      }
      if (!well || !field) continue;
      const byField = ((index[plate] = index[plate] || {})[well] = index[plate][well] || {});
      const byChannel = (byField[field] = byField[field] || {});
      for (const ch of IMG_CHANNELS) {
        const fn = row[`Image_FileName_${ch}`];
        if (fn) byChannel[ch] = { file: String(fn).trim(),
                                 path: String(row[`Image_PathName_${ch}`] ?? '').trim() };
      }
    }
    return index;
  }

  function buildFileMap(fileList) {
    const map = {};
    for (const f of fileList) map[f.name.split(/[\\/]/).pop().toLowerCase()] = f;
    return map;
  }

  // ── Decodificación TIFF + metadatos ─────────────────────────────────────────
  function decodeTiff(ab) {
    const ifds = UTIF.decode(ab);
    const ifd = ifds[0];
    UTIF.decodeImage(ab, ifd, ifds);
    const bits = (ifd.t258 && ifd.t258[0]) || 8;
    const spp  = (ifd.t277 && ifd.t277[0]) || 1;
    const cmpr = (ifd.t259 && ifd.t259[0]) || 1;
    const sfmt = (ifd.t339 && ifd.t339[0]) || 1;
    const photo = (ifd.t262 && ifd.t262[0]);
    const px = ifd.width * ifd.height;
    let raw, maxPossible;
    if (bits === 16) { raw = new Uint16Array(ifd.data.buffer, ifd.data.byteOffset, px*spp); maxPossible = 65535; }
    else if (bits === 8) { raw = new Uint8Array(ifd.data.buffer, ifd.data.byteOffset, px*spp); maxPossible = 255; }
    else { raw = new Float32Array(ifd.data.buffer, ifd.data.byteOffset, px*spp); maxPossible = null; }
    return {width:ifd.width, height:ifd.height, bits, samplesPerPixel:spp,
            compression:cmpr, sampleFormat:sfmt, photometric:photo, raw, maxPossible};
  }

  function computeHistogram(raw, spp, maxPossible, nBins) {
    nBins = nBins || 256;
    const n = spp === 1 ? raw.length : Math.floor(raw.length / spp);
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (let i = 0; i < n; i++) { const v = raw[i*spp]; if (v<mn)mn=v; if (v>mx)mx=v; sum+=v; }
    const hi = maxPossible != null ? maxPossible : mx;
    const lo = maxPossible != null ? 0 : mn;
    const span = (hi - lo) || 1;
    const bins = new Float64Array(nBins);
    for (let i = 0; i < n; i++) {
      let b = Math.floor(((raw[i*spp] - lo) / span) * nBins);
      if (b < 0) b = 0; if (b >= nBins) b = nBins - 1; bins[b]++;
    }
    return {min:mn, max:mx, mean:sum/n, nPixels:n, bins, nBins, lo, hi,
            satFrac:bins[nBins-1]/n, zeroFrac:bins[0]/n, maxPossible};
  }

  // Histograma sobre una región rectangular [x0,y0]-[x1,y1] (píxeles imagen).
  function computeHistogramRegion(raw, spp, width, maxPossible, x0, y0, x1, y1, nBins) {
    nBins = nBins || 256;
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(width, Math.ceil(x1)); y1 = Math.ceil(y1);
    let mn = Infinity, mx = -Infinity, sum = 0, n = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const v = raw[(y*width + x)*spp];
        if (v<mn)mn=v; if (v>mx)mx=v; sum+=v; n++;
      }
    }
    if (n === 0) return computeHistogram(raw, spp, maxPossible, nBins);
    const hi = maxPossible != null ? maxPossible : mx;
    const lo = maxPossible != null ? 0 : mn;
    const span = (hi - lo) || 1;
    const bins = new Float64Array(nBins);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        let b = Math.floor(((raw[(y*width + x)*spp] - lo) / span) * nBins);
        if (b < 0) b = 0; if (b >= nBins) b = nBins - 1; bins[b]++;
      }
    }
    return {min:mn, max:mx, mean:sum/n, nPixels:n, bins, nBins, lo, hi,
            satFrac:bins[nBins-1]/n, zeroFrac:bins[0]/n, maxPossible, region:true};
  }

  function percentilesFromHist(hist, plo, phi) {
    const total = hist.nPixels, tLo = total*(plo/100), tHi = total*(phi/100);
    const binW = (hist.hi - hist.lo) / hist.nBins;
    let acc = 0, vLo = hist.lo, vHi = hist.hi, foundLo = false;
    for (let b = 0; b < hist.nBins; b++) {
      acc += hist.bins[b];
      if (!foundLo && acc >= tLo) { vLo = hist.lo + b*binW; foundLo = true; }
      if (acc >= tHi) { vHi = hist.lo + (b+1)*binW; break; }
    }
    if (vHi <= vLo) vHi = vLo + binW;
    return {lo:vLo, hi:vHi};
  }

  function renderRGBA(raw, spp, w, h, win, brightness, contrast) {
    const n = w*h, out = new Uint8ClampedArray(n*4), span = (win.hi - win.lo) || 1;
    for (let i = 0; i < n; i++) {
      let g = ((raw[i*spp] - win.lo) / span) * 255;
      g = (g - 128) * contrast + 128; g = g * brightness;
      if (g < 0) g = 0; else if (g > 255) g = 255;
      const o = i*4; out[o]=out[o+1]=out[o+2]=g; out[o+3]=255;
    }
    return out;
  }

  // ── DOM refs ────────────────────────────────────────────────────────────────
  const el = id => document.getElementById(id);
  const refs = {};
  ['iv-csv-input','iv-files-input','iv-files-label','iv-status','iv-body',
   'iv-plate','iv-well','iv-field','iv-channel-tabs','iv-canvas','iv-canvas-msg',
   'iv-transform-detail','iv-meta-rows','iv-hist-canvas','iv-hist-stats','iv-alerts',
   'iv-plo','iv-phi','iv-auto','iv-wlo','iv-whi','iv-brightness','iv-contrast',
   'iv-brightness-val','iv-contrast-val','iv-reset',
   'iv-regex-plate','iv-regex-name','iv-regex-hint','iv-loadnote','iv-loadnote-path',
   'iv-canvas-wrap','iv-zoom-hud','iv-zoom-level','iv-zoom-reset','iv-hist-region'
  ].forEach(id => refs[id] = el(id));

  if (!refs['iv-csv-input']) return;  // sección no presente → salir

  // ── Carga de CSV ────────────────────────────────────────────────────────────
  refs['iv-csv-input'].addEventListener('change', ev => {
    const file = ev.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const rows = parseCSV(reader.result);
        S.rawRows = rows;
        S.imgIndex = buildImageIndex(rows);
        const nPlates = Object.keys(S.imgIndex).length;
        const nRows = rows.length;
        setStatus(`CSV loaded: ${nRows} rows, ${nPlates} plate(s). Now select the .tif files.`, 'ok');
        refs['iv-files-input'].disabled = false;
        refs['iv-files-label'].classList.remove('iv-btn-disabled');
        populatePlateSelect();
        refs['iv-body'].style.display = 'block';
      } catch (e) { setStatus('Could not parse CSV: ' + e.message, 'warn'); }
    };
    reader.readAsText(file);
  });

  // ── Selección de archivos TIFF ──────────────────────────────────────────────
  refs['iv-files-input'].addEventListener('change', ev => {
    const files = Array.from(ev.target.files);
    if (!files.length) return;
    S.fileMap = buildFileMap(files);
    setStatus(`${files.length} image file(s) ready. Select well · site · channel below.`, 'ok');
    tryRenderCurrent();
  });

  function setStatus(msg, cls) {
    refs['iv-status'].textContent = msg;
    refs['iv-status'].className = 'iv-status' + (cls ? ' ' + cls : '');
  }

  // ── Selectores encadenados ──────────────────────────────────────────────────
  function populatePlateSelect() {
    const plates = Object.keys(S.imgIndex).sort();
    refs['iv-plate'].innerHTML = plates.map(p => `<option value="${p}">${p}</option>`).join('');
    S.plate = plates[0];
    populateWellSelect();
  }
  function populateWellSelect() {
    const wells = Object.keys(S.imgIndex[S.plate] || {}).sort();
    refs['iv-well'].innerHTML = wells.map(w => `<option value="${w}">${w}</option>`).join('');
    S.well = wells[0];
    populateFieldSelect();
  }
  function populateFieldSelect() {
    const fields = Object.keys((S.imgIndex[S.plate] || {})[S.well] || {})
      .sort((a,b) => (+a) - (+b));
    refs['iv-field'].innerHTML = fields.map(f => `<option value="${f}">${f}</option>`).join('');
    S.field = fields[0];
    populateChannelTabs();
  }
  function populateChannelTabs() {
    const avail = ((S.imgIndex[S.plate] || {})[S.well] || {})[S.field] || {};
    refs['iv-channel-tabs'].innerHTML = '';
    IMG_CHANNELS.forEach(ch => {
      const b = document.createElement('button');
      const has = !!avail[ch];
      b.className = 'iv-ch-tab' + (ch === S.channel ? ' active' : '') + (has ? '' : ' missing');
      b.textContent = ch;
      b.title = has ? avail[ch].file : 'no file name in CSV for this channel';
      b.onclick = () => { S.channel = ch; refreshChannelTabs(); tryRenderCurrent(); };
      refs['iv-channel-tabs'].appendChild(b);
    });
    tryRenderCurrent();
  }
  function refreshChannelTabs() {
    [...refs['iv-channel-tabs'].children].forEach(b =>
      b.classList.toggle('active', b.textContent === S.channel));
  }

  refs['iv-plate'].onchange = e => { S.plate = e.target.value; populateWellSelect(); };
  refs['iv-well'].onchange  = e => { S.well  = e.target.value; populateFieldSelect(); };
  refs['iv-field'].onchange = e => { S.field = e.target.value; populateChannelTabs(); };

  // ── Nota "Load file" con estado (provided/missing) ──────────────────────────
  function updateLoadNote() {
    const entry = ((S.imgIndex?.[S.plate]?.[S.well]?.[S.field]) || {})[S.channel];
    const note = refs['iv-loadnote'], pathEl = refs['iv-loadnote-path'];
    // limpiar tag previo
    const oldTag = note.querySelector('.iv-loadnote-tag'); if (oldTag) oldTag.remove();
    note.classList.remove('provided', 'missing');
    if (!entry) { pathEl.textContent = '— no file name in CSV for this channel —'; return null; }
    const full = entry.path ? `${entry.path}/${entry.file}` : entry.file;
    pathEl.textContent = full;
    const provided = !!(S.fileMap && S.fileMap[entry.file.toLowerCase()]);
    note.classList.add(provided ? 'provided' : 'missing');
    const tag = document.createElement('span');
    tag.className = 'iv-loadnote-tag ' + (provided ? 'provided' : 'missing');
    tag.textContent = provided ? 'provided ✓' : 'not provided';
    note.appendChild(tag);
    return entry;
  }

  // ── Render de la imagen seleccionada ────────────────────────────────────────
  function tryRenderCurrent() {
    const entry = updateLoadNote();
    const canvas = refs['iv-canvas'], msg = refs['iv-canvas-msg'];
    if (!entry) { showMsg('No file name in the CSV for this channel.'); return; }
    if (!S.fileMap) { showMsg('Select the .tif files (button 2) to display images.'); return; }
    const file = S.fileMap[entry.file.toLowerCase()];
    if (!file) { showMsg(`File "${entry.file}" not among the selected files.`); return; }

    showMsg('Decoding…');
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const meta = decodeTiff(reader.result);
        S.meta = meta;
        // reset de zoom/pan al cargar nueva imagen
        S.zoom = 1; S.panX = 0; S.panY = 0;
        recomputeHist();
        applyAutoWindow();
        renderMeta(entry);
        drawHistogram();
        paintCanvas();
        renderAlerts();
        canvas.style.display = 'block'; msg.style.display = 'none';
        refs['iv-zoom-hud'].style.display = 'flex';
        updateZoomLabel();
      } catch (e) { showMsg('Decode error: ' + e.message); }
    };
    reader.readAsArrayBuffer(file);
  }

  // Histograma sobre imagen completa o región visible según toggle.
  function recomputeHist() {
    const m = S.meta;
    if (S.histRegion && S.zoom > 1) {
      const r = visibleRegion();
      S.hist = computeHistogramRegion(m.raw, m.samplesPerPixel, m.width, m.maxPossible,
                                      r.x0, r.y0, r.x1, r.y1, 256);
    } else {
      S.hist = computeHistogram(m.raw, m.samplesPerPixel, m.maxPossible, 256);
    }
  }


  function showMsg(t) {
    refs['iv-canvas'].style.display = 'none';
    refs['iv-canvas-msg'].style.display = 'block';
    refs['iv-canvas-msg'].textContent = t;
  }

  function applyAutoWindow() {
    const plo = parseFloat(refs['iv-plo'].value), phi = parseFloat(refs['iv-phi'].value);
    S.win = percentilesFromHist(S.hist, isNaN(plo)?0.5:plo, isNaN(phi)?99.5:phi);
    refs['iv-wlo'].value = Math.round(S.win.lo);
    refs['iv-whi'].value = Math.round(S.win.hi);
  }

  // Canvas fuente offscreen con la imagen a 8-bit (se rehace solo al cambiar
  // windowing/brillo/contraste, no al hacer zoom/pan).
  let _srcCanvas = null;
  function rebuildSource() {
    const m = S.meta;
    if (!_srcCanvas) _srcCanvas = document.createElement('canvas');
    _srcCanvas.width = m.width; _srcCanvas.height = m.height;
    const sctx = _srcCanvas.getContext('2d');
    const rgba = renderRGBA(m.raw, m.samplesPerPixel, m.width, m.height, S.win, S.brightness, S.contrast);
    sctx.putImageData(new ImageData(rgba, m.width, m.height), 0, 0);
  }

  // Viewport fijo (el canvas visible). La imagen se dibuja escalada por zoom con
  // offset pan. clamp del pan para no salir de la imagen.
  const VIEW = 512;
  function clampPan() {
    const m = S.meta;
    const scale = baseScale() * S.zoom;
    const imgW = m.width * scale, imgH = m.height * scale;
    // permitir pan solo si la imagen excede el viewport
    const maxX = Math.max(0, (imgW - VIEW) / 2), maxY = Math.max(0, (imgH - VIEW) / 2);
    S.panX = Math.max(-maxX, Math.min(maxX, S.panX));
    S.panY = Math.max(-maxY, Math.min(maxY, S.panY));
  }
  function baseScale() {
    const m = S.meta;
    return Math.min(VIEW / m.width, VIEW / m.height);  // fit inicial
  }

  function paintCanvas(skipSrc) {
    const m = S.meta, canvas = refs['iv-canvas'];
    if (!skipSrc) rebuildSource();
    canvas.width = VIEW; canvas.height = VIEW;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#050810'; ctx.fillRect(0, 0, VIEW, VIEW);
    clampPan();
    const scale = baseScale() * S.zoom;
    const imgW = m.width * scale, imgH = m.height * scale;
    const dx = (VIEW - imgW) / 2 + S.panX, dy = (VIEW - imgH) / 2 + S.panY;
    ctx.drawImage(_srcCanvas, dx, dy, imgW, imgH);
    refs['iv-transform-detail'].textContent =
      `16-bit → 8-bit · window [${Math.round(S.win.lo)}, ${Math.round(S.win.hi)}]` +
      ` · brightness ${S.brightness.toFixed(2)}× · contrast ${S.contrast.toFixed(2)}×`;
  }

  // Región de la imagen (en píxeles fuente) actualmente visible en el viewport.
  function visibleRegion() {
    const m = S.meta;
    const scale = baseScale() * S.zoom;
    const imgW = m.width * scale, imgH = m.height * scale;
    const dx = (VIEW - imgW) / 2 + S.panX, dy = (VIEW - imgH) / 2 + S.panY;
    // esquina superior izquierda del viewport en coords imagen
    const x0 = Math.max(0, (0 - dx) / scale), y0 = Math.max(0, (0 - dy) / scale);
    const x1 = Math.min(m.width, (VIEW - dx) / scale), y1 = Math.min(m.height, (VIEW - dy) / scale);
    return {x0, y0, x1, y1};
  }

  function updateZoomLabel() { refs['iv-zoom-level'].textContent = S.zoom.toFixed(1) + '×'; }

  function renderMeta(entry) {
    const m = S.meta;
    // ¿la imagen mostrada está en la lista pedida (CSV) para la selección actual?
    // Siempre lo está por construcción (entry viene del índice), pero marcamos si
    // el archivo entregado no coincide exactamente con lo pedido.
    const offList = !(S.fileMap && S.fileMap[entry.file.toLowerCase()]);
    const nameCell = entry.file +
      (offList ? ' <span class="iv-meta-tag off-list">not in requested list</span>' : '');
    const rows = [
      ['File name', nameCell],
      ['Source path', entry.path || '—'],
      ['Dimensions', `${m.width} × ${m.height} px`],
      ['Bit depth', `${m.bits}-bit ${SAMPLEFORMAT_NAMES[m.sampleFormat] || ''}`.trim()],
      ['Samples/px', m.samplesPerPixel],
      ['Compression', COMPRESSION_NAMES[m.compression] || `code ${m.compression}`],
      ['Photometric', PHOTOMETRIC_NAMES[m.photometric] ?? '—'],
    ];
    refs['iv-meta-rows'].innerHTML = rows.map(([k,v]) =>
      `<div class="iv-meta-row"><span class="iv-meta-key">${k}</span><span class="iv-meta-val">${v}</span></div>`
    ).join('');
  }

  // ── Histograma con marcas de ventana + zona de saturación ───────────────────
  function drawHistogram() {
    const h = S.hist, cv = refs['iv-hist-canvas'];
    const W = cv.width, H = cv.height, ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const pad = 2, plotH = H - 8;
    // escala log para ver colas (saturación/fondo)
    const maxCount = Math.max(...h.bins);
    const logMax = Math.log10(maxCount + 1);
    const bw = (W - pad*2) / h.nBins;
    for (let b = 0; b < h.nBins; b++) {
      const val = Math.log10(h.bins[b] + 1) / logMax;
      const barH = val * plotH;
      const x = pad + b * bw;
      // color: último bin (saturación) en rojo, primer bin (cero) en gris, resto azul
      if (b === h.nBins - 1 && h.bins[b] > 0) ctx.fillStyle = '#ff4444';
      else if (b === 0) ctx.fillStyle = '#5a6a80';
      else ctx.fillStyle = '#3a7bd5';
      ctx.fillRect(x, H - 4 - barH, Math.max(1, bw - 0.3), barH);
    }
    // líneas de ventana lo/hi (mapear valor → x)
    const toX = v => pad + ((v - h.lo) / ((h.hi - h.lo) || 1)) * (W - pad*2);
    [['#4bd760', S.win.lo], ['#4bd760', S.win.hi]].forEach(([c, v]) => {
      const x = toX(v); ctx.strokeStyle = c; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, H - 4); ctx.stroke();
    });
    // stats
    const satPct = (h.satFrac*100), zeroPct = (h.zeroFrac*100);
    refs['iv-hist-stats'].innerHTML =
      `<b>min</b> ${h.min} · <b>max</b> ${h.max} · <b>mean</b> ${h.mean.toFixed(1)}` +
      ` · <b>range</b> 0–${h.maxPossible ?? '—'}<br>` +
      `<b>saturated</b> ${satPct.toFixed(2)}% · <b>zero</b> ${zeroPct.toFixed(1)}%` +
      ` · <span style="color:#4bd760">▏</span> window [${Math.round(S.win.lo)}, ${Math.round(S.win.hi)}]`;
  }

  // ── Alertas automáticas ─────────────────────────────────────────────────────
  function renderAlerts() {
    const h = S.hist, out = [];
    const satPct = h.satFrac*100, zeroPct = h.zeroFrac*100;
    if (satPct >= 1) out.push(['danger', `⚠ ${satPct.toFixed(1)}% of pixels are saturated (clipped at max). Quantitative intensity is unreliable for this image.`]);
    else if (satPct >= 0.1) out.push(['warn', `⚠ ${satPct.toFixed(2)}% of pixels are saturated — check exposure.`]);
    if (zeroPct >= 80) out.push(['warn', `⚠ ${zeroPct.toFixed(0)}% of pixels are zero — image may be underexposed or mostly background.`]);
    // rango dinámico bajo: max muy por debajo del tope de bits
    if (h.maxPossible && h.max < h.maxPossible * 0.05)
      out.push(['info', `Low dynamic range: brightest pixel is ${h.max} of ${h.maxPossible} (${(h.max/h.maxPossible*100).toFixed(1)}% of range). Windowing is stretching a narrow band.`]);
    refs['iv-alerts'].innerHTML = out.map(([lvl, msg]) =>
      `<div class="iv-alert iv-alert-${lvl}">${msg}</div>`).join('');
  }

  // ── Controles ───────────────────────────────────────────────────────────────
  refs['iv-auto'].onclick = () => { if (!S.hist) return; applyAutoWindow(); drawHistogram(); paintCanvas(); };
  refs['iv-wlo'].onchange = () => { S.win.lo = parseFloat(refs['iv-wlo'].value); drawHistogram(); paintCanvas(); };
  refs['iv-whi'].onchange = () => { S.win.hi = parseFloat(refs['iv-whi'].value); drawHistogram(); paintCanvas(); };
  refs['iv-brightness'].oninput = () => {
    S.brightness = parseFloat(refs['iv-brightness'].value);
    refs['iv-brightness-val'].textContent = S.brightness.toFixed(2) + '×';
    if (S.meta) paintCanvas();
  };
  refs['iv-contrast'].oninput = () => {
    S.contrast = parseFloat(refs['iv-contrast'].value);
    refs['iv-contrast-val'].textContent = S.contrast.toFixed(2) + '×';
    if (S.meta) paintCanvas();
  };
  refs['iv-reset'].onclick = () => {
    S.brightness = 1; S.contrast = 1;
    refs['iv-brightness'].value = 1; refs['iv-contrast'].value = 1;
    refs['iv-brightness-val'].textContent = '1.00×'; refs['iv-contrast-val'].textContent = '1.00×';
    if (S.hist) { applyAutoWindow(); drawHistogram(); }
    if (S.meta) paintCanvas();
  };

  // ── Regex opcionales: re-indexan el CSV al cambiar ──────────────────────────
  function reindexWithRegex() {
    if (!S.rawRows) return;
    S.regexPlate = refs['iv-regex-plate'].value.trim();
    S.regexName  = refs['iv-regex-name'].value.trim();
    // validar sintaxis de cada regex; marcar campo si es inválido
    let ok = true, hint = [];
    [['iv-regex-plate', S.regexPlate], ['iv-regex-name', S.regexName]].forEach(([id, pat]) => {
      const inp = refs[id];
      if (!pat) { inp.classList.remove('iv-regex-bad'); return; }
      try { new RegExp(pat); inp.classList.remove('iv-regex-bad'); }
      catch { inp.classList.add('iv-regex-bad'); ok = false; hint.push(id.includes('plate')?'plate':'name'); }
    });
    if (!ok) {
      refs['iv-regex-hint'].textContent = 'Invalid regex: ' + hint.join(', ');
      refs['iv-regex-hint'].className = 'iv-regex-hint bad';
      return;
    }
    S.imgIndex = buildImageIndex(S.rawRows);
    const nPlates = Object.keys(S.imgIndex).length;
    refs['iv-regex-hint'].textContent =
      (S.regexPlate || S.regexName ? 'Override active — ' : 'Using CSV metadata — ') +
      `${nPlates} plate(s) detected`;
    refs['iv-regex-hint'].className = 'iv-regex-hint ok';
    populatePlateSelect();
  }
  refs['iv-regex-plate'].addEventListener('change', reindexWithRegex);
  refs['iv-regex-name'].addEventListener('change', reindexWithRegex);

  // ── Toggle histograma región/completa ───────────────────────────────────────
  refs['iv-hist-region'].addEventListener('change', () => {
    S.histRegion = refs['iv-hist-region'].checked;
    if (S.meta) { recomputeHist(); drawHistogram(); }
  });

  // ── Zoom (rueda) + pan (arrastre) ───────────────────────────────────────────
  const wrap = refs['iv-canvas-wrap'] || document.getElementById('iv-canvas-wrap');
  if (wrap) {
    wrap.addEventListener('wheel', ev => {
      if (!S.meta) return;
      ev.preventDefault();
      const factor = ev.deltaY < 0 ? 1.15 : 1/1.15;
      S.zoom = Math.max(1, Math.min(20, S.zoom * factor));
      if (S.zoom === 1) { S.panX = 0; S.panY = 0; }
      paintCanvas(true);   // solo re-dibuja (no reconstruye source)
      updateZoomLabel();
      if (S.histRegion) { recomputeHist(); drawHistogram(); }
    }, { passive: false });

    let dragging = false, lastX = 0, lastY = 0;
    wrap.addEventListener('mousedown', ev => {
      if (!S.meta || S.zoom <= 1) return;
      dragging = true; lastX = ev.clientX; lastY = ev.clientY;
      wrap.classList.add('panning');
    });
    window.addEventListener('mousemove', ev => {
      if (!dragging) return;
      S.panX += (ev.clientX - lastX); S.panY += (ev.clientY - lastY);
      lastX = ev.clientX; lastY = ev.clientY;
      paintCanvas(true);
    });
    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false; wrap.classList.remove('panning');
      if (S.histRegion) { recomputeHist(); drawHistogram(); }
    });
  }
  refs['iv-zoom-reset'].onclick = () => {
    S.zoom = 1; S.panX = 0; S.panY = 0;
    paintCanvas(true); updateZoomLabel();
    if (S.histRegion) { recomputeHist(); drawHistogram(); }
  };
})();