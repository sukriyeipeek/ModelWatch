'use strict';

const $ = (id) => document.getElementById(id);
const state = { models: [], runs: [], overview: null, classLimit: 10, eventOffset: 0, eventTotal: 0, eventLimit: 20, generation: 0 };
const integer = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
const number = (value) => typeof value === 'number' && Number.isFinite(value);
const pct = (value) => number(value) ? `${(value * 100).toFixed(1)}%` : '—';
const score = (value) => number(value) ? value.toFixed(3) : '—';
const count = (value) => number(value) ? integer.format(value) : '—';
const milliseconds = (value) => number(value) ? `${decimal.format(value)} ms` : '—';
const human = (value) => String(value ?? '—').replaceAll('_', ' ');
function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
function set(id, value) { $(id).textContent = value; }
function timestamp(value) { const date = new Date(value); return Number.isNaN(date.getTime()) ? 'Timestamp unavailable' : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }); }
function announce(text, error = false) { $('message').hidden = !text; $('message').className = `message${error ? ' error' : ''}`; $('message').textContent = text; }
async function api(path) { const response = await fetch(path, { headers: { Accept: 'application/json' } }); if (!response.ok) { let detail = ''; try { const body = await response.json(); detail = typeof body.detail === 'string' ? body.detail : ''; } catch {} throw new Error(detail || `Request failed (${response.status}).`); } return response.json(); }
function busy(value) { $('refresh').disabled = value; $('model-select').disabled = value || !state.models.length; $('run-select').disabled = value || !state.runs.length; $('main').setAttribute('aria-busy', String(value)); }
function selectOptions(id, entries, selected) { const select = $(id); select.replaceChildren(); for (const item of entries) { const option = el('option', '', item.label); option.value = item.value; select.append(option); } if (entries.some((item) => item.value === selected)) select.value = selected; }

async function initialize() {
  const generation = ++state.generation;
  const previousModel = $('model-select').value;
  const previousRun = $('run-select').value;
  busy(true); announce('Loading recorded observations…');
  try {
    const payload = await api('/api/models'); if (generation !== state.generation) return;
    state.models = payload.models || [];
    selectOptions('model-select', state.models.map((model) => ({ value: model.model_id, label: `${model.name || model.model_id} · ${model.version}` })), previousModel);
    if (!state.models.length) { showEmpty('No models registered yet.'); state.runs = []; selectOptions('run-select', [{ value: '', label: 'No observation runs' }]); return; }
    await loadRuns(previousRun, generation);
  } catch (error) { announce(`Unable to load ModelWatch. ${error.message} Check that the API is running, then refresh.`, true); }
  finally { if (generation === state.generation) busy(false); }
}

function showEmpty(message) { $('dashboard').hidden = true; $('empty-state').hidden = false; announce(message); set('updated-at', 'Ready for observations'); }
async function loadRuns(preferred, generation) {
  const payload = await api(`/api/runs?model_id=${encodeURIComponent($('model-select').value)}`); if (generation !== state.generation) return;
  state.runs = (payload.runs || []).slice().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  if (!state.runs.length) { selectOptions('run-select', [{ value: '', label: 'No observation runs' }]); showEmpty('This model has no observation runs yet.'); return; }
  const suggested = preferred || state.runs.find((run) => run.role === 'monitor')?.run_id || state.runs[0].run_id;
  selectOptions('run-select', state.runs.map((run) => ({ value: run.run_id, label: `${run.role === 'reference' ? 'Reference' : 'Monitor'} · ${run.run_id}` })), suggested);
  await loadOverview(generation);
}

async function loadOverview(generation) {
  const runId = $('run-select').value;
  const overview = await api(`/api/overview?run_id=${encodeURIComponent(runId)}`); if (generation !== state.generation) return;
  state.overview = overview; state.classLimit = 10; state.eventOffset = 0;
  $('class-search').value = '';
  $('empty-state').hidden = true; $('dashboard').hidden = false;
  renderOverview(overview);
  await loadEvents(generation);
  if (generation !== state.generation) return;
  announce(''); set('updated-at', `Updated ${new Date().toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`);
}

function renderOverview(data) {
  const { metrics = {}, model = {}, run = {}, reference_run: reference = null } = data;
  const traffic = metrics.traffic || {}; const latency = metrics.latency || {}; const quality = metrics.quality || {};
  set('request-count', count(traffic.requests));
  set('error-detail', `${count(traffic.errors)} errors · ${pct(traffic.error_rate)} error rate`);
  set('latency-value', milliseconds(latency.p95_ms));
  set('latency-detail', `p50 ${milliseconds(latency.p50_ms)} · ${count(latency.count)} timed requests`);
  set('coverage-value', pct(quality.label_coverage));
  set('coverage-detail', `${count(quality.labeled_count)} of ${count(traffic.successes)} successful requests labeled`);
  set('accuracy-value', pct(quality.accuracy));
  set('accuracy-detail', number(quality.accuracy) ? 'Evaluated on observed ground-truth labels' : 'Ground-truth labels are needed');
  set('macro-f1', score(quality.macro_f1));
  set('source-badge', run.source === 'benchmark_replay' ? 'BENCHMARK REPLAY' : run.source === 'live' ? 'LIVE OBSERVATIONS' : human(run.source).toUpperCase());
  set('model-version', `${model.version || 'Unversioned'} · ${model.model_id || ''}`);
  set('run-detail', `${count(traffic.requests)} recorded events · ${run.role === 'reference' ? 'Reference run' : 'Monitor run'}`);
  set('quality-description', `${count(quality.labeled_count)} labeled successful requests. Macro F1 gives equal weight to all ${count(model.classes?.length)} registered classes, including classes with no observed labels.`);
  renderDrift(metrics.drift || {}, reference);
  renderClasses(); renderProvenance(data);
  $('warnings').replaceChildren(...(metrics.warnings || []).map((warning) => el('div', 'warning-line', warning)));
}

function renderDrift(drift, reference) {
  const badge = $('drift-status');
  const statuses = { stable: ['No shift flagged', ''], shift_detected: ['Shift to investigate', 'warning'], insufficient_data: ['More evidence needed', 'neutral'] };
  const [text, style] = statuses[drift.status] || ['Awaiting comparison', 'neutral']; badge.textContent = text; badge.className = `badge ${style}`;
  const content = $('drift-content'); content.replaceChildren();
  if (!reference || drift.status === 'insufficient_data' || !(drift.features || []).length) {
    content.append(el('div', 'drift-empty', !reference ? 'No reference run is available. Record a reference distribution before comparing observations.' : `A comparison needs at least ${count(drift.min_samples)} successful events in each run. Reference: ${count(drift.reference_count)} · Current: ${count(drift.current_count)}.`));
  } else {
    const grid = el('div', 'drift-grid');
    const items = [...drift.features, ...(drift.predictions ? [{ name: 'predicted_class', ...drift.predictions }] : [])];
    const names = { text_length: 'Message length', word_count: 'Word count', predicted_class: 'Predicted class mix' };
    for (const feature of items) {
      const item = el('div', 'drift-item'); const title = el('div', 'drift-name'); title.append(el('span', '', names[feature.name] || human(feature.name)), el('span', 'drift-number', score(feature.js_divergence))); item.append(title);
      const track = el('div', 'drift-track'); const fill = el('div', `drift-fill${feature.shift_detected ? ' alert' : ''}`); fill.style.width = `${number(feature.js_divergence) ? Math.min(100, Math.max(0, feature.js_divergence * 100)) : 0}%`; track.append(fill);
      if (number(drift.threshold)) { const threshold = el('span', 'threshold-mark'); threshold.style.left = `${Math.min(100, Math.max(0, drift.threshold * 100))}%`; threshold.title = `Investigation threshold ${drift.threshold}`; track.append(threshold); }
      item.append(track); const meta = el('div', 'drift-meta'); meta.append(el('span', '', feature.shift_detected ? 'Threshold exceeded' : 'Below threshold'), el('span', '', number(feature.ks_statistic) ? `KS ${score(feature.ks_statistic)}` : 'Class distribution')); item.append(meta); grid.append(item);
    }
    content.append(grid);
  }
  set('drift-caption', `Jensen–Shannon divergence · scale 0–1, higher means more different · investigation threshold ${number(drift.threshold) ? drift.threshold : 'not configured'}. Reference ${reference?.run_id || 'not set'} (${count(drift.reference_count)} successful events) vs selected run (${count(drift.current_count)}). ${drift.method || ''}`);
}

function renderClasses() {
  const rows = state.overview?.metrics?.quality?.per_class || [];
  const query = $('class-search').value.trim().toLowerCase(); const sortBy = $('class-sort').value;
  const filtered = rows.filter((row) => row.label.toLowerCase().includes(query) || human(row.label).toLowerCase().includes(query)).sort((a, b) => {
    if (sortBy === 'label') return a.label.localeCompare(b.label);
    if (sortBy === 'support') return b.support - a.support || a.label.localeCompare(b.label);
    if (!a.support || !b.support) return Number(!a.support) - Number(!b.support) || a.label.localeCompare(b.label);
    return (a[sortBy] ?? 0) - (b[sortBy] ?? 0) || b.support - a.support;
  });
  const body = $('class-rows'); body.replaceChildren();
  for (const row of filtered.slice(0, state.classLimit)) {
    const tr = el('tr'); const name = el('td', 'class-name', human(row.label)); name.title = row.label; tr.append(name, el('td', '', count(row.support)), el('td', '', pct(row.precision)));
    const recall = el('td'); const inner = el('div', 'recall-cell'); const bar = el('span', 'tiny-track'); const fill = el('span', 'tiny-fill'); fill.style.width = `${number(row.recall) ? Math.min(100, Math.max(0, row.recall * 100)) : 0}%`; bar.append(fill); inner.append(bar, el('span', '', pct(row.recall))); recall.append(inner);
    tr.append(recall, el('td', '', score(row.f1)), el('td', '', count(row.fn))); body.append(tr);
  }
  if (!filtered.length) { const tr = el('tr'); const td = el('td', 'table-empty', rows.length ? 'No classes match this search.' : 'No ground-truth labels have been recorded for this run. Performance will appear when labels arrive.'); td.colSpan = 6; tr.append(td); body.append(tr); }
  set('class-count', rows.length ? `Showing ${Math.min(state.classLimit, filtered.length)} of ${filtered.length} classes · zero-support scores follow the zero-division convention` : 'Performance is unavailable without labels');
  $('show-classes').hidden = filtered.length <= state.classLimit;
}

async function loadEvents(generation = state.generation) {
  const runId = $('run-select').value;
  const data = await api(`/api/events?run_id=${encodeURIComponent(runId)}&limit=${state.eventLimit}&offset=${state.eventOffset}`);
  if (generation !== state.generation) return;
  state.eventTotal = data.total || 0;
  const body = $('event-rows'); body.replaceChildren();
  for (const event of data.events || []) {
    const row = el('tr'); const id = el('td'); const title = el('span', 'event-id', event.event_id); title.title = event.event_id; id.append(title, el('span', 'event-time', timestamp(event.occurred_at))); row.append(id);
    row.append(el('td', '', event.predicted_label ? human(event.predicted_label) : '—'), el('td', '', score(event.confidence)), el('td', '', milliseconds(event.latency_ms)), el('td', event.ground_truth ? '' : 'muted', event.ground_truth ? human(event.ground_truth) : 'Awaiting label'), el('td', event.status === 'ok' ? 'event-good' : 'event-bad', event.status === 'ok' ? 'Success' : 'Error')); body.append(row);
  }
  if (!(data.events || []).length) { const row = el('tr'); const td = el('td', 'table-empty', 'No recorded requests in this run.'); td.colSpan = 6; row.append(td); body.append(row); }
  const end = Math.min(state.eventOffset + (data.events || []).length, state.eventTotal);
  set('event-count', state.eventTotal ? `Events ${state.eventOffset + 1}–${end} of ${count(state.eventTotal)}` : 'No recorded events');
  $('previous-events').disabled = state.eventOffset === 0; $('next-events').disabled = end >= state.eventTotal;
}

function renderProvenance(data) {
  const run = data.run || {}; const metadata = data.provenance || run.metadata || {}; const modelMetadata = data.model?.metadata || {};
  set('provenance-summary', run.source === 'benchmark_replay' ? 'These observations come from a repeatable benchmark replay. Predictions and latency were measured during local inference; labels come from the source dataset.' : 'These observations were submitted through the ingestion API. Inspect source metadata and label coverage before drawing conclusions.');
  const entries = [ ['Source type', human(run.source)], ['Run', run.run_id], ['Recorded', timestamp(run.created_at)], ['Model version', data.model?.version], ['Reference run', data.reference_run?.run_id || 'Not available'], ['Dataset', metadata.dataset || modelMetadata.dataset || 'Not specified'], ['Dataset split', metadata.split || metadata.dataset_split || 'See metadata'], ['Label source', metadata.label_source || (run.source === 'benchmark_replay' ? 'Official dataset labels' : 'Submitted ground truth')] ];
  $('provenance-details').replaceChildren();
  for (const [label, value] of entries) { $('provenance-details').append(el('dt', '', label), el('dd', '', typeof value === 'object' ? JSON.stringify(value) : String(value ?? 'Not specified'))); }
  set('metadata-json', JSON.stringify({ run: metadata, model: modelMetadata }, null, 2));
}

$('refresh').addEventListener('click', initialize);
$('model-select').addEventListener('change', async () => { const generation = ++state.generation; busy(true); announce('Loading observation runs…'); try { await loadRuns(null, generation); } catch (error) { announce(error.message, true); } finally { if (generation === state.generation) busy(false); } });
$('run-select').addEventListener('change', async () => { const generation = ++state.generation; busy(true); announce('Loading selected run…'); try { await loadOverview(generation); } catch (error) { announce(error.message, true); } finally { if (generation === state.generation) busy(false); } });
$('class-search').addEventListener('input', () => { state.classLimit = 10; renderClasses(); });
$('class-sort').addEventListener('change', renderClasses);
$('show-classes').addEventListener('click', () => { state.classLimit = Infinity; renderClasses(); });
for (const [id, direction] of [['previous-events', -1], ['next-events', 1]]) $(id).addEventListener('click', async () => { const previous = state.eventOffset; state.eventOffset = Math.max(0, state.eventOffset + direction * state.eventLimit); $('previous-events').disabled = true; $('next-events').disabled = true; try { await loadEvents(); } catch (error) { state.eventOffset = previous; announce(`Unable to load requests. ${error.message}`, true); $('previous-events').disabled = state.eventOffset === 0; $('next-events').disabled = state.eventOffset + state.eventLimit >= state.eventTotal; } });
document.querySelectorAll('.nav-link').forEach((link) => link.addEventListener('click', () => { document.querySelectorAll('.nav-link').forEach((item) => item.classList.remove('active')); link.classList.add('active'); }));
initialize();
