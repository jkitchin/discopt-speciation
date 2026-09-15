// discopt-speciation web app: the page. Python runs in worker.js.
"use strict";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];

function h(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) e.append(c.nodeType ? c : String(c));
  return e;
}

const fmt = (v, d = 4) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : Number(v).toFixed(d));
const sci = (v, d = 3) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : Number(v).toExponential(d));
const log10 = (v) => (v > 0 ? Math.log10(v) : null);
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)));
const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };

// ---------------------------------------------------------------- Python worker
class Backend {
  constructor() { this.info = null; this.start(); }

  start() {
    this.worker = new Worker("worker.js");
    this.pending = new Map();
    this.next = 1;
    this.ready = new Promise((resolve, reject) => { this._resolve = resolve; this._reject = reject; });
    this.ready.catch(() => {});
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "status") setStatus(m.text, "loading");
      else if (m.type === "ready") { this.info = m.info; this._resolve(m.info); }
      else if (m.type === "error") this._reject(new Error(m.text));
      else {
        const p = this.pending.get(m.id);
        if (!p) return;
        if (m.progress) { p.onProgress && p.onProgress(...m.progress); return; }
        this.pending.delete(m.id);
        if (m.error) { const err = new Error(m.error); err.detail = m.detail; p.reject(err); }
        else p.resolve(m.result);
      }
    };
    this.worker.onerror = (e) => this._reject(new Error(e.message || "worker failed to start"));
  }

  call(fn, arg, onProgress) {
    return new Promise((resolve, reject) => {
      const id = this.next++;
      this.pending.set(id, { resolve, reject, onProgress });
      const a = arg === undefined || typeof arg === "string" ? arg : JSON.stringify(arg);
      this.worker.postMessage({ id, fn, arg: a, progress: !!onProgress });
    });
  }

  // Python cannot be interrupted without cross-origin isolation, so stopping restarts the worker.
  restart() {
    this.worker.terminate();
    for (const p of this.pending.values()) { const e = new Error("Stopped."); e.stopped = true; p.reject(e); }
    this.start();
    watchReady();
  }
}

// ---------------------------------------------------------------- state
const ACTIVITY = [
  ["auto", "database default"], ["ideal", "ideal"], ["davies", "Davies"], ["debye-huckel", "Debye–Hückel"],
  ["extended-debye-huckel", "extended Debye–Hückel"], ["truesdell-jones", "Truesdell-Jones (PHREEQC)"],
  ["bdot", "B-dot"], ["llnl", "LLNL B-dot"], ["sit", "SIT"], ["pitzer", "Pitzer"],
];

const BASE_STATE = {
  database: "phreeqc.dat", activity: "auto", components: [], amounts: {}, solids: {}, gases: {}, fixedAq: [],
  pHMode: "free", pH: 7, peMode: "free", pe: 4, T_C: 25, charge_balance: "", method: "complementarity",
};

const PRESETS = {
  calcite: {
    label: "Calcite open to air",
    state: { components: ["Ca", "C(4)"], solids: { Calcite: 1, Aragonite: 0 }, gases: { "CO2(g)": -3.5 } },
    sweep: { var: "fixed:CO2(g)", from: -6, to: 0 },
  },
  iron: {
    label: "Iron in chloride solution (Pourbaix)",
    state: { components: ["Fe", "Cl"], amounts: { Fe: 1e-3, Cl: 3e-3 }, solids: { "Fe(OH)3(a)": 0 },
             pHMode: "fixed", pH: 7, peMode: "fixed", pe: 4 },
    sweep: { var: "pH", from: 2, to: 12 },
    tab: "diagram",
  },
  gypsum: {
    label: "Gypsum or anhydrite at 60 °C",
    state: { components: ["Ca", "S(6)"], solids: { Gypsum: 0.5, Anhydrite: 0 }, T_C: 60 },
    sweep: { var: "T_C", from: 0, to: 100 },
  },
  halite: {
    label: "Halite solubility (Pitzer)",
    state: { database: "pitzer.dat", components: ["Na", "Cl"], solids: { Halite: 10 } },
  },
  carbonate: {
    label: "Sodium bicarbonate, charge balance on Cl⁻",
    state: { components: ["Na", "Cl", "C(4)"], amounts: { Na: 0.01, Cl: 0.01, "C(4)": 2e-3 },
             pHMode: "fixed", pH: 7.5, charge_balance: "Cl-" },
    sweep: { var: "pH", from: 4, to: 11 },
  },
  llnl: {
    label: "Calcite in llnl.dat",
    state: { database: "llnl.dat", components: ["Ca", "C(4)"], solids: { Calcite: 0.1 }, gases: { "CO2(g)": -3.5 } },
    sweep: { var: "T_C", from: 0, to: 200 },
  },
};

let S = structuredClone(BASE_STATE);
let DB = null;   // database_info
let SYS = null;  // system_info
let lastResult = null, lastSweep = null;
const backend = new Backend();
const dbCache = {};

function cfg() {
  const amounts = {};
  for (const c of S.components) if (+S.amounts[c] > 0) amounts[c] = +S.amounts[c];
  for (const [k, v] of Object.entries(S.solids)) if (+v > 0) amounts[k] = +v;
  const fixed = {};
  for (const [k, v] of Object.entries(S.gases)) fixed[k] = +v;
  for (const f of S.fixedAq) if (f.species && f.value !== "" && Number.isFinite(+f.value)) fixed[f.species] = +f.value;
  const c = {
    database: S.database, components: [...S.components], solids: Object.keys(S.solids), gases: Object.keys(S.gases),
    amounts, fixed, T_C: +S.T_C, method: S.method, activity: S.activity === "auto" ? null : S.activity,
  };
  if (S.pHMode === "fixed") c.pH = +S.pH;
  if (S.peMode === "fixed") c.pe = +S.pe;
  if (S.charge_balance) c.charge_balance = S.charge_balance;
  return c;
}

// ---------------------------------------------------------------- status and errors
function setStatus(text, kind) {
  $("#status-text").textContent = text;
  $("#status").className = "status " + kind;
}

function showAlert(text, kind = "error") {
  const a = $("#alert");
  a.textContent = text;
  a.className = "alert" + (kind === "warn" ? " warn" : "");
  a.hidden = !text;
}

function readyStatus() {
  const i = backend.info;
  setStatus(`Ready · discopt ${i.discopt} · pounce ${i.pounce}`, "ready");
}

let busy = 0;
async function run(button, fn) {
  busy++;
  button && button.classList.add("busy");
  syncButtons();
  setStatus("Computing…", "busy");
  showAlert("");
  try {
    await backend.ready;
    return await fn();
  } catch (err) {
    if (!err.stopped) {
      showAlert(err.message);
      if (err.detail) console.error(err.detail);
    }
    return undefined;
  } finally {
    busy--;
    button && button.classList.remove("busy");
    syncButtons();
    if (backend.info && !busy) readyStatus();
  }
}

function syncButtons() {
  const ok = !!backend.info && !busy && S.components.length > 0;
  for (const id of ["#compute", "#sw-run", "#dg-run", "#lk-run"]) $(id).disabled = !ok;
  $("#sw-csv").disabled = !lastSweep;
}

function watchReady() {
  syncButtons();
  backend.ready.then(() => { readyStatus(); syncButtons(); })
    .catch((err) => { setStatus("Could not load Python", "error"); showAlert(err.message); });
}

// ---------------------------------------------------------------- database and components
async function loadDatabase(name) {
  if (!dbCache[name]) dbCache[name] = await backend.call("database_info", name);
  DB = dbCache[name];
  const sel = $("#activity");
  sel.innerHTML = "";
  for (const [v, label] of ACTIVITY) {
    sel.append(h("option", { value: v }, v === "auto" ? `database default (${DB.activity})` : label));
  }
  sel.value = S.activity;
  renderElements();
}

function renderElements() {
  const box = $("#elements");
  box.innerHTML = "";
  if (!DB) return;
  const q = $("#element-filter").value.trim().toLowerCase();
  for (const e of DB.elements) {
    const keys = [e.name, ...e.states];
    if (q && !keys.some((k) => k.toLowerCase().startsWith(q))) continue;
    const group = h("span", { class: "el" });
    group.append(h("button", {
      class: S.components.includes(e.name) ? "on" : "", disabled: !e.master,
      title: e.master ? `${e.name} total (master species ${e.master})` : `${e.name}: choose a valence state`,
      onclick: () => toggleComponent(e.name),
    }, e.name));
    for (const st of e.states) {
      group.append(h("button", {
        class: S.components.includes(st) ? "on" : "", title: st, onclick: () => toggleComponent(st),
      }, st.slice(e.name.length)));
    }
    box.append(group);
  }
  if (!box.children.length) box.append(h("p", { class: "muted" }, "No element matches."));
}

function toggleComponent(c) {
  const base = c.split("(")[0];
  if (S.components.includes(c)) {
    S.components = S.components.filter((x) => x !== c);
  } else {
    // a total and its valence states are alternatives
    S.components = S.components.filter((x) => x.split("(")[0] !== base || (x !== base && c !== base));
    S.components.push(c);
    if (!(c in S.amounts)) S.amounts[c] = 1e-3;
  }
  renderElements();
  renderComponents();
  refreshSystem();
}

function renderComponents() {
  const box = $("#components");
  box.innerHTML = "";
  if (!S.components.length) {
    box.append(h("p", { class: "muted" }, "Click elements above. H, O and the electron are always included."));
    return;
  }
  box.append(h("div", { class: "subhead" }, h("span", {}, "Component"), h("span", {}, "total (mol / kg water)")));
  for (const c of S.components) {
    const master = SYS && SYS.masters ? SYS.masters[c] : null;
    const input = h("input", { type: "number", step: "any", min: "0", value: S.amounts[c] ?? 0,
                               "aria-label": `amount of ${c}` });
    input.addEventListener("input", () => { S.amounts[c] = input.value; saveHash(); });
    box.append(h("div", { class: "comp" },
      h("span", { class: "name" }, c, master ? h("small", {}, master) : null),
      input,
      h("button", { title: `remove ${c}`, onclick: () => toggleComponent(c) }, "×")));
  }
}

let sysToken = 0;
async function refreshSystem() {
  const token = ++sysToken;
  syncButtons();
  if (!S.components.length) {
    SYS = null;
    renderPhases(); renderConditionOptions(); renderTabOptions(); saveHash();
    return;
  }
  try {
    await backend.ready;
    const info = await backend.call("system_info", { database: S.database, components: S.components });
    if (token !== sysToken) return;
    SYS = info;
    const names = new Set(info.phases.map((p) => p.name));
    for (const k of Object.keys(S.solids)) if (!names.has(k)) delete S.solids[k];
    for (const k of Object.keys(S.gases)) if (!names.has(k)) delete S.gases[k];
    const aq = new Set(info.aqueous.map((s) => s.name));
    if (S.charge_balance && !aq.has(S.charge_balance)) S.charge_balance = "";
    S.fixedAq = S.fixedAq.filter((f) => !f.species || aq.has(f.species));
    showAlert("");
  } catch (err) {
    if (token !== sysToken) return;
    SYS = null;
    showAlert(err.message);
  }
  renderComponents(); renderPhases(); renderConditionOptions(); renderTabOptions(); saveHash(); syncButtons();
}

const GAS_DEFAULT = { "CO2(g)": -3.5, "O2(g)": -0.68, "N2(g)": -0.11, "H2O(g)": -1.5, "CH4(g)": -5.7, "H2S(g)": -6 };

function renderPhases() {
  const box = $("#phases");
  box.innerHTML = "";
  if (!SYS) { box.append(h("p", { class: "muted" }, "Choose components first.")); return; }
  const q = $("#phase-filter").value.trim().toLowerCase();
  const match = (p) => !q || p.name.toLowerCase().includes(q) || (p.formula || "").toLowerCase().includes(q);
  const solids = SYS.phases.filter((p) => !p.gas);
  const gases = SYS.phases.filter((p) => p.gas);

  const group = (title, list, store, isGas) => {
    if (!list.length) return;
    const all = h("button", { class: "ghost small", onclick: () => {
      const on = list.filter(match).some((p) => !(p.name in store));
      for (const p of list.filter(match)) {
        if (on && !(p.name in store)) store[p.name] = isGas ? (GAS_DEFAULT[p.name] ?? -3) : 0;
        if (!on) delete store[p.name];
      }
      renderPhases(); renderTabOptions(); saveHash();
    } }, "toggle all");
    box.append(h("div", { class: "phase-group" },
      h("span", {}, `${title} · ${list.length}`), isGas ? h("span", {}, "log10 P (atm)") : h("span", {}, "initial mol"), all));
    for (const p of list.filter(match)) {
      const checked = p.name in store;
      const val = h("input", { type: "number", step: "any", value: checked ? store[p.name] : (isGas ? (GAS_DEFAULT[p.name] ?? -3) : 0),
                               disabled: !checked, "aria-label": `${p.name} ${isGas ? "log10 P" : "initial amount"}` });
      val.addEventListener("input", () => { store[p.name] = val.value; saveHash(); });
      const cb = h("input", { type: "checkbox", checked, "aria-label": p.name });
      cb.addEventListener("change", () => {
        if (cb.checked) store[p.name] = val.value === "" ? 0 : val.value; else delete store[p.name];
        val.disabled = !cb.checked;
        renderTabOptions(); saveHash();
      });
      const logK = Number.isFinite(p.logK) ? ` · log K ${p.logK.toFixed(2)}` : "";
      box.append(h("label", { class: "phase", title: `${p.formula}${logK}` }, cb,
        h("span", { class: "pname" }, p.name, p.formula && p.formula !== p.name ? h("small", {}, p.formula) : null), val));
    }
  };
  group("Minerals", solids, S.solids, false);
  group("Gases (fixed partial pressure)", gases, S.gases, true);
  if (!box.children.length) box.append(h("p", { class: "muted" }, "No phases can form from these components."));
}

function renderConditionOptions() {
  const cb = $("#charge_balance");
  cb.innerHTML = "";
  cb.append(h("option", { value: "" }, "none"));
  for (const sp of SYS ? SYS.aqueous.filter((s) => s.charge) : []) cb.append(h("option", { value: sp.name }, sp.name));
  cb.value = S.charge_balance;
  for (const k of ["T_C", "pH", "pe", "pHMode", "peMode", "method"]) $("#" + k).value = S[k];
  $("#pH").disabled = S.pHMode !== "fixed";
  $("#pe").disabled = S.peMode !== "fixed";
  renderFixed();
}

function renderFixed() {
  const box = $("#fixed-list");
  box.innerHTML = "";
  S.fixedAq.forEach((f, i) => {
    const sel = h("select", { "aria-label": "species" }, h("option", { value: "" }, "species…"),
      ...(SYS ? SYS.aqueous : []).map((s) => h("option", { value: s.name }, s.name)));
    sel.value = f.species || "";
    sel.addEventListener("change", () => { f.species = sel.value; renderTabOptions(); saveHash(); });
    const val = h("input", { type: "number", step: "any", value: f.value, "aria-label": "log10 activity" });
    val.addEventListener("input", () => { f.value = val.value; saveHash(); });
    box.append(h("div", { class: "fixed-row" }, sel, val,
      h("button", { title: "remove", onclick: () => { S.fixedAq.splice(i, 1); renderFixed(); renderTabOptions(); saveHash(); } }, "×")));
  });
}

// ---------------------------------------------------------------- speciation tab
async function compute() {
  await run($("#compute"), async () => {
    const t0 = performance.now();
    const r = await backend.call("equilibrate", cfg());
    r.seconds = (performance.now() - t0) / 1000;
    lastResult = r;
    renderResult();
    selectTab("speciation");
  });
}

function renderResult() {
  const r = lastResult;
  if (!r) return;
  $("#spec-empty").hidden = true;
  $("#spec-out").hidden = false;
  const ehPerPe = 8.314462618 * (+S.T_C + 273.15) * Math.LN10 / 96485.33212;
  const card = (k, v, cls = "", small = false) =>
    h("div", { class: "card " + cls }, h("div", { class: "k" }, k), h("div", { class: "v" + (small ? " small" : "") }, v));
  const cards = $("#cards");
  cards.innerHTML = "";
  cards.append(...[
    card("status", r.converged ? "converged" : "not converged", r.converged ? "good" : "bad", true),
    card("pH", fmt(r.pH, 3)),
    r.pe !== null && r.pe !== undefined ? card("pe · Eh", `${fmt(r.pe, 2)} · ${fmt(r.pe * ehPerPe * 1000, 0)} mV`, "", true) : null,
    card("ionic strength", sci(r.ionic_strength, 3), "", true),
    card("present", r.present.length ? r.present.join(", ") : "none", "", true),
    card("water", `${fmt(r.water_mass, 5)} kg`, "", true),
    card("charge imbalance", sci(r.charge_imbalance, 1), "", true),
    card("time", `${r.seconds.toFixed(2)} s · ${r.method}`, "", true),
  ].filter(Boolean));
  if (r.charge_balance_adjustment !== null && r.charge_balance_adjustment !== undefined) {
    cards.append(card(`${S.charge_balance} adjusted`, sci(r.charge_balance_adjustment, 3), "", true));
  }
  if (!r.converged) showAlert("The solver did not converge. Try other initial amounts, fewer phases or the enumerate method.", "warn");

  renderSpeciesTable();

  const phaseRows = Object.keys(r.SI).map((k) => ({ name: k, SI: r.SI[k], amount: r.solids[k], P: r.partial_pressure[k], gas: r.gases[k] }));
  const pt = $("#phase-table");
  pt.innerHTML = "";
  pt.append(h("tr", {}, h("th", {}, "phase"), h("th", {}, "SI"), h("th", {}, "amount (mol)"), h("th", {}, "log10 P")));
  for (const p of phaseRows) {
    const present = r.present.includes(p.name);
    const cls = p.SI === null ? "" : p.SI > 1e-6 ? "pos" : Math.abs(p.SI) <= 1e-6 ? "zero" : "";
    pt.append(h("tr", { class: present ? "present" : "" }, h("td", {}, p.name), h("td", { class: cls }, fmt(p.SI, 4)),
      h("td", {}, p.amount !== undefined ? sci(p.amount) : "—"), h("td", {}, p.P ? fmt(Math.log10(p.P), 3) : "—")));
  }
  if (!phaseRows.length) pt.append(h("tr", {}, h("td", { colspan: 4 }, "No phases in the system.")));
  $("#notes").innerHTML = "";
  for (const n of r.notes || []) $("#notes").append(h("div", {}, "• " + n));
  $("#summary").textContent = r.summary || "";

  const species = Object.entries(r.molality).filter(([, m]) => m > 1e-15).sort((a, b) => b[1] - a[1]).slice(0, 30).reverse();
  plot("chart-molality", [{
    type: "bar", orientation: "h", y: species.map(([k]) => k), x: species.map(([, m]) => Math.log10(m)),
    marker: { color: species.map(([k]) => chargeColor(r.species_charge[k])) },
    hovertemplate: "%{y}: log10 m = %{x:.3f}<extra></extra>",
  }], { title: "log<sub>10</sub> molality (top 30)", xaxis: { title: "log<sub>10</sub> m" }, margin: { l: 110 },
        height: Math.max(340, 18 * species.length + 90) });

  const si = Object.entries(r.SI).filter(([, v]) => v !== null);
  plot("chart-si", si.length ? [{
    type: "bar", x: si.map(([k]) => k), y: si.map(([, v]) => v),
    marker: { color: si.map(([k, v]) => (r.present.includes(k) ? cssVar("--good") : v > 1e-6 ? cssVar("--bad") : cssVar("--muted"))) },
    hovertemplate: "%{x}: SI = %{y:.4f}<extra></extra>",
  }] : [], { title: "Saturation indices (green: present)", yaxis: { title: "SI = log<sub>10</sub>(IAP/K)", zeroline: true } });
}

function renderSpeciesTable() {
  const r = lastResult;
  if (!r) return;
  const trace = $("#show-trace").checked;
  const rows = Object.entries(r.molality).filter(([, m]) => trace || m >= 1e-15).sort((a, b) => b[1] - a[1]);
  const t = $("#species-table");
  t.innerHTML = "";
  t.append(h("tr", {}, h("th", {}, "species"), h("th", {}, "charge"), h("th", {}, "molality"), h("th", {}, "log10 a"), h("th", {}, "γ")));
  for (const [k, m] of rows) {
    t.append(h("tr", {}, h("td", {}, k), h("td", {}, r.species_charge[k] ?? ""), h("td", {}, sci(m)),
      h("td", {}, fmt(r.log10_activity[k], 4)), h("td", {}, fmt(r.gamma[k], 4))));
  }
}

function chargeColor(z) {
  if (!z) return cssVar("--muted");
  return z > 0 ? "#d9534f" : "#3b82f6";
}

// ---------------------------------------------------------------- variables for sweeps and diagrams
function variables() {
  const v = [{ value: "pH", label: "pH" }];
  if (!SYS || SYS.has_electron) v.push({ value: "pe", label: "pe" });
  v.push({ value: "T_C", label: "T (°C)" });
  for (const c of S.components) v.push({ value: `amount:${c}`, label: `total ${c} (mol)` });
  for (const s of Object.keys(S.solids)) v.push({ value: `amount:${s}`, label: `initial ${s} (mol)` });
  for (const g of Object.keys(S.gases)) v.push({ value: `fixed:${g}`, label: `log10 P ${g}` });
  for (const f of S.fixedAq) if (f.species) v.push({ value: `fixed:${f.species}`, label: `log10 a ${f.species}` });
  return v;
}

function varLabel(value) {
  const f = variables().find((v) => v.value === value);
  return f ? f.label : value;
}

function defaultRange(value) {
  if (value === "pH") return [2, 12, false];
  if (value === "pe") return [-10, 15, false];
  if (value === "T_C") return [0, 100, false];
  if (value.startsWith("amount:")) return [1e-5, 1, true];
  return [-6, 0, false];
}

function fillSelect(sel, options, keep) {
  const old = keep ?? sel.value;
  sel.innerHTML = "";
  for (const o of options) sel.append(h("option", { value: o.value }, o.label));
  if (options.some((o) => o.value === old)) sel.value = old;
  return sel.value;
}

function renderTabOptions() {
  const vars = variables();
  const before = $("#sw-var").value;
  const now = fillSelect($("#sw-var"), vars);
  if (now !== before) setSweepRange(now);
  // the diagram axes follow the user's choice when it is available, else pH and pe (or another variable)
  const setAxis = (id, want, avoid) => {
    const sel = $(id);
    const before = sel.value;
    let v = fillSelect(sel, vars, vars.some((o) => o.value === want) ? want : sel.value);
    if (v === avoid) { const alt = vars.find((o) => o.value !== avoid); if (alt) v = sel.value = alt.value; }
    if (v !== before) setDiagramRange(id === "#dg-x" ? "x" : "y", v);
    return v;
  };
  const dx = setAxis("#dg-x", $("#dg-x").dataset.user || "pH", null);
  setAxis("#dg-y", $("#dg-y").dataset.user || "pe", dx);
  const els = [...new Set(S.components.map((c) => c.split("(")[0]))].map((e) => ({ value: e, label: e }));
  fillSelect($("#dg-el"), els);
  renderCode();
}

function setDiagramRange(axis, value) {
  const [a, b] = value === "pH" ? [1, 13] : value === "pe" ? [-12, 20] : defaultRange(value);
  $(`#dg-${axis}0`).value = a;
  $(`#dg-${axis}1`).value = b;
}

function setSweepRange(value) {
  const [a, b, lg] = defaultRange(value);
  $("#sw-from").value = a;
  $("#sw-to").value = b;
  $("#sw-log").checked = lg;
}

// ---------------------------------------------------------------- sweep tab
function progressBar(id) {
  const box = $(id);
  const bar = box.querySelector("div");
  const label = box.querySelector("span");
  const t0 = performance.now();
  box.hidden = false;
  bar.style.setProperty("--p", "0%");
  label.textContent = "starting…";
  return {
    update(done, total) {
      bar.style.setProperty("--p", `${(100 * done) / total}%`);
      const el = (performance.now() - t0) / 1000;
      const eta = done ? (el * (total - done)) / done : 0;
      label.textContent = `${done} / ${total} · ${el.toFixed(0)} s${done < total ? ` · ~${eta.toFixed(0)} s left` : ""}`;
    },
    done() { box.hidden = true; },
  };
}

async function runSweep() {
  const variable = $("#sw-var").value;
  let a = +$("#sw-from").value, b = +$("#sw-to").value;
  const n = Math.max(2, Math.min(401, Math.round(+$("#sw-n").value || 41)));
  const lg = $("#sw-log").checked;
  if (lg && (a <= 0 || b <= 0)) { showAlert("Log spacing needs positive limits."); return; }
  const values = lg ? linspace(Math.log10(a), Math.log10(b), n).map((x) => 10 ** x) : linspace(a, b, n);
  const pb = progressBar("#sw-progress");
  await run($("#sw-run"), async () => {
    const out = await backend.call("sweep", { ...cfg(), variable, values }, pb.update);
    out.log = lg;
    out.label = varLabel(variable);
    lastSweep = out;
    renderSweep();
  });
  pb.done();
}

function renderSweep() {
  const sw = lastSweep;
  if (!sw) return;
  $("#sw-empty").hidden = true;
  $("#sw-out").hidden = false;
  const xs = sw.values;
  const pts = sw.points;
  const bad = pts.filter((p) => !p.converged).length;
  if (bad) showAlert(`${bad} of ${pts.length} points did not converge; they are left out of the plots.`, "warn");
  const xaxis = { title: sw.label, type: sw.log ? "log" : "linear" };
  const get = (f) => pts.map((p) => (p.converged ? f(p) : null));

  const species = new Map();
  for (const p of pts) if (p.converged) for (const [k, m] of Object.entries(p.molality)) species.set(k, Math.max(species.get(k) || 0, m));
  const top = [...species.entries()].filter(([, m]) => m > 1e-14).sort((x, y) => y[1] - x[1]).slice(0, +$("#sw-top").value).map(([k]) => k);
  plot("chart-sw-species", top.map((k) => ({
    x: xs, y: get((p) => log10(p.molality[k])), name: k, mode: "lines", connectgaps: false,
  })), { title: "Aqueous species", xaxis, yaxis: { title: "log<sub>10</sub> molality" } });

  const solidNames = new Set();
  for (const p of pts) if (p.converged) Object.keys(p.solids || {}).forEach((k) => solidNames.add(k));
  const gasNames = new Set();
  for (const p of pts) if (p.converged) Object.keys(p.gases || {}).forEach((k) => gasNames.add(k));
  const solidTraces = [...solidNames].map((k) => ({ x: xs, y: get((p) => p.solids[k]), name: k, mode: "lines" }))
    .concat([...gasNames].map((k) => ({ x: xs, y: get((p) => p.gases[k]), name: k, mode: "lines", line: { dash: "dot" } })));
  plot("chart-sw-solids", solidTraces, { title: "Solids and gas exchanged (mol)", xaxis, yaxis: { title: "mol" } });

  const siNames = new Set();
  for (const p of pts) if (p.converged) Object.keys(p.SI || {}).forEach((k) => siNames.add(k));
  plot("chart-sw-si", [...siNames].map((k) => ({ x: xs, y: get((p) => p.SI[k]), name: k, mode: "lines" })),
    { title: "Saturation indices", xaxis, yaxis: { title: "SI" } });

  const lines = [];
  if (sw.variable !== "pH") lines.push({ x: xs, y: get((p) => p.pH), name: "pH", mode: "lines" });
  if (sw.variable !== "pe" && pts.some((p) => p.pe !== null && p.pe !== undefined)) lines.push({ x: xs, y: get((p) => p.pe), name: "pe", mode: "lines" });
  lines.push({ x: xs, y: get((p) => log10(p.I)), name: "log10 I", mode: "lines", line: { dash: "dash" } });
  plot("chart-sw-ph", lines, { title: "pH, pe and ionic strength", xaxis });
  syncButtons();
}

function sweepCSV() {
  const sw = lastSweep;
  const sp = new Set(), so = new Set(), si = new Set();
  for (const p of sw.points) if (p.converged) {
    Object.keys(p.molality).forEach((k) => sp.add(k));
    Object.keys(p.solids).forEach((k) => so.add(k));
    Object.keys(p.SI).forEach((k) => si.add(k));
  }
  const head = [sw.variable, "converged", "pH", "pe", "I", ...[...sp].map((k) => `m ${k}`), ...[...so].map((k) => `n ${k}`), ...[...si].map((k) => `SI ${k}`)];
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const rows = [head.map(q).join(",")];
  sw.points.forEach((p, i) => {
    const c = p.converged;
    rows.push([sw.values[i], c, c ? p.pH : "", c ? p.pe ?? "" : "", c ? p.I : "",
      ...[...sp].map((k) => (c ? p.molality[k] ?? "" : "")), ...[...so].map((k) => (c ? p.solids[k] ?? "" : "")),
      ...[...si].map((k) => (c ? p.SI[k] ?? "" : ""))].join(","));
  });
  const url = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" }));
  const a = h("a", { href: url, download: "sweep.csv" });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- diagram tab
const PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
  "#bab0ac", "#86bcb6", "#d37295", "#8cd17d", "#b6992d", "#499894", "#fabfd2", "#79706e", "#d4a6c8", "#a0cbe8", "#ffbe7d"];

async function runDiagram() {
  const n = +$("#dg-n").value;
  const x = $("#dg-x").value, y = $("#dg-y").value;
  if (x === y) { showAlert("Choose two different variables."); return; }
  const lin = (a, b) => linspace(+$(a).value, +$(b).value, n);
  const body = { ...cfg(), element: $("#dg-el").value, x, y, x_values: lin("#dg-x0", "#dg-x1"), y_values: lin("#dg-y0", "#dg-y1") };
  const pb = progressBar("#dg-progress");
  await run($("#dg-run"), async () => {
    const out = await backend.call("predominance", body, pb.update);
    out.xlab = varLabel(x); out.ylab = varLabel(y);
    lastDiagram = out;
    renderDiagram();
  });
  pb.done();
}

let lastDiagram = null;
function renderDiagram() {
  const d = lastDiagram;
  if (!d) return;
  $("#dg-empty").hidden = true;
  $("#dg-out").hidden = false;
  const eh = $("#dg-eh").checked && d.y_label === "pe";
  const ys = d.y.map((v) => (eh ? v * d.eh_per_pe : v));
  const used = [...new Set(d.grid.flat().filter((v) => v >= 0))].sort((a, b) => a - b);
  const index = new Map(used.map((v, i) => [v, i]));
  const k = Math.max(used.length, 1);
  const z = d.grid.map((row) => row.map((v) => (v >= 0 ? index.get(v) : null)));
  const text = d.grid.map((row) => row.map((v) => (v >= 0 ? d.labels[v] : "not converged")));
  const colorscale = [];
  used.forEach((v, i) => {
    const c = PALETTE[i % PALETTE.length];
    colorscale.push([i / k, c], [(i + 1) / k, c]);
  });
  if (!colorscale.length) colorscale.push([0, "#ccc"], [1, "#ccc"]);
  const traces = [{
    type: "heatmap", x: d.x, y: ys, z, text, zmin: -0.5, zmax: k - 0.5, colorscale, showscale: false,
    hovertemplate: `%{text}<br>${d.xlab} = %{x:.3g}<br>${eh ? "Eh (V)" : d.ylab} = %{y:.3g}<extra></extra>`,
  }];
  const annotations = used.map((v) => {
    let sx = 0, sy = 0, c = 0;
    d.grid.forEach((row, j) => row.forEach((w, i) => { if (w === v) { sx += d.x[i]; sy += ys[j]; c++; } }));
    const solid = d.phase[d.labels[v]] === "solid";
    return { x: sx / c, y: sy / c, text: solid ? `<b>${d.labels[v]}</b>` : d.labels[v], showarrow: false,
             font: { size: 12, color: "#111" }, bgcolor: "rgba(255,255,255,.55)", borderpad: 2 };
  });
  const shapes = [];
  if ($("#dg-water").checked && d.x_label === "pH" && d.y_label === "pe") {
    const f = eh ? d.eh_per_pe : 1;
    const x0 = d.x[0], x1 = d.x[d.x.length - 1];
    for (const [c0, name] of [[20.78, "O₂/H₂O"], [0, "H₂O/H₂"]]) {
      traces.push({ x: [x0, x1], y: [(c0 - x0) * f, (c0 - x1) * f], mode: "lines", name, hoverinfo: "name",
                    line: { color: "#111", dash: "dash", width: 1.5 }, showlegend: false });
    }
  }
  plot("chart-diagram", traces, {
    title: `${$("#dg-el").value}: ${d.x_label === "pH" && d.y_label === "pe" ? "Pourbaix" : "predominance"} diagram`,
    xaxis: { title: d.xlab, range: [d.x[0], d.x[d.x.length - 1]] },
    yaxis: { title: eh ? "Eh (V)" : d.ylab, range: [ys[0], ys[ys.length - 1]] },
    annotations, shapes,
  });
  const legend = $("#dg-legend");
  legend.innerHTML = "";
  used.forEach((v, i) => legend.append(h("span", {}, h("i", { style: `background:${PALETTE[i % PALETTE.length]}` }),
    d.labels[v] + (d.phase[d.labels[v]] === "solid" ? " (solid)" : ""))));
  if (d.grid.flat().some((v) => v < 0)) legend.append(h("span", { class: "muted" }, "blank: not converged or none of the element"));
}

// ---------------------------------------------------------------- log K tab
async function runLogK() {
  const phases = [...Object.keys(S.solids), ...Object.keys(S.gases)];
  if (!phases.length) { showAlert("Select some phases on the left first."); return; }
  const T = linspace(+$("#lk-t0").value, +$("#lk-t1").value, 61);
  await run($("#lk-run"), async () => {
    const out = await backend.call("logk_table", { ...cfg(), T_C: T, phases });
    plot("chart-logk", Object.entries(out.logK).map(([k, v]) => ({ x: out.T_C, y: v, name: k, mode: "lines" })),
      { title: "log<sub>10</sub> K of dissolution", xaxis: { title: "T (°C)" }, yaxis: { title: "log<sub>10</sub> K" } });
  });
}

// ---------------------------------------------------------------- Python tab
function py(v) {
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(py).join(", ")}]`;
  if (v && typeof v === "object") return `{${Object.entries(v).map(([k, x]) => `${py(k)}: ${py(x)}`).join(", ")}}`;
  if (typeof v === "number") return Number.isInteger(v) ? v.toFixed(1) : String(v);
  return String(v);
}

function renderCode() {
  const c = cfg();
  const phases = [...c.solids, ...c.gases];
  const sysArgs = [py(c.components)];
  if (phases.length) sysArgs.push(`phases=${py(phases)}`);
  if (c.activity) sysArgs.push(`activity=${py(c.activity)}`);
  const eqArgs = [py(c.amounts)];
  if (Object.keys(c.fixed).length) eqArgs.push(`fixed=${py(c.fixed)}`);
  if (c.pH !== undefined) eqArgs.push(`pH=${py(c.pH)}`);
  if (c.pe !== undefined) eqArgs.push(`pe=${py(c.pe)}`);
  if (c.T_C !== 25) eqArgs.push(`T=${py(+(c.T_C + 273.15).toFixed(2))}`);
  if (c.charge_balance) eqArgs.push(`charge_balance=${py(c.charge_balance)}`);
  if (c.method !== "complementarity") eqArgs.push(`method=${py(c.method)}`);
  $("#code").textContent = [
    "from discopt.speciation import PhreeqcDatabase",
    "",
    `db = PhreeqcDatabase.load(${py(c.database)})`,
    `s = db.system(${sysArgs.join(", ")})`,
    `r = s.equilibrate(${eqArgs.join(", ")})`,
    "print(r.summary())",
  ].join("\n");
  const i = backend.info;
  $("#versions").textContent = i ? `This page runs discopt ${i.discopt}, pounce ${i.pounce} and numpy ${i.numpy} in Pyodide ${i.manifest.pyodide_version}.` : "";
}

// ---------------------------------------------------------------- plotting
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function plot(id, traces, layout = {}) {
  if (!window.Plotly) return;
  const text = cssVar("--text"), grid = cssVar("--line");
  const axis = (a = {}) => ({ gridcolor: grid, zerolinecolor: grid, linecolor: grid, ...a,
                               title: typeof a.title === "string" ? { text: a.title } : a.title });
  Plotly.react(id, traces, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: text, size: 12 },
    margin: { l: 60, r: 20, t: 40, b: 50, ...(layout.margin || {}) }, legend: { orientation: "v" },
    ...layout,
    title: layout.title ? { text: layout.title, font: { size: 14 } } : undefined,
    xaxis: axis(layout.xaxis), yaxis: axis(layout.yaxis),
  }, { responsive: true, displaylogo: false });
}

// ---------------------------------------------------------------- tabs, presets, sharing
function selectTab(name) {
  for (const b of $$(".tabs button")) b.classList.toggle("active", b.dataset.tab === name);
  for (const p of $$(".tab-panel")) p.hidden = p.id !== `tab-${name}`;
  if (name === "python") renderCode();
  resizeVisiblePlots();
}

// Plotly's `responsive` only follows window resizes, and hidden tabs keep their old width
function resizeVisiblePlots() {
  if (!window.Plotly) return;
  for (const c of $$(".tab-panel:not([hidden]) .js-plotly-plot")) Plotly.Plots.resize(c);
}
let resizeFrame = 0;
new ResizeObserver(() => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(resizeVisiblePlots);
}).observe(document.querySelector(".results"));

function saveHash() {
  const data = JSON.stringify(S);
  history.replaceState(null, "", "#s=" + btoa(unescape(encodeURIComponent(data))));
  renderCode();
}

function loadHash() {
  const m = location.hash.match(/#s=([^&]+)/);
  if (!m) return null;
  try { return JSON.parse(decodeURIComponent(escape(atob(m[1])))); } catch { return null; }
}

async function applyState(next, { computeNow = true, tab = null } = {}) {
  const oldDb = S.database;
  S = { ...structuredClone(BASE_STATE), ...structuredClone(next) };
  $("#database").value = S.database;
  if (!DB || oldDb !== S.database) await loadDatabase(S.database);
  else { $("#activity").value = S.activity; renderElements(); }
  renderComponents();
  renderConditionOptions();
  await refreshSystem();
  if (tab) selectTab(tab);
  if (computeNow && S.components.length) await compute();
  if (tab) selectTab(tab);
}

async function applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  await applyState(p.state, { tab: p.tab || "speciation" });
  if (p.sweep && variables().some((v) => v.value === p.sweep.var)) {
    $("#sw-var").value = p.sweep.var;
    setSweepRange(p.sweep.var);
    $("#sw-from").value = p.sweep.from;
    $("#sw-to").value = p.sweep.to;
  }
}

// ---------------------------------------------------------------- wiring
function wire() {
  for (const [k, p] of Object.entries(PRESETS)) $("#preset").append(h("option", { value: k }, p.label));
  $("#preset").addEventListener("change", (e) => { if (e.target.value) applyPreset(e.target.value); });
  $("#share").addEventListener("click", async () => {
    saveHash();
    try { await navigator.clipboard.writeText(location.href); $("#share").textContent = "Copied"; }
    catch { $("#share").textContent = "See address bar"; }
    setTimeout(() => ($("#share").textContent = "Copy link"), 1500);
  });
  $("#database").addEventListener("change", async (e) => {
    S.database = e.target.value;
    await run(null, () => loadDatabase(S.database));
    const names = new Set(DB.elements.flatMap((x) => [x.name, ...x.states]));
    S.components = S.components.filter((c) => names.has(c));
    renderElements(); renderComponents(); refreshSystem();
  });
  $("#activity").addEventListener("change", (e) => { S.activity = e.target.value; saveHash(); });
  $("#element-filter").addEventListener("input", renderElements);
  $("#element-filter").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const q = e.target.value.trim();
    const all = DB ? DB.elements.flatMap((x) => [...(x.master ? [x.name] : []), ...x.states]) : [];
    const hit = all.find((k) => k.toLowerCase() === q.toLowerCase());
    if (hit) { toggleComponent(hit); e.target.value = ""; renderElements(); }
  });
  $("#phase-filter").addEventListener("input", renderPhases);
  for (const k of ["T_C", "pH", "pe"]) $("#" + k).addEventListener("input", (e) => { S[k] = e.target.value; saveHash(); });
  for (const k of ["pHMode", "peMode", "method", "charge_balance"]) {
    $("#" + k).addEventListener("change", (e) => { S[k] = e.target.value; renderConditionOptions(); saveHash(); });
  }
  $("#add-fixed").addEventListener("click", () => { S.fixedAq.push({ species: "", value: 0 }); renderFixed(); });
  $("#compute").addEventListener("click", compute);
  $("#show-trace").addEventListener("change", renderSpeciesTable);
  for (const b of $$(".tabs button")) b.addEventListener("click", () => selectTab(b.dataset.tab));
  $("#sw-var").addEventListener("change", (e) => setSweepRange(e.target.value));
  $("#sw-run").addEventListener("click", runSweep);
  $("#sw-top").addEventListener("change", renderSweep);
  $("#sw-csv").addEventListener("click", sweepCSV);
  $("#dg-run").addEventListener("click", runDiagram);
  for (const axis of ["x", "y"]) {
    $(`#dg-${axis}`).addEventListener("change", (e) => {
      e.target.dataset.user = e.target.value;
      setDiagramRange(axis, e.target.value);
    });
  }
  for (const id of ["#dg-eh", "#dg-water"]) $(id).addEventListener("change", renderDiagram);
  $("#lk-run").addEventListener("click", runLogK);
  $("#copy-code").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#code").textContent); $("#copy-code").textContent = "Copied"; } catch { /* ignore */ }
    setTimeout(() => ($("#copy-code").textContent = "Copy"), 1500);
  });
  for (const b of $$(".progress .stop")) b.addEventListener("click", () => backend.restart());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    renderResult(); renderSweep(); renderDiagram();
  });
}

async function main() {
  wire();
  renderComponents();
  watchReady();
  try {
    const info = await backend.ready;
    for (const d of info.databases) $("#database").append(h("option", { value: d }, d));
    const saved = loadHash();
    if (saved) await applyState(saved);
    else { $("#preset").value = "calcite"; await applyPreset("calcite"); }
  } catch (err) {
    console.error(err);
  }
}

main();
