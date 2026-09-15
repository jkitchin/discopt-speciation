// Web Worker: runs Python (Pyodide) with discopt, pounce and discopt.speciation.
//
// Messages in:  {id, fn, arg, progress}  -> calls web_backend.<fn>(arg[, progress])
// Messages out: {type: "status", text} | {type: "ready", info} | {type: "error", text}
//               {id, result} | {id, error} | {id, progress: [done, total]}

const BASE = new URL("./", self.location.href).href;
let backend = null;

function status(text) {
  self.postMessage({ type: "status", text });
}

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r;
}

async function init() {
  const manifest = await (await fetchOk(BASE + "py/manifest.json")).json();
  status(`Loading Python (Pyodide ${manifest.pyodide_version})…`);
  importScripts(`https://cdn.jsdelivr.net/pyodide/v${manifest.pyodide_version}/full/pyodide.js`);
  const py = await loadPyodide();

  status("Loading numpy, scipy and pydantic…");
  await py.loadPackage(["numpy", "scipy", "pyyaml", "pydantic", "micropip"]);

  status(`Loading the pounce solver ${manifest.pounce_version}…`);
  const micropip = py.pyimport("micropip");
  await micropip.install.callKwargs(BASE + "py/" + manifest.pounce_wheel, { deps: false });

  status(`Loading discopt ${manifest.discopt_version} and discopt.speciation…`);
  const site = py.runPython("import site; site.getsitepackages()[0]");
  for (const name of manifest.archives) {
    const buf = await (await fetchOk(BASE + "py/" + name)).arrayBuffer();
    py.unpackArchive(buf, "zip", { extractDir: site });
  }
  const src = await (await fetchOk(BASE + "web_backend.py")).text();
  py.FS.mkdirTree("/home/pyodide/app");
  py.FS.writeFile("/home/pyodide/app/web_backend.py", src);
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");

  status("Importing the package…");
  backend = py.pyimport("web_backend");
  const info = JSON.parse(backend.ready());
  self.postMessage({ type: "ready", info: { ...info, manifest } });
}

const started = init().catch((err) => {
  self.postMessage({ type: "error", text: String(err && err.message ? err.message : err) });
  throw err;
});

self.onmessage = async (event) => {
  const { id, fn, arg, progress } = event.data;
  try {
    await started;
    const f = backend[fn];
    let last = 0;
    const report = (done, total) => {
      const now = Date.now();
      if (done === total || now - last > 100) {
        last = now;
        self.postMessage({ id, progress: [done, total] });
      }
    };
    const out = arg === undefined ? f() : progress ? f(arg, report) : f(arg);
    f.destroy();
    self.postMessage({ id, result: JSON.parse(out) });
  } catch (err) {
    let text = String(err && err.message ? err.message : err);
    // keep the last line of a Python traceback, which names the error
    const lines = text.trim().split("\n");
    self.postMessage({ id, error: lines[lines.length - 1], detail: text });
  }
};
