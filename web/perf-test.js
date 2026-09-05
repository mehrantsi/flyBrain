import { createSceneRenderer } from './scene.js';

let worker;
let renderer;
let animation;
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const reportElement = document.querySelector('#report');
document.querySelector('#capabilities').textContent = `WASM stack switching: ${typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function'}`;
function stop() {
  worker?.terminate();
  worker = undefined;
  cancelAnimationFrame(animation);
  renderer?.dispose();
  renderer = undefined;
}
document.querySelector('#stop').onclick = stop;
for (const [id, brain, rendered] of [['cns', true, false], ['rendered', true, true], ['world', false, false]]) {
  document.querySelector(`#${id}`).onclick = () => start(brain, rendered);
}
function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    p50: sorted[Math.floor(sorted.length * 0.5)],
    p95: sorted[Math.floor(sorted.length * 0.95)],
    max: sorted.at(-1),
  };
}
function start(brain, rendered) {
  stop();
  results.textContent = '';
  reportElement.textContent = '';
  const windows = Number(document.querySelector('#windows').value);
  const baseline = document.querySelector('#baseline').checked;
  const vision = rendered && document.querySelector('#vision').checked;
  const timestamps = document.querySelector('#timestamps').checked;
  if (!Number.isInteger(windows) || windows < 100 || windows > 50000) throw new Error('Invalid window count');
  worker = new Worker('./simulation-worker.js', { type: 'module' });
  if (rendered) {
    renderer = createSceneRenderer(document.querySelector('#scene'), {
      onVision: vision ? pixels => worker.postMessage({ type: 'vision', pixels }, [pixels]) : null,
      onError: error => { results.textContent = String(error.stack ?? error); stop(); },
    });
    const render = () => { renderer.render(); animation = requestAnimationFrame(render); };
    render();
  }
  worker.onmessage = async ({ data }) => {
    if (data.type === 'status') status.textContent = data.message;
    if (data.type === 'scene') renderer?.setScene(data.scene);
    if (data.type === 'frame') {
      renderer?.updateFrame(data.poses, data.snapshot);
      status.textContent = `t=${data.snapshot.time_seconds.toFixed(3)} s · ${data.snapshot.realtime_factor.toFixed(3)}×`;
    }
    if (data.type === 'error') { results.textContent = data.message; stop(); }
    if (data.type === 'benchmark') {
      const report = data.report;
      const signatureBytes = new TextEncoder().encode(JSON.stringify([report.final.qpos, report.final.qvel, report.final.total_spikes]));
      const stateSha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', signatureBytes)), b => b.toString(16).padStart(2, '0')).join('');
      let stateFnv32 = 2166136261;
      for (const byte of signatureBytes) stateFnv32 = Math.imul(stateFnv32 ^ byte, 16777619) >>> 0;
      const summary = {
        brain, rendered, vision, timestamps, windows, baseline,
        render_info: renderer ? { render: { ...renderer.renderer.info.render }, memory: { ...renderer.renderer.info.memory }, objects: renderer.objects.length } : null,
        state_sha256: stateSha256, state_fnv32: stateFnv32,
        realtime: report.final.time_seconds / (report.wall_ms / 1000),
        timings_ms: Object.fromEntries(Object.keys(report.samples[0]).map(key => [key, stats(report.samples.map(sample => sample[key]))])),
        final: {
          time_seconds: report.final.time_seconds, total_spikes: report.final.total_spikes,
          root_position: report.final.root_position, flight_mode: report.final.flight_mode,
          physics_warnings: report.final.physics_warnings,
        },
      };
      status.textContent = 'Complete';
      results.textContent = JSON.stringify(summary, null, 2);
      reportElement.textContent = JSON.stringify({ ...report, summary });
      stop();
    }
  };
  worker.postMessage({ type: 'start', brain, benchmark: { windows, baseline, timestamps } });
}
