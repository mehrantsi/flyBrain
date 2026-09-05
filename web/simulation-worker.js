import createFlyBrain from './dist/flybrain.js';
import { installNeuralEngine } from './neural-engine.js';
import { fetchFile, fetchManifestFile } from './asset-loader.js';

const EYE_BYTES = 450 * 512 * 3;
let Module;
let running = false;
let paused = false;
let starting = false;
let latestVision;
let retinaVisible = false;
let retinaDisplayPending = false;
let commands = [];
let lastFrame = 0;
let clockStarted = 0;
let pausedAt = 0;
let pausedMilliseconds = 0;
let benchmark;
const decoder = new TextDecoder();
const taskChannel = new MessageChannel();
let resumeTask;
taskChannel.port1.onmessage = () => resumeTask();
const yieldTask = () => new Promise((resolve) => {
  resumeTask = resolve;
  taskChannel.port2.postMessage(null);
});

const status = (message) => postMessage({ type: 'status', message });
const response = () => JSON.parse(decoder.decode(Module.HEAPU8.subarray(Module._fb_response_ptr(), Module._fb_response_ptr() + Module._fb_response_len())));

async function call(name, values = []) {
  const result = await Module.ccall(name, 'number', values.map(() => 'number'), values, { async: true });
  if (result !== 1) throw new Error(response().error + (Module.gpuLastError ? `\n${Module.gpuLastError}` : ''));
}

async function loadDirectory(url, destination, field, runtimeAssets = false) {
  const manifestBytes = await fetchFile(runtimeAssets ? './dist/runtime-assets.json' : `${url}/manifest.json`);
  const manifest = JSON.parse(decoder.decode(manifestBytes));
  Module.FS.mkdirTree(destination);
  if (!runtimeAssets) Module.FS.writeFile(`${destination}/manifest.json`, manifestBytes, { canOwn: true });
  const files = Object.entries(manifest[field]);
  let finished = 0;
  const queue = [...files];
  async function consume() {
    while (queue.length) {
      const [name, hash] = queue.shift();
      const bytes = await fetchManifestFile(url, name, hash, manifest.browser_chunks?.[name]);
      const path = `${destination}/${name}`;
      Module.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
      Module.FS.writeFile(path, bytes, { canOwn: true });
      status(`Loading ${destination.endsWith('pack') ? 'connectome' : 'body and room'}: ${++finished}/${files.length} files verified`);
    }
  }
  await Promise.all([consume(), consume()]);
}

function sendFrame() {
  if (Module._fb_frame() !== 1) throw new Error(response().error);
  const snapshot = response();
  snapshot.wall_elapsed_seconds = Math.max(0, ((paused ? pausedAt : performance.now()) - clockStarted - pausedMilliseconds) / 1000);
  snapshot.realtime_factor = snapshot.wall_elapsed_seconds > 0 ? snapshot.time_seconds / snapshot.wall_elapsed_seconds : 0;
  const pointer = Module._fb_poses_ptr() / 4;
  const poses = Module.HEAPF32.slice(pointer, pointer + Module._fb_poses_len()).buffer;
  postMessage({ type: 'frame', poses, snapshot }, [poses]);
  lastFrame = performance.now();
}

async function run() {
  while (running) {
    const iterationStart = performance.now();
    let visionMs = 0;
    for (const command of commands.splice(0)) {
      await call('fb_command', [{ reset: 0, sugar: 1, groom: 2, flight: 3, food: 4 }[command]]);
      if (command === 'reset') resetClock();
      sendFrame();
    }
    if (latestVision) {
      const started = performance.now();
      Module.HEAPU8.set(new Uint8Array(latestVision), Module._fb_vision_ptr());
      latestVision = undefined;
      await call('fb_set_vision');
      retinaDisplayPending = true;
      visionMs = performance.now() - started;
    }
    if (retinaVisible && retinaDisplayPending) {
      if (!benchmark?.baseline) await call('fb_update_retina_display');
      const pointer = Module._fb_display_ptr();
      const pixels = Module.HEAPU8.slice(pointer, pointer + 2 * EYE_BYTES).buffer;
      postMessage({ type: 'retina', pixels }, [pixels]);
      retinaDisplayPending = false;
    }
    if (!paused) {
      const started = performance.now();
      await call('fb_step', [1]);
      const stepMs = performance.now() - started;
      if (benchmark) {
        const metrics = benchmark.baseline
          ? [response().time_seconds, response().brain_wall_seconds, 0, 0]
          : new Float64Array(Module.HEAPU8.buffer, Module._fb_metrics_ptr(), 4);
        benchmark.samples.push({
          step_ms: stepMs, vision_ms: visionMs,
          encode_ms: Module.gpuTiming?.encode_ms ?? 0,
          wait_ms: Module.gpuTiming?.wait_ms ?? 0,
          gpu_ms: Module.gpuTiming?.gpu_ms ?? 0,
          brain_ms: metrics[1] * 1000,
          sensory_ms: metrics[2] * 1000,
          engine_ms: metrics[3] * 1000,
          total_ms: performance.now() - iterationStart,
        });
        if (benchmark.samples.length % 50 === 0) {
          if (Module._fb_frame() !== 1) throw new Error(response().error);
          benchmark.trace.push(response());
        }
        if (benchmark.samples.length === benchmark.windows) {
          sendFrame();
          postMessage({ type: 'benchmark', report: {
            samples: benchmark.samples, trace: benchmark.trace,
            wall_ms: performance.now() - clockStarted,
            final: response(),
          }});
          running = false;
          return;
        }
      }
      if (performance.now() - lastFrame >= 33) sendFrame();
    }
    if (paused) await new Promise((resolve) => setTimeout(resolve, 30));
    else await yieldTask();
  }
}

async function start(brain, profile) {
  if (starting || running) return;
  starting = true;
  benchmark = profile ? { windows: profile.windows, baseline: Boolean(profile.baseline), samples: [], trace: [] } : null;
  status('Loading Rust/MuJoCo WASM…');
  const factory = benchmark?.baseline ? (await import('/baseline/flybrain.js')).default : createFlyBrain;
  Module = await factory({
    locateFile: (name) => benchmark?.baseline ? `/baseline/${name}` : new URL(`./dist/${name}`, import.meta.url).href,
    print: (message) => console.log(message),
    printErr: (message) => console.error(message),
  });
  if (brain) {
    status('Initializing WebGPU neural backend…');
    const install = benchmark?.baseline ? (await import('/baseline/neural-engine.js')).installNeuralEngine : installNeuralEngine;
    await install(Module, { timestamps: Boolean(profile?.timestamps) });
  }
  await loadDirectory('/assets/neuromechfly', '/data/assets', 'files', true);
  if (brain) await loadDirectory('/pack', '/data/pack', 'array_sha256');
  status('Compiling the physical model and settling the articulated body…');
  await call('fb_init', [brain ? 1 : 0]);
  postMessage({ type: 'scene', scene: response() });
  await call('fb_frame');
  resetClock();
  sendFrame();
  starting = false;
  running = true;
  status(brain ? 'Running: Rust + MuJoCo WASM + WebGPU connectome' : 'World-only diagnostic — neural engine disabled');
  await run();
}

function fail(error) {
  running = false;
  starting = false;
  postMessage({ type: 'error', message: String(error?.stack ?? error) });
}

function resetClock() {
  clockStarted = performance.now();
  pausedMilliseconds = 0;
  pausedAt = clockStarted;
}

self.onmessage = ({ data }) => {
  if (data.type === 'start') start(Boolean(data.brain), data.benchmark).catch(fail);
  else if (data.type === 'pause' && paused !== Boolean(data.paused)) {
    if (data.paused) pausedAt = performance.now();
    else pausedMilliseconds += performance.now() - pausedAt;
    paused = Boolean(data.paused);
  }
  else if (data.type === 'vision' && data.pixels?.byteLength === 2 * EYE_BYTES) latestVision = data.pixels;
  else if (data.type === 'retina-display') retinaVisible = Boolean(data.visible);
  else if (data.type === 'command' && ['reset', 'sugar', 'groom', 'flight', 'food'].includes(data.command)) commands.push(data.command);
};
