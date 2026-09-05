import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import createFlyBrain from './dist/flybrain.js';

const web = path.dirname(fileURLToPath(import.meta.url));
const project = path.dirname(web);
const Module = await createFlyBrain({ wasmBinary: fs.readFileSync(path.join(web, 'dist/flybrain_browser.wasm')) });
const source = path.join(project, 'assets/neuromechfly');
const manifest = JSON.parse(fs.readFileSync(path.join(source, 'manifest.json')));
for (const name of ['manifest.json', ...Object.keys(manifest.files)]) {
  const destination = `/data/assets/${name}`;
  Module.FS.mkdirTree(path.dirname(destination));
  Module.FS.writeFile(destination, new Uint8Array(fs.readFileSync(path.join(source, name))), { canOwn: true });
}
const response = () => JSON.parse(new TextDecoder().decode(Module.HEAPU8.subarray(Module._fb_response_ptr(), Module._fb_response_ptr() + Module._fb_response_len())));
const invoke = async (name, values = []) => {
  const result = await Module.ccall(name, 'number', values.map(() => 'number'), values, { async: true });
  assert.equal(result, 1, JSON.stringify(response()));
  if (name === 'fb_step') assert.equal(Module._fb_frame(), 1);
  return response();
};
const scene = await invoke('fb_init', [0]);
assert.equal(scene.bodyCount, 71);
assert.equal(scene.cameras.length, 4);
assert.equal(scene.brain.neurons, 0);
assert(scene.geoms.some((geom) => geom.name === 'fly/detail_bristles_c_thorax'));
const frames = [await invoke('fb_frame')];
for (let i = 0; i < 10; i++) frames.push(await invoke('fb_step', [1]));
for (const frame of frames) {
  assert.equal(frame.qpos.length, 133);
  assert.equal(frame.qvel.length, 132);
  assert(frame.qpos.every(Number.isFinite));
  assert(frame.physics_warnings.every((warning) => warning === 0));
}
const reset = await invoke('fb_command', [0]);
assert.equal(reset.time_seconds, 0);
assert.deepEqual(reset.qpos, frames[0].qpos);
Module.HEAPU8.fill(127, Module._fb_vision_ptr(), Module._fb_vision_ptr() + 2 * 450 * 512 * 3);
await invoke('fb_set_vision');
await invoke('fb_update_retina_display');
const display = Module.HEAPU8.subarray(Module._fb_display_ptr(), Module._fb_display_ptr() + 2 * 450 * 512 * 3);
assert(display.some((value) => value !== 0));
assert.deepEqual(display.slice(0, display.length / 2), display.slice(display.length / 2));
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify({ frames }, null, 2) + '\n');
console.log('PASS: real MuJoCo WASM loads the detailed model, advances 10 controller windows, resets deterministically, and samples both retinas. No physics warnings.');
