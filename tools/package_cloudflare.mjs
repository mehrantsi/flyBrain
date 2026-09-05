import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rename, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateAssetPath } from '../web/asset-loader.js';

const project = fileURLToPath(new URL('../', import.meta.url));
const buildRoot = path.join(project, 'work/cloudflare');
const limit = 25 * 1024 * 1024;
const chunkSize = 24 * 1024 * 1024;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
await mkdir(buildRoot, { recursive: true });
const staging = await mkdtemp(path.join(buildRoot, 'staging-'));
const entries = [];

async function emit(name, bytes) {
  validateAssetPath(name);
  if (bytes.length > limit) throw new Error(`Cloudflare asset is too large: ${name}`);
  const destination = path.join(staging, name);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, bytes);
  entries.push({ path: name, bytes: bytes.length, sha256: sha256(bytes) });
}

async function copy(source, destination, expectedHash) {
  const bytes = await readFile(path.join(project, source));
  if (expectedHash && sha256(bytes) !== expectedHash) throw new Error(`Source hash mismatch: ${source}`);
  await emit(destination, bytes);
}

for (const name of [
  'index.html', 'style.css', 'app.js', 'scene.js', 'simulation-worker.js',
  'asset-loader.js', 'neural-engine.js', 'neural.wgsl', 'credits.html', '_headers',
  'dist/flybrain.js', 'dist/flybrain_browser.wasm', 'dist/runtime-assets.json',
  'node_modules/three/build/three.module.js', 'node_modules/three/build/three.core.js',
  'node_modules/three/examples/jsm/controls/OrbitControls.js',
]) await copy(`web/${name}`, name);

const runtime = JSON.parse(await readFile(path.join(project, 'web/dist/runtime-assets.json'), 'utf8'));
for (const [name, hash] of Object.entries(runtime.files)) {
  validateAssetPath(name);
  await copy(`assets/neuromechfly/${name}`, `assets/neuromechfly/${name}`, hash);
}

const packRoot = 'outputs/packs/male_cns_v1';
const pack = JSON.parse(await readFile(path.join(project, packRoot, 'manifest.json'), 'utf8'));
pack.browser_chunks = {};
for (const [name, hash] of Object.entries(pack.array_sha256)) {
  validateAssetPath(name);
  const bytes = await readFile(path.join(project, packRoot, name));
  if (sha256(bytes) !== hash) throw new Error(`Connectome hash mismatch: ${name}`);
  if (bytes.length <= limit) {
    await emit(`pack/${name}`, bytes);
    continue;
  }
  const parts = [];
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    const name = `chunks/${hash}/${parts.length}.bin`;
    await emit(`pack/${name}`, chunk);
    parts.push({ path: name, bytes: chunk.length, sha256: sha256(chunk) });
  }
  pack.browser_chunks[name] = { bytes: bytes.length, parts };
}
await emit('pack/manifest.json', Buffer.from(JSON.stringify(pack, null, 2) + '\n'));

for (const [source, destination] of [
  ['LICENSE', 'licenses/FlyBrain.txt'],
  ...[
    'Drosophila_brain_model.txt', 'iFish.txt', 'CC-BY-4.0.txt',
    'CC-BY-NC-4.0.txt', 'GPL-3.0.txt',
  ].map(name => [`licenses/${name}`, `licenses/${name}`]),
  ['assets/neuromechfly/LICENSE-FLYGYM', 'licenses/FlyGym.txt'],
  ['licenses/Three.js.txt', 'licenses/Three.js.txt'],
  ['vendor/mujoco-rs/LICENSE', 'licenses/mujoco-rs.txt'],
  ['licenses/MuJoCo.txt', 'licenses/MuJoCo.txt'],
  ['licenses/FlyBody.txt', 'licenses/FlyBody.txt'],
  ...[
    ['ccd', 'BSD-LICENSE'], ['lodepng', 'LICENSE'], ['miniz', 'LICENSE'],
    ['qhull', 'COPYING.txt'], ['tinyobjloader', 'LICENSE'], ['tinyxml2', 'LICENSE.txt'],
  ].map(([name, file]) => [
    `work/browser/build-mujoco-3.9.0/_deps/${name}-src/${file}`, `licenses/${name}.txt`,
  ]),
]) await copy(source, destination);

const metadata = JSON.parse(execFileSync('cargo', [
  'metadata', '--locked', '--offline', '--format-version', '1',
  '--filter-platform', 'wasm32-unknown-emscripten',
], { cwd: project, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 }));
const resolved = new Set(metadata.resolve.nodes.map(node => node.id));
const notices = [];
for (const pkg of metadata.packages.filter(pkg => resolved.has(pkg.id))) {
  const directory = path.dirname(pkg.manifest_path);
  const files = (await readdir(directory)).filter(name => /^(LICENSE|LICENCE|COPYING|NOTICE)([.\-_]|$)/i.test(name));
  if (pkg.license_file && !files.includes(pkg.license_file)) files.push(pkg.license_file);
  let notice = `${pkg.name} ${pkg.version}\nLicense: ${pkg.license ?? 'See license text'}\n`;
  if (pkg.repository) notice += `${pkg.repository}\n`;
  for (const file of files) {
    if ((await stat(path.join(directory, file))).isFile()) notice += `\n${await readFile(path.join(directory, file), 'utf8')}\n`;
  }
  notices.push(notice);
}
await emit('licenses/Rust-dependencies.txt', Buffer.from(notices.join('\n----------------------------------------\n\n')));

const report = {
  schema: 'flybrain.cloudflare-static-build.v1',
  files: entries,
  total_bytes: entries.reduce((sum, file) => sum + file.bytes, 0),
  chunked_connectome_files: Object.keys(pack.browser_chunks),
};
await writeFile(path.join(staging, 'build-info.json'), JSON.stringify(report, null, 2) + '\n');
const output = path.join(buildRoot, 'public');
try {
  await stat(output);
  await rename(output, path.join(buildRoot, `previous-${Date.now()}`));
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
await rename(staging, output);
console.log(`Packaged ${entries.length + 1} public files (${(report.total_bytes / 1024 / 1024).toFixed(1)} MiB)`);
console.log(`Connectome transport chunks: ${Object.keys(pack.browser_chunks).join(', ')}`);
console.log(`Output: ${output}`);
