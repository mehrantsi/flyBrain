import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { fetchManifestFile, validateAssetPath } from './asset-loader.js';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const content = new Uint8Array([1, 3, 5, 7, 9, 11]);
const parts = [content.slice(0, 3), content.slice(3)];
const transport = {
  bytes: content.length,
  parts: parts.map((bytes, index) => ({ path: `chunks/${index}.bin`, bytes: bytes.length, sha256: hash(bytes) })),
};

test('direct and chunked downloads reconstruct identical verified bytes', async t => {
  t.mock.method(globalThis, 'fetch', async url => new Response(
    url === '/pack/data.npy' ? content : parts[Number(url.match(/(\d+)\.bin$/)[1])],
  ));
  assert.deepEqual(await fetchManifestFile('/pack', 'data.npy', hash(content)), content);
  assert.deepEqual(await fetchManifestFile('/pack', 'data.npy', hash(content), transport), content);
});

test('corrupt chunks, wrong order, truncation, size mismatches and HTTP failures fail closed', async t => {
  let corrupt = false;
  let fail = false;
  t.mock.method(globalThis, 'fetch', async url => {
    if (fail) return new Response('', { status: 404 });
    const chunk = parts[Number(url.match(/(\d+)\.bin$/)[1])].slice();
    if (corrupt) chunk[0] ^= 1;
    return new Response(chunk);
  });
  const load = value => fetchManifestFile('/pack', 'data.npy', hash(content), value);
  await assert.rejects(load({ ...transport, parts: [...transport.parts].reverse() }), /hash mismatch/);
  await assert.rejects(load({ ...transport, parts: transport.parts.slice(0, 1) }), /Incomplete/);
  await assert.rejects(load({ ...transport, parts: [{ ...transport.parts[0], bytes: 2 }] }), /size mismatch/);
  corrupt = true;
  await assert.rejects(load(transport), /hash mismatch/);
  corrupt = false;
  fail = true;
  await assert.rejects(load(transport), /HTTP 404/);
});

test('manifest paths cannot escape the asset directory', () => {
  for (const value of ['', '/etc/passwd', '../secret', 'a/../b', 'https://example.com/a',
    '//example.com/x', 'a\\b', 'a?b', 'a#b', '%2e%2e/secret', './a']) {
    assert.throws(() => validateAssetPath(value), /Invalid manifest asset path/);
  }
  validateAssetPath('chunks/abc/0.bin');
});

test('the staged full connectome retains its original metadata and all four array hashes', async t => {
  const root = new URL('../work/cloudflare/public/', import.meta.url);
  const staged = JSON.parse(await readFile(new URL('pack/manifest.json', root)));
  const original = JSON.parse(await readFile(new URL('../outputs/packs/male_cns_v1/manifest.json', import.meta.url)));
  const { browser_chunks, ...metadata } = staged;
  assert.deepEqual(metadata, original);
  assert.deepEqual(Object.keys(browser_chunks).sort(), ['destinations.npy', 'signed_counts.npy']);
  t.mock.method(globalThis, 'fetch', async url => new Response(await readFile(new URL(url.slice(1), root))));
  for (const [name, expectedHash] of Object.entries(staged.array_sha256)) {
    const reconstructed = await fetchManifestFile('/pack', name, expectedHash, browser_chunks[name]);
    assert.equal(hash(reconstructed), original.array_sha256[name]);
  }
  const build = JSON.parse(await readFile(new URL('build-info.json', root)));
  for (const file of build.files) {
    assert(file.bytes <= 25 * 1024 * 1024, file.path);
    assert(!/(^|\/)(\.env|\.git|\.wrangler|test|perf-test|serve\.mjs)/.test(file.path), file.path);
    assert.equal(hash(await readFile(new URL(file.path, root))), file.sha256, file.path);
  }
});
