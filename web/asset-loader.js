export function validateAssetPath(name) {
  if (typeof name !== 'string' || !name || /[\\:%?#]/.test(name)
      || name.split('/').some(part => !part || part === '.' || part === '..')) {
    throw new Error('Invalid manifest asset path');
  }
}

export async function verifyHash(bytes, expectedHash, label) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hash = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== expectedHash) throw new Error(`Asset hash mismatch: ${label}`);
}

export async function fetchFile(url, expectedHash) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (expectedHash) await verifyHash(bytes, expectedHash, url);
  return bytes;
}

export async function fetchManifestFile(baseUrl, name, expectedHash, transport) {
  validateAssetPath(name);
  if (!transport) return fetchFile(`${baseUrl}/${name}`, expectedHash);
  if (!Number.isSafeInteger(transport.bytes) || transport.bytes <= 0
      || !Array.isArray(transport.parts) || !transport.parts.length) {
    throw new Error(`Invalid chunk manifest: ${name}`);
  }
  const bytes = new Uint8Array(transport.bytes);
  let offset = 0;
  for (const part of transport.parts) {
    validateAssetPath(part.path);
    if (!Number.isSafeInteger(part.bytes) || part.bytes <= 0
        || offset + part.bytes > bytes.length || !/^[a-f0-9]{64}$/.test(part.sha256)) {
      throw new Error(`Invalid chunk: ${name}`);
    }
    const chunk = await fetchFile(`${baseUrl}/${part.path}`, part.sha256);
    if (chunk.length !== part.bytes) throw new Error(`Chunk size mismatch: ${part.path}`);
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (offset !== bytes.length) throw new Error(`Incomplete chunked asset: ${name}`);
  await verifyHash(bytes, expectedHash, name);
  return bytes;
}
