# Cloudflare deployment

Public URL: https://flybrain.mehran.dk

The `flybrain` Worker serves static assets only. Rust/MuJoCo WASM, the full
MaleCNS WebGPU engine, sensory rendering and the viewer all run in the visitor's
browser. There is no server-side simulation, database, R2 bucket or credential
in the browser build. WebGPU and a sufficiently capable device are required;
the uncompressed deployment is about 164 MiB, mostly connectome data.

## Publish

With the existing WASM build, runtime assets and MaleCNS pack available:

```sh
npm --prefix web run deploy
```

This packages the current build and runs the installed Wrangler CLI. The
wrapper reads `orvena-cloudflare-token` from macOS Keychain and pins the personal
account from `web/wrangler.jsonc`. It fails closed if the token cannot be read;
it never falls back to a potentially unrelated Wrangler OAuth account. The
custom-domain route lets Wrangler manage the hostname and certificate.

Rust/physics changes still require `tools/build_browser.sh` before packaging.
The configuration's compatibility date matches the installed Wrangler 4.80.0
local runtime. Publishing does not upgrade development dependencies.

## Packaging and verification

```sh
npm --prefix web run build:cloudflare
node --test web/asset-loader-test.mjs web/scene-test.mjs
node tools/cloudflare.mjs deploy --dry-run
```

`tools/package_cloudflare.mjs` stages an explicit public-file allowlist in
`work/cloudflare/public`. It includes runtime assets referenced by the hashed
manifest, only the required Three.js modules, data provenance and dependency
licenses. Development servers, test pages, credentials, source workspace files
and unrelated datasets are excluded. Previous bundles are retained as
`work/cloudflare/previous-*` when repackaging.

The two large arrays exceed Cloudflare's 25 MiB static-file limit. Packaging
splits them into at most 24 MiB transport chunks and adds `browser_chunks` only
to the deployed pack manifest. Original local arrays and metadata are not
modified. The browser verifies each chunk, reconstructs the original array,
then verifies its original SHA-256 before passing it to Rust. Local development
continues to use whole files through the same loader; download failures never
silently switch transport or skip integrity checks.

All resources are same-origin, with COOP/COEP isolation headers and cache
revalidation. Missing files return 404, not an HTML SPA fallback. No special
server code runs on asset requests. `_headers` supplies security headers and
Cloudflare supplies the WASM MIME type.

For deployment history:

```sh
node tools/cloudflare.mjs deployments list
```

Before publishing, the chunk loader is tested for corruption, reordering,
truncation, path traversal and exact reconstruction of all four real arrays.
The packaged site is also verified with the full neural engine under Wrangler's
local server. After publishing, verify the actual HTTPS host, asset headers,
download/hash checks and advancing full-CNS telemetry in a WebGPU browser.

## Attribution and licenses

The package includes the browser credits page and retained upstream license texts
from `licenses/`, including the original Shiu model, iFish, the Creative Commons
data licenses, and the separately licensed FlyBody flight dataset. See the
[repository's source history](../REFERENCES.md) and
[third-party inventory](../THIRD_PARTY_NOTICES.md) for attribution, modifications,
and distribution caveats. Editing these files does not itself publish a new build.
