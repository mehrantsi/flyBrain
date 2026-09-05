import http from 'node:http';
import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const web = path.dirname(fileURLToPath(import.meta.url));
const project = path.dirname(web);
const port = Number(process.env.FLYBRAIN_PORT ?? 8080);
const routes = [
  ['/baseline/', path.join(project, 'outputs/performance/browser-baseline')],
  ['/assets/neuromechfly/', path.join(project, 'assets/neuromechfly')],
  ['/pack/', path.join(project, 'outputs/packs/male_cns_v1')],
  ['/fixtures/', path.join(project, 'fixtures')],
  ['/', web],
];
const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png', '.wgsl': 'text/plain' };

http.createServer(async (request, response) => {
  try {
    if (!['GET', 'HEAD'].includes(request.method)) {
      response.writeHead(405).end();
      return;
    }
    const url = new URL(request.url, 'http://localhost');
    const pathname = decodeURIComponent(url.pathname);
    const [prefix, root] = routes.find(([prefix]) => pathname.startsWith(prefix));
    const relative = pathname.slice(prefix.length) || 'index.html';
    const target = await realpath(path.resolve(root, relative));
    if (!target.startsWith(`${root}${path.sep}`)) {
      response.writeHead(403).end();
      return;
    }
    const info = await stat(target);
    if (!info.isFile()) { response.writeHead(404).end(); return; }
    response.writeHead(200, {
      'Content-Type': mime[path.extname(target)] ?? 'application/octet-stream',
      'Content-Length': info.size,
      'Cache-Control': 'no-cache',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'X-Content-Type-Options': 'nosniff',
    });
    if (request.method === 'HEAD') response.end();
    else createReadStream(target).pipe(response);
  } catch (error) {
    response.writeHead(error.code === 'ENOENT' ? 404 : 500).end('File unavailable');
  }
}).listen(port, '127.0.0.1', () => console.log(`FlyBrain browser: http://localhost:${port}`));
