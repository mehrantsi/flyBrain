import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const project = fileURLToPath(new URL('../', import.meta.url));
let token;
try {
  token = execFileSync('security', [
    'find-generic-password', '-s', 'orvena-cloudflare-token', '-w',
  ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
} catch {
  console.error('Cannot read orvena-cloudflare-token from Keychain; no OAuth fallback.');
  process.exit(1);
}
if (!token) throw new Error('Keychain returned an empty Cloudflare token');
if (process.argv.length < 3) throw new Error('Provide a Wrangler command');
const result = spawnSync('wrangler', [...process.argv.slice(2), '--config', 'web/wrangler.jsonc'], {
  cwd: project,
  stdio: 'inherit',
  env: {
    ...process.env,
    CLOUDFLARE_API_TOKEN: token,
    CLOUDFLARE_ACCOUNT_ID: '27f59462492dee86a0c2b9a8929f2e13',
    WRANGLER_SEND_METRICS: 'false',
    WRANGLER_LOG_PATH: `${project}/work/cloudflare/logs`,
  },
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
