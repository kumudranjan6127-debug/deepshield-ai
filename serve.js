/* ============================================================
   DeepShield AI — tiny local dev server (zero dependencies)
   Usage: node serve.js  →  http://localhost:8000
   ============================================================ */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8000;
const ROOT = path.join(__dirname, 'frontend');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.json': 'application/json',
  '.mp4': 'video/mp4',
};

const TEXT_HEADERS = {
  'Content-Type': 'text/plain; charset=utf-8',
  'Cache-Control': 'no-store',
  'X-Content-Type-Options': 'nosniff',
};

const DEMO_HEALTH = {
  ok: true,
  status: 'healthy',
  engine: 'echo',
  model: {
    model_name: 'DeepShield',
    architecture: '—',
    version: '—',
    runtime: 'simulated',
    input_size: null,
    name: 'MobileNetV3',
    params: '—',
    input: '—',
    backend: 'none',
    device: 'CPU',
  },
  certainty_bands: [],
  calibrated: false,
};

http.createServer((req, res) => {
  const rawPath = String(req.url || '/').split('?')[0];

  // The static server is an intentional demo environment. Advertise that
  // explicitly so the frontend can distinguish it from a production backend
  // that is simply unreachable.
  if (rawPath === '/api/health') {
    const body = JSON.stringify(DEMO_HEALTH);
    res.writeHead(200, {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    return res.end(body);
  }

  let urlPath;
  try {
    urlPath = decodeURIComponent(rawPath);
  } catch (err) {
    if (err instanceof URIError) {
      res.writeHead(400, TEXT_HEADERS);
      return res.end('400 Bad Request');
    }
    throw err;
  }

  // Modern Node rejects NUL bytes in filesystem paths. Reject them at the
  // HTTP boundary so malformed input cannot escape the normal error flow.
  if (urlPath.includes('\0')) {
    res.writeHead(400, TEXT_HEADERS);
    return res.end('400 Bad Request');
  }

  if (urlPath === '/') urlPath = '/landing.html';

  const filePath = path.resolve(ROOT, `.${urlPath}`);
  const insideRoot = filePath === ROOT || filePath.startsWith(`${ROOT}${path.sep}`);
  if (!insideRoot) {
    res.writeHead(403, TEXT_HEADERS);
    return res.end('Forbidden');
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, TEXT_HEADERS);
      return res.end('404 Not Found: ' + urlPath);
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    res.end(data);
  });
}).listen(PORT, () => {
  console.log(`DeepShield AI running at http://localhost:${PORT}`);
});
