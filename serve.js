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
    });
    return res.end(body);
  }

  let urlPath;
  try {
    urlPath = decodeURIComponent(rawPath);
  } catch (err) {
    if (err instanceof URIError) {
      res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end('400 Bad Request');
    }
    throw err;
  }

  // Modern Node rejects NUL bytes in filesystem paths. Reject them at the
  // HTTP boundary so malformed input cannot escape the normal error flow.
  if (urlPath.includes('\0')) {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('400 Bad Request');
  }

  if (urlPath === '/') urlPath = '/landing.html';

  const filePath = path.join(ROOT, path.normalize(urlPath));
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('Forbidden');
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end('404 Not Found: ' + urlPath);
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(data);
  });
}).listen(PORT, () => {
  console.log(`DeepShield AI running at http://localhost:${PORT}`);
});
