/* DeepShield AI — tiny zero-dependency local demo server. */
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8000;
const ROOT = path.resolve(__dirname, 'frontend');
const MIME = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.webp': 'image/webp', '.ico': 'image/x-icon', '.woff2': 'font/woff2',
  '.json': 'application/json', '.mp4': 'video/mp4',
};

function resolvePath(rawUrl) {
  let urlPath;
  try {
    urlPath = decodeURIComponent(String(rawUrl || '/').split('?')[0]);
  } catch {
    return { status: 400, message: '400 Bad Request: malformed URL encoding' };
  }
  if (urlPath === '/') urlPath = '/landing.html';

  // Resolve against the real frontend root and require a path-boundary match;
  // simple startsWith checks can confuse /frontend with /frontend-evil.
  const filePath = path.resolve(ROOT, '.' + urlPath);
  const relative = path.relative(ROOT, filePath);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    return { status: 403, message: 'Forbidden' };
  }
  return { status: 200, urlPath, filePath };
}

function handler(req, res) {
  const target = resolvePath(req.url);
  if (target.status !== 200) {
    res.writeHead(target.status, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
    return res.end(target.message);
  }

  fs.readFile(target.filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
      return res.end('404 Not Found: ' + target.urlPath);
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(target.filePath).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    res.end(data);
  });
}

if (require.main === module) {
  http.createServer(handler).listen(PORT, () => {
    console.log(`DeepShield AI demo running at http://localhost:${PORT}`);
    console.log('Simulated analysis is enabled on this explicit demo port.');
  });
}

module.exports = { resolvePath, handler };
