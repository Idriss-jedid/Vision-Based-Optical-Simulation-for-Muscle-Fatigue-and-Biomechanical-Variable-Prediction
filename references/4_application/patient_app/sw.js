/* VisionBiomech service worker — app shell cache-first, API always network. */
const VERSION = 'vbiomech-v13';
const SHELL = [
  '/',
  '/index.html',
  '/manifest.webmanifest',
  '/css/app.css',
  '/css/fit.css',
  '/js/app.js',
  '/icons/icon.svg',
  '/icons/maskable.svg',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))
  ).then(() => self.clients.claim()));
});

/* Never cache the pipeline / health calls — they must always hit the backend. */
const NETWORK = ['/analyze', '/health'];

/* NETWORK-FIRST: always serve the freshest file from the server; fall back to
   cache only when offline. Avoids stale app.js / index.html during development. */
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  if (NETWORK.some((p) => url.pathname.startsWith(p))) return;

  e.respondWith(
    fetch(e.request).then((res) => {
      if (res && res.status === 200 && url.origin === self.location.origin) {
        const clone = res.clone();
        caches.open(VERSION).then((c) => c.put(e.request, clone));
      }
      return res;
    }).catch(() => caches.match(e.request))
  );
});