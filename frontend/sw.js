// Service worker: makes the resident app installable and opens it instantly.
// Pages and app files are cached; reports (/api) always go to the server.
const CACHE = "wildwatch-v2";
const SHELL = [
  "/app",
  "/static/vendor/leaflet.css",
  "/static/vendor/leaflet.js",
  "/static/vendor/images/marker-icon.png",
  "/static/vendor/images/marker-icon-2x.png",
  "/static/vendor/images/marker-shadow.png",
  "/static/icons/icon-192.png",
];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api")) return;
  // Network first, so updates show straight away; fall back to cache when offline.
  e.respondWith(
    fetch(e.request)
      .then(res => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
